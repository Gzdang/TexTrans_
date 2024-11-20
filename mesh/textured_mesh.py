import os

import kaolin as kal
import numpy as np
import xatlas
import torch
import torch.nn as nn

from omegaconf import OmegaConf
from PIL import Image

from mesh.mesh import Mesh
from mesh.render import Render
from mesh.unet.lipis import LPIPS
from mesh.unet.skip import skip
from torchvision.utils import save_image

class TexturedMeshModel(nn.Module):
    def __init__(
        self,
        opt: OmegaConf,
        render_grid_size,
        initial_texture_path=None,
        cache_path=None,
        device=torch.device("cpu"),
        augmentations=False,
        augment_prob=0.5,
        use_unet=True
    ):

        super().__init__()
        self.render_size = render_grid_size
        self.device = device
        self.augmentations = augmentations
        self.augment_prob = augment_prob
        self.opt = opt
        self.dy = self.opt.dy
        self.mesh_scale = self.opt.shape_scale
        self.texture_resolution = self.opt.texture_resolution
        self.use_unet = use_unet
        self.initial_texture_path = initial_texture_path
        self.cache_path = cache_path

        # uv特征数量
        self.num_features = 3

        self.renderer = Render(
            device=self.device,
            render_size=(render_grid_size, render_grid_size),
            interpolation_mode=self.opt.texture_interpolation_mode,
        )
        self.render_cache = None

        self.mesh = self.init_meshes()

        # uv其他部分的默认颜色
        # self.default_color = [0.8, 0.1, 0.8]

        # 初始化纹理图
        if use_unet:
            self.texture_seed = torch.randn(1, 3, self.texture_resolution, self.texture_resolution).to(self.device)
            self.texture_unet = skip(
                3,
                3,
                num_channels_down=[128] * 5,
                num_channels_up=[128] * 5,
                num_channels_skip=[128] * 5,
                filter_size_up=3,
                filter_size_down=3,
                upsample_mode="nearest",
                filter_skip_size=1,
                need_sigmoid=False,
                # need_tanh=True,
                need_bias=True,
                pad="reflection",
                act_fun="LeakyReLU",
            ).to(self.device)
            if "texture_unet_path" in self.opt:
                state_dict = torch.load(self.opt.texture_unet_path, weights_only=True, map_location=torch.device(self.device))
                self.texture_seed = state_dict["seed"]
                self.texture_unet.load_state_dict(state_dict["unet"])
            self.texture_map = None
        else:
            self.init_textures()

        # 初始化模型uv坐标
        self.vt, self.ft = self.init_texture_map()
        self.face_attributes = self.mesh.face_uv_matrix
        self.xyz_attributes = self.mesh.face_xyz_matrix

        self.perceptual_loss = LPIPS(True).cuda().eval()

    def init_meshes(self):
        mesh = Mesh(self.opt.shape_path, self.device)
        return mesh

    def init_textures(self):
        if self.initial_texture_path is not None:
            texture = (
                torch.Tensor(
                    np.array(
                        Image.open(self.initial_texture_path)
                        .convert("RGB")
                        .resize((self.texture_resolution, self.texture_resolution))
                    )
                )
                .permute(2, 0, 1)
                .to(self.device)
                .unsqueeze(0)
                / 255.0
            )

        else:
            texture = torch.ones(1, self.num_features, self.texture_resolution, self.texture_resolution).to(self.device)
            # texture = torch.randn(1, self.num_features, self.texture_resolution, self.texture_resolution).to(self.device)

        self.texture_map = nn.Parameter(texture)

    def init_texture_map(self):
        cache_path = self.cache_path
        if cache_path is None:
            cache_exists_flag = False
        else:
            vt_cache, ft_cache = cache_path / "vt.pth", cache_path / "ft.pth"
            cache_exists_flag = vt_cache.exists() and ft_cache.exists()

        if (
            self.mesh.vertex_uvs is not None
            and self.mesh.face_uvs is not None
            and self.mesh.vertex_uvs.shape[0] > 0
            and self.mesh.face_uvs.min() > -1
        ):
            # logger.info('Mesh includes UV map')
            vt = self.mesh.vertex_uvs.to(self.device)
            ft = self.mesh.face_uvs.to(self.device)
        elif cache_exists_flag:
            # logger.info(f'running cached UV maps from {vt_cache}')
            vt = torch.load(vt_cache).to(self.device)
            ft = torch.load(ft_cache).to(self.device)
        else:
            # logger.info(f'running xatlas to unwrap UVs for mesh')
            # unwrap uvs
            v_np = self.mesh.vertices.cpu().numpy()
            f_np = self.mesh.faces.int().cpu().numpy()
            atlas = xatlas.Atlas()
            atlas.add_mesh(v_np, f_np)
            chart_options = xatlas.ChartOptions()
            chart_options.max_iterations = 4
            atlas.generate(chart_options=chart_options)
            vmapping, ft_np, vt_np = atlas[0]  # [N], [M, 3], [N, 2]

            vt = torch.from_numpy(vt_np.astype(np.float32)).float().to(self.device)
            ft = torch.from_numpy(ft_np.astype(np.int64)).int().to(self.device)
            if cache_path is not None:
                os.makedirs(cache_path, exist_ok=True)
                torch.save(vt.cpu(), vt_cache)
                torch.save(ft.cpu(), ft_cache)
        return vt, ft

    def get_last_layer(self):
        return self.texture_unet[-1][-1].weight
    
    def forward(self, x):
        raise NotImplementedError

    def get_params(self):
        return self.texture_unet.parameters()
    
    def get_texture(self):
        if self.texture_map is not None:
            return self.texture_map
        return self.texture_unet(self.texture_seed)
    
    def save_texture_unet(self, out_path):
        state_dict = {
            "seed": self.texture_seed,
            "unet": self.texture_unet.state_dict()
        }
        torch.save(state_dict, out_path)

    def render(
        self,
        theta=None,
        phi=None,
        radius=None,
        background=None,
        dim=512,
    ):
        augmented_vertices = self.mesh.vertices

        if self.use_unet:
            texture_img = self.texture_unet(self.texture_seed)
            self.texture_map = texture_img.detach()
        else:
            texture_img = self.texture_map

        # render_cache = self.render_cache
        render_cache = None

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=False):
            images, mask, depth, normals, render_cache = self.renderer.render_multi_view_texture(
                augmented_vertices,
                self.mesh.faces,
                self.face_attributes,
                texture_img.float(),
                elev=theta,
                azim=phi,
                radius=radius,
                look_at_height=self.dy,
                render_cache=render_cache,
                dims=(dim, dim),
            )
        # self.render_cache = render_cache
        mask = mask.detach()

        if background != None:
            if background == "white":
                background = torch.ones_like(images)
            elif background == "black":
                background = torch.zeros_like(images)
            else:
                background = background
            images = background * (1 - mask) + images * mask

        return {
            "image": images,
            "mask": mask,
            "depth": depth,
            "normals": normals,
            "render_cache": render_cache,
            "texture_map": texture_img,
        }

    def reshape_image(self, image_batch):
        image_row = image_batch.chunk(3)
        cols = []
        for row in image_row:
            cols.append(torch.cat(row.chunk(3), -1))
        return torch.cat(cols, -2)

    def render_all(self):
        # elev_list = [t*np.pi for t in (1/4, 1/4, 1/4, 1/2, 1/2, 1/2, 3/4, 3/4, 3/4)]
        # azim_list = [t*np.pi for t in (5/3, 1, 1/3, 4/3, 0, 2/3, 5/3, 1, 1/3)]

        elev_list = [t*np.pi for t in (1/4, 1/4, 1/4, 1/2, 1/2, 1/2, 3/4, 3/4, 3/4)]
        azim_list = [t*np.pi for t in (5/3, 1, 1/3, 4/3, 0, 2/3, 5/3, 1, 1/3)]
        
        res = self.render(elev_list, azim_list, 3, dim=self.render_size)
        img_res = self.reshape_image(res["image"])
        mask_res = self.reshape_image(res["mask"])
        return img_res, mask_res
    
    def project_all(self, target):
        # self.init_texture_map()
        optimizer = torch.optim.Adam(self.parameters(), 1e-4)
        scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500, 700], gamma=0.5)
        for _ in range(800):
            image, _ = self.render_all()
            loss = torch.nn.functional.l1_loss(image, target.detach())
            loss += self.perceptual_loss(target, image)[0][0][0][0]
            # print(loss, end="\r")
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        # print()
        res, _ = self.render_all()
        return res.detach()
    
    def project(self, target, elev, azim):
        optimizer = torch.optim.Adam(self.parameters(), 1e-4)
        for _ in range(200):
            res = self.render([elev], [azim], 3, dim=self.render_size)
            loss = torch.nn.functional.l1_loss(res["image"], target.detach())
            # loss = torch.nn.functional.l1_loss(res["image"], target.detach())
            # print(loss, end="\r")
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # print()
        res = self.render([elev], [azim], 3, dim=self.render_size)["image"]
        return res.detach()

