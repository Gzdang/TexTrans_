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
from mesh.unet.skip import skip


class TexturedMeshModel(nn.Module):
    def __init__(
        self,
        opt: OmegaConf,
        render_grid_size=512,
        texture_resolution=1024,
        initial_texture_path=None,
        cache_path=None,
        device=torch.device("cpu"),
        augmentations=False,
        augment_prob=0.5,
    ):

        super().__init__()
        self.device = device
        self.augmentations = augmentations
        self.augment_prob = augment_prob
        self.opt = opt
        self.dy = self.opt.dy
        self.mesh_scale = self.opt.shape_scale
        self.texture_resolution = texture_resolution
        if initial_texture_path is not None:
            self.initial_texture_path = initial_texture_path
        else:
            self.initial_texture_path = self.opt.initial_texture
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
        self.texture_map = self.init_textures()

        self.texture_seed = torch.randn(1, 3, texture_resolution, texture_resolution).to(self.device)
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
            need_sigmoid=True,
            need_bias=True,
            pad="reflection",
            act_fun="LeakyReLU",
        ).to(self.device)

        # self.texture_img = self.init_paint()

        # 初始化模型uv坐标
        self.vt, self.ft = self.init_texture_map()
        self.face_attributes = self.mesh.face_uv_matrix
        self.xyz_attributes = self.mesh.face_xyz_matrix

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
                .cuda()
                .unsqueeze(0)
                / 255.0
            )

        else:
            texture = torch.ones(1, self.num_features, self.texture_resolution, self.texture_resolution).cuda()
            # texture = torch.randn(1, self.num_features, self.texture_resolution, self.texture_resolution).cuda()

        return nn.Parameter(texture)

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
            vt = self.mesh.vertex_uvs.cuda()
            ft = self.mesh.face_uvs.cuda()
        elif cache_exists_flag:
            # logger.info(f'running cached UV maps from {vt_cache}')
            vt = torch.load(vt_cache).cuda()
            ft = torch.load(ft_cache).cuda()
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

            vt = torch.from_numpy(vt_np.astype(np.float32)).float().cuda()
            ft = torch.from_numpy(ft_np.astype(np.int64)).int().cuda()
            if cache_path is not None:
                os.makedirs(cache_path, exist_ok=True)
                torch.save(vt.cpu(), vt_cache)
                torch.save(ft.cpu(), ft_cache)
        return vt, ft

    def forward(self, x):
        raise NotImplementedError

    def get_params(self):
        return self.texture_unet.parameters()
    
    def get_texture(self):
        return self.texture_unet(self.texture_seed)

    def render(
        self,
        theta=None,
        phi=None,
        radius=None,
        background=None,
        dim=512,
    ):
        augmented_vertices = self.mesh.vertices

        texture_img = self.texture_unet(self.texture_seed)
        render_cache = self.render_cache

        images, mask, depth, normals, render_cache = self.renderer.render_multi_view_texture(
            augmented_vertices,
            self.mesh.faces,
            self.face_attributes,
            texture_img,
            elev=theta,
            azim=phi,
            radius=radius,
            look_at_height=self.dy,
            render_cache=render_cache,
            dims=(dim, dim),
        )

        self.render_cache = render_cache
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
        # elev_list = [t*np.pi for t in (1/6, 1/6, 1/6, 5/12, 5/12, 5/12, 5/6, 5/6, 5/6)]
        # azim_list = [t*np.pi for t in (5/3, 0, 2/3, 5/3, 0, 2/3, 5/3, 0, 2/3)]
        elev_list = [t * np.pi for t in (1 / 4, 1 / 4, 1 / 4, 1 / 2, 1 / 2, 1 / 2, 3 / 4, 3 / 4, 3 / 4)]
        azim_list = [t * np.pi for t in (5 / 3, 0, 2 / 3, 5 / 3, 0, 2 / 3, 5 / 3, 0, 2 / 3)]

        res = self.render(elev_list, azim_list, 3, dim=341)
        img_res = self.reshape_image(res["image"])
        mask_res = self.reshape_image(res["mask"])
        return img_res, mask_res
