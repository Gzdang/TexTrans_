if __name__ == "__main__":
    import sys
    sys.path.append("./")

import os
import numpy as np
import torch
import PIL.Image as Image

from mesh.mesh import Mesh
from mesh.render import Render

from torchvision.utils import save_image


class RenderHandle:
    def __init__(self, dataset_path, mesh_type, render_size=1024, image_size=256, is_load_mesh=False):
        self.dataset_path = dataset_path
        self.obj_path = os.path.join(self.dataset_path, "obj")
        self.textures_path = os.path.join(self.dataset_path, "texture")
        self.obj_list_path = os.path.join(self.dataset_path, "split", f"{mesh_type}.txt")

        self.render = Render(
            render_size=(render_size, render_size),
            image_size=(image_size, image_size),
            interpolation_mode="bilinear",
        )

        self.mesh_path_list = []
        self.texture_path_list = []
        f = open(self.obj_list_path, "r")
        for obj_name in f.readlines():
            obj_name = obj_name.strip()
            self.mesh_path_list.append(os.path.join(self.obj_path, obj_name + ".obj"))
            self.texture_path_list.append(os.path.join(self.textures_path, obj_name + ".png"))

        self.mesh_num = len(self.mesh_path_list)
        self.mesh_combination = self.mesh_num * (self.mesh_num - 1)
        self.elev_list = [t*np.pi for t in (1/4, 1/4, 1/4, 1/2, 1/2, 1/2, 3/4, 3/4, 3/4)]
        self.azim_list = [t*np.pi for t in (5/3, 0, 2/3, 5/3, 0, 2/3, 5/3, 0, 2/3)]
        # self.elev_list = [t*np.pi for t in (1/6, 1/6, 1/6, 5/12, 5/12, 5/12, 5/6, 5/6, 5/6)]
        # self.azim_list = [t*np.pi for t in (5/3, 0, 2/3, 5/3, 0, 2/3, 5/3, 0, 2/3)]
        self.camera_list = list(zip(self.elev_list, self.azim_list))
        self.camera_num = len(self.elev_list)
        self.camera_combination = self.camera_num * (self.camera_num - 1)

        if is_load_mesh:
            self.load_mesh()
        self.is_load_mesh = is_load_mesh

    def load_mesh(self):
        self.is_load_mesh = True
        self.mesh_list = []
        for mesh_path in self.mesh_path_list:
            self.mesh_list.append(Mesh(mesh_path))

    def load_texture_map(self, texture_path):
        texture = torch.tensor(
            np.array(Image.open(texture_path).convert("RGB")) / 255.0,
            dtype=torch.float32,
        )
        texture = texture.permute(2, 0, 1).unsqueeze(0)
        # texture_img = nn.Parameter(texture)
        return texture

    def render_images(self, out_path):
        for mesh_index, mesh_path in enumerate(self.mesh_path_list):
            mesh = Mesh(mesh_path)
            texture = self.load_texture_map(self.texture_path_list[mesh_index])
            view_list = self.camera_list
            res_rgb_list = []
            res_depth_list = []
            res_normal_list = []
            for index, view in enumerate(view_list):
                camera_view = self.render.get_camera_from_view(view[0], view[1])
                render = self.render(mesh.to("cuda"), texture.cuda(), camera_view.cuda())
                mesh.cpu()
                render_out_path = os.path.join(out_path, f"{mesh_index}")
                if not os.path.exists(render_out_path):
                    os.makedirs(render_out_path, exist_ok=True)
                
                render_rgb_path = os.path.join(render_out_path, f"{index}_rgb.png")
                render_depth_path = os.path.join(render_out_path, f"{index}_depth.png")
                render_normal_path = os.path.join(render_out_path, f"{index}_normal.png")
                render_mask_path = os.path.join(render_out_path, f"{index}_mask.png")
                render_xyz_path = os.path.join(render_out_path, f"{index}_xyz.png")

                res_rgb_list.append(render["image"].cpu())
                res_depth_list.append(render["depth"].cpu())
                res_normal_list.append(render["normal"].cpu())
                save_image(render["image"], render_rgb_path)
                save_image(render["depth"], render_depth_path)
                save_image(render["normal"], render_normal_path)
                save_image(render["mask"], render_mask_path)
                # save_image(render["xyz"].unsqueeze().cpu(), render_xyz_path)
            save_image(torch.cat(res_rgb_list), os.path.join(render_out_path, f"all_rgb.png"), nrow=3, padding=0)
            save_image(torch.cat(res_depth_list), os.path.join(render_out_path, f"all_depth.png"), nrow=3, padding=0)
            save_image(torch.cat(res_normal_list), os.path.join(render_out_path, f"all_normal.png"), nrow=3, padding=0)

if __name__ == "__main__":
    handler = RenderHandle(f"{os.environ['HOME']}/dataset/3D_Future", "chair", render_size=341, image_size=341, is_load_mesh=False)
    handler.render_images("./dataset/new")
