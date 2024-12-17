if __name__ == "__main__":
    import sys
    sys.path.append("./")

import math
import os
import numpy as np
import torch
import json
import PIL.Image as Image

from mesh.mesh import Mesh
from mesh.render import Render

from torchvision.utils import save_image


class RenderHandle:
    def __init__(self, dataset_path, cls_name, n_c, render_size=1024, image_size=256, is_load_mesh=False):
        self.dataset_path = dataset_path
        mesh_dict_file = os.path.join(self.dataset_path, "model_info.json")
        assert os.path.exists(mesh_dict_file), "model_info.json not found"

        with open(mesh_dict_file, "r") as f:
            mesh_dict = json.loads(f.read())
        mesh_info = []
        for mesh in mesh_dict:
            if mesh["category"] == cls_name:
                mesh_info.append(mesh)

        self.mesh_path_list = []
        for cur_mesh in mesh_info:
            mesh_path = os.path.join(self.dataset_path, cur_mesh["model_id"])
            if os.path.exists(mesh_path):
                self.mesh_path_list.append(mesh_path)

        self.render = Render(
            render_size=(render_size, render_size),
            image_size=(image_size, image_size),
            interpolation_mode="bilinear",
        )

        assert n_c in [9, 6, 4], "num of camera is not defined"
        self.n_c = n_c
        if n_c == 9:
            self.elev_list = [t*np.pi for t in (1/4, 1/4, 1/4, 1/2, 1/2, 1/2, 3/4, 3/4, 3/4)]
            self.azim_list = [t*np.pi for t in (5/3, 1, 1/3, 4/3, 0, 2/3, 5/3, 1, 1/3)]
        elif n_c == 6:
            self.elev_list = [t*np.pi for t in (1/4, 1/4, 1/4, 1/2, 1/2, 1/2, 3/4, 3/4, 3/4)]
            self.azim_list = [t*np.pi for t in (5/3, 1, 1/3, 4/3, 0, 2/3, 5/3, 1, 1/3)]
        elif n_c == 4:
            self.elev_list = [t*np.pi for t in (1/2, 1/2, 1/2, 1/2)]
            self.azim_list = [t*np.pi for t in (0, 1/2, 1, 3/2)]
        
        self.camera_list = list(zip(self.elev_list, self.azim_list))

        self.is_load_mesh = is_load_mesh
        if is_load_mesh:
            self.load_mesh()
        
    def load_mesh(self):
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
        mesh_index = 0
        os.makedirs(os.path.join(out_path, "preview"), exist_ok=True)
        for mesh_path in self.mesh_path_list:
            try:
                mesh = Mesh(os.path.join(mesh_path, "normalized_model.obj"))
            except:
                print(f"load mesh error: {mesh_path}")
                continue
            texture = self.load_texture_map(os.path.join(mesh_path, "texture.png"))
            view_list = self.camera_list
            res_rgb_list = []
            res_depth_list = []
            res_normal_list = []
            for index, view in enumerate(view_list):
                camera_view = self.render.get_camera_from_view(view[0], view[1])
                render = self.render(mesh.to("cuda"), texture.cuda(), camera_view.cuda())
                mesh.cpu()
                render_out_path = os.path.join(out_path, f"{mesh_index}/render_images")
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

            save_image(torch.cat(res_rgb_list), os.path.join(render_out_path, f"all_rgb.png"), nrow=math.floor(self.n_c**0.5), padding=0)
            save_image(torch.cat(res_depth_list), os.path.join(render_out_path, f"all_depth.png"), nrow=math.floor(self.n_c**0.5), padding=0)
            save_image(torch.cat(res_normal_list), os.path.join(render_out_path, f"all_normal.png"), nrow=math.floor(self.n_c**0.5), padding=0)
            os.system(f"cp {mesh_path}/* {out_path}/{mesh_index}/")
            os.system(f"cp {mesh_path}/image.jpg {out_path}/preview/{mesh_index}.jpg")
            mesh_index += 1

if __name__ == "__main__":
    cls_name = "Lazy Sofa"
    handler = RenderHandle(f"resource/dataset/3D-FUTURE-model", cls_name, 9, render_size=512, image_size=512, is_load_mesh=False)
    
    sub_folder = cls_name.replace(' ', '_')
    sub_folder = sub_folder.replace('-', '_')
    handler.render_images(f"./dataset/future/{sub_folder.lower()}")
