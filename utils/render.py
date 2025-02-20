import os
import math
import torch
import numpy as np
import PIL.Image as Image
from torchvision.utils import save_image

from mesh.mesh import Mesh
from mesh.render import Render

def get_camera_list(n_c=9):
    assert n_c in [9, 6, 4], "num of camera is not defined"
    if n_c == 9:
        elev_list = [t*np.pi for t in (1/4, 1/4, 1/4, 1/2, 1/2, 1/2, 3/4, 3/4, 3/4)]
        azim_list = [t*np.pi for t in (5/3, 1, 1/3, 4/3, 0, 2/3, 5/3, 1, 1/3)]
    elif n_c == 6:
        elev_list = [t*np.pi for t in (1/3, 11/18, 1/3, 11/18, 1/3, 11/18,)]
        azim_list = [t*np.pi for t in (30/180, 90/180, 150/180, 210/180, 270/180, 330/180)]
    elif n_c == 4:
        elev_list = [t*np.pi for t in (1/2, 1/2, 1/2, 1/2)]
        azim_list = [t*np.pi for t in (0, 1/2, 1, 3/2)]
    
    camera_list = list(zip(elev_list, azim_list))
    return camera_list

def load_texture_map(texture_path):
        texture = torch.tensor(
            np.array(Image.open(texture_path).convert("RGB")) / 255.0,
            dtype=torch.float32,
        )
        texture = texture.permute(2, 0, 1).unsqueeze(0)
        return texture

def render_images(mesh_path, texture_path=None, size=512, n_c=9, out_path=".cache"):
    os.makedirs(out_path, exist_ok=True)
    mesh = Mesh(mesh_path)
    render = Render(
        render_size=(512, 512),
        image_size=(size, size),
        interpolation_mode="bilinear",
    )

    texture = torch.ones([1, 3, 1024, 1024])
    if texture_path is not None:
        texture = load_texture_map(texture_path)
    view_list = get_camera_list(n_c)
    res_rgb_list = []
    res_depth_list = []
    for view in view_list:
        camera_view = render.get_camera_from_view(view[0], view[1])
        res = render(mesh.to("cuda"), texture.cuda(), camera_view.cuda())
        res_rgb_list.append(res["image"].cpu())
        res_depth_list.append(res["depth"].cpu())

    save_image(torch.cat(res_rgb_list), os.path.join(out_path, f"all_rgb.png"), nrow=math.floor(n_c**0.5), padding=0)
    save_image(torch.cat(res_depth_list), os.path.join(out_path, f"all_depth.png"), nrow=math.floor(n_c**0.5), padding=0)

    return Image.open(os.path.join(out_path, f"all_rgb.png")), Image.open(os.path.join(out_path, f"all_depth.png"))