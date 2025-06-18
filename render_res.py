import os
import torch
import numpy as np
import PIL.Image as Image

from torchvision.utils import save_image
from mesh.mesh import Mesh
from mesh.render import Render


def load_texture_map(texture_path):
    texture = torch.tensor(
        np.array(Image.open(texture_path).convert("RGB")) / 255.0,
        dtype=torch.float32,
    )
    texture = texture.permute(2, 0, 1).unsqueeze(0)
    return texture

def render_images(mesh_path, texture_path=None, size=512, n_c=9, out_path=".cache", is_depthanything=False):
    os.makedirs(out_path, exist_ok=True)
    mesh = Mesh(mesh_path, with_materials=False)
    render = Render(
        render_size=(512, 512),
        image_size=(size, size),
        interpolation_mode="bilinear",
    )

    texture = torch.ones([1, 3, 1024, 1024])
    if texture_path is not None:
        texture = load_texture_map(texture_path)

    elev_list = [np.pi * (75/180) for t in range(n_c)]
    azim_list = [(np.pi * (t/n_c)* 2) for t in range(n_c)]

    render_res = render(mesh.to("cuda"), texture.cuda(), elev_list, azim_list)

    for i in range(n_c):
        mask = render_res["mask"][i]
        image = render_res["image"][i]
        save_image(image*mask + (1-mask), f"{out_path}/image_{i}.png")


lt = ["", "_l", "_h"]
for tag in lt:
    tar_idx = "163786646ae2578749a5fc144a874235"
    # tar_idx = "1a19271683597db4fe7e6f8a8e38f62d"
    mesh = f"resource/dataset/shapenetv1/02958343/{tar_idx}/model.obj"
    texture = f"car_mark/output-car-1d4b2404a00ef4bb627014ff98c41eb1-163786646ae2578749a5fc144a874235/output/texture{tag}.png"
    # texture = f"./car_mark/output-car-1a1dcd236a1e6133860800e6696b8284-1a19271683597db4fe7e6f8a8e38f62d/output/texture{tag}.png"
    render_images(mesh, texture, size=1024, n_c=16, out_path=f".render_res{tag}")