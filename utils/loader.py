import os
import torch

from PIL import Image
from omegaconf import OmegaConf

from diffusers import DDIMScheduler, ControlNetModel

from masactrl.pipeline_15 import MyPipeline
from mesh.config import GuideConfig
from mesh.textured_mesh import TexturedMeshModel


def load_model(model_path, device):
    scheduler = DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="linear",
        clip_sample=False,
        set_alpha_to_one=False,
    )

    model = MyPipeline.from_pretrained(
        model_path, scheduler=scheduler, torch_dtype=torch.float16
    ).to(device)
    model.vae.requires_grad_(False)
    model.vae.to(dtype=torch.float32)

    depth_controlnet = ControlNetModel.from_pretrained(
        "/home/lrz/diffuser/controlnet/depth_15", torch_dtype=torch.float16
    ).eval()

    model.depth_controlnet = depth_controlnet.to(device)

    return model


def load_imgs(resource_dir, ref_idx, tar_idx, size):
    ref_img = Image.open(os.path.join(resource_dir, f"{ref_idx}/all_rgb.png")).resize(
        (size, size)
    )
    ref_depth = Image.open(
        os.path.join(resource_dir, f"{ref_idx}/all_depth.png")
    ).resize((size, size))
    tar_img = Image.open(os.path.join(resource_dir, f"{tar_idx}/all_rgb.png")).resize(
        (size, size)
    )
    tar_depth = Image.open(
        os.path.join(resource_dir, f"{tar_idx}/all_depth.png")
    ).resize((size, size))

    return ref_img, ref_depth, tar_img, tar_depth

def load_uv_model(object_list_file, obj_idx, render_size, uv_size, use_unet, init_texture = None):
    object_list = []
    with open(object_list_file) as f:
        for obj_name in f.readlines():
            object_list.append(f"{os.environ['HOME']}/dataset/3D_Future/obj/{obj_name.strip()}.obj")

    opt = OmegaConf.create(GuideConfig)
    opt.shape_path = object_list[obj_idx]
    uv_model = TexturedMeshModel(opt, render_size, uv_size, init_texture, device="cuda", use_unet=use_unet)
    return uv_model
