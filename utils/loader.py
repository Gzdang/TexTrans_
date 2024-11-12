import os
import torch

from PIL import Image
from omegaconf import OmegaConf

from diffusers import DDIMScheduler, ControlNetModel

from masactrl.pipeline_15 import MyPipeline
from mesh.textured_mesh import TexturedMeshModel

from diffusers.image_processor import VaeImageProcessor
from transformers import CLIPImageProcessor


def load_model(cfg, device):
    scheduler = DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="linear",
        clip_sample=False,
        set_alpha_to_one=False,
    )

    model = MyPipeline.from_pretrained(
        cfg.base_model, scheduler=scheduler, torch_dtype=torch.float16
    ).to(device)
    model.vae.requires_grad_(False)
    model.unet.requires_grad_(False)
    model.text_encoder.requires_grad_(False)
    # model.vae.to(dtype=torch.float32)

    controlnet = ControlNetModel.from_pretrained(
        cfg.controlnet, torch_dtype=torch.float16
    ).eval()

    model.controlnet = controlnet.to(device)
    model.controlnet.requires_grad_(False)
    # model.vae_image_processor = VaeImageProcessor(do_convert_rgb=True, do_normalize=True)
    # model.control_image_processor = VaeImageProcessor(do_convert_rgb=True, do_normalize=False)
    # model.clip_image_processor = CLIPImageProcessor()

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

def load_uv_model(cfg, obj_idx, render_size, use_unet, init_texture = None):
    object_list_file = f"{cfg.path}/split/chair.txt"
    object_list = []
    with open(object_list_file) as f:
        for obj_name in f.readlines():
            object_list.append(f"{cfg.path}/obj/{obj_name.strip()}.obj")

    cfg.shape_path = object_list[obj_idx]
    uv_model = TexturedMeshModel(cfg, render_size, init_texture, device="cuda", use_unet=use_unet)
    return uv_model
