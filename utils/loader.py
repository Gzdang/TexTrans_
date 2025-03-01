import os
import torch
import numpy as np

from PIL import Image
from omegaconf import OmegaConf

from diffusers import DDIMScheduler, ControlNetModel

from masactrl.pipeline import MyPipeline
from masactrl.pipeline_xl import MyPipelineXL
from mesh.textured_mesh import TexturedMeshModel

from diffusers.image_processor import VaeImageProcessor
from transformers import CLIPImageProcessor

from masactrl.controlnet_union import ControlNetModel_Union


def load_confg(base_path, cfg_name):
    cfg_dir = os.path.join(base_path, cfg_name)
    if not os.path.exists(cfg_dir):
        raise FileNotFoundError()
    cfg = OmegaConf.load(cfg_dir)

    if os.path.exists(os.path.join(base_path, "global.yaml")):
        cfg = OmegaConf.merge(cfg, OmegaConf.load(os.path.join(base_path, "global.yaml")))

    args = OmegaConf.from_cli()
    print(args)
    # assert args.ref_mesh is None or args.ref_mesh is None or args.tar_mesh is None
    cfg = OmegaConf.merge(cfg, args)

    return cfg


def load_model(cfg, device):
    scheduler = DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="linear",
        clip_sample=False,
        set_alpha_to_one=False,
    )
    if cfg.get("type", "sd15") == "sdxl":
        model = MyPipelineXL.from_pretrained(cfg.base_model, scheduler=scheduler, torch_dtype=torch.float16).to(device)
        model.vae.to(dtype=torch.float32)
        # controlnet = ControlNetModel.from_pretrained(cfg.controlnet, variant="fp16", torch_dtype=torch.float16).eval()
        controlnet = ControlNetModel_Union.from_pretrained(cfg.controlnet, torch_dtype=torch.float16, use_safetensors=True)
    else:
        model = MyPipeline.from_pretrained(cfg.base_model, scheduler=scheduler, torch_dtype=torch.float16).to(device)
        controlnet = ControlNetModel.from_pretrained(cfg.controlnet, torch_dtype=torch.float16).eval()

    model.vae.requires_grad_(False)
    model.unet.requires_grad_(False)
    model.text_encoder.requires_grad_(False)

    model.controlnet = controlnet.to(device)
    model.controlnet.requires_grad_(False)
    # model.vae_image_processor = VaeImageProcessor(do_convert_rgb=True, do_normalize=True)
    # model.control_image_processor = VaeImageProcessor(do_convert_rgb=True, do_normalize=False)
    # model.clip_image_processor = CLIPImageProcessor()

    return model


def load_imgs(resource_dir, ref_idx, tar_idx, size):
    ref_img = Image.open(os.path.join(resource_dir, f"{ref_idx}/render_images/all_rgb.png")).resize(size)
    ref_depth = Image.open(os.path.join(resource_dir, f"{ref_idx}/render_images/all_depth.png")).resize(size)
    tar_img = Image.open(os.path.join(resource_dir, f"{tar_idx}/render_images/all_rgb.png")).resize(size)
    tar_depth = Image.open(os.path.join(resource_dir, f"{tar_idx}/render_images/all_depth.png")).resize(size)

    return ref_img, ref_depth, tar_img, tar_depth


def load_uv_model(cfg, mesh_path, render_size, use_unet, init_texture=None, with_materials=True, device="cuda"):
    cfg.shape_path = mesh_path
    uv_model = TexturedMeshModel(cfg, render_size, init_texture, device=device, use_unet=use_unet, with_materials=with_materials)
    return uv_model


def preprocess_mask(cfg, texture_path):
    texture = np.array(Image.open(texture_path).resize([cfg.texture_resolution] * 2))
    mask = (texture[:, :, -1] == 255).astype(np.uint8)
    Image.fromarray(mask * 255).save(".cache/mask.png")
    Image.fromarray((1 - mask) * 255).save(".cache/_mask.png")
    return mask
