import math
import os
import torch

from masactrl.attn_processor import set_masactrl_attn, set_local_attn
from masactrl.utils import image_transfer
from utils import *

def main(cfg):
    device = cfg.device
    mesh_cfg = cfg.mesh
    render_size = mesh_cfg.render_size
    
    col = math.floor(cfg.n_c**0.5)
    row = cfg.n_c // col
    img_size = (render_size*row, render_size*col)

    ref_img, ref_depth = render_images(cfg.ref_mesh, cfg.ref_texture, size=render_size, n_c=cfg.n_c, out_path=".cache/ref_up")
    tar_img, tar_depth = render_images(cfg.tar_mesh, cfg.tar_texture, size=render_size, n_c=cfg.n_c, out_path=".cache/tar_up")
    
    model = load_model(cfg.model, device)
    control = {"depth": [ref_depth, ref_depth, tar_depth]}

    ref_prompt = ""
    target_prompt = ""
    prompts = [ref_prompt, ref_prompt, target_prompt]

    num_step = cfg.model.num_step
    strength = cfg.strength
    set_local_attn(model, render_size//8)

    # invert the source image
    style_code, latents_list = model.invert(
        ref_img,
        ref_prompt,
        num_inference_steps=num_step,
        guidance_scale=1,
        base_resolution=img_size,
        strength=strength
    )

    # tar_image_ = image_transfer(tar_img, ref_img)
    start_code, _ = model.invert(
        tar_img,
        target_prompt,
        num_inference_steps=num_step,
        guidance_scale=1,
        base_resolution=img_size,
        strength=strength
    )
    start_code = start_code.expand(len(prompts), -1, -1, -1)

    set_masactrl_attn(model, render_size//8)
    masa_imgs = model(
        prompts,
        latents=start_code,
        num_inference_steps=num_step,
        guidance_scale=1,
        ref_intermediate_latents=latents_list,
        control=control,
        control_scale=4,
        base_resolution=img_size,
        strength=strength
    )

    # save the synthesized image
    save_res(ref_img, tar_img, masa_imgs[-1:], device)

if __name__ == "__main__":
    cfg = load_confg("configs", "config_upscale.yaml")
    main(cfg)

