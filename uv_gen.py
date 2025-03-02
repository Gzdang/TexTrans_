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
    print(img_size)

    ref_img, ref_depth, ref_normal = render_images(cfg.ref_mesh, cfg.ref_texture, size=render_size, n_c=cfg.n_c, out_path=".cache/ref")
    tar_img, tar_depth, tar_normal = render_images(cfg.tar_mesh, cfg.tar_texture, size=render_size, n_c=cfg.n_c, out_path=".cache/tar")

    model = load_model(cfg.model, device)
    control = {"depth": [ref_depth, ref_depth, tar_depth]}
    # control = {"depth": [ref_normal, ref_normal, tar_normal]}

    ref_prompt, target_prompt = "", ""
    prompts = [ref_prompt, ref_prompt, target_prompt]

    num_step = cfg.model.num_step
    # set_local_attn(model)
    set_local_attn(model, render_size//8, scale_factor=(col, row))

    # invert the source image
    style_code, latents_list = model.invert(
        ref_img,
        ref_prompt,
        num_inference_steps=num_step,
        guidance_scale=1,
        base_resolution=img_size,
    )

    if cfg.tar_texture is not None:
        start_code, _ = model.invert(
            tar_img,
            target_prompt,
            num_inference_steps=num_step,
            guidance_scale=1,
            base_resolution=img_size,
        )
    else:
        start_code = style_code
    start_code = start_code.expand(len(prompts), -1, -1, -1)

    # set_masactrl_attn(model)
    set_masactrl_attn(model, render_size//8, scale_factor=(col, row))
    masa_imgs = model(
        prompts,
        latents=start_code,
        num_inference_steps=num_step,
        guidance_scale=1,
        ref_intermediate_latents=latents_list,
        control=control,
        control_scale=4,
        base_resolution=img_size,
        # uv_model = tar_uv_model
    )

    # save the synthesized image
    save_res(ref_img, tar_img, masa_imgs[-1:], device)

if __name__ == "__main__":
    cfg = load_confg("configs", "config_gen.yaml")
    main(cfg)
