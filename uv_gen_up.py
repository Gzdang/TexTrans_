import os
import torch

from masactrl.attn_processor import set_masactrl_attn
from masactrl.utils import image_transfer
from utils import *

def main(cfg):
    device = cfg.device
    mesh_cfg = cfg.mesh
    render_size = mesh_cfg.render_size
    img_size = render_size * 3

    model = load_model(cfg.model, device)

    ref_img, ref_depth, tar_img, tar_depth = load_imgs(cfg.dataset.path, cfg.ref_idx, cfg.tar_idx, img_size)
    control = {"depth": [ref_depth, tar_depth]}

    # load obj
    init_texture = "output/proj/texture_l.png"
    tar_uv_model = load_uv_model(cfg.mesh, cfg.tar_idx, render_size, False, init_texture)

    image, _ = tar_uv_model.render_all()
    save_image(image, "render.png")
    tar_img = Image.open("./render.png").resize((img_size, img_size))
    # tar_img = Image.open("output/gen/sample_3/masactrl_step.png").resize((img_size, img_size))

    ref_prompt = ""
    target_prompt = ""
    prompts = [ref_prompt, target_prompt]

    num_step = cfg.model.num_step
    strength = 0.6

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
        control_scale=0.75,
        base_resolution=img_size,
        strength=strength
    )

    # save the synthesized image
    save_res(ref_img, tar_img, masa_imgs[-1:], device)

if __name__ == "__main__":
    cfg = load_confg("configs", "config_upscale.yaml")
    main(cfg)

