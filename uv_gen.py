import os
import torch

from masactrl.attn_processor import set_masactrl_attn
from masactrl.utils import image_transfer
from utils import *

def main(cfg):
    device = cfg.device
    masa_cfg = cfg.masa
    render_size = masa_cfg.size
    img_size = render_size * 3

    model = load_model(cfg.model, device)

    ref_img, ref_depth, tar_img, tar_depth = load_imgs(cfg.dataset.path, masa_cfg.ref_idx, masa_cfg.tar_idx, img_size)
    control = {"depth": [ref_depth, tar_depth]}

    # load obj
    tar_uv_model = load_uv_model(cfg.mesh, masa_cfg.tar_idx, render_size, True)

    ref_prompt, target_prompt = "", ""
    prompts = [ref_prompt, target_prompt]

    num_step = cfg.model.num_step

    # invert the source image
    style_code, latents_list = model.invert(
        ref_img,
        ref_prompt,
        num_inference_steps=num_step,
        guidance_scale=1,
        base_resolution=img_size,
        # control={"depth": ref_depth}
    )

    # tar_image_ = image_transfer(tar_img, ref_img)
    # start_code, _ = model.invert(
    #     tar_image_,
    #     target_prompt,
    #     num_inference_steps=num_step,
    #     guidance_scale=1,
    #     base_resolution=img_size,
    #     # control={"depth": tar_depth}
    # )
    # start_code = start_code.expand(len(prompts), -1, -1, -1)
    start_code = style_code.expand(len(prompts), -1, -1, -1)

    set_masactrl_attn(model, render_size//8)
    masa_imgs = model(
        prompts,
        latents=start_code,
        num_inference_steps=num_step,
        guidance_scale=1,
        ref_intermediate_latents=latents_list,
        control=control,
        control_scale=2,
        base_resolution=img_size,
        # uv_model = tar_uv_model
    )

    # save the synthesized image
    save_res(ref_img, tar_img, masa_imgs[-1:], device)

if __name__ == "__main__":
    cfg = OmegaConf.load("configs/config.yaml")
    main(cfg)
