import os
import numpy as np

from PIL import Image
from torchvision.utils import save_image

from masactrl.attn_processor import set_masactrl_attn
from utils import *

cfg = OmegaConf.load("configs/config_xl.yaml")

device = cfg.device
masa_cfg = cfg.masa
render_size = masa_cfg.size
img_size = render_size * 3

model = load_model(cfg.model, device)

# load image
ref_uv_model = load_uv_model(cfg.mesh, cfg.masa.ref_idx, render_size, False)
tar_uv_model = load_uv_model(cfg.mesh, cfg.masa.tar_idx, render_size, False, "output/proj/texture.png")

elev_list = [t * np.pi for t in (1/2,)]
azim_list = [t * np.pi for t in (1/3,)]

tar_render = tar_uv_model.render(elev_list, azim_list, 3, "black", render_size)
ref_render = ref_uv_model.render(elev_list, azim_list, 3, "black", render_size)

ref_depth = ref_render["depth"]
tar_depth = tar_render["depth"]
# mean_ref = torch.mean(ref_depth, dim=(2,3), keepdim=True)
# std_ref = torch.std(ref_depth, dim=(2,3), keepdim=True)
# mean_tar = torch.mean(tar_depth, dim=(2,3), keepdim=True)
# std_tar = torch.std(tar_depth, dim=(2,3), keepdim=True)
# tar_depth = ((tar_depth - mean_tar)/std_tar)*std_ref + mean_ref

save_image(tar_render["image"], "temp/tar.png")
save_image(tar_depth, "temp/tar_depth.png")
save_image(ref_render["image"], "temp/ref.png")
save_image(ref_depth, "temp/ref_depth.png")

source_prompt = "multi-view from single chair, high quality, 4K"
target_prompt = "multi-view from single chair, high quality, 4K"
prompts = [source_prompt, target_prompt]

num_step = 50

# invert the source image

style_code, latents_list = model.invert(
    (ref_render["image"] * 2 - 1).half(),
    source_prompt,
    guidance_scale=1,
    num_inference_steps=num_step,
    return_intermediates=True,
    base_resolution=render_size,
)

start_code, _ = model.invert(
    (tar_render["image"] * 2 - 1).half(),
    source_prompt,
    guidance_scale=1,
    num_inference_steps=num_step,
    return_intermediates=True,
    base_resolution=render_size,
)

start_code = start_code.expand(len(prompts), -1, -1, -1)

set_masactrl_attn(model)

control = {"depth": [ref_depth.repeat(1, 3, 1, 1), tar_depth.repeat(1, 3, 1, 1)]}

image_masactrl = model(
    prompts,
    latents=start_code,
    num_inference_steps=num_step,
    guidance_scale=1,
    ref_intermediate_latents=latents_list,
    control=control,
    control_scale=2,
    base_resolution=render_size,
)

save_image(image_masactrl, "temp/res.png")

