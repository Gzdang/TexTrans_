import os
import torch

from masactrl.attn_processor import set_masactrl_attn
from masactrl.utils import image_transfer
from utils import *
from config import *

def format_img(image):
    image = np.array(image)
    image = torch.from_numpy(image).float() / 127.5 - 1
    image = image.permute(2, 0, 1).unsqueeze(0).cuda()
    if image.shape[1] == 4:
        image = image[:, :-1, :, :]
    return image

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model_path = "/home/lrz/diffuser/diffusion/15"
img_size = high_size * 3
render_size = high_size

model = load_model(model_path, device)

ref_idx, tar_idx = 21, 5
ref_img, ref_depth, tar_img, tar_depth = load_imgs("./dataset/new", ref_idx, tar_idx, img_size)
control = {"depth": [ref_depth, tar_depth]}

ref_prompt = ""
target_prompt = ""
prompts = [ref_prompt, target_prompt]

num_step = 30

# invert the source image
style_code, latents_list = model.invert(
    torch.cat([format_img(ref_depth), format_img(ref_img)]),
    ref_prompt,
    num_inference_steps=num_step,
    guidance_scale=1,
    base_resolution=img_size,
    # control_scale=0.75,
    # control={"depth": ref_depth}
)

tar_image_ = image_transfer(tar_img, ref_img)
start_code, _ = model.invert(
    torch.cat([format_img(tar_depth), format_img(tar_img)]),
    target_prompt,
    num_inference_steps=num_step,
    guidance_scale=1,
    base_resolution=img_size,
    # control={"depth": tar_depth}
)
start_code = torch.cat([start_code]*2)

set_masactrl_attn(model.unet)
masa_imgs = model(
    prompts,
    latents=start_code,
    num_inference_steps=num_step,
    guidance_scale=1,
    ref_intermediate_latents=latents_list,
    control=control,
    control_scale=0.75,
    base_resolution=img_size,
    # uv_model = tar_uv_model
)

# save the synthesized image
save_res(ref_img, tar_img, masa_imgs[-1:], device)
