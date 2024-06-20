import os
import numpy as np
import torch

from diffusers import DDIMScheduler, ControlNetModel
from diffusers.image_processor import VaeImageProcessor

from masactrl.pipeline import MyPipeline
from PIL import Image
from torchvision.utils import save_image

from masactrl.attn_processor import set_masactrl_attn
from masactrl.utils import image_transfer


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model_path = "/home/lrz/diffuser/diffusion/xl"
scheduler = DDIMScheduler(
    beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False
)

model = MyPipeline.from_pretrained(model_path, scheduler=scheduler, torch_dtype=torch.float16).to(device)
model.upcast_vae()

depth_controlnet = ControlNetModel.from_pretrained(
    "/home/lrz/diffuser/controlnet/depth_xl", variant="fp16", use_safetensors=True, torch_dtype=torch.float16
)

model.depth_controlnet = depth_controlnet.to(device)
model.control_image_processor = VaeImageProcessor(
    vae_scale_factor=model.vae_scale_factor, do_convert_rgb=True, do_normalize=False
)


base_res = 1024
source_idx = 3
tar_idx = 5

# load image
resource_dir = "./dataset/new"
source_image = Image.open(os.path.join(resource_dir, f"{source_idx}/all_rgb.png")).resize((1024, 1024))
source_depth = Image.open(os.path.join(resource_dir, f"{source_idx}/all_depth.png")).resize((1024, 1024))
tar_image = Image.open(os.path.join(resource_dir, f"{tar_idx}/all_rgb.png")).resize((1024, 1024))
# tar_image = Image.open("tar.png").resize((1024, 1024))
tar_depth = Image.open(os.path.join(resource_dir, f"{tar_idx}/all_depth.png")).resize((1024, 1024))

source_prompt = ""
target_prompt = ""
prompts = [source_prompt, target_prompt]

num_step = 30

# invert the source image

style_code, latents_list = model.invert(
    source_image,
    source_prompt,
    num_inference_steps=num_step,
    # control={"depth": source_depth},
    # control_scale=1,
    base_resolution=base_res,
)


tar_image_ = image_transfer(tar_image, source_image)
start_code, _ = model.invert(
    tar_image_,
    source_prompt,
    num_inference_steps=num_step,
    # control={"depth": tar_depth},
    # control_scale=-1,
    base_resolution=base_res,
)

start_code = start_code.expand(len(prompts), -1, -1, -1)

set_masactrl_attn(model.unet)

control = {"depth": [source_depth, tar_depth]}

image_masactrl = model(
    prompts,
    latents=start_code,
    num_inference_steps=num_step,
    ref_intermediate_latents=latents_list,
    control=control,
    control_scale=1,
    base_resolution=base_res,
)

# save the synthesized image
out_dir = "./output/gen"
os.makedirs(out_dir, exist_ok=True)
sample_count = len(os.listdir(out_dir))
out_dir = os.path.join(out_dir, f"sample_{sample_count}")
os.makedirs(out_dir, exist_ok=True)

out_image = torch.cat(
    [
        torch.tensor(np.array(source_image), dtype=torch.float).permute(2, 0, 1).unsqueeze(0).to(device) / 255,
        image_masactrl[-1:],
        torch.tensor(np.array(tar_image), dtype=torch.float).permute(2, 0, 1).unsqueeze(0).to(device) / 255,
    ],
    dim=0,
)
save_image(out_image, os.path.join(out_dir, f"all_step.png"))
save_image(out_image[0], os.path.join(out_dir, f"source_step.png"))
save_image(out_image[1], os.path.join(out_dir, f"masactrl_step.png"))
save_image(out_image[2], os.path.join(out_dir, f"tar_step.png"))

print("Syntheiszed images are saved in", out_dir)
