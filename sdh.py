import os
import torch

from masactrl.attn_processor import set_masactrl_attn
from masactrl.utils import image_transfer
from utils import *
from masactrl.pipeline import MyPipeline

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

def load_uv_model(cfg, obj_idx, render_size, use_unet, init_texture = None):
    object_list_file = f"{cfg.path}/split/chair.txt"
    object_list = []
    texture_list = []
    with open(object_list_file) as f:
        for obj_name in f.readlines():
            object_list.append(f"{cfg.path}/obj/{obj_name.strip()}.obj")
            texture_list.append(f"{cfg.path}/texture/{obj_name.strip()}.png")

    cfg.shape_path = object_list[obj_idx]
    if init_texture is None:
        init_texture = texture_list[obj_idx]
    uv_model = TexturedMeshModel(cfg, render_size, init_texture, device="cuda", use_unet=use_unet)
    return uv_model

cfg = OmegaConf.load("configs/config_h.yaml")
device = cfg.device
masa_cfg = cfg.masa
render_size = masa_cfg.size
img_size = render_size * 3
model = load_model(cfg.model, device)

# load obj
ref_uv_model = load_uv_model(cfg.mesh, cfg.masa.ref_idx, render_size, False)
tar_uv_model = load_uv_model(cfg.mesh, cfg.masa.tar_idx, render_size, False, "output/proj/texture.png")

elev_list = [t * np.pi for t in (1/2,)]
azim_list = [t * np.pi for t in (1,)]

tar_render = tar_uv_model.render(elev_list, azim_list, 3, "black", render_size)
ref_render = ref_uv_model.render(elev_list, azim_list, 3, "black", render_size)

ref_depth = ref_render["depth"]
tar_depth = tar_render["depth"]
mean_ref = torch.mean(ref_depth, dim=(2,3), keepdim=True)
std_ref = torch.std(ref_depth, dim=(2,3), keepdim=True)
mean_tar = torch.mean(tar_depth, dim=(2,3), keepdim=True)
std_tar = torch.std(tar_depth, dim=(2,3), keepdim=True)
# tar_depth = ((tar_depth - mean_tar)/std_tar)*std_ref + mean_ref

save_image(tar_render["image"], "temp/tar.png")
save_image(tar_depth, "temp/tar_depth.png")
save_image(ref_render["image"], "temp/ref.png")
save_image(ref_depth, "temp/ref_depth.png")

ref_prompt = ""
target_prompt = ""
prompts = [ref_prompt, target_prompt] * len(elev_list)
# control = {"depth": [ref_render["depth"].repeat(1, 3, 1, 1), tar_render["depth"].repeat(1, 3, 1, 1)]}
control = {"depth": torch.cat([ref_depth.repeat(1, 3, 1, 1), tar_depth.repeat(1, 3, 1, 1)])}
num_step = 50
# invert the source image
style_code, latents_list = model.invert(
    (ref_render["image"] * 2 - 1).half(),
    ref_prompt,
    num_inference_steps=num_step,
    guidance_scale=1,
    base_resolution=render_size,
    # control_scale=0.75,
    # control={"depth": ref_depth}
)
start_code, _ = model.invert(
    (tar_render["image"] * 2 - 1).half(),
    target_prompt,
    num_inference_steps=num_step,
    guidance_scale=1,
    base_resolution=render_size,
    # control={"depth": tar_depth}
)
start_code = start_code.repeat(2, 1, 1, 1)
set_masactrl_attn(model)
masa_imgs = model(
    prompts,
    latents=start_code,
    num_inference_steps=num_step,
    guidance_scale=1,
    ref_intermediate_latents=latents_list,
    control=control,
    control_scale=1,
    base_resolution=render_size,
    # uv_model = tar_uv_model
)

save_image(masa_imgs, "temp/res.png")
