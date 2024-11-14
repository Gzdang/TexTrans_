import os
import numpy as np
import cv2

from PIL import Image, ImageFilter
from torchvision.utils import save_image

from masactrl.attn_processor import set_masactrl_attn, reset_attn
from mesh.unet.lipis import LPIPS
from utils import *


def sde(model, ref_image, tar_image, control, num_step, size):
    source_prompt, target_prompt = "", ""
    prompts = [source_prompt, target_prompt]

    reset_attn(model)
    _, latents_list = model.invert(
        ref_image,
        source_prompt,
        guidance_scale=1,
        num_inference_steps=num_step,
        return_intermediates=True,
        base_resolution=size,
    )
    start_code, _ = model.invert(
        tar_image,
        source_prompt,
        guidance_scale=1,
        num_inference_steps=num_step,
        return_intermediates=True,
        base_resolution=size,
    )
    start_code = start_code.expand(len(prompts), -1, -1, -1)

    set_masactrl_attn(model)
    image_masactrl = model(
        prompts,
        latents=start_code,
        num_inference_steps=num_step,
        guidance_scale=1,
        ref_intermediate_latents=latents_list,
        control=control,
        control_scale=2,
        base_resolution=size,
    )

    return image_masactrl

def image_transfer(tar_image, style_image):
    eps: float = 1e-5
    mean = tar_image.mean(dim=[-2, -1], keepdim=True)
    var = (tar_image.var(dim=[-2, -1], keepdims=True) + eps).sqrt()
    style_mean = style_image.mean(dim=[-2, -1], keepdim=True)
    style_var = (style_image.var(dim=[-2, -1], keepdims=True) + eps).sqrt()

    tar_image = (tar_image - mean) / var
    res = (tar_image * style_var + style_mean)

    return res


def main(cfg):
    device = cfg.device
    masa_cfg = cfg.masa
    render_size = masa_cfg.size

    model = load_model(cfg.model, device)

    # load image
    cfg.mesh.texture_unet_path = "output/proj/unet.pth"
    ref_uv_model = load_uv_model(cfg.mesh, cfg.masa.ref_idx, render_size, False)
    tar_uv_model = load_uv_model(cfg.mesh, cfg.masa.tar_idx, render_size, True)
    # mask_model = load_uv_model(cfg.mesh, cfg.masa.tar_idx, render_size, False, "temp/mask.png")
    mask_model = load_uv_model(cfg.mesh, cfg.masa.tar_idx, int(render_size * 1.5), False, "temp/_mask.png")

    base_mask = (mask_model.texture_map != 0).detach()
    last_mask = base_mask

    elev_list = [t * np.pi for t in ( 3/4, 3/4, 3/4, 1/4, 1/4, 1/4, 1/2, 1/2, 1/2,)]
    azim_list = [t * np.pi for t in ( 4/3, 0, 2/3, 4/3, 0, 2/3, 1/3, 5/3, 1, )]

    base_images = tar_uv_model.render(elev_list, azim_list, 3, "black", render_size)
    save_image(base_images["image"], "temp/base.png")

    optim_texture = torch.optim.Adam(tar_uv_model.parameters(), 1e-4)
    optim_mask = torch.optim.Adam(mask_model.parameters(), 1e-1)
    perceptual_loss = LPIPS(True).cuda().eval()

    for elev, azim in zip(elev_list, azim_list):
        for _ in range(10):
            mask_out = mask_model.render([elev], [azim], 3, dim=int(render_size * 1.5))
            loss = torch.nn.functional.l1_loss(mask_out["image"], mask_out["mask"].detach())
            loss.backward()
            optim_mask.step()
            optim_mask.zero_grad()
        save_image((mask_model.texture_map!=0).float(), "temp/mask_res.png")

        tar_render = tar_uv_model.render([elev], [azim], 3, "black", render_size)
        ref_render = ref_uv_model.render([elev], [azim], 3, "black", render_size)

        ref_image = (ref_render["image"] * 2 - 1).half()
        ref_depth = ref_render["depth"]
        tar_image = (tar_render["image"] * 2 - 1).half()
        tar_depth = tar_render["depth"]
        control = {"depth": [ref_depth.repeat(1, 3, 1, 1), tar_depth.repeat(1, 3, 1, 1)]}
        # mean_ref = torch.mean(ref_depth, dim=(2,3), keepdim=True)
        # std_ref = torch.std(ref_depth, dim=(2,3), keepdim=True)
        # mean_tar = torch.mean(tar_depth, dim=(2,3), keepdim=True)
        # std_tar = torch.std(tar_depth, dim=(2,3), keepdim=True)
        # tar_depth = ((tar_depth - mean_tar)/std_tar)*std_ref + mean_ref

        save_image(tar_render["image"], "temp/tar.png")
        save_image(tar_depth, "temp/tar_depth.png")
        save_image(ref_render["image"], "temp/ref.png")
        save_image(ref_depth, "temp/ref_depth.png")

        target = sde(model, ref_image, tar_image, control, cfg.model.num_step, render_size).detach()
        # save_image(target, "temp/before.png")
        # image_transfer(target, base_images["image"][i:i+1])
        # save_image(target, "temp/after.png")


        change_mask = ((mask_model.texture_map != 0) != last_mask).int().detach()
        save_image(change_mask.float(), "temp/change_mask.png")
        last_texture = tar_uv_model.get_texture().detach()
        save_image(last_texture, "temp/last_texture.png")

        for i in range(200):
            texture_out = tar_uv_model.render([elev], [azim], 3, dim=render_size)
            # loss = torch.nn.functional.l1_loss(texture_out["image"], target.detach())
            loss = perceptual_loss(texture_out["image"], target.detach())[0][0][0][0]
            loss += torch.nn.functional.l1_loss((1-base_mask.int())*texture_out["texture_map"], (1-base_mask.int())*last_texture.detach())
            print(f"{i}: {loss}", end="\r")
            loss.backward()
            optim_texture.step()
            optim_texture.zero_grad()

        last_mask = mask_model.texture_map != 0
        save_image(texture_out["image"], "./temp/mesa_res.png")
        save_image(tar_uv_model.texture_map, "./temp/texture.png")


if __name__ == "__main__":
    cfg = OmegaConf.load("configs/config_h.yaml")
    main(cfg)

    texture = np.array(Image.open("temp/texture.png"))
    mask = (np.array(Image.open("temp/mask_res.png"))[:, :, 0] == 0).astype(np.uint8)* 255
    inpaint = cv2.inpaint(
        texture, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA
    )

    texture = Image.fromarray(inpaint)
    texture.save("temp/inpaint.png")

