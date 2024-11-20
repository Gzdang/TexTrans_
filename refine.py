import os
import numpy as np
import cv2

from PIL import Image, ImageFilter
from torch import GradScaler
from torchvision.utils import save_image

from masactrl.attn_processor import set_masactrl_attn, reset_attn
from mesh.unet.lipis import LPIPS
from utils import *
from torchvision.transforms.functional import gaussian_blur


def sde(model, ref_image, tar_image, control, num_step, size):
    source_prompt, target_prompt = "", ""
    prompts = [source_prompt, target_prompt]
    strength = 0.6

    reset_attn(model)
    style_code, latents_list = model.invert(
        ref_image,
        source_prompt,
        guidance_scale=1,
        num_inference_steps=num_step,
        return_intermediates=True,
        base_resolution=size,
        strength = strength
    )
    start_code, _ = model.invert(
        tar_image,
        source_prompt,
        guidance_scale=1,
        num_inference_steps=num_step,
        return_intermediates=True,
        base_resolution=size,
        strength = strength
    )
    start_code = start_code.expand(len(prompts), -1, -1, -1)
    # start_code = style_code.expand(len(prompts), -1, -1, -1)

    set_masactrl_attn(model)
    image_masactrl = model(
        prompts,
        latents=start_code,
        num_inference_steps=num_step,
        guidance_scale=1,
        ref_intermediate_latents=latents_list,
        control=control,
        control_scale=0.75,
        base_resolution=size,
        strength = strength
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
    ref_uv_model.requires_grad_(False)
    tar_uv_model = load_uv_model(cfg.mesh, cfg.masa.tar_idx, render_size, True, device="cuda:1")
    # mask_model = load_uv_model(cfg.mesh, cfg.masa.tar_idx, render_size, False, "temp/mask.png")
    mask_model = load_uv_model(cfg.mesh, cfg.masa.tar_idx, render_size, False, "temp/_mask.png")

    base_texture = tar_uv_model.get_texture().detach()
    base_mask = (mask_model.texture_map != 0).detach().cpu().to("cuda:1")
    last_mask = base_mask

    # elev_list = [t * np.pi for t in (1/2, 1/2, 1/4, )]
    # azim_list = [t * np.pi for t in (1/3, 5/3, 0, )]
    elev_list = [t * np.pi for t in (3/4, 1/2, 1/2, 1/2, 1/4, )]
    azim_list = [t * np.pi for t in (0, 1, 1/3, 5/3, 0, )]

    # base_images = tar_uv_model.render(elev_list, azim_list, 3, "black", render_size)
    # save_image(base_images["image"], "temp/base.png")

    optim_texture = torch.optim.Adam(tar_uv_model.parameters(), 1e-3)
    scaler = GradScaler()
    perceptual_loss = LPIPS(True).to("cuda:1").eval()

    for elev, azim in zip(elev_list, azim_list):
        mask_model.init_textures()
        optim_mask = torch.optim.Adam(mask_model.parameters(), 1e-1)
        for _ in range(10):
            mask_out = mask_model.render([elev], [azim], 3, dim=render_size)
            loss = torch.nn.functional.l1_loss(mask_out["image"], mask_out["mask"].repeat(1, 3, 1, 1).detach())
            loss.backward()
            optim_mask.step()
            optim_mask.zero_grad()
        save_image((mask_model.texture_map!=0).float(), "temp/mask_res.png")

        tar_render = tar_uv_model.render([elev], [azim], 3, "black", render_size)
        ref_render = ref_uv_model.render([elev], [azim], 3, "black", render_size)

        ref_image = (ref_render["image"] * 2 - 1).half()
        ref_depth = ref_render["depth"]
        tar_image = (tar_render["image"].detach() * 2 - 1).half().cpu().to(device)
        tar_depth = tar_render["depth"].detach().cpu().to(device)
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

        target = sde(model, ref_image, tar_image, control, cfg.model.num_step, render_size).detach().cpu().to("cuda:1")
        save_image(target, "temp/target.png")
        # image_transfer(target, base_images["image"][i:i+1])
        # save_image(target, "temp/after.png")


        proj_mask = (mask_model.texture_map != 0).detach().cpu().to("cuda:1")
        change_mask = (proj_mask != base_mask).int().detach()
        # change_mask = (proj_mask != last_mask).int().detach()
        save_image(change_mask.float(), "temp/change_mask.png")
        _change_mask = gaussian_blur(change_mask.float(), 15, 9)
        change_mask = (_change_mask>0.2).int().detach()
        unchange_mask = (_change_mask<0.8).int().detach()
        save_image(change_mask.float(), "temp/_change_mask.png")
        save_image(unchange_mask.float(), "temp/unchange_mask.png")
        last_texture = tar_uv_model.get_texture().detach()
        save_image(last_texture, "temp/last_texture.png")

        # blur_texture = gaussian_blur(last_texture,9,9)

        scheduler=torch.optim.lr_scheduler.MultiStepLR(optim_texture, milestones=[600, 800], gamma=0.5)
        for i in range(1000):
            optim_texture.zero_grad()
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                texture_out = tar_uv_model.render([elev], [azim], 3, dim=render_size)
                # loss = torch.nn.functional.l1_loss(texture_out["image"], target.detach())
                pl = perceptual_loss(texture_out["image"], target[-1:])[0][0][0][0]
                ll = torch.nn.functional.l1_loss(texture_out["image"], target[-1:])
                # blured = gaussian_blur(texture_out["texture_map"],9,9)
                cl = torch.nn.functional.l1_loss(change_mask*texture_out["texture_map"], change_mask*last_texture)
                ul = torch.nn.functional.l1_loss(unchange_mask*texture_out["texture_map"], unchange_mask*last_texture)
                loss = pl + cl + ul
            print(f"{i} pl:{pl}, ll:{ll}, cl:{cl}, ul:{ul}, loss:{loss}", end="\r")
            scaler.scale(loss).backward()
            scaler.step(optim_texture)
            scaler.update()
            scheduler.step()

        last_mask = (mask_model.texture_map != 0).detach().cpu().to("cuda:1")
        save_image(texture_out["image"], "./temp/mesa_res.png")
        save_image(tar_uv_model.texture_map, "./temp/texture.png")
    with torch.no_grad():
        res, _ = tar_uv_model.render_all()
        save_image(res, "./temp/image_all.png")


if __name__ == "__main__":
    cfg = OmegaConf.load("configs/config_refine.yaml")
    main(cfg)

    texture = np.array(Image.open("temp/texture.png"))
    mask = (np.array(Image.open("temp/mask_res.png"))[:, :, 0] == 0).astype(np.uint8)* 255
    inpaint = cv2.inpaint(
        texture, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA
    )

    texture = Image.fromarray(inpaint)
    texture.save("temp/inpaint.png")

