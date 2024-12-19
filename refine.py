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


def sde(model, ref_image, tar_image, control, num_step, strength, size):
    source_prompt, target_prompt = "", ""
    # prompts = [source_prompt, target_prompt] 
    prompts = [source_prompt, source_prompt, target_prompt] 

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

    set_masactrl_attn(model)
    image_masactrl = model(
        prompts,
        latents=start_code,
        num_inference_steps=num_step,
        guidance_scale=1,
        ref_intermediate_latents=latents_list,
        control=control,
        control_scale=0.5,
        base_resolution=size,
        strength = strength
    )

    return image_masactrl

def gen_mask(cfg, mesh_path, render_size):
    mask_model = load_uv_model(cfg, mesh_path, render_size, False)
    mask_model.texture_map = torch.nn.Parameter(torch.zeros_like(mask_model.texture_map))
    optim_mask = torch.optim.Adam(mask_model.parameters(), 1e-2)
    for _ in range(5):
        image, mask = mask_model.render_all()
        loss = torch.nn.functional.l1_loss(image, mask.detach())
        loss.backward()
        optim_mask.step()
        optim_mask.zero_grad()
    save_image((mask_model.texture_map==0).float(), ".cache/_mask.png")
    

def main(cfg):
    device = cfg.device
    render_size = 1024

    model = load_model(cfg.model, device)

    # load image
    cfg.mesh.n_c = cfg.n_c
    gen_mask(cfg.mesh, cfg.tar_mesh, 512)
    tar_uv_model = load_uv_model(cfg.mesh, cfg.tar_mesh, render_size, True, init_texture=cfg.tar_texture, device="cuda:1")
    mask_model = load_uv_model(cfg.mesh, cfg.tar_mesh, render_size, False, init_texture=".cache/_mask.png")
    ref_uv_model = load_uv_model(cfg.mesh, cfg.ref_mesh, render_size, False, init_texture=cfg.ref_texture)
    ref_uv_model.requires_grad_(False)

    base_texture = tar_uv_model.get_texture().detach()
    base_mask = (mask_model.texture_map != 0).detach().cpu().to("cuda:1")
    last_mask = base_mask

    # elev_list = [t * np.pi for t in (3/4, 1/2, 1/2, 1/2, 1/4, )]
    # azim_list = [t * np.pi for t in (0, 1, 1/3, 5/3, 0, )]

    # elev_list = [t*np.pi for t in (1/2, 1/2, 1/2, 1/2)]
    # azim_list = [t*np.pi for t in (1/4, 3/4, 5/4, 7/4)]
    # elev_list = [t*np.pi for t in (1/3, 11/18, 1/3, 11/18, 1/3, 11/18,)]
    # azim_list = [t*np.pi for t in (30/180, 90/180, 150/180, 210/180, 270/180, 330/180)]

    elev_list = [t*np.pi for t in (1/4, 1/4, 1/4, 1/2, 1/2, 1/2, 3/4, 3/4, 3/4)]
    azim_list = [t*np.pi for t in (5/3, 1, 1/3, 4/3, 0, 2/3, 5/3, 1, 1/3)]

    optim_texture = torch.optim.Adam(tar_uv_model.parameters(), 1e-3)
    scaler = GradScaler()
    perceptual_loss = LPIPS(True).to("cuda:1").eval()

    for elev, azim in zip(elev_list, azim_list):
        mask_model.init_textures()
        optim_mask = torch.optim.Adam(mask_model.parameters(), 1e-2)
        save_image((mask_model.texture_map!=0).float(), ".cache/mask_res.png")
        for _ in range(5):
            mask_out = mask_model.render([elev], [azim], 3, dim=render_size)
            loss = torch.nn.functional.l1_loss(mask_out["image"], mask_out["mask"].repeat(1, 3, 1, 1).detach())
            loss.backward()
            optim_mask.step()
            optim_mask.zero_grad()
        save_image((mask_model.texture_map!=0).float(), ".cache/mask_res.png")

        tar_render = tar_uv_model.render([elev], [azim], 3, "black", render_size)
        ref_render = ref_uv_model.render([elev], [azim], 3, "black", render_size)

        ref_image = (ref_render["image"] * 2 - 1).half()
        ref_depth = ref_render["depth"]
        tar_image = (tar_render["image"].detach() * 2 - 1).half().cpu().to(device)
        tar_depth = tar_render["depth"].detach().cpu().to(device)
        control = {"depth": [ref_depth.repeat(1, 3, 1, 1), ref_depth.repeat(1, 3, 1, 1), tar_depth.repeat(1, 3, 1, 1)]}

        save_image(tar_render["image"], ".cache/tar.png")
        save_image(tar_depth, ".cache/tar_depth.png")
        save_image(ref_render["image"], ".cache/ref.png")
        save_image(ref_depth, ".cache/ref_depth.png")

        target = sde(model, ref_image, tar_image, control, cfg.model.num_step, cfg.strength, render_size).detach().cpu().to("cuda:1")
        save_image(target, ".cache/target.png")


        proj_mask = (mask_model.texture_map != 0).detach().cpu().to("cuda:1")
        change_mask = (proj_mask != base_mask).int().detach()
        # change_mask = (proj_mask != last_mask).int().detach()
        save_image(change_mask.float(), ".cache/change_mask.png")
        _change_mask = gaussian_blur(change_mask.float(), 15, 9)
        unchange_mask = (_change_mask<0.8).int().detach()
        _change_mask = (_change_mask>0.2).int().detach()
        bound_mask = _change_mask * unchange_mask
        fix_mask=(1-_change_mask) * change_mask
        change_mask = _change_mask
        save_image(change_mask.float(), ".cache/_change_mask.png")
        save_image(unchange_mask.float(), ".cache/unchange_mask.png")
        save_image(bound_mask.float(), ".cache/bound_mask.png")
        save_image(fix_mask.float(), ".cache/fix_mask.png")

        last_texture = tar_uv_model.get_texture().detach()
        save_image(last_texture, ".cache/last_texture.png")
        # blur_texture = gaussian_blur(last_texture,9,9)

        scheduler=torch.optim.lr_scheduler.MultiStepLR(optim_texture, milestones=[600, 800], gamma=0.5)
        for i in range(1000):
            optim_texture.zero_grad()
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                texture_out = tar_uv_model.render([elev], [azim], 3, dim=render_size)
                pl = perceptual_loss(texture_out["image"], target[-1:])[0][0][0][0]
                ll = torch.nn.functional.l1_loss(texture_out["image"], target[-1:])
                ul = torch.nn.functional.l1_loss(unchange_mask*texture_out["texture_map"], unchange_mask*last_texture)
                bl = 20 * torch.nn.functional.l1_loss(bound_mask*texture_out["texture_map"], bound_mask*last_texture)
                ol = 100*torch.nn.functional.l1_loss(fix_mask*texture_out["texture_map"], fix_mask*last_texture)
                loss = pl + ll + bl + ul + ol

            print(f"{i} pl:{pl}, ll:{ll}, bl:{bl}, ul:{ul}, loss:{loss}", end="\r")
            scaler.scale(loss).backward()
            scaler.step(optim_texture)
            scaler.update()
            scheduler.step()

        print()
        last_mask = (mask_model.texture_map != 0).detach().cpu().to("cuda:1")
        save_image(texture_out["image"], ".cache/mesa_res.png")
        save_image(tar_uv_model.texture_map, ".cache/texture.png")
    with torch.no_grad():
        res, _ = tar_uv_model.render_all()
        save_image(res, "output/image_all.png")
        save_image(tar_uv_model.texture_map, "output/texture.png")
        tar_uv_model.save_texture_unet(".cache/unet.pth")


if __name__ == "__main__":
    cfg = load_confg("configs", "config_refine.yaml")
    main(cfg)
