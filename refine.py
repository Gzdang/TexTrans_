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
    prompts = [source_prompt, source_prompt, target_prompt] 

    reset_attn(model) 
    _, latents_list = model.invert(
            ref_image,
            source_prompt,
            guidance_scale=1,
            num_inference_steps=num_step,
            return_intermediates=True,
            base_resolution=size,
            strength = strength
        )       
    # save_image(model.latent2image(latents_list[0].float(), "pt"), "test.png")

    for _i in range(5):
        reset_attn(model) 

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
            control_scale=1.0,
            base_resolution=size,
            strength = strength
        )

        tar_image = image_masactrl

        if (_i+1)in (1, 2, 3, 5, 10, 20, 30, 40, 50):
            save_image(image_masactrl.float(), f".cache/_tar_{_i+1}.png")
        # strength /= 2

    return image_masactrl

def gen_mask(cfg, mesh_path, render_size):
    mask_model = load_uv_model(cfg, mesh_path, render_size, False)
    mask_model.texture_map = torch.nn.Parameter(torch.zeros_like(mask_model.texture_map))
    optim_mask = torch.optim.Adam(mask_model.parameters(), 1e-2)
    for _ in range(5):
        render_res = mask_model.render_all()
        loss = torch.nn.functional.l1_loss(render_res["image"], render_res["mask"].detach().repeat((1,3,1,1)))
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
    tar_uv_model = load_uv_model(cfg.mesh, cfg.tar_mesh, render_size, True, init_texture=cfg.tar_texture, device="cuda:1")
    ref_uv_model = load_uv_model(cfg.mesh, cfg.ref_mesh, render_size, False, init_texture=cfg.ref_texture)
    ref_uv_model.requires_grad_(False)

    gen_mask(cfg.mesh, cfg.tar_mesh, 512)
    mask_model = load_uv_model(cfg.mesh, cfg.tar_mesh, render_size, False, init_texture=".cache/_mask.png")

    base_texture = tar_uv_model.get_texture().detach()
    base_mask = (mask_model.texture_map != 0).detach().cpu().to("cuda:1")
    last_mask = base_mask

    # if cfg.n_c == 9:
    #     elev_list = [t*np.pi for t in (1/4, 1/4, 1/4, 1/2, 1/2, 1/2, 3/4, 3/4, 3/4)]
    #     azim_list = [t*np.pi for t in (5/3, 1, 1/3, 4/3, 0, 2/3, 5/3, 1, 1/3)]
    # elif cfg.n_c == 6:
    #     elev_list = [t*np.pi for t in (1/4, 1/4, 1/4, 1/2, 1/2, 1/2, 3/4, 3/4, 3/4)]
    #     azim_list = [t*np.pi for t in (5/3, 1, 1/3, 4/3, 0, 2/3, 5/3, 1, 1/3)]
    # elif cfg.n_c == 4:
    #     elev_list = [t*np.pi for t in (1/2, 1/2, 1/2, 1/2)]
    #     azim_list = [t*np.pi for t in (0, 1/2, 1, 3/2)]

    # elev_list = [t * np.pi for t in (3/4, 1/2, 1/2, 1/2, 1/4, )]
    # azim_list = [t * np.pi for t in (0, 1, 1/3, 5/3, 0, )]

    # elev_list = [t*np.pi for t in (1/2, 1/2, 1/2, 1/2)]
    # azim_list = [t*np.pi for t in (1/4, 3/4, 5/4, 7/4)]
    # elev_list = [t*np.pi for t in (1/3, 11/18, 1/3, 11/18, 1/3, 11/18,)]
    # azim_list = [t*np.pi for t in (30/180, 90/180, 150/180, 210/180, 270/180, 330/180)]

    # elev_list = [t*np.pi for t in (3/4, 3/4, 3/4, 1/4, 1/4, 1/4, 1/2, 1/2, 1/3, )]
    # azim_list = [t*np.pi for t in (1, 1/3, 5/3, 5/3, 1/3, 1, 4/3, 2/3, 0, )]
    # elev_list = [t*np.pi for t in (1/3, 3/4, 1/4, 1/4, 1/4, 1/2, 1/2, 1/3, )]
    # azim_list = [t*np.pi for t in (0, 1, 5/3, 1/3, 1, 4/3, 2/3, 0, )]
    elev_list = [t*np.pi for t in (1/3, 1/3, 1/3, 1/3, 1/3, 1/3, )]
    azim_list = [t*np.pi for t in (1, 4/3, 2/3, 5/3, 1/3, 0, )]
    # elev_list = [t*np.pi for t in (1/3,)]
    # azim_list = [t*np.pi for t in (0,)]

    # elev_list = [t*np.pi for t in (1/3, 1/3, 1/4, )]
    # azim_list = [t*np.pi for t in (2/3, 4/3, 0, )]

    # elev_list = [t*np.pi for t in (5/6, 1/3, 1/4, 1/4, 1/3)]
    # azim_list = [t*np.pi for t in (0, 1, 3/2, 1/2, 0)]


    optim_mask = torch.optim.Adam(mask_model.parameters(), 1e-2)
    mask_sc=torch.optim.lr_scheduler.MultiStepLR(optim_mask, milestones=[100], gamma=0.1)
    optim_texture = torch.optim.Adam(tar_uv_model.parameters(), 1e-3)
    scaler = GradScaler()
    perceptual_loss = LPIPS(True).to("cuda:1").eval()

    for elev, azim in zip(elev_list, azim_list):
        ref_render = ref_uv_model.render([elev], [azim], 3, "white", render_size)
        ref_image = ref_render["image"].half()
        ref_depth = ref_render["depth"]
        ref_normal = ref_render["normal"]

        tar_render = tar_uv_model.render([elev], [azim], 3, "white", render_size)
        tar_image = tar_render["image"].clamp(0,1).detach().half().cpu().to(device)
        tar_depth = tar_render["depth"].detach().cpu().to(device)
        tar_normal = tar_render["normal"].detach().cpu().to(device)

        save_image(tar_render["image"], ".cache/tar.png")
        save_image(tar_depth, ".cache/tar_depth.png")
        save_image(tar_normal, ".cache/tar_normal.png")
        save_image(ref_render["image"], ".cache/ref.png")
        save_image(ref_depth, ".cache/ref_depth.png")
        save_image(ref_normal, ".cache/ref_normal.png")

        mask_out = mask_model.render([elev], [azim], 3, dim=render_size)
        last_normal_tex = mask_model.texture_map.detach().clone()
        last_normal = mask_out["image"][:, -1:, :, :].clamp(0, 1)
        cur_normal=tar_render["normal"][:, -1:, :, :].clamp(0, 1).cpu().to("cuda:0")
        cm = cur_normal < last_normal
        cur_normal[cm] = 0
        update_normal = last_normal * cm + cur_normal
        # save_image((cur_normal.cpu() + 1-tar_render["mask"].cpu()), ".cache/lcur_normal.png")
        # save_image(last_normal, ".cache/last_normal.png")
        # save_image(update_normal, ".cache/update_normal.png")
        # save_image(cur_normal, ".cache/cur_normal.png")
        
        # save_image(last_normal_tex, ".cache/last_normal_tex.png")

        for _ in range(200):
            mask_out = mask_model.render([elev], [azim], 3, dim=render_size)
            loss = torch.nn.functional.l1_loss(mask_out["image"], update_normal.repeat(1, 3, 1, 1).detach())
            loss.backward()
            optim_mask.step()
            optim_mask.zero_grad()
            mask_sc.step()
        save_image(mask_model.texture_map, ".cache/mask_res.png")

        normal_mask = gaussian_blur(cur_normal, 15, 9)
        normal_mask = (tar_render["mask"]*(normal_mask>0.2).to("cuda:1"))
        save_image(normal_mask, ".cache/normal_mask.png")

        # sleep(2)
        
        # 0 -- openpose
        # 1 -- depth
        # 2 -- hed/pidi/scribble/ted
        # 3 -- canny/lineart/anime_lineart/mlsd
        # 4 -- normal
        # 5 -- segment
        control = [0, 0, 0, 0, 0, 0]
        control[1] = torch.cat([ref_depth.repeat(1, 3, 1, 1), ref_depth.repeat(1, 3, 1, 1), tar_depth.repeat(1, 3, 1, 1)])
        # control[4] = torch.cat([ref_normal, ref_normal, tar_normal])
        # control = {"depth": [ref_depth.repeat(1, 3, 1, 1), ref_depth.repeat(1, 3, 1, 1), tar_depth.repeat(1, 3, 1, 1)]}

        change_mask = (torch.abs(mask_model.texture_map[:, -1:, :, :] - last_normal_tex[:, -1:, :, :])>0.05).detach().int().cpu().to("cuda:1")
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

        last_texture = tar_uv_model.get_texture().detach().clone()
        save_image(last_texture, ".cache/last_texture.png")

        target = sde(model, ref_image, tar_image, control, cfg.model.num_step, cfg.strength, render_size).detach().cpu().to("cuda:1")
        save_image(target, ".cache/target.png")

        scheduler=torch.optim.lr_scheduler.MultiStepLR(optim_texture, milestones=[400, 600, 700], gamma=0.5)
        for i in range(800):
            optim_texture.zero_grad()
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                texture_out = tar_uv_model.render([elev], [azim], 3, dim=render_size, background="white", render_cache=tar_render["render_cache"])
                
                pl = perceptual_loss(texture_out["image"]*normal_mask, target[-1:]*normal_mask)[0][0][0][0]
                ll = torch.nn.functional.l1_loss(texture_out["image"]*normal_mask, target[-1:]*normal_mask)
                ul = torch.nn.functional.l1_loss(unchange_mask*texture_out["texture_map"], unchange_mask*last_texture, reduction="sum")/torch.sum(unchange_mask)
                bl = torch.nn.functional.l1_loss(bound_mask*texture_out["texture_map"], bound_mask*last_texture, reduction="sum")/torch.sum(bound_mask)
                ol = torch.nn.functional.l1_loss(fix_mask*texture_out["texture_map"], fix_mask*last_texture, reduction="sum")/torch.sum(fix_mask)
                loss = pl + ll + bl + ul + ol

            print(f"{i} pl:{pl:.4f}, ll:{ll:.4f}, bl:{bl:.4f}, ul:{ul:.4f}, ol:{ol:.4f}, loss:{loss:.4f}", end="\r")
            scaler.scale(loss).backward()
            scaler.step(optim_texture)
            scaler.update()
            scheduler.step()

        print()
        save_image(texture_out["image"], ".cache/mesa_res.png")
        save_image(tar_uv_model.texture_map, ".cache/texture.png")

        tar_uv_model.save_texture_unet(".cache/unet.pth")
        cfg.mesh.texture_unet_path = ".cache/unet.pth"
        tar_uv_model = load_uv_model(cfg.mesh, cfg.tar_mesh, render_size, True, init_texture=".cache/texture.png", device="cuda:1")
        # tar_uv_model.refresh()
        optim_texture = torch.optim.Adam(tar_uv_model.parameters(), 1e-3)
    with torch.no_grad():
        render_res = tar_uv_model.render_all()
        save_image(render_res["image"], "output/image_all.png")
        save_image(tar_uv_model.texture_map, "output/texture.png")
        tar_uv_model.save_texture_unet(".cache/unet.pth")


if __name__ == "__main__":
    cfg = load_confg("configs", "config_refine.yaml")
    main(cfg)
