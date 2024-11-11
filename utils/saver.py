import os
import torch
import numpy as np

from torchvision.utils import save_image

def save_res(ref_img, tar_img, masa_img, device):
    out_dir = "./output/gen"
    os.makedirs(out_dir, exist_ok=True)
    sample_count = len(os.listdir(out_dir))
    out_dir = os.path.join(out_dir, f"sample_{sample_count}")
    os.makedirs(out_dir, exist_ok=True)

    ref_img = torch.tensor(np.array(ref_img), dtype=torch.float).permute(2, 0, 1).unsqueeze(0).to(device) / 255
    if ref_img.shape[1] == 4:
        ref_img = ref_img[:,:-1,:,:]
    tar_img = torch.tensor(np.array(tar_img), dtype=torch.float).permute(2, 0, 1).unsqueeze(0).to(device) / 255
    out_image = torch.cat([ref_img, masa_img, tar_img], dim=0)

    save_image(out_image, os.path.join(out_dir, f"all_step.png"))
    save_image(out_image[0], os.path.join(out_dir, f"source_step.png"))
    save_image(out_image[1], os.path.join(out_dir, f"masactrl_step.png"))
    save_image(out_image[2], os.path.join(out_dir, f"tar_step.png"))

    print("Syntheiszed images are saved in", out_dir)