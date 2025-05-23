import os
import random
import torch
import numpy as np

from PIL import Image

from cl.cl import Contextual_Loss
from torchvision.models.vgg import vgg19

base_size = 512

def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def to_tensor(path, device="cuda"):
    image = Image.open(path).resize((base_size, base_size))
    image = np.array(image)
    image = 2 * (torch.from_numpy(image).float() / 255) - 1
    image = image.permute(2, 0, 1).unsqueeze(0).to(device)
    if image.shape[1] == 4:
        image = image[:, :-1, :, :]
    return image

def get_mask(path, device="cuda"):
    image = Image.open(path).resize((base_size, base_size))
    image = np.array(image)
    image = 2 * (torch.from_numpy(image).float() / 255) - 1
    image = image.permute(2, 0, 1).unsqueeze(0).to(device)
    image = image[:, -1, :, :]!=0
    return image

def main(tar_path):
    seed_everything()
    ours = []
    tex = []
    painter = []
    paint3d = []
    for p in os.listdir(tar_path):
        _, cls, ref_idx, tar_idx = p.split("-")
        # print(cls, ref_idx, tar_idx)

        if cls == "chair":
            ref_texture=f"/home/lrz/TexTrans_/dataset/chair/{ref_idx}/texture.png"
            tar_texture=f"/home/lrz/TexTrans_/dataset/chair/{tar_idx}/texture.png"
        else:
            ref_texture=f"/home/lrz/TexTrans_/dataset/future/{cls}/{ref_idx}/texture.png"
            tar_texture=f"/home/lrz/TexTrans_/dataset/future/{cls}/{tar_idx}/texture.png"
        ref_texture = to_tensor(ref_texture)
        tar_mask = get_mask(tar_texture)
        our_texture=to_tensor(f"{tar_path}/{p}/output/texture.png")
        tex_texture=to_tensor(f"/home/lrz/baseline/TEXTurePaper/experiments/{cls}_{ref_idx}_{tar_idx}/results/step_00010_texture.png")
        painter_texture=to_tensor(f"/home/lrz/baseline/TexPainter/compare_res/{p}/tex_result.png")
        paint_texture=to_tensor(f"/home/lrz/baseline/Paint3D/compare_res/{p}/img_stage2/UV_inpaint_res_0.png")

        # print(ref_texture.shape ,our_texture.shape ,tex_texture.shape ,painter_texture.shape ,paint_texture.shape)

        losser = Contextual_Loss({'conv_2_2': 1.0}).cuda()
        with torch.no_grad():
            ours.append(losser(tar_mask*our_texture, ref_texture))
            tex.append(losser(tar_mask*tex_texture, ref_texture))
            painter.append(losser(tar_mask*painter_texture, ref_texture))
            paint3d.append(losser(tar_mask*paint_texture, ref_texture))

    print(torch.mean(torch.stack(ours)))
    print(torch.mean(torch.stack(tex)))
    print(torch.mean(torch.stack(painter)))
    print(torch.mean(torch.stack(paint3d)))


if __name__ == "__main__":
    main("/home/lrz/TexTrans_/choice")