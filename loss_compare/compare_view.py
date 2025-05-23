import os
import random
import torch
import numpy as np

from PIL import Image

from mesh.loss import Losser
from torchvision.models.vgg import vgg19

def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def to_tensor(path, device="cuda"):
    image = Image.open(path)
    image = np.array(image)
    image = 2 * (torch.from_numpy(image).float() / 255) - 1
    image = image.permute(2, 0, 1).unsqueeze(0).half().to(device)
    if image.shape[1] == 4:
        image = image[:, :-1, :, :]
    return image

def main(tar_path):
    seed_everything()
    ours = []
    tex = []
    painter = []
    paint3d = []

    vgg = vgg19(pretrained=True).to("cuda").features.eval()
    for cls in os.listdir(tar_path):
        cls_path = os.path.join(tar_path, cls)
        ref_path = os.path.join(tar_path, cls, "result_ref")
        sub_list = os.listdir(cls_path)
        sub_list.remove("result_ref")
        for idx in sub_list:
            our_path = os.path.join(cls_path, idx, "result_ours")
            tex_path = os.path.join(cls_path, idx, "result_tex")
            painter_path = os.path.join(cls_path, idx, "result_painter")
            paint_path = os.path.join(cls_path, idx, "result_3d")

            for i in range(9):
                ref_img = to_tensor(f"{ref_path}/view{i*12}_rgb_render.png")
                our_img=to_tensor(f"{our_path}/view{i*12}_rgb_render.png")
                tex_img=to_tensor(f"{tex_path}/view{i*12}_rgb_render.png")
                painter_img=to_tensor(f"{painter_path}/view{i*12}_rgb_render.png")
                paint_img=to_tensor(f"{paint_path}/view{i*12}_rgb_render.png")

                losser = Losser(ref_img, None, vgg)
                ours.append(losser.get_loss(our_img))
                tex.append(losser.get_loss(tex_img))
                painter.append(losser.get_loss(painter_img))
                paint3d.append(losser.get_loss(paint_img))

    print(len(ours))
    print(torch.mean(torch.stack(ours)))
    print(torch.mean(torch.stack(tex)))
    print(torch.mean(torch.stack(painter)))
    print(torch.mean(torch.stack(paint3d)))


if __name__ == "__main__":
    main("/home/lrz/baseline/Paint3D/user_study/images")