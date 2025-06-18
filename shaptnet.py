import os
import torch
import numpy as np

from transformers import pipeline
from PIL import Image
from torchvision.utils import save_image

# load pipe
pipe = pipeline(task="depth-estimation", model="resource/diffuser/depth_anything")

def i2t(image):
    image = np.array(image)
    image = torch.from_numpy(image).float() / 255
    image = image.permute(2, 0, 1).unsqueeze(0)
    if image.shape[1] == 4:
        image = image[:, :-1, :, :]
    return image

def get_depth(img):
    return pipe(img)["depth"]

def get_ref(ref_idx, size):
    image_path = f"resource/dataset/out/img/02958343/{ref_idx}"
    image_list = []
    depth_list = []
    for i in [2,1,0,5,4,3,8,7,6]:
        image = Image.open(f"{image_path}/00{str(i)}.png").transpose(Image.FLIP_LEFT_RIGHT)
        depth = get_depth(image).convert("RGB")

        image_list.append(i2t(image.resize((size, size))))
        depth_list.append(i2t(depth.resize((size, size))))

    os.makedirs(".cache/ref", exist_ok=True)
    save_image(torch.cat(image_list), ".cache/ref/all_rgb.png", nrow=3, padding=0)
    save_image(torch.cat(depth_list), ".cache/ref/all_depth.png", nrow=3, padding=0)

    return Image.open(".cache/ref/all_rgb.png"), Image.open(".cache/ref/all_depth.png")

def get_tar(tar_idx, size):
    image_path = f"resource/dataset/out/img/02958343/{tar_idx}"
    image_list = []
    depth_list = []
    for i in [2,1,0,5,4,3,8,7,6]:
        image = Image.open(f"{image_path}/00{str(i)}.png").transpose(Image.FLIP_LEFT_RIGHT)
        depth = get_depth(image).convert("RGB")

        image_list.append(i2t(image.resize((size, size))))
        depth_list.append(i2t(depth.resize((size, size))))

    os.makedirs(".cache/tar", exist_ok=True)
    save_image(torch.cat(image_list), ".cache/tar/all_rgb.png", nrow=3, padding=0)
    save_image(torch.cat(depth_list), ".cache/tar/all_depth.png", nrow=3, padding=0)

    return Image.open(".cache/tar/all_rgb.png"), Image.open(".cache/tar/all_depth.png")

if __name__ == "__main__":
     get_ref("1a0bc9ab92c915167ae33d942430658c")
