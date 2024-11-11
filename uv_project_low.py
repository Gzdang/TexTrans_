import os
import numpy as np
import torch

from omegaconf import OmegaConf
from PIL import Image
from torchvision.utils import save_image

from mesh.textured_mesh import TexturedMeshModel
from mesh.config import GuideConfig
from mesh.unet.lipis import LPIPS

from config import *
from utils.loader import load_uv_model


def reshape_image(image_batch):
    image_col = image_batch.chunk(2)
    rows = []
    for col in image_col:
        rows.append(torch.cat(col.chunk(3), -2))
    return torch.cat(rows, -1)


if __name__ == "__main__":
    opt = OmegaConf.create(GuideConfig)

    img_size = low_size * 3
    render_size = low_size

    object_list_file = f"{os.environ['HOME']}/dataset/3D_Future/split/chair.txt"
    model = load_uv_model(object_list_file, tar_idx, render_size, uv_size_low, True)

    out_dir = "./output/gen"
    sample_count = len(os.listdir(out_dir))
    out_dir = os.path.join(out_dir, f"sample_{sample_count-1}")
    target_path = os.path.join(out_dir, "masactrl_step.png")
    target = (
        torch.Tensor(np.array(Image.open(target_path).resize((img_size, img_size)).convert("RGB")))
        .permute(2, 0, 1)
        .cuda()
        .unsqueeze(0)
        / 255.0
    )

    perceptual_loss = LPIPS(True).cuda().eval()
    optimizer = torch.optim.Adam(model.parameters(), 1e-5)

    for i in range(1000):
        image, _ = model.render_all()
        loss = torch.nn.functional.l1_loss(image, target)
        loss += perceptual_loss(target, image)[0][0][0][0]
        print(loss)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # save_image(image, "output/proj/test.png")

    res, res_ = model.render_all()
    save_image(res, "./output/proj/image.png")
    save_image(model.texture_map, "./output/proj/texture.png")
    print()