import os
import numpy as np
import torch

from omegaconf import OmegaConf
from PIL import Image
from torchvision.utils import save_image

from mesh.textured_mesh import TexturedMeshModel
from mesh.unet.lipis import LPIPS
from utils.loader import load_uv_model


def reshape_image(image_batch):
    image_col = image_batch.chunk(2)
    rows = []
    for col in image_col:
        rows.append(torch.cat(col.chunk(3), -2))
    return torch.cat(rows, -1)


def main(cfg):
    render_size = cfg.masa.size
    img_size = render_size * 3
    
    model = load_uv_model(cfg.mesh, cfg.masa.tar_idx, render_size, True)

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
    optimizer = torch.optim.AdamW(model.parameters(), 1e-4)
    scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1000, 1500], gamma=0.5)
    for i in range(2000):
        image, _ = model.render_all()
        loss = torch.nn.functional.l1_loss(image, target)
        loss += perceptual_loss(target, image)[0][0][0][0]
        print(loss)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        # save_image(image, "output/proj/test.png")

    res, res_ = model.render_all()
    save_image(res, "./output/proj/image.png")
    save_image(model.texture_map, "./output/proj/texture.png")

if __name__ == "__main__":
    cfg = OmegaConf.load("configs/config.yaml")
    main(cfg)
