import os
import numpy as np
import torch
from PIL import Image
from torchvision.utils import save_image

from mesh.textured_mesh import TexturedMeshModel
from mesh.config import GuideConfig
from omegaconf import OmegaConf


def reshape_image(image_batch):
    image_col = image_batch.chunk(2)
    rows = []
    for col in image_col:
        rows.append(torch.cat(col.chunk(3), -2))
    return torch.cat(rows, -1)


if __name__ == "__main__":
    opt = OmegaConf.create(GuideConfig)

    opt.shape_path = f"{os.environ['HOME']}/dataset/3D_Future/obj/0000017.obj"
    model = TexturedMeshModel(opt, device="cuda", use_unet=False)

    out_dir = "./output/gen"
    sample_count = len(os.listdir(out_dir))
    out_dir = os.path.join(out_dir, f"sample_{sample_count-1}")
    target_path = os.path.join(out_dir, "masactrl_step.png")
    target = (
        torch.Tensor(np.array(Image.open(target_path).resize((1023, 1023)).convert("RGB")))
        .permute(2, 0, 1)
        .cuda()
        .unsqueeze(0)
        / 255.0
    )

    optimizer = torch.optim.Adam(model.parameters(), 0.01)

    for i in range(300):
        image, _ = model.render_all()
        loss = torch.nn.functional.l1_loss(image, target)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    res, res_ = model.render_all()
    save_image(res, "./output/proj/image.png")
    save_image(model.texture_map, "./output/proj/texture.png")
    print()
