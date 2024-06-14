import re

import numpy as np
import torch
from torchvision.transforms.functional import to_pil_image

from masactrl.attn_processor import MasaProcessor


def calc_mean_std(feat, dim=-2, eps: float = 1e-5):
    feat_std = (feat.var(dim=dim, keepdims=True) + eps).sqrt()
    feat_mean = feat.mean(dim=dim, keepdims=True)
    return feat_mean, feat_std


def attn_adain(feat):
    b = feat.shape[0]
    feat_style = feat[: b // 2]
    feat_content = feat[b // 2 :]

    feat_style_mean, feat_style_std = calc_mean_std(feat_style)
    feat_mean, feat_std = calc_mean_std(feat_content)

    feat_content = (feat_content - feat_mean) / feat_std
    feat_content = feat_content * feat_style_std + feat_style_mean
    return torch.cat([feat_style, feat_content])


def feat_adain(feat_content, feat_style):
    feat_style_mean, feat_style_std = calc_mean_std(feat_style, 1)
    feat_mean, feat_std = calc_mean_std(feat_content, 1)

    feat_content = (feat_content - feat_mean) / feat_std
    feat_content = feat_content * feat_style_std + feat_style_mean
    return feat_content


def image_transfer(tar_image, style_image):
    tar_image = torch.from_numpy(np.array(tar_image)).float() / 255
    style_image = torch.from_numpy(np.array(style_image)).float() / 255

    eps: float = 1e-5
    mean = tar_image.mean(dim=[0, 1], keepdim=True)
    var = (tar_image.var(dim=[0, 1], keepdims=True) + eps).sqrt()
    style_mean = style_image.mean(dim=[0, 1], keepdim=True)
    style_var = (style_image.var(dim=[0, 1], keepdims=True) + eps).sqrt()

    tar_image = (tar_image - mean) / var

    res = (tar_image * style_var + style_mean).permute(2, 0, 1).clip(0, 1)
    # res = (res-res.min())/(res.max()-res.min())
    # save_image(res, "./test.png")

    res = to_pil_image(res)

    return res


def set_masactrl_attn(model):
    attn_processor = model.attn_processors
    pattern = "(up|mid)\w.*attn1"

    masactrl_processor = MasaProcessor()
    for k, v in attn_processor.items():
        if re.match(pattern, k) is not None:
            attn_processor[k] = masactrl_processor

    model.set_attn_processor(attn_processor)
