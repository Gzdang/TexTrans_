from dataclasses import dataclass
import os
from pathlib import Path
from typing import Optional


@dataclass
class GuideConfig:
    """Parameters defining the guidance"""

    # Guiding text prompt
    text: str = ""
    # The mesh to paint
    shape_path: str = f"{os.environ['HOME']}/dataset/3D_Future/obj/0000002.obj"
    # Append direction to text prompts
    # append_direction: bool = True
    # A Textual-Inversion concept to use
    concept_name: Optional[str] = None
    # Path to the TI embedding
    concept_path: Optional[Path] = None
    # A huggingface diffusion model to use
    diffusion_name: str = "stabilityai/stable-diffusion-2-depth"
    # Scale of mesh in 1x1x1 cube
    shape_scale: float = 1
    # height of mesh
    dy: float = 0
    # texture image resolution
    texture_resolution: int = 128
    # texture mapping interpolation mode from texture image, options: 'nearest', 'bilinear', 'bicubic'
    texture_interpolation_mode: str = "bilinear"
    # Guidance scale for score distillation
    guidance_scale: float = 7.5
    # Use inpainting in relevant iterations
    use_inpainting: bool = True
    # The texture before editing
    reference_texture: Optional[Path] = None
    # The edited texture
    # initial_texture: Optional[Path] = f"{os.environ['HOME']}/dataset/3D_Future/texture/0000002.png"
    initial_texture: Optional[Path] = None
    # Whether to use background color or image
    use_background_color: bool = True
    # Background image to use
    # background_img: str = 'textures/brick_wall.png'
    # Threshold for defining refine regions
    z_update_thr: float = 0.2
    # Some more strict masking for projecting back
    strict_projection: bool = True
