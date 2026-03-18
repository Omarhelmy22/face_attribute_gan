"""
Visualization helpers for building comparison grids.
"""

from pathlib import Path
from typing import List, Optional

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from utils.image_utils import tensor_to_pil


def _get_font(size: int = 24):
    """Try loading a TTF font; fall back to default bitmap font."""
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
        except OSError:
            return ImageFont.load_default()


def make_comparison_grid(
    original: torch.Tensor,
    reconstruction: torch.Tensor,
    edited: torch.Tensor,
    labels: Optional[List[str]] = None,
    resize: int = 512,
) -> Image.Image:
    """
    Create a side-by-side comparison: Original | Reconstructed | Edited.

    Parameters
    ----------
    original, reconstruction, edited : Tensor  (1, 3, H, W) in [-1, 1]
    labels : list of 3 strings, or None for defaults
    resize : target height/width for each panel

    Returns
    -------
    grid : PIL.Image
    """
    if labels is None:
        labels = ["Original", "Reconstruction", "Edited"]

    panels = []
    for t in [original, reconstruction, edited]:
        img = tensor_to_pil(t).resize((resize, resize), Image.LANCZOS)
        panels.append(img)

    header_h = 36
    grid_w = resize * len(panels)
    grid_h = resize + header_h
    grid = Image.new("RGB", (grid_w, grid_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(grid)
    font = _get_font(22)

    for i, (panel, label) in enumerate(zip(panels, labels)):
        x_offset = i * resize
        grid.paste(panel, (x_offset, header_h))
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        draw.text(
            (x_offset + (resize - tw) // 2, 6),
            label, fill=(0, 0, 0), font=font,
        )

    return grid


def make_strength_strip(
    original: torch.Tensor,
    edited_images: List[torch.Tensor],
    strengths: List[float],
    attribute: str,
    resize: int = 256,
) -> Image.Image:
    """
    Create a strip showing the effect of varying α.

    Layout:  Original | α=-5 | α=-3 | … | α=+3 | α=+5
    """
    n = 1 + len(edited_images)
    header_h = 36
    grid_w = resize * n
    grid_h = resize + header_h
    grid = Image.new("RGB", (grid_w, grid_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(grid)
    font = _get_font(18)

    orig_pil = tensor_to_pil(original).resize((resize, resize), Image.LANCZOS)
    grid.paste(orig_pil, (0, header_h))
    label = "Original"
    bbox = draw.textbbox((0, 0), label, font=font)
    tw = bbox[2] - bbox[0]
    draw.text(((resize - tw) // 2, 6), label, fill=(0, 0, 0), font=font)

    for i, (img_t, alpha) in enumerate(zip(edited_images, strengths)):
        x = (i + 1) * resize
        img = tensor_to_pil(img_t).resize((resize, resize), Image.LANCZOS)
        grid.paste(img, (x, header_h))
        label = f"{attribute} α={alpha:+.1f}"
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        draw.text((x + (resize - tw) // 2, 6), label, fill=(0, 0, 0), font=font)

    return grid


def save_grid(grid: Image.Image, path: str):
    """Save a PIL grid image to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    grid.save(path, quality=95)
