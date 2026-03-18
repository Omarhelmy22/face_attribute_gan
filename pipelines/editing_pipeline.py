"""
End-to-end face attribute editing pipeline.

    image  →  align / preprocess  →  e4e encode  →  edit latent  →  decode  →  save
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
import torch

from models.e4e_model import E4E
from editing.latent_editor import LatentEditor
from utils.image_utils import load_image, save_image, align_face, TRANSFORM
from utils.visualization import make_comparison_grid, make_strength_strip, save_grid


class EditingPipeline:
    """
    High-level API that wires together the model, editor, and I/O.

    Parameters
    ----------
    config_path : str
        Path to ``configs/config.yaml``.
    device : str or None
        Override the device specified in config.
    """

    def __init__(self, config_path: str = "configs/config.yaml", device: str = None):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        self.device = device or self.cfg["inference"]["device"]
        if self.device == "cuda" and not torch.cuda.is_available():
            print("CUDA unavailable — falling back to CPU (this will be slow).")
            self.device = "cpu"

        self.model = E4E(
            checkpoint_path=self.cfg["paths"]["checkpoint"],
            device=self.device,
            output_size=self.cfg["model"]["output_size"],
        )

        self.editor = LatentEditor(
            directions_dir=self.cfg["paths"]["directions_dir"],
            device=self.device,
        )

        self.results_dir = Path(self.cfg["paths"]["results_dir"])
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def _get_attr_config(self, attribute: str) -> dict:
        attrs = self.cfg["editing"]["attributes"]
        if attribute not in attrs:
            return {"default_strength": 3.0, "layer_range": None}
        cfg = attrs[attribute]
        lr = cfg.get("layer_range")
        if isinstance(lr, list) and len(lr) == 2:
            lr = tuple(lr)
        else:
            lr = None
        return {
            "default_strength": cfg.get("default_strength", 3.0),
            "layer_range": lr,
        }

    # ── Core editing ────────────────────────────────────────────────

    def run(
        self,
        image_path: str,
        attribute: str,
        strength: float = None,
        layer_range: Optional[Tuple[int, int]] = None,
        save: bool = True,
        align: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Full pipeline: load → invert → edit → decode → save.

        Returns dict with keys: original, reconstruction, edited, latent, edited_latent.
        """
        attr_cfg = self._get_attr_config(attribute)
        if strength is None:
            strength = attr_cfg["default_strength"]
        if layer_range is None:
            layer_range = attr_cfg["layer_range"]

        pil_image = load_image(image_path)

        if align:
            aligned_pil = align_face(pil_image)
            if aligned_pil is not None:
                pil_image = aligned_pil
        input_tensor = TRANSFORM(pil_image).unsqueeze(0).to(self.device)

        latent, reconstruction = self.model.invert(input_tensor)
        edited_latent = self.editor.edit(
            latent, attribute, strength, layer_range=layer_range,
        )
        edited_image = self.model.decode(edited_latent)

        result = {
            "original": input_tensor,
            "reconstruction": reconstruction,
            "edited": edited_image,
            "latent": latent,
            "edited_latent": edited_latent,
        }

        if save:
            self._save_results(image_path, attribute, strength, result)

        return result

    def run_multi_strength(
        self,
        image_path: str,
        attribute: str,
        strengths: List[float],
        layer_range: Optional[Tuple[int, int]] = None,
        save: bool = True,
        align: bool = True,
    ) -> Dict:
        """Run the same image through multiple editing strengths."""
        attr_cfg = self._get_attr_config(attribute)
        if layer_range is None:
            layer_range = attr_cfg["layer_range"]

        pil_image = load_image(image_path)
        if align:
            aligned_pil = align_face(pil_image)
            if aligned_pil is not None:
                pil_image = aligned_pil
        input_tensor = TRANSFORM(pil_image).unsqueeze(0).to(self.device)
        latent, reconstruction = self.model.invert(input_tensor)

        edited_images = []
        for alpha in strengths:
            edited_latent = self.editor.edit(
                latent, attribute, alpha, layer_range=layer_range,
            )
            edited_images.append(self.model.decode(edited_latent))

        result = {
            "original": input_tensor,
            "reconstruction": reconstruction,
            "edited_images": edited_images,
            "strengths": strengths,
        }

        if save:
            stem = Path(image_path).stem
            strip = make_strength_strip(
                reconstruction, edited_images, strengths, attribute,
            )
            out_dir = self.results_dir / attribute
            save_grid(strip, str(out_dir / f"{stem}_strip.png"))

            for img_t, alpha in zip(edited_images, strengths):
                fname = f"{stem}_{attribute}_{alpha:+.1f}.png"
                save_image(img_t, str(out_dir / fname))

        return result

    # ── Save helpers ────────────────────────────────────────────────

    def _save_results(
        self, image_path: str, attribute: str, strength: float,
        result: Dict[str, torch.Tensor],
    ):
        stem = Path(image_path).stem
        out_dir = self.results_dir / attribute
        out_dir.mkdir(parents=True, exist_ok=True)

        save_image(result["original"],
                   str(out_dir / f"{stem}_input_preprocessed.png"))
        save_image(result["reconstruction"],
                   str(out_dir / f"{stem}_reconstruction.png"))
        save_image(result["edited"],
                   str(out_dir / f"{stem}_{attribute}_{strength:+.1f}.png"))

        grid = make_comparison_grid(
            result["original"], result["reconstruction"], result["edited"],
            labels=["Input (aligned)", "Reconstruction", f"{attribute} α={strength:+.1f}"],
        )
        save_grid(grid, str(out_dir / f"{stem}_{attribute}_{strength:+.1f}_grid.png"))
