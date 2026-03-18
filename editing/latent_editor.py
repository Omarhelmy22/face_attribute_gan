"""
Latent-space editing via direction vectors.

Supports two families of directions:
  1. InterfaceGAN  — supervised linear boundaries (age, smile, pose)
  2. SeFa          — unsupervised eigenvector directions (hair, etc.)

Both reduce to the same arithmetic:  w_new = w + α · d
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import numpy as np


class LatentEditor:
    """
    Load direction vectors and apply them to W+ latent codes.

    Parameters
    ----------
    directions_dir : str or Path
        Folder containing ``*.pt`` or ``*.npy`` direction files.
    device : str
        Target device.
    """

    def __init__(self, directions_dir: str, device: str = "cuda"):
        self.device = device
        self.directions: Dict[str, torch.Tensor] = {}
        self.directions_dir = Path(directions_dir)
        self._load_all()

    def _load_all(self):
        """Auto-discover and load every .pt / .npy file in the directory."""
        if not self.directions_dir.exists():
            return
        for p in sorted(self.directions_dir.iterdir()):
            if p.suffix == ".pt":
                vec = torch.load(p, map_location="cpu", weights_only=False)
                if not isinstance(vec, torch.Tensor):
                    continue
            elif p.suffix == ".npy":
                vec = torch.from_numpy(np.load(str(p))).float()
            else:
                continue
            if vec.ndim >= 2 and vec.shape[0] == 1:
                vec = vec.squeeze(0)
            self.directions[p.stem] = vec.to(self.device)

    @property
    def available_attributes(self):
        return list(self.directions.keys())

    def edit(
        self,
        latent: torch.Tensor,
        attribute: str,
        strength: float = 3.0,
        layer_range: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        Apply an attribute edit to a W+ latent code.

        Parameters
        ----------
        latent : Tensor (B, n_styles, 512)
            Original W+ code.
        attribute : str
            Name matching a loaded direction file (e.g. 'age', 'smile').
        strength : float
            Scalar multiplier α.  Positive / negative for opposite effects.
        layer_range : (start, end) or None
            If given, only modify W+ layers in [start, end).
            Use coarse layers (0–6) for structural changes like hair.
            Use all layers (None) for holistic changes like age.

        Returns
        -------
        edited : Tensor  same shape as *latent*
        """
        if attribute not in self.directions:
            raise ValueError(
                f"Unknown attribute '{attribute}'. "
                f"Available: {self.available_attributes}"
            )

        direction = self.directions[attribute]

        # Normalize to (1, n_styles, 512) regardless of stored shape:
        #   (512,)          → shared across all layers
        #   (1, 512)        → shared across all layers
        #   (18, 512)       → per-layer direction
        #   (1, 18, 512)    → per-layer direction
        if direction.ndim == 1:
            delta = direction.unsqueeze(0).unsqueeze(0)
        elif direction.ndim == 2 and direction.shape[0] == latent.shape[1]:
            delta = direction.unsqueeze(0)
        elif direction.ndim == 3:
            delta = direction
        else:
            delta = direction.view(1, 1, -1)
        delta = delta.expand_as(latent)

        if layer_range is not None:
            mask = torch.zeros_like(delta)
            mask[:, layer_range[0]:layer_range[1], :] = 1.0
            delta = delta * mask

        return latent + strength * delta

    def edit_multi(
        self,
        latent: torch.Tensor,
        edits: list,
    ) -> torch.Tensor:
        """
        Apply multiple edits sequentially.

        Parameters
        ----------
        edits : list of dict
            Each dict has keys: attribute, strength, layer_range (optional).
        """
        result = latent
        for e in edits:
            result = self.edit(
                result,
                attribute=e["attribute"],
                strength=e["strength"],
                layer_range=e.get("layer_range"),
            )
        return result
