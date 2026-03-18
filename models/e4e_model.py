"""
Top-level e4e (encoder4editing) model.

Combines:
  - Encoder4Editing (IR-SE50 → W+ codes)
  - StyleGAN2 Generator (W+ codes → 1024×1024 images)
  - latent_avg (mean W code for residual prediction)

The pretrained checkpoint bundles all three components, so no separate
StyleGAN2 weight file is needed.
"""

import math

import torch
import torch.nn as nn

from models.stylegan2.model import Generator
from models.encoders.psp_encoders import Encoder4Editing


def _filter_state_dict(state_dict, prefix):
    """Extract keys with a given prefix and strip the prefix."""
    return {
        k[len(prefix) + 1:]: v
        for k, v in state_dict.items()
        if k.startswith(prefix + ".")
    }


class E4E(nn.Module):
    """
    Encoder-for-Editing wrapper.

    Parameters
    ----------
    checkpoint_path : str or None
        Path to the e4e_ffhq_encode.pt checkpoint.
    device : str
        Target device ('cuda' or 'cpu').
    output_size : int
        Generator output resolution (1024 for FFHQ).
    """

    def __init__(self, checkpoint_path=None, device="cuda", output_size=1024):
        super().__init__()
        self.device = device
        self.output_size = output_size
        self.n_styles = int(math.log(output_size, 2)) * 2 - 2  # 18 for 1024

        self.encoder = Encoder4Editing(
            num_layers=50, mode="ir_se", n_styles=self.n_styles,
        )
        self.decoder = Generator(output_size, 512, 8)
        self.latent_avg = None

        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path)

        self.to(device)
        self.eval()

    def _load_checkpoint(self, path):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)

        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt

        self.encoder.load_state_dict(
            _filter_state_dict(state_dict, "encoder"), strict=True,
        )
        self.decoder.load_state_dict(
            _filter_state_dict(state_dict, "decoder"), strict=True,
        )

        if "latent_avg" in state_dict:
            self.latent_avg = state_dict["latent_avg"].to(self.device)
        elif "latent_avg" in ckpt:
            self.latent_avg = ckpt["latent_avg"].to(self.device)

    @torch.no_grad()
    def encode(self, x):
        """
        Encode a preprocessed image tensor to W+ latent codes.

        Parameters
        ----------
        x : Tensor  (B, 3, 256, 256)  in [-1, 1]

        Returns
        -------
        codes : Tensor  (B, n_styles, 512)
        """
        codes = self.encoder(x)
        if self.latent_avg is not None:
            codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)
        return codes

    @torch.no_grad()
    def decode(self, codes, randomize_noise=False):
        """
        Decode W+ latent codes to images.

        Parameters
        ----------
        codes : Tensor  (B, n_styles, 512)

        Returns
        -------
        images : Tensor  (B, 3, output_size, output_size)  in [-1, 1]
        """
        images, _ = self.decoder(
            [codes], input_is_latent=True, randomize_noise=randomize_noise,
        )
        return images

    @torch.no_grad()
    def invert(self, x):
        """
        Full inversion: image → latent code + reconstructed image.

        Parameters
        ----------
        x : Tensor  (B, 3, 256, 256)  in [-1, 1]

        Returns
        -------
        codes : Tensor  (B, n_styles, 512)
        reconstructed : Tensor  (B, 3, output_size, output_size)  in [-1, 1]
        """
        codes = self.encode(x)
        reconstructed = self.decode(codes)
        return codes, reconstructed
