"""
Encoder4Editing encoder architecture.
Ported from omertov/encoder4editing for checkpoint compatibility.

Uses an IR-SE50 backbone with an FPN-like feature pyramid and
GradualStyleBlock heads to map features → W+ latent codes.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import (
    Sequential, BatchNorm2d, Conv2d, PReLU,
)

from models.encoders.helpers import bottleneck_IR_SE
from models.encoders.model_irse import get_blocks
from models.stylegan2.model import EqualLinear


class GradualStyleBlock(nn.Module):
    """
    Progressively downsample spatial features to a single style vector (512-d).
    Uses a chain of strided convolutions followed by an equalized linear layer.
    """

    def __init__(self, in_c, out_c, spatial):
        super().__init__()
        self.out_c = out_c
        self.spatial = spatial
        num_pools = int(np.log2(spatial))
        modules = [Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
                   nn.LeakyReLU()]
        for _ in range(num_pools - 1):
            modules += [Conv2d(out_c, out_c, kernel_size=3, stride=2, padding=1),
                        nn.LeakyReLU()]
        self.convs = Sequential(*modules)
        self.linear = EqualLinear(out_c, out_c, lr_mul=1)

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self.out_c)
        x = self.linear(x)
        return x


class Encoder4Editing(nn.Module):
    """
    e4e encoder that maps a 256×256 face image to an 18×512 W+ latent code.

    Architecture:
      - IR-SE50 backbone produces multi-scale features
      - FPN-like lateral connections merge scales
      - GradualStyleBlock heads map each scale to W+ style vectors

    Feature extraction indices (for 50-layer IR-SE):
      - c1 @ body index 6  → 128-ch, 64×64  (end of block 1)
      - c2 @ body index 20 → 256-ch, 32×32  (end of block 2)
      - c3 @ body index 23 → 512-ch, 16×16  (end of block 3)
    """

    def __init__(self, num_layers, mode="ir_se", n_styles=18):
        super().__init__()
        assert num_layers in (50, 100, 152)
        assert mode == "ir_se"
        blocks = get_blocks(num_layers)

        self.input_layer = Sequential(
            Conv2d(3, 64, (3, 3), 1, 1, bias=False),
            BatchNorm2d(64),
            PReLU(64),
        )

        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    bottleneck_IR_SE(bottleneck.in_channel,
                                    bottleneck.depth,
                                    bottleneck.stride)
                )
        self.body = Sequential(*modules)

        self.styles = nn.ModuleList()
        self.style_count = n_styles
        self.coarse_ind = 3
        self.middle_ind = 7

        for i in range(self.style_count):
            if i < self.coarse_ind:
                style = GradualStyleBlock(512, 512, 16)
            elif i < self.middle_ind:
                style = GradualStyleBlock(512, 512, 32)
            else:
                style = GradualStyleBlock(512, 512, 64)
            self.styles.append(style)

        self.latlayer1 = Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = Conv2d(128, 512, kernel_size=1, stride=1, padding=0)

    @staticmethod
    def _upsample_add(x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode="bilinear",
                             align_corners=True) + y

    def forward(self, x):
        x = self.input_layer(x)

        modulelist = list(self.body._modules.values())
        for i, layer in enumerate(modulelist):
            x = layer(x)
            if i == 6:
                c1 = x
            elif i == 20:
                c2 = x
            elif i == 23:
                c3 = x

        # styles[0] predicts the BASE W code, duplicated across all layers.
        # styles[1..N] predict DELTAS added to that base (e4e architecture).
        w0 = self.styles[0](c3)
        w = w0.repeat(self.style_count, 1, 1).permute(1, 0, 2)

        features = c3
        for i in range(1, self.style_count):
            if i == self.coarse_ind:
                p2 = self._upsample_add(c3, self.latlayer1(c2))
                features = p2
            elif i == self.middle_ind:
                p1 = self._upsample_add(p2, self.latlayer2(c1))
                features = p1
            delta_i = self.styles[i](features)
            w[:, i] += delta_i

        return w
