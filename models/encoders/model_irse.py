"""
IR-SE backbone block configuration.
Defines how many bottleneck units per stage for IR-SE50/100/152.
"""

from collections import namedtuple

Bottleneck = namedtuple("Bottleneck", ["in_channel", "depth", "stride"])


def get_block(in_channel, depth, num_units, stride=2):
    return (
        [Bottleneck(in_channel, depth, stride)]
        + [Bottleneck(depth, depth, 1) for _ in range(num_units - 1)]
    )


def get_blocks(num_layers):
    if num_layers == 50:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14),
            get_block(in_channel=256, depth=512, num_units=3),
        ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3),
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=8),
            get_block(in_channel=128, depth=256, num_units=36),
            get_block(in_channel=256, depth=512, num_units=3),
        ]
    else:
        raise ValueError(f"Unsupported num_layers={num_layers}. Use 50, 100, or 152.")
    return blocks
