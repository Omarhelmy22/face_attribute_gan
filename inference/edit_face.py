"""
CLI tool for face attribute editing.

Usage
-----
    python inference/edit_face.py \
        --image  data/input_images/person.jpg \
        --attribute age \
        --strength 3

    python inference/edit_face.py \
        --image  data/input_images/person.jpg \
        --attribute smile \
        --strength -2 \
        --no-align
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path so imports work when
# running the script directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipelines.editing_pipeline import EditingPipeline


def parse_args():
    p = argparse.ArgumentParser(
        description="Face attribute editing via GAN latent-space manipulation.",
    )
    p.add_argument(
        "--image", type=str, required=True,
        help="Path to the input face image.",
    )
    p.add_argument(
        "--attribute", type=str, required=True,
        help="Attribute to edit (e.g. age, smile, pose, hair).",
    )
    p.add_argument(
        "--strength", type=float, default=None,
        help="Editing strength α (positive or negative). "
             "Defaults to the value in config.yaml.",
    )
    p.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Path to config YAML.",
    )
    p.add_argument(
        "--device", type=str, default=None,
        help="Override device (cuda / cpu).",
    )
    p.add_argument(
        "--layer-start", type=int, default=None,
        help="Start of W+ layer range for editing.",
    )
    p.add_argument(
        "--layer-end", type=int, default=None,
        help="End of W+ layer range for editing.",
    )
    p.add_argument(
        "--no-align", action="store_true",
        help="Skip face alignment (use if image is already aligned).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    layer_range = None
    if args.layer_start is not None and args.layer_end is not None:
        layer_range = (args.layer_start, args.layer_end)

    pipeline = EditingPipeline(config_path=args.config, device=args.device)

    print(f"Available attributes: {pipeline.editor.available_attributes}")
    print(f"Editing '{args.attribute}' on {args.image} ...")

    result = pipeline.run(
        image_path=args.image,
        attribute=args.attribute,
        strength=args.strength,
        layer_range=layer_range,
        align=not args.no_align,
    )

    print(f"Done. Results saved to {pipeline.results_dir / args.attribute}/")


if __name__ == "__main__":
    main()
