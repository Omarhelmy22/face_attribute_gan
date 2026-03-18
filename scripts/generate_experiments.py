"""
Generate multi-strength editing experiments for the report.

For each input image and each attribute, produces a strip of edits
with varying α values so the report can show gradual changes.

Usage
-----
    python scripts/generate_experiments.py
    python scripts/generate_experiments.py --image data/input_images/person.jpg
    python scripts/generate_experiments.py --attributes age smile
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipelines.editing_pipeline import EditingPipeline

DEFAULT_STRENGTHS = {
    "age":   [-5, -3, -1, 1, 3, 5],
    "smile": [-4, -2, -1, 1, 2, 4],
    "pose":  [-5, -3, -1, 1, 3, 5],
    "hair":  [-6, -3, -1, 1, 3, 6],
}


def parse_args():
    p = argparse.ArgumentParser(description="Batch experiment generator.")
    p.add_argument("--image", type=str, default=None,
                   help="Single image path.  If omitted, process all images "
                        "in data/input_images/.")
    p.add_argument("--attributes", nargs="+", default=None,
                   help="Attributes to test.  Defaults to all available.")
    p.add_argument("--config", type=str, default="configs/config.yaml")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    pipeline = EditingPipeline(config_path=args.config, device=args.device)

    attributes = args.attributes or pipeline.editor.available_attributes
    if not attributes:
        print("No direction files found in weights/directions/. "
              "Please download them first (see README).")
        return

    if args.image:
        images = [args.image]
    else:
        input_dir = Path("data/input_images")
        images = sorted(
            str(p) for p in input_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        )
        if not images:
            print(f"No images found in {input_dir}/")
            return

    for img_path in images:
        for attr in attributes:
            strengths = DEFAULT_STRENGTHS.get(attr, [-3, -1, 1, 3])
            print(f"  {Path(img_path).name} × {attr}  "
                  f"α={strengths}")
            pipeline.run_multi_strength(
                image_path=img_path,
                attribute=attr,
                strengths=strengths,
            )

    print(f"\nAll experiments saved to {pipeline.results_dir}/")


if __name__ == "__main__":
    main()
