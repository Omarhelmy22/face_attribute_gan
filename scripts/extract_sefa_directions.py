"""
Extract SeFa (Closed-Form Factorization) directions from the StyleGAN2
generator bundled inside the e4e checkpoint.

SeFa decomposes the modulation weight matrices of the styled convolutions
to discover semantically meaningful editing directions — no labelled data
required.  This is the recommended way to obtain a *hair* direction, which
InterfaceGAN does not provide.

Usage
-----
    python scripts/extract_sefa_directions.py

    # Keep top 20 components, restrict to layers 0-6 (coarse / structural):
    python scripts/extract_sefa_directions.py --n-components 20 --layer-start 0 --layer-end 6

After running, inspect the visualisation grid saved in results/ to identify
which component index corresponds to hair (or other attributes).  Then copy
the matching direction into weights/directions/hair.pt.

Usage with visualisation
------------------------
    python scripts/extract_sefa_directions.py --visualise --n-samples 5
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
from tqdm import tqdm

from models.e4e_model import E4E
from utils.image_utils import tensor_to_pil
from utils.visualization import save_grid


def extract_sefa(generator, layer_range=(0, 6), n_components=10):
    """
    Compute SeFa directions from the generator's modulation weights.

    Parameters
    ----------
    generator : models.stylegan2.model.Generator
    layer_range : (start, end)
        Which styled convolution layers to include.
    n_components : int
        Number of principal directions to return.

    Returns
    -------
    directions : Tensor (n_components, 512)
    eigenvalues : Tensor (n_components,)
    """
    weights = []

    # conv1 is the first styled conv (layer index 0 in W+)
    mod_weight = generator.conv1.conv.modulation.weight  # (in_ch, 512)
    if layer_range[0] == 0:
        weights.append(mod_weight)

    for idx, conv in enumerate(generator.convs):
        layer_idx = idx + 1  # conv1 is 0, convs[0] is 1, etc.
        if layer_range[0] <= layer_idx <= layer_range[1]:
            weights.append(conv.conv.modulation.weight)

    W = torch.cat(weights, dim=0).detach().cpu().float()  # (N, 512)
    # Eigendecomposition of W^T W
    WtW = W.T @ W
    eigenvalues, eigenvectors = torch.linalg.eigh(WtW)

    # Sort descending
    idx = eigenvalues.argsort(descending=True)
    eigenvalues = eigenvalues[idx][:n_components]
    directions = eigenvectors[:, idx][:, :n_components].T  # (n_comp, 512)

    return directions, eigenvalues


def visualise_directions(model, directions, n_samples=5, strength=5.0,
                         output_dir="results/sefa"):
    """Generate a grid showing what each SeFa component does."""
    from PIL import Image

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = next(model.parameters()).device
    n_comp = directions.shape[0]

    # Sample random W+ codes via the mapping network
    with torch.no_grad():
        z = torch.randn(n_samples, 512, device=device)
        w = model.decoder.style(z).unsqueeze(1).repeat(1, model.n_styles, 1)

    for comp_idx in tqdm(range(n_comp), desc="Visualising SeFa components"):
        d = directions[comp_idx].to(device)
        panels = []
        for s_idx in range(n_samples):
            latent = w[s_idx:s_idx + 1]  # (1, 18, 512)

            imgs = []
            for alpha in [-strength, 0, strength]:
                edited = latent + alpha * d.view(1, 1, 512)
                img, _ = model.decoder(
                    [edited], input_is_latent=True, randomize_noise=False,
                )
                imgs.append(tensor_to_pil(img).resize((256, 256), Image.LANCZOS))

            row = Image.new("RGB", (256 * 3, 256))
            for i, im in enumerate(imgs):
                row.paste(im, (i * 256, 0))
            panels.append(row)

        grid = Image.new("RGB", (256 * 3, 256 * n_samples))
        for i, row in enumerate(panels):
            grid.paste(row, (0, i * 256))
        grid.save(str(output_dir / f"component_{comp_idx:03d}.png"))

    print(f"Saved visualisation grids to {output_dir}/")
    print("Inspect the grids and identify the component that corresponds to hair.")
    print("Then save that direction:")
    print("  torch.save(directions[INDEX], 'weights/directions/hair.pt')")


def main():
    p = argparse.ArgumentParser(description="Extract SeFa directions.")
    p.add_argument("--checkpoint", type=str, default="weights/e4e_ffhq_encode.pt")
    p.add_argument("--n-components", type=int, default=15)
    p.add_argument("--layer-start", type=int, default=0)
    p.add_argument("--layer-end", type=int, default=6)
    p.add_argument("--output", type=str, default="weights/directions/sefa_directions.pt")
    p.add_argument("--visualise", action="store_true")
    p.add_argument("--n-samples", type=int, default=5)
    p.add_argument("--strength", type=float, default=5.0)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("CUDA unavailable — using CPU.")

    print("Loading e4e checkpoint ...")
    model = E4E(checkpoint_path=args.checkpoint, device=device)

    print(f"Extracting SeFa directions (layers {args.layer_start}–{args.layer_end}, "
          f"top {args.n_components} components) ...")
    directions, eigenvalues = extract_sefa(
        model.decoder,
        layer_range=(args.layer_start, args.layer_end),
        n_components=args.n_components,
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "directions": directions,
        "eigenvalues": eigenvalues,
        "layer_range": (args.layer_start, args.layer_end),
    }, args.output)
    print(f"Saved {args.n_components} directions to {args.output}")

    if args.visualise:
        visualise_directions(
            model, directions,
            n_samples=args.n_samples,
            strength=args.strength,
        )

    print("\nTo use a discovered direction for hair editing:")
    print("  import torch")
    print(f"  data = torch.load('{args.output}')")
    print("  hair_dir = data['directions'][INDEX]  # pick the right INDEX")
    print("  torch.save(hair_dir, 'weights/directions/hair.pt')")


if __name__ == "__main__":
    main()
