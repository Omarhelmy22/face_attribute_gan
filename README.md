# Face Attribute Editing via GAN Latent Space Manipulation

Modify facial attributes (age, smile, pose, hair) in real photographs while preserving identity and realism.

**Approach:**

- **Generator:** StyleGAN2 pretrained on FFHQ (1024x1024)
- **Inversion:** e4e (encoder4editing) — encoder-based GAN inversion into W+ space
- **Editing:** InterfaceGAN direction vectors (age, smile, pose) + SeFa closed-form factorization (hair)

---

## Repository Structure

```
face_attribute_gan/
├── configs/config.yaml              # All hyperparameters and paths
├── data/input_images/               # input face photos here
├── models/
│   ├── stylegan2/                   # StyleGAN2 generator
│   │   ├── op.py                    # Pure-Python custom ops
│   │   └── model.py                 # Generator architecture
│   ├── encoders/                    # e4e encoder components
│   │   ├── helpers.py               # IR-SE bottleneck + SE block
│   │   ├── model_irse.py            # Block configuration for IR-SE50
│   │   └── psp_encoders.py          # Encoder4Editing architecture
│   └── e4e_model.py                 # Top-level model (encoder + decoder)
├── editing/latent_editor.py         # Direction-based W+ latent editing
├── pipelines/editing_pipeline.py    # End-to-end pipeline
├── inference/edit_face.py           # CLI inference script
├── scripts/
│   ├── generate_experiments.py      # Batch multi-strength experiments
│   └── extract_sefa_directions.py   # Discover hair direction via SeFa
├── utils/
│   ├── image_utils.py               # Preprocessing, alignment, I/O
│   └── visualization.py             # Comparison grids and strips
├── weights/                         # ** Downloaded model weights go here **
│   ├── e4e_ffhq_encode.pt
│   └── directions/
│       ├── age.pt
│       ├── smile.pt
│       ├── pose.pt
│       └── hair.pt
├── results/                         # Auto-generated output images
├── requirements.txt
└── README.md
```

---

## Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd face_attribute_gan

# 2. Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Install face-alignment for FFHQ-style alignment
#    Without this, a simple centre-crop fallback is used.
pip install face-alignment
```

---

## Model Weights Download

The pretrained weights are **not** included in this repository.
You must download them manually and place them in the `weights/` folder.

### Step 1 — e4e Encoder + StyleGAN2 Generator (single file)


| File                 | Size    | Description                            |
| -------------------- | ------- | -------------------------------------- |
| `e4e_ffhq_encode.pt` | ~1.2 GB | e4e encoder + StyleGAN2 FFHQ generator |


**Download from HuggingFace (recommended):**

```
https://huggingface.co/zhengchong/e4e/resolve/main/e4e_ffhq_encode.pt
```

**Place the file at:**

```
weights/e4e_ffhq_encode.pt
```

### Step 2 — InterfaceGAN Editing Directions (age, smile, pose)

These small `.pt` files come from the official [encoder4editing](https://github.com/omertov/encoder4editing) repository, inside the `editings/` folder.


| File       | Source                                                      |
| ---------- | ----------------------------------------------------------- |
| `age.pt`   | `editings/interfacegan_directions/age.pt` in the e4e repo   |
| `smile.pt` | `editings/interfacegan_directions/smile.pt` in the e4e repo |
| `pose.pt`  | `editings/interfacegan_directions/pose.pt` in the e4e repo  |


**Place the files at:**

```
weights/directions/age.pt
weights/directions/smile.pt
weights/directions/pose.pt
```

### Step 3 — Hair Direction (extracted via SeFa)

InterfaceGAN does **not** provide a hair direction. Instead, we use SeFa
(Closed-Form Factorization) to discover one from the generator weights.

**After completing Steps 1–2**, run:

```bash
python scripts/extract_sefa_directions.py --visualise --n-samples 5
```

This will:

1. Extract the top 15 SeFa directions from the StyleGAN2 generator
2. Save visualisation grids to `results/sefa/`
3. Save all directions to `weights/directions/sefa_directions.pt`

**Then inspect the grids** in `results/sefa/` to identify which component
corresponds to hair changes. Once you find the right index (e.g., component 3):

```python
import torch
data = torch.load('weights/directions/sefa_directions.pt')
torch.save(data['directions'][3], 'weights/directions/hair.pt')
```

---

## Quick Start

### Single edit via CLI

```bash
python inference/edit_face.py \
    --image data/input_images/person.jpg \
    --attribute age \
    --strength 3
```

### Available CLI options


| Argument      | Description                                      |
| ------------- | ------------------------------------------------ |
| `--image`     | Path to input face image                         |
| `--attribute` | One of: `age`, `smile`, `pose`, `hair`           |
| `--strength`  | Editing intensity α (positive or negative)       |
| `--no-align`  | Skip face alignment (use for pre-aligned images) |
| `--device`    | `cuda` or `cpu`                                  |


### Batch experiments

Generate multi-strength edits for all images and all attributes:

```bash
python scripts/generate_experiments.py
```

Or for a specific image and attribute:

```bash
python scripts/generate_experiments.py \
    --image data/input_images/person.jpg \
    --attributes age smile
```

---

## How It Works

### Pipeline

```
Input Image
    │
    ▼
Face Alignment (FFHQ-style, 68 landmarks)
    │
    ▼
Resize to 256×256, normalise to [-1, 1]
    │
    ▼
e4e Encoder  →  W+ latent code (1, 18, 512)
    │
    ▼
Latent Editing:  w_new = w + α · direction
    │
    ▼
StyleGAN2 Decoder  →  Edited image (1024×1024)
    │
    ▼
Save comparison grid (Original | Reconstruction | Edited)
```

### Editing Directions


| Attribute | Method       | Source         | Layer Range  |
| --------- | ------------ | -------------- | ------------ |
| Age       | InterfaceGAN | e4e repo       | All (0–17)   |
| Smile     | InterfaceGAN | e4e repo       | All (0–17)   |
| Pose      | InterfaceGAN | e4e repo       | All (0–17)   |
| Hair      | SeFa         | Self-extracted | Coarse (0–6) |


**Layer-selective editing:** W+ has 18 style vectors controlling different
levels of detail. Coarse layers (0–6) control structure (shape, pose, hair
geometry). Fine layers (7–17) control texture and colour. For hair, restricting
edits to coarse layers preserves facial identity better.

### Architecture Dependency

The model architecture code in `models/` is ported from:

- [rosinality/stylegan2-pytorch](https://github.com/rosinality/stylegan2-pytorch) — StyleGAN2 generator
- [omertov/encoder4editing](https://github.com/omertov/encoder4editing) — e4e encoder (IR-SE50 backbone)

These are vendored into the project (not pip-installable) to ensure exact
weight compatibility. No modifications to the architecture code are needed.

---

## Output Format

All results are saved to `results/<attribute>/`:

- `<name>_reconstruction.png` — GAN inversion output
- `<name>_<attribute>_<strength>.png` — Edited image
- `<name>_<attribute>_<strength>_grid.png` — Side-by-side comparison

---

