"""
Image I/O, preprocessing, and face alignment utilities.

The e4e encoder expects:
  - 256×256 RGB tensor normalised to [-1, 1]
  - FFHQ-style aligned faces (eyes centred, mouth at fixed position)

Alignment priority:
  1. ``face-alignment`` package  →  full 68-landmark FFHQ alignment (best)
  2. OpenCV Haar cascade         →  detect + FFHQ-style crop (good fallback)
  3. Simple centre crop           →  last resort (poor for non-cropped photos)
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

ENCODER_INPUT_SIZE = 256
FFHQ_TRANSFORM_SIZE = 1024

TRANSFORM = T.Compose([
    T.Resize((ENCODER_INPUT_SIZE, ENCODER_INPUT_SIZE)),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


def load_image(path: str) -> Image.Image:
    """Load an image from disk as RGB PIL Image."""
    return Image.open(path).convert("RGB")



def align_face(image: Image.Image) -> Optional[Image.Image]:
    """
    Face alignment with cascading fallbacks:
      1. face_alignment (68-landmark FFHQ alignment)
      2. OpenCV Haar cascade (face-detect + FFHQ-style expand & crop)
      3. Simple centre crop
    """
    # --- Try face_alignment (best quality) ---
    try:
        import face_alignment
        fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D, flip_input=False, device="cpu",
        )
        img_np = np.array(image)
        landmarks = fa.get_landmarks(img_np)
        if landmarks is not None and len(landmarks) > 0:
            return _align_from_landmarks(image, landmarks[0])
    except ImportError:
        pass

    # --- Fallback: OpenCV face detection + FFHQ-style crop ---
    detected = _opencv_face_crop(image)
    if detected is not None:
        return detected

    # --- Last resort: centre crop ---
    print("WARNING: No face detected. Using centre crop — results may be poor.")
    return _centre_crop(image)


def _opencv_face_crop(image: Image.Image, expand: float = 2.0) -> Optional[Image.Image]:
    """
    Detect the largest face with OpenCV Haar cascade, expand the bounding box
    to approximate FFHQ framing, and return a square crop.

    The expand factor of ~2.0 adds forehead, chin, and margin so the face
    fills roughly 50-70 % of the frame — matching FFHQ training distribution.
    """
    img_np = np.array(image)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
    )

    if len(faces) == 0:
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20),
        )

    if len(faces) == 0:
        return None

    # Pick the largest detected face
    areas = [w * h for (_, _, w, h) in faces]
    x, y, w, h = faces[np.argmax(areas)]

    # Expand the box to approximate FFHQ framing
    cx = x + w / 2.0
    cy = y + h / 2.0
    side = max(w, h) * expand

    # Shift the centre upward slightly (FFHQ has eyes at ~1/3 from top)
    cy -= h * 0.1

    x1 = int(cx - side / 2)
    y1 = int(cy - side / 2)
    x2 = int(cx + side / 2)
    y2 = int(cy + side / 2)

    # Pad if the box extends outside the image
    img_h, img_w = img_np.shape[:2]
    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - img_w)
    pad_bottom = max(0, y2 - img_h)

    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        img_np = np.pad(
            img_np,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="reflect",
        )
        x1 += pad_left
        y1 += pad_top
        x2 += pad_left
        y2 += pad_top

    cropped = img_np[y1:y2, x1:x2]
    return Image.fromarray(cropped)


def _align_from_landmarks(
    image: Image.Image, lm: np.ndarray, output_size: int = FFHQ_TRANSFORM_SIZE
) -> Image.Image:
    """
    Compute a similarity transform from 68-landmark positions,
    matching the FFHQ dataset alignment procedure.
    """
    left_eye = lm[36:42].mean(axis=0)
    right_eye = lm[42:48].mean(axis=0)
    mouth_left = lm[48]
    mouth_right = lm[54]
    mouth_center = (mouth_left + mouth_right) / 2.0

    eye_center = (left_eye + right_eye) / 2.0
    eye_to_eye = right_eye - left_eye
    eye_to_mouth = mouth_center - eye_center

    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    center = eye_center + eye_to_mouth * 0.1

    quad = np.stack([center - x - y, center - x + y,
                     center + x + y, center + x - y])
    qsize = np.hypot(*x) * 2

    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (
            int(np.rint(float(image.size[0]) / shrink)),
            int(np.rint(float(image.size[1]) / shrink)),
        )
        image = image.resize(rsize, Image.LANCZOS)
        quad /= shrink
        qsize /= shrink

    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (
        int(np.floor(min(quad[:, 0]))),
        int(np.floor(min(quad[:, 1]))),
        int(np.ceil(max(quad[:, 0]))),
        int(np.ceil(max(quad[:, 1]))),
    )
    crop = (
        max(crop[0] - border, 0),
        max(crop[1] - border, 0),
        min(crop[2] + border, image.size[0]),
        min(crop[3] + border, image.size[1]),
    )
    if crop[2] - crop[0] < image.size[0] or crop[3] - crop[1] < image.size[1]:
        image = image.crop(crop)
        quad -= [crop[0], crop[1]]

    pad = (
        int(np.floor(min(quad[:, 0]))),
        int(np.floor(min(quad[:, 1]))),
        int(np.ceil(max(quad[:, 0]))),
        int(np.ceil(max(quad[:, 1]))),
    )
    pad = (
        max(-pad[0] + border, 0),
        max(-pad[1] + border, 0),
        max(pad[2] - image.size[0] + border, 0),
        max(pad[3] - image.size[1] + border, 0),
    )
    if max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img_np = np.array(image)
        img_np = np.pad(
            img_np,
            ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)),
            mode="reflect",
        )
        image = Image.fromarray(img_np)
        quad += [pad[0], pad[1]]

    image = image.transform(
        (output_size, output_size), Image.QUAD,
        (quad + 0.5).flatten(), Image.BILINEAR,
    )

    return image


def _centre_crop(image: Image.Image) -> Image.Image:
    """Simple centre square crop (fallback when alignment is unavailable)."""
    w, h = image.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return image.crop((left, top, left + side, top + side))


# ── Tensor ↔ image conversion ──────────────────────────────────────

def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    Convert a single-image tensor in [-1, 1] to a PIL Image.
    Accepts shapes (3, H, W) or (1, 3, H, W).
    """
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)
    img = (tensor.clamp(-1, 1) + 1) / 2 * 255
    img = img.permute(1, 2, 0).cpu().byte().numpy()
    return Image.fromarray(img)


def save_image(tensor: torch.Tensor, path: str):
    """Save a tensor image to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tensor_to_pil(tensor).save(path)
