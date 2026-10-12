"""Train a *tougher* softmax regression: same model, better training data.

The model is unchanged (10 x 65 weights, softmax, gradient descent), so the slides and the API
stay the same.  What changes is what it learns from.  The plain model (train.py) only saw the
clean 8x8 dataset images, but the website feeds it hand-drawn digits that went through
app/preprocess.py.  So here we train on exactly that kind of input:

  1. the original 8x8 images (as before)
  2. each dataset image blown up to a 280x280 "canvas", randomly rotated / sheared / resized /
     moved, with random brush thickness, then pushed through the SAME preprocessing as the site
  3. digits drawn with many system fonts (random size, thickness, tilt), same preprocessing

Evaluation uses data the model never trained on, including fonts that were held out.

Run:  python train_augmented.py     (needs scikit-learn and Windows fonts; ~1-2 minutes)
      -> writes model/weights.npy   (the old weights are kept in model/weights_plain.npy)
"""
from __future__ import annotations

import io
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from app.preprocess import DrawingError, to_vector

ROOT = Path(__file__).parent
WEIGHTS = ROOT / "model" / "weights.npy"
BACKUP = ROOT / "model" / "weights_plain.npy"

LR, STEPS, L2 = 1.0, 3000, 1e-4
AUG_PER_IMAGE = 12
FONT_DIR = Path(r"C:\Windows\Fonts")
TRAIN_FONTS = ["arial", "times", "cour", "tahoma", "segoeui", "segoepr", "segoesc", "Inkfree",
               "georgia", "trebuc", "consola", "BRADHITC", "LHANDW", "ariblk", "framd"]
HELD_OUT_FONTS = ["comic", "calibri", "verdana"]  # never used for training

rng = np.random.default_rng(0)


# ---------- making fake "canvas drawings" ----------------------------------------------------
def affine(img: Image.Image, angle: float, shear: float) -> Image.Image:
    img = img.rotate(angle, resample=Image.BILINEAR, expand=True)
    w, h = img.size
    return img.transform(img.size, Image.AFFINE, (1, shear, -shear * h / 2, 0, 1, 0), Image.BILINEAR)


def paste_random(img: Image.Image) -> np.ndarray:
    canvas = Image.new("L", (280, 280), 0)
    w, h = img.size
    x = int(rng.integers(0, max(1, 280 - w)))
    y = int(rng.integers(0, max(1, 280 - h)))
    canvas.paste(img, (x, y))
    return np.asarray(canvas, dtype=np.float32) / 255.0


def canvas_from_grid(grid: np.ndarray) -> np.ndarray:
    """8x8 dataset image -> a messy 280x280 'drawing' (white ink on black)."""
    im = Image.fromarray((grid / 16 * 255).astype(np.uint8)).resize((200, 200), Image.BICUBIC)
    a = np.asarray(im, dtype=np.float32) / 255
    a = np.clip((a - rng.uniform(0.25, 0.55)) * rng.uniform(4, 10) + 0.5, 0, 1)  # soft threshold
    im = Image.fromarray((a * 255).astype(np.uint8))
    k = int(rng.choice([1, 3, 3, 5, 7, 9]))  # brush thickness
    if k > 1:
        im = im.filter(ImageFilter.MaxFilter(k))
    side = int(200 * rng.uniform(0.45, 1.0))
    im = im.resize((side, side))
    im = affine(im, rng.uniform(-15, 15), rng.uniform(-0.25, 0.25))
    return paste_random(im)


def canvas_from_font(font_path: Path, digit: int) -> np.ndarray:
    size = int(rng.integers(100, 240))
    font = ImageFont.truetype(str(font_path), size)
    img = Image.new("L", (size * 2, size * 2), 0)
    ImageDraw.Draw(img).text((size, size), str(digit), font=font, fill=255, anchor="mm",
                             stroke_width=int(rng.integers(2, 9)), stroke_fill=255)
    img = affine(img, rng.uniform(-12, 12), rng.uniform(-0.2, 0.2))
    bbox = img.getbbox()
    img = img.crop(bbox)
    img.thumbnail((270, 270))
    return paste_random(img)


def to_grid(ink: np.ndarray) -> np.ndarray | None:
    try:
        return to_vector(ink).ravel()
    except DrawingError:
        return None


# ---------- training --------------------------------------------------------------------------
def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def add_bias(X: np.ndarray) -> np.ndarray:
    return np.hstack([np.ones((len(X), 1)), X])


def accuracy(W: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    return float((softmax(add_bias(X / 16.0) @ W.T).argmax(1) == y).mean())


def fonts(names: list[str]) -> list[Path]:
    found = []
    for n in names:
        for ext in (".ttf", ".TTF"):
            if (FONT_DIR / f"{n}{ext}").exists():
                found.append(FONT_DIR / f"{n}{ext}")
                break
    return found


def font_set(paths: list[Path], per_digit: int) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for p in paths:
        for d in range(10):
            for _ in range(per_digit):
                v = to_grid(canvas_from_font(p, d))
                if v is not None:
                    X.append(v)
                    y.append(d)
    return np.array(X), np.array(y)


def main() -> None:
    digits = load_digits()
    X8, y = digits.data, digits.target
    idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=0.2, random_state=42, stratify=y)

    tr_fonts, ho_fonts = fonts(TRAIN_FONTS), fonts(HELD_OUT_FONTS)
    print(f"train fonts: {len(tr_fonts)}   held-out fonts: {[p.stem for p in ho_fonts]}")

    # training data
    Xs, ys = [X8[idx_tr]], [y[idx_tr]]
    aug_X, aug_y = [], []
    for i in idx_tr:
        for _ in range(AUG_PER_IMAGE):
            v = to_grid(canvas_from_grid(digits.images[i]))
            if v is not None:
                aug_X.append(v)
                aug_y.append(y[i])
    Xs.append(np.array(aug_X)); ys.append(np.array(aug_y))
    fx, fy = font_set(tr_fonts, per_digit=30)
    Xs.append(fx); ys.append(fy)
    X_tr, y_tr = np.vstack(Xs), np.concatenate(ys)
    print(f"training samples: {len(y_tr)}  (plain {len(idx_tr)}, augmented {len(aug_y)}, fonts {len(fy)})")

    # evaluation data (never trained on)
    te_clean = X8[idx_te]
    te_canvas = np.array([v for i in idx_te if (v := to_grid(canvas_from_grid(digits.images[i]))) is not None])
    te_canvas_y = np.array([y[i] for i in idx_te][: len(te_canvas)])
    te_fx, te_fy = font_set(ho_fonts, per_digit=40)

    # gradient descent (same update rule as in the slides, plus a tiny L2 pull toward 0)
    A, Y = add_bias(X_tr / 16.0), np.eye(10)[y_tr]
    W = np.zeros((10, 65))
    for step in range(STEPS + 1):
        P = softmax(A @ W.T)
        if step in (0, 100, 1000, STEPS):
            loss = -np.log(P[np.arange(len(y_tr)), y_tr]).mean()
            print(f"step {step:4d}  loss {loss:.4f}")
        W -= LR * ((P - Y).T @ A / len(A) + L2 * W)

    old = np.load(BACKUP if BACKUP.exists() else WEIGHTS)  # always compare with the plain model
    print("\naccuracy            old(plain)   new(augmented)")
    for name, X, yy in [("dataset test 8x8", te_clean, y[idx_te]),
                        ("test as canvas", te_canvas, te_canvas_y),
                        ("held-out fonts", te_fx, te_fy)]:
        print(f"{name:18s}  {accuracy(old, X, yy):9.1%}  {accuracy(W, X, yy):12.1%}")

    if not BACKUP.exists():
        shutil.copy(WEIGHTS, BACKUP)
    np.save(WEIGHTS, W)
    print(f"\nsaved {WEIGHTS}  shape={W.shape}  (old weights: {BACKUP.name})")


if __name__ == "__main__":
    main()
