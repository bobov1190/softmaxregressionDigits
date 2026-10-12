# Softmax Digits

Draw a digit in the browser → the server turns it into the 8×8 grid (64 numbers, 0..16) that
the `sklearn` digits dataset uses → a softmax regression (10 × 65 weights, plain numpy) predicts it.

One FastAPI service serves both the page and the API.

## Run locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
# open http://127.0.0.1:8000
```

## Retrain (optional, needs scikit-learn)

```bash
pip install scikit-learn
python train.py        # writes model/weights.npy
```

Recipe: pixels / 16, weights start at 0, full-batch gradient descent, step 1.0, 1000 steps.

## Deploy on Render (free)

- Build Command: `pip install -r requirements.txt`
- Start Command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- Health Check Path: `/health`

(`render.yaml` already contains all of this: use **New → Blueprint**.)

## How a drawing becomes 64 numbers (`app/preprocess.py`)

The dataset digits are stretched to the full frame height, thin-stroked and centred, so a raw
canvas drawing is normalised the same way: crop to the ink → scale → fix the stroke thickness →
centre by centre of mass → average 8×8 blocks → scale to 0..16.


## How the model was trained (and made smarter)

The model itself never changed: softmax regression, 10 × 65 weights, plain gradient descent.
What changed is **what it learns from**.

**v1: plain (`train.py`, kept as `model/weights_plain.npy`).** Trained only on the clean 8×8 images
from sklearn digits (80% train / 20% test, pixels / 16, step 1.0, 1000 steps). It gets 96.7% on the
dataset's test images, but only ~78% on hand-drawn digits that went through `app/preprocess.py`
and ~77% on digits typed in fonts it never saw. The site feeds it drawings, not dataset images,
so it was trained on the wrong kind of data.

**v2: augmented (`train_augmented.py`, current `model/weights.npy`).** Same model, new training data,
all of it passed through the *same* `preprocess.py` as the website:

1. the original 8×8 training images (1437);
2. each training image blown up to a 280×280 fake canvas and randomly rotated, sheared,
   resized, moved, with random brush thickness, 12 variants per image (17 244);
3. digits rendered in 15 system fonts with random size, thickness and tilt (4 500).

Total: 23 181 samples. Training: step 1.0, 3000 steps, tiny L2 penalty (1e-4), start from zeros.
Final loss 0.305.

Evaluation uses only data the model never trained on, including 3 fonts held out entirely
(Comic Sans, Calibri, Verdana):

| test set                    | v1 plain | v2 augmented |
|-----------------------------|---------:|-------------:|
| dataset test images (8×8)   |    96.7% |        93.6% |
| test images as fake canvas  |    78.3% |        89.2% |
| held-out fonts              |    77.0% |        97.3% |

The price: 3 points on clean 8×8 images. The gain: much better on what the site actually receives.
It is still a linear model, so thin 1s and 7s can get mixed up, and softmax is always "confident":
it has no "I don't know" answer, because the 10 probabilities must sum to 1.

Retrain v2: `pip install scikit-learn pillow`, then `python train_augmented.py` (needs Windows fonts).
