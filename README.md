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

**v2: augmented (`train_augmented.py`, kept as `model/weights_v2_fake.npy`; it was the current model until v3).** Same model, new training data,
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


## v3: trained with my own drawings (current `model/weights.npy`)

v1 and v2 were trained AND tested only on data that was made by a computer (dataset images, fake canvases,
fonts). When we drew real digits with a mouse on the site, the models did worse than those tests promised.
A test is only honest if the model never saw it and if it looks like what the model meets in real life.
So v3 is trained and tested on real drawings.

The model is the same softmax regression (10 x 65 weights, column 0 is beta_0), the site code did not change.
Only the weights and the training data changed.

**How v3 was trained.** The training code lives in the neural net repo
(https://github.com/bobov1190/neuralnetDigits, `train_mine.py`), because there the softmax and the small
neural net are trained on exactly the same data and compared:

1. I drew 150 real digits with the mouse on the site in collect mode (15 per digit); the raw canvas pictures are
   in that repo, `data/mine/`.
2. About 30% (43 drawings) were locked away as a test set. The model never trains on them.
3. The other 107 drawings were added to the v2 training data (the 23k fake samples), each one also repeated
   40 times with a small random tilt, shear and brush thickness.
4. Training: Adam, mini-batches of 256, 80 epochs, learning rate 1e-2 with cosine decay, tiny L2 penalty (1e-4),
   start from zeros. (v1 and v2 used plain full-batch gradient descent, so the recipe changed a little too.)

| test set                          | v2 (fake data only) | v3 (+ my drawings) |
|-----------------------------------|--------------------:|-------------------:|
| my 43 locked-away drawings        |               69.8% |          **86.0%** |
| dataset test images (8x8)         |               93.6% |              92.2% |
| test images as fake canvas        |               90.3% |              90.0% |
| 3 fonts never seen in training    |               97.6% |              97.2% |

(The v2 numbers here are a bit different from the v2 table above, because all tests were re-made with new random
distortions, so the old and new columns are measured on the same test pictures.)

Checked on the site code of this repo: 37 of 43 test drawings are right (86.0%). Mistakes: 0 read as 5, 1 as 6,
1 as 8, 4 as 9, 5 as 9, 9 as 0.

What to remember:
- The big gain is on real drawings (+16 points). The tiny drops on fake tests (-1.4 on clean 8x8) are the price.
- Only 43 test drawings, all drawn by one person: one drawing is about 2.3 points. Another hand can give other numbers.
- Softmax is a linear model and reaches only about 90% even on its own training data, so some digits stay hard for it.
  The small neural net trained on the same data gets 95.3% on the same 43 drawings.
- `train.py` and `train_augmented.py` still write `model/weights.npy`: running them would replace v3.
  The previous versions are kept as `model/weights_plain.npy` (v1) and `model/weights_v2_fake.npy` (v2).
