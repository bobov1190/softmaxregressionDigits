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
