"""One free Render service: draw-a-digit website + softmax regression API.

  GET  /          the drawing page (static/index.html)
  GET  /health    liveness check
  POST /predict   {"image": "<data URL or base64 PNG>"} ->
                  {"digit", "probs"[10], "grid"[8][8]}   (grid = what the model sees, 0..16)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .preprocess import DrawingError, image_to_grid

ROOT = Path(__file__).resolve().parent.parent
W = np.load(ROOT / "model" / "weights.npy")  # shape (10, 65): column 0 is beta_0
assert W.shape == (10, 65), W.shape

app = FastAPI(title="Softmax digits")


class Drawing(BaseModel):
    image: str


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def predict_grid(grid: np.ndarray) -> np.ndarray:
    """8x8 grid (0..16) -> 10 probabilities. Same recipe as in training."""
    x = np.concatenate([[1.0], grid.ravel() / 16.0])
    return softmax(W @ x)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
def predict(drawing: Drawing) -> dict:
    try:
        grid = image_to_grid(drawing.image)
    except DrawingError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    probs = predict_grid(grid)
    return {
        "digit": int(probs.argmax()),
        "probs": [round(float(p), 4) for p in probs],
        "grid": np.round(grid, 1).tolist(),
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(ROOT / "static" / "index.html")


app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
