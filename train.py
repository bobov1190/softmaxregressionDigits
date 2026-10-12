"""Train softmax regression on sklearn digits with plain numpy gradient descent.

Same recipe as in the lesson slides:
  - 1797 images of 8x8 digits, pixels divided by 16
  - 10 x 65 weights (column 0 is the bias beta_0)
  - start from all zeros, full-batch gradient descent, step 1.0, 1000 steps
  - 80% of images for training, 20% held out

Run:  python train.py     ->  writes model/weights.npy
"""
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

LR = 1.0
STEPS = 1000
OUT = Path(__file__).parent / "model" / "weights.npy"


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)  # numerical stability
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def add_bias(X: np.ndarray) -> np.ndarray:
    return np.hstack([np.ones((len(X), 1)), X])


def loss_fn(P: np.ndarray, y: np.ndarray) -> float:
    return float(-np.log(P[np.arange(len(y)), y]).mean())


def main() -> None:
    digits = load_digits()
    X = digits.data / 16.0
    y = digits.target
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    A_tr, A_te = add_bias(X_tr), add_bias(X_te)

    W = np.zeros((10, 65))
    Y = np.eye(10)[y_tr]

    for step in range(STEPS + 1):
        P = softmax(A_tr @ W.T)
        if step in (0, 10, 100, 500, STEPS):
            print(f"step {step:4d}  loss {loss_fn(P, y_tr):.4f}")
        grad = (P - Y).T @ A_tr / len(A_tr)
        W -= LR * grad

    acc_tr = (softmax(A_tr @ W.T).argmax(1) == y_tr).mean()
    acc_te = (softmax(A_te @ W.T).argmax(1) == y_te).mean()
    print(f"train accuracy {acc_tr:.3%}   test accuracy {acc_te:.3%}")

    OUT.parent.mkdir(exist_ok=True)
    np.save(OUT, W)
    print(f"saved {OUT}  shape={W.shape}")


if __name__ == "__main__":
    main()
