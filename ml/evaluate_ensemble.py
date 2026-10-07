"""
SignSpeak - try cheap accuracy boosts WITHOUT retraining.

Compares, on the SAME validation and test splits:
  1. each saved model on its own
  2. mirror test-time augmentation (TTA): average the prediction for the
     sequence and its left/right mirror (the models were trained with
     mirrored copies, so they understand both)
  3. an ensemble: average the predictions of several saved models
  4. ensemble + mirror TTA

Decide on VALIDATION accuracy, report TEST accuracy - the test split is
never used to choose.

Run from the project root (after preprocessing with ADD_WRIST = True):
    python ml/evaluate_ensemble.py
"""

import json
import os

import numpy as np
import tensorflow as tf

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "dataset", "preprocessed_data_words.npz")

# Saved 129-feature models to combine (add more folders here if you train more)
MODEL_DIRS = [
    os.path.join(PROJECT_ROOT, "ml", "artifacts", "words_run6_wrist"),
    os.path.join(PROJECT_ROOT, "ml", "artifacts", "words_run7_reg"),
]
MODEL_FILE = "sign_speak_words_lstm_s2.h5"
OUT_PATH = os.path.join(PROJECT_ROOT, "ml", "artifacts", "ensemble_results.json")

HAND_SLOTS = (slice(1, 64), slice(64, 127))


def mirror(X):
    """Left/right mirror of normalised sequences (same rule as training).

    Hands: x -> -x (wrist is at 0 after normalisation).
    Wrist position (features 127, 128): x -> 1 - x.
    Missing values (-1.0) are left unchanged.
    """
    M = X.copy()
    n, frames, feats = M.shape
    for sl in HAND_SLOTS:
        hand = M[:, :, sl].reshape(n, frames, 21, 3)
        present = ~np.isclose(hand[:, :, 0, 0], -1.0)
        hand[..., 0] = np.where(present[..., None], -hand[..., 0], hand[..., 0])
        M[:, :, sl] = hand.reshape(n, frames, 63)
    if feats > 127:
        wx = M[:, :, 127]
        seen = ~np.isclose(wx, -1.0)
        M[:, :, 127] = np.where(seen, 1.0 - wx, wx)
    return M


def accuracy(probs, y):
    return float((np.argmax(probs, axis=1) == y).mean())


def main():
    data = np.load(DATA_PATH)
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]
    print(f"Validation {X_val.shape}, Test {X_test.shape}")

    models = []
    for d in MODEL_DIRS:
        path = os.path.join(d, MODEL_FILE)
        if not os.path.exists(path):
            print(f"  (skipping, not found: {path})")
            continue
        m = tf.keras.models.load_model(path)
        if m.input_shape[-1] != X_test.shape[-1]:
            print(f"  (skipping {os.path.basename(d)}: expects {m.input_shape[-1]} features)")
            continue
        models.append((os.path.basename(d), m))

    if not models:
        raise SystemExit("No compatible models found.")

    results = {}
    val_plain, val_tta, test_plain, test_tta = [], [], [], []

    for name, m in models:
        pv = m.predict(X_val, verbose=0)
        pt = m.predict(X_test, verbose=0)
        pv_m = m.predict(mirror(X_val), verbose=0)
        pt_m = m.predict(mirror(X_test), verbose=0)
        val_plain.append(pv); test_plain.append(pt)
        val_tta.append((pv + pv_m) / 2); test_tta.append((pt + pt_m) / 2)
        results[name] = {
            "val": accuracy(pv, y_val), "test": accuracy(pt, y_test),
            "val_mirror_tta": accuracy(val_tta[-1], y_val),
            "test_mirror_tta": accuracy(test_tta[-1], y_test),
        }

    if len(models) > 1:
        results["ENSEMBLE"] = {
            "val": accuracy(np.mean(val_plain, 0), y_val),
            "test": accuracy(np.mean(test_plain, 0), y_test),
            "val_mirror_tta": accuracy(np.mean(val_tta, 0), y_val),
            "test_mirror_tta": accuracy(np.mean(test_tta, 0), y_test),
        }

    print("\n" + "=" * 72)
    print(f"{'Model':22s}{'Val':>9s}{'Test':>9s}{'Val+TTA':>11s}{'Test+TTA':>11s}")
    print("-" * 72)
    for name, r in results.items():
        print(f"{name:22s}{r['val']*100:8.2f}%{r['test']*100:8.2f}%"
              f"{r['val_mirror_tta']*100:10.2f}%{r['test_mirror_tta']*100:10.2f}%")
    print("=" * 72)

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()