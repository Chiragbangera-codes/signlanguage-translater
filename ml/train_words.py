"""
SignSpeak - Words LSTM training (replacement for ml/train_words.py)

Fixes in this version
---------------------
* Loads the WORDS data (dataset/preprocessed_data_words.npz). The old script
  loaded dataset/preprocessed_data.npz, which is the digits data.
* Builds the same architecture as the model deployed in
  frontend/public/model/words/model.json (2-layer stacked LSTM).
* Saves to ml/model/sign_speak_words_lstm_s2.h5, the path the backend loads.
* Optional data augmentation of the training split (rotation, mirror,
  speed change, noise) -- set HPARAMS["augment_copies"] = 0 to turn it off.
* Writes every number the panel / IEEE paper needs to
  ml/artifacts/words/training_report_words.json:
  split sizes and ratios, hyperparameters, loss function, train/val/test
  accuracy and loss, macro precision/recall/F1, top-3 accuracy,
  per-class accuracy, confusion matrix, and model inference latency.

Run (from the project root):
    python ml/preprocess_words.py
    python ml/train_words.py
"""

import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "dataset", "preprocessed_data_words.npz")
LABEL_ENCODER_PATH = os.path.join(PROJECT_ROOT, "ml", "model", "label_encoder_words.pkl")
MODEL_DIR = os.path.join(PROJECT_ROOT, "ml", "model")
MODEL_PATH = os.path.join(MODEL_DIR, "sign_speak_words_lstm_s2.h5")          # loaded by the backend
EXPORT_MODEL_PATH = os.path.join(MODEL_DIR, "sign_speak_words_lstm.h5")      # read by the TF.js export scripts
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "ml", "artifacts", "words")

# ============================================================
# HYPERPARAMETERS  (report these in the paper)
# ============================================================

SEED = 42
HPARAMS = {
    "lstm_1_units": 160,
    "lstm_2_units": 96,
    "dense_units": 128,
    "dropout": 0.4,
    "optimizer": "Adam",
    "learning_rate": 5e-4,
    "batch_size": 64,
    "max_epochs": 100,
    "monitor": "val_accuracy",     # epoch selection, early stopping and LR schedule
    "early_stopping_patience": 15,
    "reduce_lr_factor": 0.5,
    "reduce_lr_patience": 5,
    "min_learning_rate": 1e-6,
    "loss": "sparse_categorical_crossentropy",
    # Data augmentation (training split only; validation and test stay untouched)
    "augment_copies": 3,          # extra augmented copies of every training sequence (0 = off)
    "augment_rotation_deg": 10,   # random in-plane hand rotation, +/- degrees
    "augment_noise_std": 0.01,    # Gaussian noise on landmark coordinates
    "augment_time_scale": 0.15,   # random speed change, +/- 15%
    "augment_mirror_prob": 0.5,   # chance of a left/right mirror (left-handed signers)
}


def set_seeds(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ============================================================
# DATA AUGMENTATION
# ============================================================

HAND_SLOTS = (slice(1, 64), slice(64, 127))   # 21 landmarks x (x, y, z) per hand


def augment_sequence(seq, rng, hp=HPARAMS):
    """Return one randomly augmented copy of a normalised (frames, 127) sequence.

    Missing hands (all -1.0) are left exactly as they are.
    """
    out = seq.copy()
    frames = out.shape[0]

    # 1. Speed change: resample the 30 frames a little faster or slower.
    scale = 1.0 + rng.uniform(-hp["augment_time_scale"], hp["augment_time_scale"])
    span = (frames - 1) * scale
    start = rng.uniform(0, max(0.0, (frames - 1) - span)) if scale < 1 else 0.0
    idx = np.clip(np.rint(start + np.linspace(0, span, frames)), 0, frames - 1).astype(int)
    out = out[idx]

    theta = np.deg2rad(rng.uniform(-hp["augment_rotation_deg"], hp["augment_rotation_deg"]))
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    mirror = rng.random() < hp["augment_mirror_prob"]

    for sl in HAND_SLOTS:
        hand = out[:, sl].reshape(frames, 21, 3)
        present = ~np.isclose(hand[:, 0, 0], -1.0)          # wrist is 0 after normalisation
        if not present.any():
            continue
        h = hand[present]
        x, y = h[..., 0].copy(), h[..., 1].copy()
        # 2. Rotation in the image plane (keeps hand size, so normalisation still holds).
        h[..., 0] = x * cos_t - y * sin_t
        h[..., 1] = x * sin_t + y * cos_t
        # 3. Left/right mirror.
        if mirror:
            h[..., 0] = -h[..., 0]
        # 4. Small landmark noise.
        h += rng.normal(0.0, hp["augment_noise_std"], h.shape).astype(np.float32)
        hand[present] = h
        out[:, sl] = hand.reshape(frames, 63)

    # Optional wrist-position features (127 -> 129, see ADD_WRIST in
    # preprocess_words.py): image coordinates 0-1, -1.0 when no hand.
    if out.shape[1] > 127:
        wx, wy = out[:, 127], out[:, 128]
        seen = ~np.isclose(wx, -1.0)
        if mirror:
            wx[seen] = 1.0 - wx[seen]
        wx[seen] += rng.normal(0.0, hp["augment_noise_std"] / 2, seen.sum())
        wy[seen] += rng.normal(0.0, hp["augment_noise_std"] / 2, seen.sum())

    return out


def augment_training_set(X, y, hp=HPARAMS, seed=SEED):
    copies = hp["augment_copies"]
    if copies <= 0:
        return X, y
    rng = np.random.default_rng(seed)
    extra = [np.stack([augment_sequence(s, rng, hp) for s in X]) for _ in range(copies)]
    X_aug = np.concatenate([X] + extra).astype(np.float32)
    y_aug = np.concatenate([y] * (copies + 1))
    order = rng.permutation(len(X_aug))
    return X_aug[order], y_aug[order]


def build_model(input_shape, num_classes, hp=HPARAMS):
    model = Sequential(
        [
            Input(shape=input_shape),
            LSTM(hp["lstm_1_units"], return_sequences=True),
            Dropout(hp["dropout"]),
            LSTM(hp["lstm_2_units"], return_sequences=False),
            Dropout(hp["dropout"]),
            Dense(hp["dense_units"], activation="relu"),
            Dropout(hp["dropout"]),
            Dense(num_classes, activation="softmax"),
        ],
        name="SignSpeak_Words_LSTM",
    )
    model.compile(
        optimizer=Adam(learning_rate=hp["learning_rate"]),
        loss=hp["loss"],
        metrics=[
            "accuracy",
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3_accuracy"),
        ],
    )
    return model


def load_class_names(num_classes):
    try:
        import pickle

        with open(LABEL_ENCODER_PATH, "rb") as f:
            return [str(c) for c in pickle.load(f).classes_]
    except Exception:
        return [str(i) for i in range(num_classes)]


def measure_latency(model, input_shape, runs=200):
    """Median single-sequence inference time in ms (batch size 1)."""
    infer = tf.function(lambda x: model(x, training=False))
    x = tf.zeros((1, *input_shape), dtype=tf.float32)
    for _ in range(20):
        infer(x)
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        infer(x).numpy()
        times.append((time.perf_counter() - t) * 1000)
    return float(np.median(times)), float(np.percentile(times, 95))


def save_curves(history):
    for metric, title in (("accuracy", "Accuracy"), ("loss", "Loss")):
        plt.figure(figsize=(9, 5))
        plt.plot(history[metric], label=f"Training {title}", linewidth=2)
        plt.plot(history[f"val_{metric}"], label=f"Validation {title}", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel(title)
        plt.title(f"SignSpeak Words LSTM - {title}")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(ARTIFACTS_DIR, f"{metric}_curve_words.png"), dpi=150)
        plt.close()


def save_confusion(cm, class_names):
    fig, ax = plt.subplots(figsize=(14, 14))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
        ax=ax, values_format="d", xticks_rotation=90, colorbar=False
    )
    ax.set_title("SignSpeak Words LSTM - Test Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACTS_DIR, "confusion_matrix_words.png"), dpi=150)
    plt.close()


def main():
    set_seeds()
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} not found. Run ml/preprocess_words.py first.")

    data = np.load(DATA_PATH)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]

    num_classes = int(max(y_train.max(), y_val.max(), y_test.max()) + 1)
    input_shape = X_train.shape[1:]
    class_names = load_class_names(num_classes)
    total = len(X_train) + len(X_val) + len(X_test)

    print(f"Input shape {input_shape}, classes {num_classes}, sequences {total}")

    n_original_train = len(X_train)
    X_train, y_train = augment_training_set(X_train, y_train)
    if len(X_train) > n_original_train:
        print(f"Augmentation: {n_original_train} -> {len(X_train)} training sequences")
    print(f"Train {len(X_train)} | Val {len(X_val)} | Test {len(X_test)}")

    model = build_model(input_shape, num_classes)
    model.summary()

    # Select the epoch with the best VALIDATION ACCURACY (not lowest val loss):
    # val loss can rise while val accuracy is still improving, and accuracy is
    # what we report. The test split is never used for this choice.
    monitor = HPARAMS["monitor"]
    callbacks = [
        EarlyStopping(monitor=monitor, mode="max",
                      patience=HPARAMS["early_stopping_patience"],
                      restore_best_weights=True),
        ModelCheckpoint(MODEL_PATH, monitor=monitor, mode="max", save_best_only=True),
        ReduceLROnPlateau(monitor=monitor, mode="max", factor=HPARAMS["reduce_lr_factor"],
                          patience=HPARAMS["reduce_lr_patience"],
                          min_lr=HPARAMS["min_learning_rate"]),
    ]

    start = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=HPARAMS["max_epochs"],
        batch_size=HPARAMS["batch_size"],
        callbacks=callbacks,
        shuffle=True,
        verbose=1,
    )
    train_minutes = (time.time() - start) / 60
    hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    best_epoch = int(np.argmax(hist["val_accuracy"]))

    # ---- Evaluation on the untouched test split ----
    # Train accuracy is measured on the original (un-augmented) training split.
    X_tr0, y_tr0 = data["X_train"], data["y_train"]
    train_loss, train_acc, _ = model.evaluate(X_tr0, y_tr0, verbose=0)
    val_loss, val_acc, _ = model.evaluate(X_val, y_val, verbose=0)
    test_loss, test_acc, test_top3 = model.evaluate(X_test, y_test, verbose=0)

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred, labels=np.arange(num_classes))
    per_class = {
        class_names[i]: float(cm[i, i] / cm[i].sum()) if cm[i].sum() else 0.0
        for i in range(num_classes)
    }
    latency_median, latency_p95 = measure_latency(model, input_shape)

    report = {
        "dataset": {
            "total_sequences": int(total),
            "classes": num_classes,
            "class_names": class_names,
            "sequence_shape": list(input_shape),
            "train": int(n_original_train),
            "train_after_augmentation": int(len(X_train)),
            "validation": int(len(X_val)),
            "test": int(len(X_test)),
            "split_ratio": "80:10:10 (stratified, random_state=42)",
        },
        "hyperparameters": HPARAMS,
        "total_parameters": int(model.count_params()),
        "training": {
            "epochs_run": len(hist["loss"]),
            "best_epoch": best_epoch + 1,
            "training_time_minutes": round(train_minutes, 2),
        },
        "results": {
            "train_accuracy": float(train_acc),
            "train_loss": float(train_loss),
            "val_accuracy": float(val_acc),
            "val_loss": float(val_loss),
            "test_accuracy": float(test_acc),
            "test_top3_accuracy": float(test_top3),
            "test_loss": float(test_loss),
            "test_macro_precision": float(precision),
            "test_macro_recall": float(recall),
            "test_macro_f1": float(f1),
        },
        "inference_latency_ms": {"median": latency_median, "p95": latency_p95},
        "per_class_test_accuracy": per_class,
        "confusion_matrix": cm.tolist(),
        "history": hist,
    }

    with open(os.path.join(ARTIFACTS_DIR, "training_report_words.json"), "w") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(ARTIFACTS_DIR, "classification_report_words.txt"), "w") as f:
        f.write(classification_report(y_test, y_pred, target_names=class_names,
                                      labels=np.arange(num_classes), zero_division=0))

    save_curves(hist)
    save_confusion(cm, class_names)

    # The backend loads *_s2.h5, but scripts/export_tfjs.py and
    # scripts/keras_to_tfjs.py look for sign_speak_words_lstm.h5. Keep both in sync.
    import shutil
    shutil.copyfile(MODEL_PATH, EXPORT_MODEL_PATH)

    print("\n" + "=" * 60)
    print(f"Train accuracy : {train_acc * 100:.2f}%")
    print(f"Val accuracy   : {val_acc * 100:.2f}%")
    print(f"TEST accuracy  : {test_acc * 100:.2f}%   (top-3 {test_top3 * 100:.2f}%)")
    print(f"Macro F1       : {f1 * 100:.2f}%")
    print(f"Test loss      : {test_loss:.4f}")
    print(f"Inference      : {latency_median:.2f} ms median, {latency_p95:.2f} ms p95")
    print(f"Model saved    : {MODEL_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
