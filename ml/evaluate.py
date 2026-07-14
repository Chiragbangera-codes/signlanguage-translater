import datetime
import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

# Load env variables
env_paths = [
    os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"),
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "backend", ".env")
]
for path in env_paths:
    if os.path.exists(path):
        load_dotenv(path)
        break

DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
DEFAULT_MODEL_SAVE_PATH = os.path.join(DEFAULT_MODEL_DIR, "sign_speak_model.keras")
MODEL_SAVE_PATH = os.getenv("MODEL_SAVE_PATH", DEFAULT_MODEL_SAVE_PATH)

DEFAULT_LABEL_ENCODER_PATH = os.path.join(DEFAULT_MODEL_DIR, "label_encoder.pkl")
LABEL_ENCODER_PATH = os.getenv("LABEL_ENCODER_PATH", DEFAULT_LABEL_ENCODER_PATH)

DEFAULT_METADATA_PATH = os.path.join(DEFAULT_MODEL_DIR, "training_metadata.json")
METADATA_PATH = os.getenv("METADATA_PATH", DEFAULT_METADATA_PATH)

DEFAULT_DATASET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "dataset",
    "preprocessed_data.npz"
)

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")

def evaluate_model():
    """Evaluates the trained model on test splits, prints metrics,

    plots confusion matrix, and saves metadata.
    """
    print("Loading test data...")
    if not os.path.exists(DEFAULT_DATASET_PATH):
        raise FileNotFoundError(f"Preprocessed dataset splits not found at: {DEFAULT_DATASET_PATH}")
    data = np.load(DEFAULT_DATASET_PATH)
    X_test, y_test = data['X_test'], data['y_test']

    print(f"Loading model from: {MODEL_SAVE_PATH}...")
    if not os.path.exists(MODEL_SAVE_PATH):
        raise FileNotFoundError(f"Model file not found at: {MODEL_SAVE_PATH}")
    model = tf.keras.models.load_model(MODEL_SAVE_PATH)

    print(f"Loading label encoder from: {LABEL_ENCODER_PATH}...")
    if not os.path.exists(LABEL_ENCODER_PATH):
        raise FileNotFoundError(f"Label encoder not found at: {LABEL_ENCODER_PATH}")
    with open(LABEL_ENCODER_PATH, "rb") as f:
        le = pickle.load(f)

    print("Running predictions on test set...")
    predictions = model.predict(X_test)
    y_pred = np.argmax(predictions, axis=1)

    # Compute metrics
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    report = classification_report(y_test, y_pred, target_names=[chr(65 + int(c)) for c in le.classes_])
    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 60)
    print("             TEST EVALUATION METRICS")
    print("=" * 60)
    print(f"Accuracy  : {acc:.4f} ({acc*100:.2f}%)")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print("-" * 60)
    print("Classification Report:\n")
    print(report)
    print("=" * 60)

    # Plot and save confusion matrix
    plot_confusion_matrix(cm, [chr(65 + int(c)) for c in le.classes_])

    # Load comparison details if exist
    comparison_path = os.path.join(DEFAULT_MODEL_DIR, "architectures_comparison.json")
    architectures_comparison = {}
    if os.path.exists(comparison_path):
        try:
            with open(comparison_path, "r") as f:
                architectures_comparison = json.load(f)
        except Exception:
            pass

    # Save training_metadata.json
    metadata = {
        "project_name": "SignSpeak AI",
        "version": "1.0.0",
        "date_trained": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_architecture": {
            "name": model.name,
            "total_params": model.count_params(),
            "layers": [layer.name for layer in model.layers]
        },
        "hyperparameters": {
            "optimizer": "adam",
            "loss": "sparse_categorical_crossentropy",
            "batch_size": 64
        },
        "evaluation_metrics": {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1)
        },
        "architectures_comparison": architectures_comparison
    }

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved training metadata to: {METADATA_PATH}")

def plot_confusion_matrix(cm, classes):
    """Plots and saves confusion matrix heatmap using matplotlib."""
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Purples)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title='SignSpeak AI - Confusion Matrix (Test Set)',
        ylabel='True Label',
        xlabel='Predicted Label'
    )

    # Rotate xticks
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add values inside cells
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            # Print text only if count > 0 to keep the plot clean
            if val > 0:
                ax.text(j, i, format(val, 'd'),
                        ha="center", va="center",
                        color="white" if val > thresh else "black",
                        fontsize=8)

    fig.tight_layout()
    cm_path = os.path.join(ARTIFACTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix plot to: {cm_path}")

if __name__ == "__main__":
    try:
        evaluate_model()
    except Exception as e:
        print(f"Error in model evaluation: {e}")
