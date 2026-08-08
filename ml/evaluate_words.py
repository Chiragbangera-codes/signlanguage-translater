import datetime
import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "ml",
    "model"
)

DATASET_PATH = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "preprocessed_data_words.npz"
)

MODEL_SAVE_PATH = os.path.join(
    MODEL_DIR,
    "sign_speak_words_lstm.keras"
)

LABEL_ENCODER_PATH = os.path.join(
    MODEL_DIR,
    "label_encoder_words.pkl"
)

METADATA_PATH = os.path.join(
    MODEL_DIR,
    "training_metadata_words.json"
)

ARTIFACTS_DIR = os.path.join(
    PROJECT_ROOT,
    "ml",
    "artifacts",
    "words"
)


# ============================================================
# EVALUATE WORD MODEL
# ============================================================

def evaluate_model():

    print("\n" + "=" * 60)
    print("       SIGNSPEAK WORD MODEL EVALUATION")
    print("=" * 60)

    # --------------------------------------------------------
    # Load test data
    # --------------------------------------------------------

    print("\nLoading word test data...")

    if not os.path.exists(DATASET_PATH):

        raise FileNotFoundError(
            f"Preprocessed word dataset not found at:\n"
            f"{DATASET_PATH}\n\n"
            "Run preprocess_words.py first."
        )

    data = np.load(
        DATASET_PATH
    )

    X_test = data["X_test"]
    y_test = data["y_test"]

    print(
        f"Test data shape: {X_test.shape}"
    )

    # --------------------------------------------------------
    # Validate test data
    # --------------------------------------------------------

    if X_test.ndim != 3:

        raise ValueError(
            "Expected test sequences shaped "
            "(samples, frames, features), "
            f"but received {X_test.shape}."
        )

    # --------------------------------------------------------
    # Load trained word model
    # --------------------------------------------------------

    print(
        f"\nLoading word model from:\n"
        f"{MODEL_SAVE_PATH}"
    )

    if not os.path.exists(
        MODEL_SAVE_PATH
    ):

        raise FileNotFoundError(
            f"Word model not found at:\n"
            f"{MODEL_SAVE_PATH}\n\n"
            "Run train_words.py first."
        )

    model = tf.keras.models.load_model(
        MODEL_SAVE_PATH
    )

    print(
        f"Model loaded successfully: "
        f"{model.name}"
    )

    # --------------------------------------------------------
    # Check model input shape
    # --------------------------------------------------------

    model_input_shape = tuple(
        model.input_shape[1:]
    )

    test_input_shape = tuple(
        X_test.shape[1:]
    )

    if model_input_shape != test_input_shape:

        raise ValueError(
            f"\nShape mismatch!\n"
            f"Model expects : {model_input_shape}\n"
            f"Test data has : {test_input_shape}"
        )

    # --------------------------------------------------------
    # Load word label encoder
    # --------------------------------------------------------

    print(
        f"\nLoading word label encoder from:\n"
        f"{LABEL_ENCODER_PATH}"
    )

    if not os.path.exists(
        LABEL_ENCODER_PATH
    ):

        raise FileNotFoundError(
            f"Word label encoder not found at:\n"
            f"{LABEL_ENCODER_PATH}\n\n"
            "Run preprocess_words.py first."
        )

    with open(
        LABEL_ENCODER_PATH,
        "rb"
    ) as f:

        label_encoder = pickle.load(f)

    class_names = [
        str(label)
        for label in label_encoder.classes_
    ]

    print(
        f"\nWord classes ({len(class_names)}):"
    )

    for index, class_name in enumerate(
        class_names
    ):

        print(
            f"  {index} -> {class_name}"
        )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    print("\nRunning predictions on test set...")

    predictions = model.predict(
        X_test,
        verbose=1
    )

    y_pred = np.argmax(
        predictions,
        axis=1
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )
    )

    report = classification_report(
        y_test,
        y_pred,
        target_names=class_names,
        zero_division=0
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("             WORD MODEL METRICS")
    print("=" * 60)

    print(
        f"Accuracy  : {accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1 Score  : {f1:.4f}"
    )

    print("-" * 60)

    print(
        "Classification Report:\n"
    )

    print(
        report
    )

    print("=" * 60)

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    plot_confusion_matrix(
        cm,
        class_names
    )

    # --------------------------------------------------------
    # Save evaluation metadata
    # --------------------------------------------------------

    metadata = {

        "project_name": "SignSpeak AI",

        "model_type": "Word Sign Language Recognition",

        "version": "1.0.0",

        "date_evaluated":
            datetime.datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

        "model_architecture": {

            "name": model.name,

            "total_params":
                model.count_params(),

            "input_shape":
                list(model_input_shape),

            "layers": [
                layer.name
                for layer in model.layers
            ]
        },

        "test_dataset": {

            "samples":
                int(len(X_test)),

            "sequence_shape":
                list(X_test.shape[1:]),

            "number_of_classes":
                len(class_names),

            "classes":
                class_names
        },

        "evaluation_metrics": {

            "accuracy":
                float(accuracy),

            "precision":
                float(precision),

            "recall":
                float(recall),

            "f1_score":
                float(f1)
        },

        "hyperparameters": {

            "optimizer":
                "adam",

            "loss":
                "sparse_categorical_crossentropy",

            "batch_size":
                64
        }
    }

    with open(
        METADATA_PATH,
        "w"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4
        )

    print(
        f"\nEvaluation metadata saved to:\n"
        f"{METADATA_PATH}"
    )

    print("\n" + "=" * 60)
    print("       WORD MODEL EVALUATION COMPLETED")
    print("=" * 60)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }


# ============================================================
# CONFUSION MATRIX
# ============================================================

def plot_confusion_matrix(
    cm,
    classes
):

    os.makedirs(
        ARTIFACTS_DIR,
        exist_ok=True
    )

    fig, ax = plt.subplots(
        figsize=(12, 10)
    )

    im = ax.imshow(
        cm,
        interpolation="nearest",
        cmap=plt.cm.Purples
    )

    ax.figure.colorbar(
        im,
        ax=ax
    )

    ax.set(
        xticks=np.arange(
            cm.shape[1]
        ),

        yticks=np.arange(
            cm.shape[0]
        ),

        xticklabels=classes,

        yticklabels=classes,

        title=(
            "SignSpeak AI - "
            "Word Model Confusion Matrix"
        ),

        ylabel="True Label",

        xlabel="Predicted Label"
    )

    # Rotate x-axis labels
    plt.setp(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor"
    )

    # Add values inside cells
    threshold = cm.max() / 2.0

    for i in range(
        cm.shape[0]
    ):

        for j in range(
            cm.shape[1]
        ):

            value = cm[i, j]

            if value > 0:

                ax.text(
                    j,
                    i,
                    format(
                        value,
                        "d"
                    ),

                    ha="center",

                    va="center",

                    color=(
                        "white"
                        if value > threshold
                        else "black"
                    ),

                    fontsize=8
                )

    fig.tight_layout()

    confusion_matrix_path = os.path.join(
        ARTIFACTS_DIR,
        "confusion_matrix_words.png"
    )

    plt.savefig(
        confusion_matrix_path,
        dpi=150
    )

    plt.close()

    print(
        f"\nSaved word confusion matrix to:\n"
        f"{confusion_matrix_path}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    try:

        evaluate_model()

    except Exception as e:

        print(
            "\nError in word model evaluation:"
        )

        print(e)