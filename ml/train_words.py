import json
import os

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

env_paths = [
    os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        ".env"
    ),
    os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "backend",
        ".env"
    )
]

for path in env_paths:
    if os.path.exists(path):
        load_dotenv(path)
        break


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_MODE = os.getenv("MODEL_MODE", "numbers").lower()

DEFAULT_MODEL_DIR = os.path.join(
    os.path.dirname(__file__),
    "model"
)

DEFAULT_MODEL_SAVE_PATH = os.path.join(
    DEFAULT_MODEL_DIR,
    "sign_speak_lstm.h5"
)

MODEL_SAVE_PATH = os.getenv(
    "MODEL_SAVE_PATH",
    DEFAULT_MODEL_SAVE_PATH
)

DEFAULT_DATASET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "dataset",
    "preprocessed_data.npz"
)

ARTIFACTS_DIR = os.path.join(
    os.path.dirname(__file__),
    "artifacts"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_preprocessed_data(path=DEFAULT_DATASET_PATH):
    """
    Loads training, validation and test datasets
    from the preprocessed NPZ file.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Preprocessed data not found at:\n{path}"
        )

    data = np.load(path)

    X_train = data["X_train"]
    y_train = data["y_train"]

    X_val = data["X_val"]
    y_val = data["y_val"]

    X_test = data["X_test"]
    y_test = data["y_test"]

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test
    )


# ============================================================
# CREATE LSTM MODEL
# ============================================================

def create_lstm_model(
    input_shape,
    num_classes=10
):
    """
    Creates an LSTM classifier for
    sign-language number recognition.
    """

    model = Sequential(
        [
            Input(shape=input_shape),

            LSTM(
                64,
                return_sequences=False
            ),

            Dropout(0.2),

            Dense(
                64,
                activation="relu"
            ),

            Dropout(0.2),

            Dense(
                num_classes,
                activation="softmax"
            )
        ],
        name="SignSpeak_LSTM"
    )

    return model


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test,
    num_classes
):
    """
    Evaluates the trained model using
    completely unseen test data.
    """

    print("\n")
    print("=" * 60)
    print("                 FINAL TEST RESULTS")
    print("=" * 60)

    # --------------------------------------------------------
    # Test loss and accuracy
    # --------------------------------------------------------

    test_loss, test_accuracy = model.evaluate(
        X_test,
        y_test,
        verbose=1
    )

    print("\nTest Loss     :", f"{test_loss:.4f}")
    print(
        "Test Accuracy :",
        f"{test_accuracy * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    print("\nGenerating predictions...")

    y_pred_probability = model.predict(
        X_test,
        verbose=1
    )

    y_pred = np.argmax(
        y_pred_probability,
        axis=1
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=np.arange(num_classes)
    )

    print("\n")
    print("=" * 60)
    print("                 CONFUSION MATRIX")
    print("=" * 60)

    print(cm)

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    class_names = [
        str(i)
        for i in range(num_classes)
    ]

    report = classification_report(
        y_test,
        y_pred,
        labels=np.arange(num_classes),
        target_names=class_names,
        zero_division=0
    )

    print("\n")
    print("=" * 60)
    print("              CLASSIFICATION REPORT")
    print("=" * 60)

    print(report)

    # --------------------------------------------------------
    # Per-digit accuracy
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("                 PER-DIGIT ACCURACY")
    print("=" * 60)

    digit_accuracy = {}

    for digit in range(num_classes):

        digit_indices = (
            y_test == digit
        )

        total = np.sum(
            digit_indices
        )

        if total == 0:
            accuracy = 0
        else:
            correct = np.sum(
                y_pred[digit_indices] == digit
            )

            accuracy = (
                correct / total
            )

        digit_accuracy[str(digit)] = float(
            accuracy
        )

        print(
            f"Digit {digit}: "
            f"{accuracy * 100:.2f}% "
            f"({int(accuracy * total)}/{total})"
        )

    # --------------------------------------------------------
    # Save evaluation results
    # --------------------------------------------------------

    evaluation_results = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "test_accuracy_percentage": float(
            test_accuracy * 100
        ),
        "per_digit_accuracy": digit_accuracy,
        "confusion_matrix": cm.tolist()
    }

    evaluation_path = os.path.join(
        ARTIFACTS_DIR,
        "evaluation_results.json"
    )

    with open(
        evaluation_path,
        "w"
    ) as f:

        json.dump(
            evaluation_results,
            f,
            indent=4
        )

    print(
        f"\nEvaluation results saved to:\n"
        f"{evaluation_path}"
    )

    # --------------------------------------------------------
    # Save confusion matrix image
    # --------------------------------------------------------

    plt.figure(
        figsize=(8, 8)
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    disp.plot(
        values_format="d",
        ax=plt.gca()
    )

    plt.title(
        "SignSpeak - Number Recognition Confusion Matrix"
    )

    plt.tight_layout()

    confusion_path = os.path.join(
        ARTIFACTS_DIR,
        "confusion_matrix.png"
    )

    plt.savefig(
        confusion_path,
        dpi=150
    )

    plt.close()

    print(
        f"Confusion matrix saved to:\n"
        f"{confusion_path}"
    )

    return (
        test_loss,
        test_accuracy,
        y_pred,
        cm
    )


# ============================================================
# PLOT TRAINING CURVES
# ============================================================

def plot_curves(history):
    """
    Creates training/validation accuracy
    and loss graphs.
    """

    epochs_range = range(
        1,
        len(history["accuracy"]) + 1
    )

    # --------------------------------------------------------
    # Accuracy graph
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        epochs_range,
        history["accuracy"],
        label="Training Accuracy",
        linewidth=2
    )

    plt.plot(
        epochs_range,
        history["val_accuracy"],
        label="Validation Accuracy",
        linewidth=2
    )

    plt.title(
        "SignSpeak - Training & Validation Accuracy"
    )

    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    accuracy_path = os.path.join(
        ARTIFACTS_DIR,
        "accuracy_curve.png"
    )

    plt.savefig(
        accuracy_path,
        dpi=150
    )

    plt.close()

    print(
        f"Accuracy curve saved to:\n"
        f"{accuracy_path}"
    )

    # --------------------------------------------------------
    # Loss graph
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        epochs_range,
        history["loss"],
        label="Training Loss",
        linewidth=2
    )

    plt.plot(
        epochs_range,
        history["val_loss"],
        label="Validation Loss",
        linewidth=2
    )

    plt.title(
        "SignSpeak - Training & Validation Loss"
    )

    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    loss_path = os.path.join(
        ARTIFACTS_DIR,
        "loss_curve.png"
    )

    plt.savefig(
        loss_path,
        dpi=150
    )

    plt.close()

    print(
        f"Loss curve saved to:\n"
        f"{loss_path}"
    )


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def train_model():

    os.makedirs(
        DEFAULT_MODEL_DIR,
        exist_ok=True
    )

    os.makedirs(
        ARTIFACTS_DIR,
        exist_ok=True
    )

    print("\n")
    print("=" * 60)
    print("           SIGNSPEAK LSTM MODEL TRAINING")
    print("=" * 60)

    print(
        f"\nModel Mode: {MODEL_MODE}"
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print("\nLoading preprocessed data...")

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test
    ) = load_preprocessed_data()

    print("\nDataset shapes:")

    print(
        "Training   :",
        X_train.shape,
        y_train.shape
    )

    print(
        "Validation :",
        X_val.shape,
        y_val.shape
    )

    print(
        "Testing    :",
        X_test.shape,
        y_test.shape
    )

    # --------------------------------------------------------
    # Validate shapes
    # --------------------------------------------------------

    if (
        X_train.ndim != 3
        or X_val.ndim != 3
        or X_test.ndim != 3
    ):

        raise ValueError(
            "Expected input shape:\n"
            "(samples, frames, features)"
        )

    if (
        X_train.shape[1:]
        != X_val.shape[1:]
        or
        X_train.shape[1:]
        != X_test.shape[1:]
    ):

        raise ValueError(
            "Training, validation and test "
            "sequences must have the same shape."
        )

    # --------------------------------------------------------
    # Number of classes
    # --------------------------------------------------------

    num_classes = len(
        np.unique(
            np.concatenate(
                (
                    y_train,
                    y_val,
                    y_test
                )
            )
        )
    )

    input_shape = X_train.shape[1:]

    print("\nInput shape :", input_shape)
    print("Classes     :", num_classes)

    # --------------------------------------------------------
    # Check number model
    # --------------------------------------------------------

    if num_classes != 10:

        print(
            "\nWARNING:"
            f" Expected 10 classes for numbers, "
            f"but found {num_classes}."
        )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    print("\nCreating LSTM model...")

    model = create_lstm_model(
        input_shape=input_shape,
        num_classes=num_classes
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("\nModel Summary:")

    model.summary()

    # --------------------------------------------------------
    # Callbacks
    # --------------------------------------------------------

    callbacks = [

        EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True
        ),

        ModelCheckpoint(
            filepath=MODEL_SAVE_PATH,
            monitor="val_loss",
            save_best_only=True
        ),

        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6
        )
    ]

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("                   TRAINING")
    print("=" * 60)

    history = model.fit(

        X_train,
        y_train,

        epochs=35,

        batch_size=64,

        validation_data=(
            X_val,
            y_val
        ),

        callbacks=callbacks,

        verbose=1
    )

    # --------------------------------------------------------
    # Save training history
    # --------------------------------------------------------

    history_path = os.path.join(
        DEFAULT_MODEL_DIR,
        "training_history.json"
    )

    with open(
        history_path,
        "w"
    ) as f:

        json.dump(
            {
                metric: [
                    float(value)
                    for value in values
                ]
                for metric, values
                in history.history.items()
            },
            f,
            indent=4
        )

    print(
        f"\nTraining history saved to:\n"
        f"{history_path}"
    )

    # --------------------------------------------------------
    # Training summary
    # --------------------------------------------------------

    final_training_accuracy = (
        history.history["accuracy"][-1]
    )

    final_validation_accuracy = (
        history.history["val_accuracy"][-1]
    )

    print("\n")
    print("=" * 60)
    print("                 TRAINING SUMMARY")
    print("=" * 60)

    print(
        f"\nTraining Accuracy   : "
        f"{final_training_accuracy * 100:.2f}%"
    )

    print(
        f"Validation Accuracy : "
        f"{final_validation_accuracy * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Plot curves
    # --------------------------------------------------------

    plot_curves(
        history.history
    )

    # --------------------------------------------------------
    # Final test evaluation
    # --------------------------------------------------------

    (
        test_loss,
        test_accuracy,
        y_pred,
        confusion
    ) = evaluate_model(
        model,
        X_test,
        y_test,
        num_classes
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("                 FINAL SUMMARY")
    print("=" * 60)

    print(
        f"\nTraining Accuracy   : "
        f"{final_training_accuracy * 100:.2f}%"
    )

    print(
        f"Validation Accuracy : "
        f"{final_validation_accuracy * 100:.2f}%"
    )

    print(
        f"TEST ACCURACY       : "
        f"{test_accuracy * 100:.2f}%"
    )

    print(
        f"\nModel saved to:\n"
        f"{MODEL_SAVE_PATH}"
    )

    print("\nTraining and evaluation completed successfully!")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    try:

        train_model()

    except Exception as e:

        print(
            "\nERROR:"
        )

        print(e)