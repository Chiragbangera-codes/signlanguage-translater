import json
import os

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

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

HISTORY_PATH = os.path.join(
    MODEL_DIR,
    "training_history_words.json"
)

ARTIFACTS_DIR = os.path.join(
    PROJECT_ROOT,
    "ml",
    "artifacts",
    "words"
)


# ============================================================
# LOAD PREPROCESSED DATA
# ============================================================

def load_preprocessed_data(path=DATASET_PATH):

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Preprocessed word dataset not found at:\n{path}\n\n"
            "Run preprocess_words.py first."
        )

    data = np.load(path)

    return (
        data["X_train"],
        data["y_train"],
        data["X_val"],
        data["y_val"],
        data["X_test"],
        data["y_test"],
    )


# ============================================================
# CREATE LSTM MODEL
# ============================================================

def create_lstm_model(
    input_shape,
    num_classes
):

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
            ),
        ],
        name="SignSpeak_Words_LSTM"
    )

    return model


# ============================================================
# PLOT TRAINING CURVES
# ============================================================

def plot_curves(history):

    os.makedirs(
        ARTIFACTS_DIR,
        exist_ok=True
    )

    epochs_range = range(
        1,
        len(history["accuracy"]) + 1
    )

    # --------------------------------------------------------
    # Accuracy
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
        "SignSpeak Words - Training & Validation Accuracy"
    )

    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")

    plt.legend()
    plt.tight_layout()

    accuracy_path = os.path.join(
        ARTIFACTS_DIR,
        "accuracy_curve_words.png"
    )

    plt.savefig(
        accuracy_path,
        dpi=150
    )

    plt.close()

    print(
        f"Saved accuracy curve to:\n{accuracy_path}"
    )

    # --------------------------------------------------------
    # Loss
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
        "SignSpeak Words - Training & Validation Loss"
    )

    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    plt.legend()
    plt.tight_layout()

    loss_path = os.path.join(
        ARTIFACTS_DIR,
        "loss_curve_words.png"
    )

    plt.savefig(
        loss_path,
        dpi=150
    )

    plt.close()

    print(
        f"Saved loss curve to:\n{loss_path}"
    )


# ============================================================
# TRAIN WORD MODEL
# ============================================================

def train_words():

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    os.makedirs(
        ARTIFACTS_DIR,
        exist_ok=True
    )

    print("\n" + "=" * 60)
    print("        SIGNSPEAK WORD MODEL TRAINING")
    print("=" * 60)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print("\nLoading preprocessed word dataset...")

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test
    ) = load_preprocessed_data()

    print(
        f"Training data   : {X_train.shape}"
    )

    print(
        f"Validation data : {X_val.shape}"
    )

    print(
        f"Testing data    : {X_test.shape}"
    )

    # --------------------------------------------------------
    # Validate shapes
    # --------------------------------------------------------

    if X_train.ndim != 3:
        raise ValueError(
            f"Expected 3D training data, got {X_train.shape}"
        )

    if X_val.ndim != 3:
        raise ValueError(
            f"Expected 3D validation data, got {X_val.shape}"
        )

    if X_test.ndim != 3:
        raise ValueError(
            f"Expected 3D testing data, got {X_test.shape}"
        )

    if (
        X_train.shape[1:]
        != X_val.shape[1:]
        or
        X_train.shape[1:]
        != X_test.shape[1:]
    ):
        raise ValueError(
            "Train, validation and test "
            "sequences must have the same shape."
        )

    # --------------------------------------------------------
    # Determine model shape
    # --------------------------------------------------------

    input_shape = X_train.shape[1:]

    all_labels = np.concatenate(
        [
            y_train,
            y_val,
            y_test
        ]
    )

    num_classes = len(
        np.unique(all_labels)
    )

    print(
        f"\nInput shape : {input_shape}"
    )

    print(
        f"Word classes: {num_classes}"
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
            save_best_only=True,
            verbose=1
        ),

        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("              TRAINING WORD MODEL")
    print("=" * 60)

    history = model.fit(

        X_train,
        y_train,

        validation_data=(
            X_val,
            y_val
        ),

        epochs=35,

        batch_size=64,

        callbacks=callbacks,

        verbose=1
    )

    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------

    with open(
        HISTORY_PATH,
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
        f"{HISTORY_PATH}"
    )

    # --------------------------------------------------------
    # Plot curves
    # --------------------------------------------------------

    plot_curves(
        history.history
    )

    # --------------------------------------------------------
    # Evaluate test data
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("              TEST SET EVALUATION")
    print("=" * 60)

    test_loss, test_accuracy = model.evaluate(
        X_test,
        y_test,
        verbose=1
    )

    print(
        f"\nTest Loss     : {test_loss:.4f}"
    )

    print(
        f"Test Accuracy : {test_accuracy:.4f}"
    )

    print(
        f"Test Accuracy : {test_accuracy * 100:.2f}%"
    )

    print("\n" + "=" * 60)
    print("       WORD MODEL TRAINING COMPLETED")
    print("=" * 60)

    print(
        f"\nModel saved to:\n{MODEL_SAVE_PATH}"
    )

    return (
        model,
        history.history
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    try:

        train_words()

    except Exception as e:

        print(
            "\nError in word model training pipeline:"
        )

        print(e)
