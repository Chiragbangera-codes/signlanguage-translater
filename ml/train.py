import json
import os

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

# Load env variables
env_paths = [
    os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"),
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "backend", ".env")
]
for path in env_paths:
    if os.path.exists(path):
        load_dotenv(path)
        break

MODEL_MODE = os.getenv("MODEL_MODE", "numbers").lower()
_IS_WORDS = MODEL_MODE == "words"

DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
DEFAULT_MODEL_SAVE_PATH = os.path.join(
    DEFAULT_MODEL_DIR,
    "sign_speak_words_lstm.h5" if _IS_WORDS else "sign_speak_lstm.h5",
)
MODEL_SAVE_PATH = os.getenv(
    "WORDS_MODEL_SAVE_PATH" if _IS_WORDS else "MODEL_SAVE_PATH",
    DEFAULT_MODEL_SAVE_PATH,
)

DEFAULT_METADATA_PATH = os.path.join(
    DEFAULT_MODEL_DIR,
    "training_metadata_words.json" if _IS_WORDS else "training_metadata.json",
)
METADATA_PATH = os.getenv("METADATA_PATH", DEFAULT_METADATA_PATH)

DEFAULT_DATASET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "dataset",
    "preprocessed_data_words.npz" if _IS_WORDS else "preprocessed_data.npz",
)

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")

def load_preprocessed_data(path=DEFAULT_DATASET_PATH):
    """Loads the train, validation, and test datasets from npz file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Preprocessed splits not found at: {path}")
    data = np.load(path)
    return (
        data['X_train'], data['y_train'],
        data['X_val'], data['y_val'],
        data['X_test'], data['y_test']
    )

def create_lstm_model(input_shape: tuple[int, int] = (30, 127), num_classes: int = 10) -> Sequential:
    """Creates an LSTM classifier for landmark sequences.

    Capacity scales with the class count. The single LSTM(64) below was sized
    for 10 digits; on 33 word classes it underfit badly — training accuracy
    plateaued around 64% with validation only 4 points behind, meaning the
    model could not fit even the data it had seen. That is a capacity problem,
    not a data or regularisation one, so the wider variant stacks two
    recurrent layers and widens the head.
    """
    if num_classes > 15:
        layers = [
            Input(shape=input_shape),
            LSTM(160, return_sequences=True),
            Dropout(0.3),
            LSTM(96, return_sequences=False),
            Dropout(0.3),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(num_classes, activation='softmax'),
        ]
    else:
        layers = [
            Input(shape=input_shape),
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(num_classes, activation='softmax'),
        ]

    return Sequential(layers, name="SignSpeak_LSTM")

def train_and_compare():
    """Trains the LSTM model, saves checkpoints and training charts."""
    os.makedirs(DEFAULT_MODEL_DIR, exist_ok=True)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    print("Loading preprocessed splits...")
    print(f"Model mode: {MODEL_MODE}")
    X_train, y_train, X_val, y_val, X_test, y_test = load_preprocessed_data()

    if X_train.ndim != 3 or X_val.ndim != 3 or X_test.ndim != 3:
        raise ValueError(
            "Expected preprocessed sequences shaped (samples, frames, features). "
            f"Received train={X_train.shape}, validation={X_val.shape}, test={X_test.shape}."
        )
    if X_train.shape[1:] != X_val.shape[1:] or X_train.shape[1:] != X_test.shape[1:]:
        raise ValueError("Train, validation, and test sequences must use the same shape.")

    input_shape = X_train.shape[1:]
    num_classes = len(np.unique(np.concatenate((y_train, y_val, y_test))))


    print("\n" + "=" * 60)
    print("             LSTM MODEL TRAINING")
    print("=" * 60)

    print("\n--- Training LSTM model ---")
    final_model = create_lstm_model(input_shape=input_shape, num_classes=num_classes)
    final_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Track val_accuracy, not val_loss. The two diverge here: val_loss bottomed
    # at epoch 35 and drifted while val_accuracy kept climbing for another 17
    # epochs (the model grows overconfident on the cases it still gets wrong,
    # which costs loss but not accuracy). Selecting on loss silently discarded
    # the more accurate weights. Accuracy is what the demo is judged on.
    final_callbacks = [
        EarlyStopping(monitor='val_accuracy', mode='max', patience=25,
                      restore_best_weights=True),
        ModelCheckpoint(filepath=MODEL_SAVE_PATH, monitor='val_accuracy', mode='max',
                        save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, min_lr=1e-6)
    ]

    final_history = final_model.fit(
        X_train, y_train,
        # The previous 35-epoch cap cut training off while val_loss was still
        # falling — EarlyStopping never fired. Let the callback decide instead.
        epochs=300,
        batch_size=64,
        validation_data=(X_val, y_val),
        callbacks=final_callbacks,
        verbose=1
    )

    # Save training history
    history_filename = "training_history_words.json" if _IS_WORDS else "training_history.json"
    history_path = os.path.join(DEFAULT_MODEL_DIR, history_filename)
    with open(history_path, "w") as f:
        json.dump(
            {metric: [float(value) for value in values] for metric, values in final_history.history.items()},
            f,
            indent=4,
        )

    print(f"Model successfully trained and saved to: {MODEL_SAVE_PATH}")
    print(f"History successfully saved to: {history_path}")

    # Plot curves
    plot_curves(final_history.history)

    return final_model, final_history.history, "LSTM"

def plot_curves(history: dict):
    """Plots training/validation accuracy and loss curves."""
    epochs_range = range(1, len(history['accuracy']) + 1)

    # Accuracy Curve
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, history['accuracy'], label='Training Accuracy', color='#1abc9c', linewidth=2)
    plt.plot(epochs_range, history['val_accuracy'], label='Validation Accuracy', color='#e67e22', linewidth=2)
    plt.title('SignSpeak AI - Training & Validation Accuracy', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(loc='lower right', frameon=True)
    plt.tight_layout()
    acc_curve_path = os.path.join(ARTIFACTS_DIR, "accuracy_curve.png")
    plt.savefig(acc_curve_path, dpi=150)
    plt.close()
    print(f"Saved accuracy curve to: {acc_curve_path}")

    # Loss Curve
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, history['loss'], label='Training Loss', color='#2ecc71', linewidth=2)
    plt.plot(epochs_range, history['val_loss'], label='Validation Loss', color='#e74c3c', linewidth=2)
    plt.title('SignSpeak AI - Training & Validation Loss', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(loc='upper right', frameon=True)
    plt.tight_layout()
    loss_curve_path = os.path.join(ARTIFACTS_DIR, "loss_curve.png")
    plt.savefig(loss_curve_path, dpi=150)
    plt.close()
    print(f"Saved loss curve to: {loss_curve_path}")

if __name__ == "__main__":
    try:
        train_and_compare()
    except Exception as e:
        print(f"Error in training pipeline: {e}")
