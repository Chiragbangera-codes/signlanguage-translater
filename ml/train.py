import json
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, Input

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

DEFAULT_METADATA_PATH = os.path.join(DEFAULT_MODEL_DIR, "training_metadata.json")
METADATA_PATH = os.getenv("METADATA_PATH", DEFAULT_METADATA_PATH)

DEFAULT_DATASET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "dataset",
    "preprocessed_data.npz"
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

def create_model_a() -> Sequential:
    """Simple MLP Model Architecture (Model A)"""
    model = Sequential([
        Input(shape=(127,)),
        Dense(64, activation='relu'),
        Dropout(0.1),
        Dense(32, activation='relu'),
        Dense(26, activation='softmax')
    ], name="Simple_MLP_Model_A")
    return model

def create_model_b() -> Sequential:
    """Medium MLP Model Architecture (Model B)"""
    model = Sequential([
        Input(shape=(127,)),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(26, activation='softmax')
    ], name="Medium_MLP_Model_B")
    return model

def create_model_c() -> Sequential:
    """Complex MLP Model Architecture (Model C)"""
    model = Sequential([
        Input(shape=(127,)),
        Dense(256, activation='relu'),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(26, activation='softmax')
    ], name="Complex_MLP_Model_C")
    return model

def train_and_compare():
    """Trains and compares three models, selects the best-performing model,

    and saves model checkpoints and training charts.
    """
    os.makedirs(DEFAULT_MODEL_DIR, exist_ok=True)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    print("Loading preprocessed splits...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_preprocessed_data()

    architectures = [
        ("Model A (Simple)", create_model_a),
        ("Model B (Medium)", create_model_b),
        ("Model C (Complex)", create_model_c)
    ]

    results = {}
    best_val_accuracy = -1.0
    best_model_name = ""
    best_model_fn = None

    epochs = 20
    batch_size = 64

    print("\n" + "=" * 60)
    print("             MLP MODEL COMPARISON TRAINING")
    print("=" * 60)

    for name, model_fn in architectures:
        print(f"\n--- Training {name} ---")
        model = model_fn()
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Temp model checkpoint filepath
        temp_chk_path = os.path.join(DEFAULT_MODEL_DIR, f"temp_{model.name}.keras")

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ModelCheckpoint(filepath=temp_chk_path, monitor='val_loss', save_best_only=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5)
        ]

        model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )

        # Evaluate validation performance
        # Load the best weights from the temp checkpoint
        if os.path.exists(temp_chk_path):
            model = tf.keras.models.load_model(temp_chk_path)
            # Remove temp checkpoint file
            try:
                os.remove(temp_chk_path)
            except Exception:
                pass

        val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
        print(f"{name} Validation Accuracy: {val_acc:.4f}")
        results[name] = {"val_acc": float(val_acc), "val_loss": float(val_loss)}

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_model_name = name
            best_model_fn = model_fn

    print("\n" + "=" * 60)
    print(f"Winner Model: {best_model_name} (Val Accuracy: {best_val_accuracy:.4f})")
    print("=" * 60)

    # Save comparison results
    with open(os.path.join(DEFAULT_MODEL_DIR, "architectures_comparison.json"), "w") as f:
        json.dump(results, f, indent=4)

    # Re-train or train final selected winner model to full convergence (more patience)
    print(f"\nFinal training of winner architecture: {best_model_name}...")
    final_model = best_model_fn()
    final_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    final_callbacks = [
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
        ModelCheckpoint(filepath=MODEL_SAVE_PATH, monitor='val_loss', save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
    ]

    final_history = final_model.fit(
        X_train, y_train,
        epochs=35,  # Allow longer training for final convergence
        batch_size=64,
        validation_data=(X_val, y_val),
        callbacks=final_callbacks,
        verbose=1
    )

    # Save training history
    history_path = os.path.join(DEFAULT_MODEL_DIR, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(final_history.history, f, indent=4)

    print(f"Model successfully trained and saved to: {MODEL_SAVE_PATH}")
    print(f"History successfully saved to: {history_path}")

    # Plot curves
    plot_curves(final_history.history)

    return final_model, final_history.history, best_model_name

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
