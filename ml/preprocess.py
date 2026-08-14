import os
import pickle

import numpy as np
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

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

DEFAULT_DATASET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "dataset", "words" if MODEL_MODE == "words" else "digits"
)
DATASET_PATH = os.getenv("DATASET_PATH", DEFAULT_DATASET_PATH)

DEFAULT_LABEL_ENCODER_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "ml", "model",
    "label_encoder_words.pkl" if MODEL_MODE == "words" else "label_encoder.pkl"
)
if MODEL_MODE == "words":
    LABEL_ENCODER_PATH = os.getenv("WORDS_LABEL_ENCODER_PATH", DEFAULT_LABEL_ENCODER_PATH)
else:
    LABEL_ENCODER_PATH = os.getenv("LABEL_ENCODER_PATH", DEFAULT_LABEL_ENCODER_PATH)

def normalize_hand(landmarks_63: np.ndarray) -> np.ndarray:
    """Normalizes a single hand's 63 landmarks:
    - Translates landmarks so the wrist (index 0) is at (0, 0, 0).
    - Scales landmarks by hand size (maximum distance from wrist to any landmark).
    - Preserves missing-hand padding values (-1.0).
    """
    # Check if hand is missing (padded with exactly -1.0)
    # The dataset uses -1.0 for all coordinates of the missing hand
    if np.allclose(landmarks_63, -1.0) or landmarks_63[0] == -1.0:
        return landmarks_63

    # Reshape to (21, 3) for easier 3D operations
    coords = landmarks_63.reshape(21, 3)

    # 1. Translate wrist (index 0) to origin (0, 0, 0)
    wrist = coords[0]
    translated = coords - wrist

    # 2. Scale by hand size (max Euclidean distance from wrist to any landmark)
    distances = np.linalg.norm(translated, axis=1)
    max_dist = np.max(distances)

    if max_dist > 0:
        scaled = translated / max_dist
    else:
        scaled = translated

    # Flatten back to 63 features
    return scaled.flatten()

def load_sequence_dataset(path: str = DATASET_PATH) -> tuple[np.ndarray, np.ndarray]:
    """Loads ``<label>/sequence_*.npy`` files from the collected sequence dataset."""
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Sequence dataset directory not found: {path}")

    sequences, labels = [], []
    class_dirs = [entry for entry in os.scandir(path) if entry.is_dir()]
    class_dirs.sort(key=lambda entry: (not entry.name.isdigit(), int(entry.name) if entry.name.isdigit() else entry.name))

    for class_dir in class_dirs:
        for filename in sorted(os.listdir(class_dir.path)):
            if not filename.endswith(".npy"):
                continue
            sequence = np.load(os.path.join(class_dir.path, filename)).astype(np.float32)
            if sequence.ndim != 2 or sequence.shape[1] != 127:
                raise ValueError(
                    f"Expected a (frames, 127) sequence, got {sequence.shape} in "
                    f"{os.path.join(class_dir.path, filename)}"
                )
            sequences.append(sequence)
            labels.append(class_dir.name)

    if not sequences:
        raise ValueError(f"No .npy sequence files found in: {path}")

    frame_counts = {sequence.shape[0] for sequence in sequences}
    if len(frame_counts) != 1:
        raise ValueError(f"All sequences must have the same frame count; found {sorted(frame_counts)}")

    return np.stack(sequences), np.asarray(labels)


def preprocess_features(sequences: np.ndarray) -> np.ndarray:
    """Normalizes both hands in every frame of ``(samples, frames, 127)`` data."""
    if sequences.ndim != 3 or sequences.shape[2] != 127:
        raise ValueError(f"Expected sequences shaped (samples, frames, 127), got {sequences.shape}")

    processed = sequences.astype(np.float32, copy=True)
    for sample_index, sequence in enumerate(processed):
        for frame_index, frame in enumerate(sequence):
            frame[1:64] = normalize_hand(frame[1:64])
            frame[64:127] = normalize_hand(frame[64:127])
            processed[sample_index, frame_index] = frame
    return processed


def preprocess_and_split(sequences: np.ndarray, labels: np.ndarray, test_size=0.1, val_size=0.1, random_state=42):
    """Normalizes sequence data, encodes labels, and creates train/validation/test splits."""
    print("Normalizing hand landmarks in sequences...")
    X = preprocess_features(sequences)

    print("Encoding target labels...")
    # Clean label encoder
    le = LabelEncoder()
    y = le.fit_transform(labels.astype(str))

    # Save label encoder
    os.makedirs(os.path.dirname(LABEL_ENCODER_PATH), exist_ok=True)
    with open(LABEL_ENCODER_PATH, "wb") as f:
        pickle.dump(le, f)
    print(f"Label encoder saved to: {LABEL_ENCODER_PATH}")

    # Split: Train + Val (90%) and Test (10%)
    print(f"Splitting dataset: test_size={test_size}, val_size={val_size}...")
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Adjust val size relative to train_val (e.g. 0.1 of total is 0.1111 of 0.9)
    adjusted_val_size = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=adjusted_val_size, random_state=random_state, stratify=y_train_val
    )

    return X_train, y_train, X_val, y_val, X_test, y_test, le

if __name__ == "__main__":
    try:
        print(f"Loading sequence dataset from {DATASET_PATH}...")
        print(f"Model mode: {MODEL_MODE}")
        sequences, labels = load_sequence_dataset()

        X_train, y_train, X_val, y_val, X_test, y_test, le = preprocess_and_split(sequences, labels)

        # Save preprocessed splits for Phase 2C decoupling
        npz_dir = os.path.dirname(DEFAULT_DATASET_PATH)
        npz_filename = "preprocessed_data_words.npz" if MODEL_MODE == "words" else "preprocessed_data.npz"
        npz_path = os.path.join(npz_dir, npz_filename)
        np.savez_compressed(
            npz_path,
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            X_test=X_test, y_test=y_test
        )
        print(f"Saved preprocessed splits to: {npz_path}")

        print("\n" + "=" * 60)
        print("          PREPROCESSING REPORT STATISTICS")
        print("=" * 60)
        print(f"Total Sequences      : {len(sequences)}")
        print(f"Training Set Size    : {X_train.shape[0]} ({X_train.shape[0]/len(sequences)*100:.1f}%)")
        print(f"Validation Set Size  : {X_val.shape[0]} ({X_val.shape[0]/len(sequences)*100:.1f}%)")
        print(f"Testing Set Size     : {X_test.shape[0]} ({X_test.shape[0]/len(sequences)*100:.1f}%)")
        print(f"Sequence Shape       : {X_train.shape[1:]}")
        print(f"Label Classes        : {len(le.classes_)} ({le.classes_})")

        # Verify missing-hand preservation
        # Find a frame where the right hand is absent.
        missing_mask = (X_train[:, :, 0] == 0.0)
        if np.any(missing_mask):
            sample_idx, frame_idx = np.argwhere(missing_mask)[0]
            right_hand_sample = X_train[sample_idx, frame_idx, 64:127]
            is_padded = np.all(right_hand_sample == -1.0)
            print(f"Missing hand check (-1.0 preserved in preprocessed splits): {is_padded}")
        else:
            print("Missing hand check: No single hand samples found in training split (unexpected)")
        print("=" * 60)

    except Exception as e:
        print(f"Error during preprocessing pipeline: {e}")
