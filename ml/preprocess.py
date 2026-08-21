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

def normalize_hands_batch(hands_batch: np.ndarray) -> np.ndarray:
    """Vectorized normalization for an (N, 63) array of hand landmarks.

    Same logic as ``normalize_hand`` but applied to all frames at once
    using NumPy broadcasting for significantly faster execution.
    """
    missing_mask = np.all(np.isclose(hands_batch, -1.0), axis=1) | (hands_batch[:, 0] == -1.0)

    coords = hands_batch.reshape(-1, 21, 3)
    wrist = coords[:, 0, :]
    translated = coords - wrist[:, np.newaxis, :]

    distances = np.linalg.norm(translated, axis=2)
    max_dist = np.max(distances, axis=1, keepdims=True)
    max_dist = np.maximum(max_dist, 1e-8)

    scaled = translated / max_dist[:, :, np.newaxis]
    scaled[missing_mask] = -1.0

    return scaled.reshape(hands_batch.shape)

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

    # Split: Train + Val (90%) and Test (10%)  [augmentation applied after]
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


# ============================================================
# TRAINING-SET AUGMENTATION
# ============================================================

AUGMENT_FACTOR = int(os.getenv("AUGMENT_FACTOR", "3"))

# Every transform below must survive normalize_hand(), which moves the wrist to
# the origin and divides by hand size. That erases absolute translation and
# scale, so translating or rescaling a sequence would be silently undone. What
# survives is mirroring, rotation, time warping and per-coordinate jitter.


def _hand_present(hand_63: np.ndarray) -> bool:
    """A hand slot is padding when every value is the -1.0 sentinel."""
    return not np.all(np.isclose(hand_63, -1.0))


def _apply_to_hands(sequence: np.ndarray, fn) -> np.ndarray:
    """Runs `fn` over each present hand slot, leaving padded slots untouched."""
    out = sequence.copy()
    for frame in range(out.shape[0]):
        for start in (1, 64):
            hand = out[frame, start:start + 63]
            if _hand_present(hand):
                out[frame, start:start + 63] = fn(hand)
    return out


def augment_mirror(sequence: np.ndarray) -> np.ndarray:
    """Mirrors the signer horizontally: a right-handed sign becomes left-handed.

    Signers use either dominant hand, so this is a genuine variation rather
    than a distortion. Coordinates are wrist-centred post-normalisation, so
    negating x mirrors each hand's shape.

    For two-handed frames the slots must swap as well. Slot A holds whichever
    hand was leftmost in the camera frame, and mirroring the scene makes the
    other hand leftmost. Negating x without swapping would produce a pose that
    cannot physically occur, and the browser (which assigns slots from raw
    positions) would never present one like it.
    """
    def flip(hand):
        coords = hand.reshape(21, 3).copy()
        coords[:, 0] *= -1.0
        return coords.reshape(-1)

    out = _apply_to_hands(sequence, flip)

    for frame in range(out.shape[0]):
        slot_a = out[frame, 1:64]
        slot_b = out[frame, 64:127]
        if _hand_present(slot_a) and _hand_present(slot_b):
            a_copy = slot_a.copy()
            out[frame, 1:64] = slot_b
            out[frame, 64:127] = a_copy

    return out


def augment_rotate(sequence: np.ndarray, max_degrees: float = 15.0) -> np.ndarray:
    """Rotates the hand in the image plane — models camera tilt and wrist angle.

    One angle for the whole sequence, so the motion stays coherent.
    """
    theta = np.deg2rad(np.random.uniform(-max_degrees, max_degrees))
    cos, sin = np.cos(theta), np.sin(theta)

    def rotate(hand):
        coords = hand.reshape(21, 3).copy()
        x, y = coords[:, 0].copy(), coords[:, 1].copy()
        coords[:, 0] = x * cos - y * sin
        coords[:, 1] = x * sin + y * cos
        return coords.reshape(-1)
    return _apply_to_hands(sequence, rotate)


def augment_jitter(sequence: np.ndarray, sigma: float = 0.02) -> np.ndarray:
    """Adds small per-coordinate noise, modelling landmark detection wobble."""
    def jitter(hand):
        return hand + np.random.normal(0.0, sigma, size=hand.shape).astype(hand.dtype)
    return _apply_to_hands(sequence, jitter)


def augment_time_warp(sequence: np.ndarray, strength: float = 0.25) -> np.ndarray:
    """Resamples the sequence along a randomly warped timeline.

    Models a signer moving faster or slower through parts of the gesture.
    Nearest-frame sampling keeps padded frames intact — interpolating across a
    -1.0 sentinel would fabricate coordinates.
    """
    n = sequence.shape[0]
    base = np.linspace(0.0, 1.0, n)
    offsets = np.random.uniform(-strength, strength, size=n) / n
    warped = np.clip(np.cumsum(np.diff(base, prepend=0.0) + offsets), 0.0, 1.0)
    warped = warped / max(warped[-1], 1e-6)
    indices = np.clip(np.rint(warped * (n - 1)).astype(int), 0, n - 1)
    return sequence[indices]


def augment_training_set(X: np.ndarray, y: np.ndarray, factor: int,
                         random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Expands the training split `factor`x. Validation and test stay untouched.

    Augmenting only the training split is the point: val/test must keep
    measuring performance on unmodified data.
    """
    if factor <= 1:
        return X, y

    np.random.seed(random_state)
    augmented = [X]
    labels = [y]

    for copy in range(factor - 1):
        batch = np.empty_like(X)
        for i, sequence in enumerate(X):
            out = sequence
            # Mirror half the copies — a signer is left- or right-handed, not
            # partially so, hence a per-sequence coin flip rather than a blend.
            if np.random.rand() < 0.5:
                out = augment_mirror(out)
            out = augment_time_warp(out)
            out = augment_rotate(out)
            out = augment_jitter(out)
            batch[i] = out
        augmented.append(batch)
        labels.append(y)
        print(f"  augmented copy {copy + 1}/{factor - 1} generated")

    return np.concatenate(augmented, axis=0), np.concatenate(labels, axis=0)

if __name__ == "__main__":
    try:
        print(f"Loading sequence dataset from {DATASET_PATH}...")
        print(f"Model mode: {MODEL_MODE}")
        sequences, labels = load_sequence_dataset()

        X_train, y_train, X_val, y_val, X_test, y_test, le = preprocess_and_split(sequences, labels)

        original_train_size = X_train.shape[0]
        if AUGMENT_FACTOR > 1:
            print(f"Augmenting training split {AUGMENT_FACTOR}x "
                  f"(set AUGMENT_FACTOR=1 to disable)...")
            X_train, y_train = augment_training_set(X_train, y_train, AUGMENT_FACTOR)

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
        print(f"Training Set Size    : {X_train.shape[0]} "
              f"({original_train_size} original x{AUGMENT_FACTOR} augmented)")
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
