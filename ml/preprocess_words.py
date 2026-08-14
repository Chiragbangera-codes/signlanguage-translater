import os
import pickle

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_PATH = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "words"
)

LABEL_ENCODER_PATH = os.path.join(
    PROJECT_ROOT,
    "ml",
    "model",
    "label_encoder_words.pkl"
)

PREPROCESSED_DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "preprocessed_data_words.npz"
)


# ============================================================
# NORMALIZE SINGLE HAND
# ============================================================

def normalize_hand(landmarks_63: np.ndarray) -> np.ndarray:
    """
    Normalize a single hand containing 21 landmarks (63 values).

    Steps:
    1. Move wrist to origin.
    2. Scale landmarks based on hand size.
    3. Preserve missing-hand padding (-1.0).
    """

    # Check whether hand is missing
    if np.allclose(landmarks_63, -1.0) or landmarks_63[0] == -1.0:
        return landmarks_63

    # Reshape to 21 landmarks × 3 coordinates
    coords = landmarks_63.reshape(21, 3)

    # Move wrist (landmark 0) to origin
    wrist = coords[0]
    translated = coords - wrist

    # Calculate distance of every landmark from wrist
    distances = np.linalg.norm(translated, axis=1)

    # Maximum distance = hand size
    max_dist = np.max(distances)

    # Scale
    if max_dist > 0:
        scaled = translated / max_dist
    else:
        scaled = translated

    # Return flattened 63 features
    return scaled.flatten()


# ============================================================
# LOAD WORD SEQUENCE DATASET
# ============================================================

def load_sequence_dataset(path: str = DATASET_PATH):

    print(f"Loading word dataset from: {path}")

    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"Word dataset directory not found: {path}"
        )

    sequences = []
    labels = []

    # Get word folders
    class_dirs = [
        entry
        for entry in os.scandir(path)
        if entry.is_dir()
    ]

    # Sort alphabetically
    class_dirs.sort(
        key=lambda entry: entry.name.lower()
    )

    print(f"\nFound {len(class_dirs)} word classes:")

    for class_dir in class_dirs:

        print(f"  - {class_dir.name}")

        files = sorted(
            os.listdir(class_dir.path)
        )

        for filename in files:

            if not filename.endswith(".npy"):
                continue

            file_path = os.path.join(
                class_dir.path,
                filename
            )

            sequence = np.load(
                file_path
            ).astype(np.float32)

            # Check shape
            if sequence.ndim != 2 or sequence.shape[1] != 127:

                raise ValueError(
                    f"\nInvalid sequence shape!\n"
                    f"File: {file_path}\n"
                    f"Expected: (frames, 127)\n"
                    f"Got: {sequence.shape}"
                )

            sequences.append(sequence)
            labels.append(class_dir.name)

    if not sequences:
        raise ValueError(
            f"No .npy sequence files found in: {path}"
        )

    # Check that all sequences have same frame count
    frame_counts = {
        sequence.shape[0]
        for sequence in sequences
    }

    if len(frame_counts) != 1:

        raise ValueError(
            "All sequences must have the same "
            f"frame count. Found: {sorted(frame_counts)}"
        )

    sequences = np.stack(sequences)
    labels = np.asarray(labels)

    print("\nDataset loaded successfully!")

    print(f"Total sequences : {len(sequences)}")
    print(f"Sequence shape  : {sequences.shape}")
    print(f"Labels          : {len(set(labels))}")

    return sequences, labels


# ============================================================
# PREPROCESS FEATURES
# ============================================================

def preprocess_features(
    sequences: np.ndarray
) -> np.ndarray:

    print("\nNormalizing hand landmarks...")

    if sequences.ndim != 3 or sequences.shape[2] != 127:

        raise ValueError(
            f"Expected shape (samples, frames, 127), "
            f"got {sequences.shape}"
        )

    processed = sequences.astype(
        np.float32,
        copy=True
    )

    # Process every sample
    for sample_index, sequence in enumerate(processed):

        # Process every frame
        for frame_index, frame in enumerate(sequence):

            # First hand: features 1:64
            frame[1:64] = normalize_hand(
                frame[1:64]
            )

            # Second hand: features 64:127
            frame[64:127] = normalize_hand(
                frame[64:127]
            )

            processed[
                sample_index,
                frame_index
            ] = frame

    print("Normalization completed.")

    return processed


# ============================================================
# ENCODE LABELS
# ============================================================

def encode_labels(labels):

    print("\nEncoding word labels...")

    label_encoder = LabelEncoder()

    y = label_encoder.fit_transform(
        labels.astype(str)
    )

    # Create model directory
    os.makedirs(
        os.path.dirname(LABEL_ENCODER_PATH),
        exist_ok=True
    )

    # Save encoder
    with open(
        LABEL_ENCODER_PATH,
        "wb"
    ) as f:

        pickle.dump(
            label_encoder,
            f
        )

    print(
        f"Label encoder saved to:\n"
        f"{LABEL_ENCODER_PATH}"
    )

    print(
        "\nWord classes:"
    )

    for index, label in enumerate(
        label_encoder.classes_
    ):

        print(
            f"  {index} -> {label}"
        )

    return y, label_encoder


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

def preprocess_and_split(
    sequences,
    labels,
    test_size=0.10,
    val_size=0.10,
    random_state=42
):

    # Normalize
    X = preprocess_features(
        sequences
    )

    # Encode labels
    y, label_encoder = encode_labels(
        labels
    )

    print("\nSplitting dataset...")

    # First split:
    # 90% train+validation
    # 10% test

    X_train_val, X_test, y_train_val, y_test = train_test_split(

        X,
        y,

        test_size=test_size,

        random_state=random_state,

        stratify=y
    )

    # Convert validation size
    adjusted_val_size = (
        val_size /
        (1.0 - test_size)
    )

    # Second split:
    # 80% train
    # 10% validation

    X_train, X_val, y_train, y_val = train_test_split(

        X_train_val,
        y_train_val,

        test_size=adjusted_val_size,

        random_state=random_state,

        stratify=y_train_val
    )

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        label_encoder
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    try:

        print("\n" + "=" * 60)
        print("       WORD SIGN LANGUAGE PREPROCESSING")
        print("=" * 60)

        # ----------------------------------------------------
        # Load dataset
        # ----------------------------------------------------

        sequences, labels = load_sequence_dataset()

        # ----------------------------------------------------
        # Preprocess + split
        # ----------------------------------------------------

        (
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            label_encoder
        ) = preprocess_and_split(
            sequences,
            labels
        )

        # ----------------------------------------------------
        # Save preprocessed dataset
        # ----------------------------------------------------

        np.savez_compressed(

            PREPROCESSED_DATA_PATH,

            X_train=X_train,
            y_train=y_train,

            X_val=X_val,
            y_val=y_val,

            X_test=X_test,
            y_test=y_test
        )

        print(
            f"\nPreprocessed data saved to:\n"
            f"{PREPROCESSED_DATA_PATH}"
        )

        # ----------------------------------------------------
        # Report
        # ----------------------------------------------------

        print("\n" + "=" * 60)
        print("          PREPROCESSING REPORT")
        print("=" * 60)

        total = len(sequences)

        print(
            f"Total Sequences     : {total}"
        )

        print(
            f"Training Set Size   : "
            f"{X_train.shape[0]} "
            f"({X_train.shape[0] / total * 100:.1f}%)"
        )

        print(
            f"Validation Set Size : "
            f"{X_val.shape[0]} "
            f"({X_val.shape[0] / total * 100:.1f}%)"
        )

        print(
            f"Testing Set Size    : "
            f"{X_test.shape[0]} "
            f"({X_test.shape[0] / total * 100:.1f}%)"
        )

        print(
            f"Sequence Shape      : "
            f"{X_train.shape[1:]}"
        )

        print(
            f"Word Classes        : "
            f"{len(label_encoder.classes_)}"
        )

        print(
            f"Classes             : "
            f"{list(label_encoder.classes_)}"
        )

        print("=" * 60)

        print(
            "\nWord preprocessing completed successfully!"
        )

    except Exception as e:

        print(
            "\nERROR during word preprocessing:"
        )

        print(e)
