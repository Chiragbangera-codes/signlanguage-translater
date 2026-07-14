import os
import pickle

import numpy as np
import pandas as pd
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

DEFAULT_DATASET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "dataset",
    "Indian Sign Language Gesture Landmarks.csv"
)
DATASET_PATH = os.getenv("DATASET_PATH", DEFAULT_DATASET_PATH)

DEFAULT_LABEL_ENCODER_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "ml", "model", "label_encoder.pkl"
)
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

def preprocess_features(df: pd.DataFrame) -> np.ndarray:
    """Applies hand normalization on left and right hand coordinates of the dataframe.
    Returns a numpy array of shape (num_samples, 127).
    """
    num_samples = len(df)
    processed_features = np.zeros((num_samples, 127))

    # Feature columns mapping
    left_hand_cols = [f"left_hand_{coord}_{i}" for i in range(21) for coord in ['x', 'y', 'z']]
    right_hand_cols = [f"right_hand_{coord}_{i}" for i in range(21) for coord in ['x', 'y', 'z']]

    # Extract raw data
    left_hand_data = df[left_hand_cols].values
    right_hand_data = df[right_hand_cols].values
    uses_two_hands = df["uses_two_hands"].values

    for i in range(num_samples):
        # Normalize left hand
        norm_left = normalize_hand(left_hand_data[i])
        # Normalize right hand
        norm_right = normalize_hand(right_hand_data[i])

        # Concatenate: uses_two_hands (1), left_hand (63), right_hand (63)
        processed_features[i, 0] = uses_two_hands[i]
        processed_features[i, 1:64] = norm_left
        processed_features[i, 64:127] = norm_right

    return processed_features

def preprocess_and_split(df: pd.DataFrame, test_size=0.1, val_size=0.1, random_state=42):
    """Normalizes features, encodes labels, and splits dataset into Train, Val, and Test sets."""
    print("Normalizing hand landmarks...")
    X = preprocess_features(df)

    print("Encoding target labels...")
    # Clean label encoder
    le = LabelEncoder()
    y = le.fit_transform(df["target"].astype(str))

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
    from dataset import load_dataset

    try:
        print(f"Loading raw dataset from {DATASET_PATH}...")
        df = load_dataset()

        X_train, y_train, X_val, y_val, X_test, y_test, le = preprocess_and_split(df)

        # Save preprocessed splits for Phase 2C decoupling
        npz_dir = os.path.dirname(DEFAULT_DATASET_PATH)
        npz_path = os.path.join(npz_dir, "preprocessed_data.npz")
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
        print(f"Total Samples        : {len(df)}")
        print(f"Training Set Size    : {X_train.shape[0]} ({X_train.shape[0]/len(df)*100:.1f}%)")
        print(f"Validation Set Size  : {X_val.shape[0]} ({X_val.shape[0]/len(df)*100:.1f}%)")
        print(f"Testing Set Size     : {X_test.shape[0]} ({X_test.shape[0]/len(df)*100:.1f}%)")
        print(f"Feature Dimension    : {X_train.shape[1]}")
        print(f"Label Encoding Classes: {len(le.classes_)} classes ({le.classes_})")

        # Verify missing-hand preservation
        # Find a row where right hand is missing in X_train
        missing_mask = (X_train[:, 0] == 0.0)
        if np.any(missing_mask):
            sample_idx = np.where(missing_mask)[0][0]
            right_hand_sample = X_train[sample_idx, 64:127]
            is_padded = np.all(right_hand_sample == -1.0)
            print(f"Missing hand check (-1.0 preserved in preprocessed splits): {is_padded}")
        else:
            print("Missing hand check: No single hand samples found in training split (unexpected)")
        print("=" * 60)

    except Exception as e:
        print(f"Error during preprocessing pipeline: {e}")
