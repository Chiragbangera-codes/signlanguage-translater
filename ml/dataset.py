import os

import pandas as pd
from dotenv import load_dotenv

# Try loading env file from common locations (root first, then backend)
env_paths = [
    os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"),
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "backend", ".env")
]
for path in env_paths:
    if os.path.exists(path):
        load_dotenv(path)
        break

# Default path if environment variable is not defined
DEFAULT_DATASET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "dataset",
    "Indian Sign Language Gesture Landmarks.csv"
)
DATASET_PATH = os.getenv("DATASET_PATH", DEFAULT_DATASET_PATH)

def load_dataset(path: str = None) -> pd.DataFrame:
    """Loads the ISL hand landmarks dataset from the specified path."""
    if path is None:
        path = DATASET_PATH

    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found at: {path}")

    df = pd.read_csv(path)
    return df

def validate_schema(df: pd.DataFrame) -> dict:
    """Validates the schema of the loaded dataset.

    Checks for:
    - target column
    - uses_two_hands column
    - 63 left_hand landmark columns (left_hand_x_0 to left_hand_z_20)
    - 63 right_hand landmark columns (right_hand_x_0 to right_hand_z_20)
    """
    errors = []

    # Check essential columns
    if "target" not in df.columns:
        errors.append("Missing column: target")
    if "uses_two_hands" not in df.columns:
        errors.append("Missing column: uses_two_hands")

    # Check left hand landmark columns
    left_hand_cols = []
    for i in range(21):
        for coord in ['x', 'y', 'z']:
            col = f"left_hand_{coord}_{i}"
            left_hand_cols.append(col)
            if col not in df.columns:
                errors.append(f"Missing left hand landmark column: {col}")

    # Check right hand landmark columns
    right_hand_cols = []
    for i in range(21):
        for coord in ['x', 'y', 'z']:
            col = f"right_hand_{coord}_{i}"
            right_hand_cols.append(col)
            if col not in df.columns:
                errors.append(f"Missing right hand landmark column: {col}")

    is_valid = len(errors) == 0
    return {
        "is_valid": is_valid,
        "errors": errors,
        "left_hand_count": len([c for c in left_hand_cols if c in df.columns]),
        "right_hand_count": len([c for c in right_hand_cols if c in df.columns])
    }

def analyze_dataset(df: pd.DataFrame) -> dict:
    """Performs comprehensive analysis on the loaded dataset."""
    shape = df.shape
    columns = list(df.columns)
    data_types = df.dtypes.value_counts().to_dict()
    missing_values = df.isnull().sum().sum()
    duplicate_rows = df.duplicated().sum()

    # Class distribution
    class_dist = df["target"].value_counts().sort_index().to_dict()

    # Feature count (excluding target)
    feature_count = len(columns) - 1

    # Uses two hands distribution
    uses_two_hands_dist = df["uses_two_hands"].value_counts().to_dict()

    # Descriptive statistics
    desc_stats = df.describe()

    return {
        "shape": shape,
        "columns": columns,
        "data_types": {str(k): int(v) for k, v in data_types.items()},
        "missing_values": int(missing_values),
        "duplicate_rows": int(duplicate_rows),
        "class_distribution": class_dist,
        "feature_count": feature_count,
        "uses_two_hands_distribution": uses_two_hands_dist,
        "descriptive_statistics": desc_stats
    }

def print_summary_report(analysis: dict, schema_validation: dict):
    """Prints dataset summary analysis to the console."""
    print("=" * 60)
    print("             DATASET SUMMARY REPORT")
    print("=" * 60)
    print(f"Dataset Shape       : {analysis['shape']}")
    print(f"Feature Count       : {analysis['feature_count']}")
    print(f"Missing Values      : {analysis['missing_values']}")
    print(f"Duplicate Rows      : {analysis['duplicate_rows']}")
    print("-" * 60)
    print("SCHEMA VALIDATION:")
    print(f"  Is Schema Valid   : {schema_validation['is_valid']}")
    if not schema_validation['is_valid']:
        print(f"  Schema Errors     : {len(schema_validation['errors'])} errors found")
        for err in schema_validation['errors'][:5]:
            print(f"    - {err}")
    else:
        print(f"  Left Hand Features: {schema_validation['left_hand_count']} columns verified")
        print(f"  Right Hand Features: {schema_validation['right_hand_count']} columns verified")
        print("  uses_two_hands    : verified")

    print("-" * 60)
    print("DATA TYPES DISTRIBUTION:")
    for dt, count in analysis['data_types'].items():
        print(f"  {dt:<18}: {count} columns")

    print("-" * 60)
    print("USES_TWO_HANDS DISTRIBUTION:")
    for val, count in analysis['uses_two_hands_distribution'].items():
        hand_label = "Two Hands" if float(val) == 1.0 else "One Hand"
        print(f"  {hand_label:<18}: {count} ({count/analysis['shape'][0]*100:.2f}%)")

    print("-" * 60)
    print("CLASS DISTRIBUTION (Alphabet Indices):")
    for cls, count in sorted(analysis['class_distribution'].items(), key=lambda x: int(x[0])):
        letter = chr(65 + int(cls))  # Map index to ASCII: 0 -> A, 1 -> B, etc.
        print(f"  Class {cls:<3} ({letter})      : {count} samples ({count/analysis['shape'][0]*100:.2f}%)")
    print("=" * 60)

if __name__ == "__main__":
    print(f"Attempting to load dataset from: {DATASET_PATH}")
    try:
        df = load_dataset()
        print("Dataset loaded successfully!")
        schema_val = validate_schema(df)
        analysis = analyze_dataset(df)
        print_summary_report(analysis, schema_val)
    except Exception as e:
        print(f"Error during dataset analysis: {e}")
