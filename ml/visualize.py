import os

import matplotlib.pyplot as plt

from dataset import DATASET_PATH, load_dataset

# Define output path for artifacts
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")

def generate_visualizations():
    """Generates visualizations for the ISL landmarks dataset."""
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    print(f"Loading dataset for visualization from: {DATASET_PATH}")
    df = load_dataset()

    # Setup plotting style
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # 1. Class Distribution Plot
    plt.figure(figsize=(12, 6))
    class_counts = df["target"].value_counts().sort_index()
    letters = [chr(65 + int(idx)) for idx in class_counts.index]

    colors = plt.cm.plasma(class_counts.values / max(class_counts.values))
    bars = plt.bar(letters, class_counts.values, color=colors, edgecolor='none', alpha=0.95)

    plt.title("SignSpeak AI - ISL Dataset Class Distribution (A-Z)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Sign Language Alphabet Letter", fontsize=12, labelpad=10)
    plt.ylabel("Sample Count", fontsize=12, labelpad=10)
    plt.ylim(0, max(class_counts.values) + 200)

    # Add values on top of the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height + 30, f'{int(height)}',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    class_dist_path = os.path.join(ARTIFACTS_DIR, "class_distribution.png")
    plt.savefig(class_dist_path, dpi=150)
    plt.close()
    print(f"Saved class distribution plot to: {class_dist_path}")

    # 2. Hand Usage Plot (One Hand vs Two Hands)
    plt.figure(figsize=(7, 7))
    hand_counts = df["uses_two_hands"].value_counts()
    labels = ['Two Hands (Both Active)', 'One Hand (Left Only)']
    sizes = [hand_counts.get(1.0, 0), hand_counts.get(0.0, 0)]
    colors = ['#8e44ad', '#3498db']

    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140,
            colors=colors, textprops={'fontsize': 12, 'fontweight': 'bold'},
            wedgeprops={'edgecolor': 'white', 'linewidth': 2, 'antialiased': True})

    plt.title("SignSpeak AI - ISL Dataset Hand Usage Pattern", fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    hand_usage_path = os.path.join(ARTIFACTS_DIR, "hand_usage.png")
    plt.savefig(hand_usage_path, dpi=150)
    plt.close()
    print(f"Saved hand usage plot to: {hand_usage_path}")

    # 3. Coordinate Value Histogram (illustrating the -1.0 values)
    plt.figure(figsize=(10, 6))

    # Extract index wrist x and right wrist x coordinates for comparison
    left_wrist_x = df['left_hand_x_0']
    right_wrist_x = df['right_hand_x_0']

    plt.hist(left_wrist_x, bins=50, alpha=0.6, label='Left Hand Wrist X Coord', color='#2ecc71')
    plt.hist(right_wrist_x, bins=50, alpha=0.6, label='Right Hand Wrist X Coord', color='#e74c3c')

    plt.title("Landmark Coordinates Distribution (Left vs Right Wrist X)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Coordinate Value", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.yscale('log') # Use log scale to clearly see the -1.0 peak
    plt.legend(loc='upper right', frameon=True, fontsize=11)

    # Add annotation for the -1.0 missing peak
    plt.annotate('Missing Hand Padding (-1.0)', xy=(-1.0, 11000), xytext=(-0.5, 20000),
                 arrowprops={"facecolor": 'black', "shrink": 0.05, "width": 1.5, "headwidth": 8},
                 fontsize=11, fontweight='bold', color='#c0392b')

    plt.tight_layout()
    coords_path = os.path.join(ARTIFACTS_DIR, "landmark_coordinates_distribution.png")
    plt.savefig(coords_path, dpi=150)
    plt.close()
    print(f"Saved coordinate distribution plot to: {coords_path}")

if __name__ == "__main__":
    generate_visualizations()
