
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    # Pasted into a notebook cell rather than run as a file.
    PROJECT_ROOT = Path.cwd()

FRAMES_PER_SEQUENCE = 30
FEATURES_PER_FRAME = 127
LANDMARKS_PER_HAND = 21
COORDS_PER_HAND = LANDMARKS_PER_HAND * 3
MISSING = -1.0

VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


# ============================================================
# FRAME ASSEMBLY
# ============================================================

def assemble_frame(hands: list[np.ndarray]) -> np.ndarray:
    """Builds one 127-feature frame from 0-2 hands of 63 coordinates each.

    Hands are ordered by mean x so the slot assignment is reproducible without
    relying on left/right labels. See the module docstring.
    """
    slot_a = np.full(COORDS_PER_HAND, MISSING, dtype=np.float32)
    slot_b = np.full(COORDS_PER_HAND, MISSING, dtype=np.float32)

    present = [h for h in hands if h is not None and not np.all(np.isnan(h))]

    if len(present) == 1:
        slot_a = present[0]
    elif len(present) >= 2:
        # x is every third value starting at 0.
        ordered = sorted(present[:2], key=lambda h: float(np.nanmean(h[0::3])))
        slot_a, slot_b = ordered[0], ordered[1]

    uses_two_hands = 1.0 if len(present) >= 2 else 0.0
    frame = np.concatenate(([uses_two_hands], slot_a, slot_b)).astype(np.float32)

    return np.nan_to_num(frame, nan=MISSING)


def resample_to_length(frames: np.ndarray, length: int = FRAMES_PER_SEQUENCE) -> np.ndarray:
    """Resamples a (n, 127) sequence to exactly `length` frames.

    Nearest-index sampling rather than interpolation: blending a real frame
    with a -1.0 padded one would invent coordinates that never occurred.
    """
    n = frames.shape[0]
    if n == length:
        return frames
    indices = np.linspace(0, n - 1, num=length)
    return frames[np.rint(indices).astype(int)]


def has_any_hand(sequence: np.ndarray) -> bool:
    """True when at least one frame contains a detected hand."""
    return bool(np.any(~np.isclose(sequence[:, 1:1 + COORDS_PER_HAND], MISSING)))


def trim_to_active_span(frames: np.ndarray) -> np.ndarray:
    """Drops leading and trailing frames that contain no hand.

    Source clips include dead air before the sign starts and after it ends.
    Resampling that whole span means roughly a third of the 30 output frames
    carry nothing but -1.0 padding, and the sign itself lands at a different
    offset in every sample — the model spends capacity learning to ignore
    padding instead of learning the gesture. Gaps *inside* the sign (where
    tracking dropped mid-motion) are left alone: those are real signal about
    the movement, and live inference produces them too.
    """
    present = ~np.all(
        np.isclose(frames[:, 1:1 + COORDS_PER_HAND], MISSING), axis=1
    )
    if not present.any():
        return frames

    first = int(np.argmax(present))
    last = len(present) - 1 - int(np.argmax(present[::-1]))
    return frames[first:last + 1]


# ============================================================
# PARQUET SOURCE (MediaPipe landmark tables)
# ============================================================

def find_index_csv(root: Path) -> Path:
    """Locates the index CSV mapping each parquet file to its sign."""
    candidates = sorted(root.glob("*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No index CSV found in {root}. Expected a CSV with 'path' and 'sign' columns."
        )
    # Prefer a file named like train.csv when several are present.
    for candidate in candidates:
        if "train" in candidate.stem.lower():
            return candidate
    return candidates[0]


def load_parquet_index(root: Path):
    import pandas as pd

    index_path = find_index_csv(root)
    index = pd.read_csv(index_path)

    missing = {"path", "sign"} - set(index.columns)
    if missing:
        raise ValueError(
            f"{index_path.name} is missing column(s) {sorted(missing)}. "
            f"Found: {list(index.columns)}."
        )

    print(f"Index: {index_path.name}  ({len(index)} sequences, {index['sign'].nunique()} signs)")
    return index


def sequence_from_parquet(parquet_path: Path) -> np.ndarray | None:
    """Reads one landmark parquet file into an (n, 127) sequence.

    Keeps only the two hand landmark groups; face and pose rows are dropped
    because the model's 127-feature input is hands-only.
    """
    import pandas as pd

    table = pd.read_parquet(parquet_path)

    required = {"frame", "type", "landmark_index", "x", "y", "z"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(
            f"{parquet_path.name} is missing column(s) {sorted(missing)}. "
            f"Found: {list(table.columns)}."
        )

    hands = table[table["type"].isin(["left_hand", "right_hand"])]
    if hands.empty:
        return None

    frames: list[np.ndarray] = []
    for _, group in hands.groupby("frame", sort=True):
        per_hand: list[np.ndarray] = []
        for _, hand in group.groupby("type", sort=True):
            hand = hand.sort_values("landmark_index")
            if len(hand) != LANDMARKS_PER_HAND:
                continue
            coords = hand[["x", "y", "z"]].to_numpy(dtype=np.float32).reshape(-1)
            if np.all(np.isnan(coords)):
                continue
            per_hand.append(coords)
        frames.append(assemble_frame(per_hand))

    if not frames:
        return None
    return np.stack(frames).astype(np.float32)


def iter_parquet_sequences(root: Path, index, sign: str, limit: int, trim: bool = True):
    # Match case-insensitively: the label list is lowercased, the CSV may not be.
    rows = index[index["sign"].astype(str).str.lower() == sign.lower()]
    produced = 0

    for _, row in rows.iterrows():
        if produced >= limit:
            return

        parquet_path = root / str(row["path"])
        if not parquet_path.exists():
            continue

        try:
            raw = sequence_from_parquet(parquet_path)
        except Exception as e:  # a single corrupt file shouldn't kill the import
            print(f"    ! skipped {parquet_path.name}: {e}")
            continue

        if raw is None or raw.shape[0] < 2:
            continue

        if trim:
            raw = trim_to_active_span(raw)
            if raw.shape[0] < 2:
                continue

        sequence = resample_to_length(raw)
        if not has_any_hand(sequence):
            continue

        produced += 1
        yield sequence


# ============================================================
# VIDEO SOURCE (run MediaPipe locally)
# ============================================================

def iter_video_sequences(class_dir: Path, limit: int, trim: bool = True):
    import cv2
    import mediapipe as mp

    clips = sorted(p for p in class_dir.iterdir() if p.suffix.lower() in VIDEO_SUFFIXES)
    produced = 0

    with mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as detector:
        for clip in clips:
            if produced >= limit:
                return

            capture = cv2.VideoCapture(str(clip))
            frames: list[np.ndarray] = []

            while True:
                ok, image = capture.read()
                if not ok:
                    break
                results = detector.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                hands = [
                    np.asarray(
                        [v for lm in hand.landmark for v in (lm.x, lm.y, lm.z)],
                        dtype=np.float32,
                    )
                    for hand in (results.multi_hand_landmarks or [])
                ]
                frames.append(assemble_frame(hands))

            capture.release()

            if len(frames) < 2:
                print(f"    ! skipped {clip.name}: no usable frames")
                continue

            stacked = np.stack(frames).astype(np.float32)
            if trim:
                stacked = trim_to_active_span(stacked)
                if stacked.shape[0] < 2:
                    print(f"    ! skipped {clip.name}: no hand detected")
                    continue
            sequence = resample_to_length(stacked)
            if not has_any_hand(sequence):
                print(f"    ! skipped {clip.name}: no hand detected")
                continue

            produced += 1
            yield sequence

def write_sequences(out_dir: Path, sequences, label: str, clear: bool, dry_run: bool) -> int:
    label_dir = out_dir / label

    if clear and label_dir.exists() and not dry_run:
        shutil.rmtree(label_dir)

    existing = 0
    if not dry_run:
        label_dir.mkdir(parents=True, exist_ok=True)
        existing = len(list(label_dir.glob("sequence_*.npy")))

    written = 0
    for sequence in sequences:
        if sequence.shape != (FRAMES_PER_SEQUENCE, FEATURES_PER_FRAME):
            raise ValueError(
                f"Refusing to write {label}: expected "
                f"({FRAMES_PER_SEQUENCE}, {FEATURES_PER_FRAME}), got {sequence.shape}."
            )
        written += 1
        if not dry_run:
            path = label_dir / f"sequence_{existing + written:03d}.npy"
            np.save(path, sequence.astype(np.float32))

    return written


def read_word_list(args) -> list[str] | None:
    if args.words:
        return [w.strip().lower() for w in args.words.split(",") if w.strip()]
    if args.words_file:
        text = Path(args.words_file).read_text(encoding="utf-8")
        words = []
        for line in text.splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                words.append(line.lower())
        return words
    return None

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Import a public sign-language dataset into dataset/words/<label>/sequence_*.npy"
    )
    parser.add_argument("--mode", choices=["parquet", "video"], required=True)
    parser.add_argument("--input", required=True, help="Dataset root directory.")
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "dataset" / "words"),
        help="Where to write sequences (default: dataset/words).",
    )
    parser.add_argument("--words", help="Comma-separated labels to import.")
    parser.add_argument("--words-file", help="File with one label per line ('#' comments allowed).")
    parser.add_argument("--max-per-word", type=int, default=400, help="Cap per label (default 400).")
    parser.add_argument("--min-per-word", type=int, default=20,
                        help="Warn when a label yields fewer than this many sequences (default 20).")
    parser.add_argument("--list-signs", action="store_true",
                        help="Print the available labels with their sample counts, then exit.")
    parser.add_argument("--clear", action="store_true",
                        help="Delete each label's existing sequences before importing.")
    parser.add_argument("--no-trim", action="store_true",
                        help="Keep leading/trailing hand-less frames instead of trimming to the "
                             "active span before resampling.")
    parser.add_argument("--dry-run", action="store_true", help="Report what would be written.")
    args = parser.parse_args(argv)

    root = Path(args.input).expanduser()
    if not root.is_dir():
        print(f"Input directory not found: {root}")
        return 1

    out_dir = Path(args.output).expanduser()
    wanted = read_word_list(args)

    if args.mode == "parquet":
        index = load_parquet_index(root)
        class_dir_for = {}
        available = sorted(index["sign"].astype(str).str.lower().unique())
        counts = index["sign"].astype(str).str.lower().value_counts().to_dict()
    else:
        index = None
        class_dirs = sorted(p for p in root.iterdir() if p.is_dir())
        class_dir_for = {p.name.lower(): p for p in class_dirs}
        available = [p.name.lower() for p in class_dirs]
        counts = {
            p.name.lower(): sum(1 for f in p.iterdir() if f.suffix.lower() in VIDEO_SUFFIXES)
            for p in class_dirs
        }

    if args.list_signs:
        print(f"\n{len(available)} labels available:\n")
        for name in available:
            print(f"  {name:<24} {counts.get(name, 0)} samples")
        return 0

    if wanted is None:
        print("Pass --words or --words-file to choose which labels to import "
              "(use --list-signs to see what is available).")
        return 1

    unknown = [w for w in wanted if w not in available]
    if unknown:
        print(f"\nNot in this dataset ({len(unknown)}): {', '.join(unknown)}")
        print("Continuing with the rest.\n")

    selected = [w for w in wanted if w in available]
    if not selected:
        print("None of the requested labels exist in this dataset.")
        return 1

    print(f"Importing {len(selected)} labels into {out_dir}"
          f"{'  (dry run)' if args.dry_run else ''}\n")

    total = 0
    thin: list[tuple[str, int]] = []

    for label in selected:
        print(f"  {label}")

        if args.mode == "parquet":
            sequences = iter_parquet_sequences(
                root, index, label, args.max_per_word, trim=not args.no_trim
            )
        else:
            sequences = iter_video_sequences(
                class_dir_for[label], args.max_per_word, trim=not args.no_trim
            )

        written = write_sequences(out_dir, sequences, label, args.clear, args.dry_run)
        total += written
        print(f"    {written} sequences")

        if written < args.min_per_word:
            thin.append((label, written))

    print(f"\nDone: {total} sequences across {len(selected)} labels.")

    if thin:
        print("\nToo few samples to train reliably:")
        for label, count in thin:
            print(f"  {label}: {count}")
        print("Drop these labels or record more with scripts/collect_sequences.py.")

    if not args.dry_run:
        print("\nNext:")
        print("  MODEL_MODE=words python ml/preprocess.py")
        print("  MODEL_MODE=words python ml/train.py")

    return 0


def _in_notebook() -> bool:
    """True inside Jupyter/Kaggle, where sys.argv belongs to the kernel."""
    return "ipykernel" in sys.modules


if __name__ == "__main__" and not _in_notebook():
    sys.exit(main())
