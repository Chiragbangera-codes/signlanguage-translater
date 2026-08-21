from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

COORDS_PER_HAND = 63
SLOT_A = slice(1, 1 + COORDS_PER_HAND)
SLOT_B = slice(1 + COORDS_PER_HAND, 1 + COORDS_PER_HAND * 2)
MISSING = -1.0


def hand_present(hand: np.ndarray) -> bool:
    return not np.all(np.isclose(hand, MISSING))


def mean_x(hand: np.ndarray) -> float:
    """Mean of the x coordinates — every third value starting at index 0."""
    return float(np.mean(hand[0::3]))


def migrate_sequence(sequence: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Returns (migrated, frames_swapped, flags_repaired)."""
    out = sequence.copy()
    swapped = 0
    repaired = 0

    for i in range(out.shape[0]):
        slot_a = out[i, SLOT_A]
        slot_b = out[i, SLOT_B]
        a_present = hand_present(slot_a)
        b_present = hand_present(slot_b)

        # A lone hand belongs in slot A under both conventions; if it somehow
        # sits in slot B, move it.
        if b_present and not a_present:
            out[i, SLOT_A] = slot_b.copy()
            out[i, SLOT_B] = MISSING
            swapped += 1
            a_present, b_present = True, False

        if a_present and b_present:
            if mean_x(out[i, SLOT_A]) > mean_x(out[i, SLOT_B]):
                a_copy = out[i, SLOT_A].copy()
                out[i, SLOT_A] = out[i, SLOT_B]
                out[i, SLOT_B] = a_copy
                swapped += 1

        # Keep the uses_two_hands flag honest about what the slots contain.
        expected_flag = 1.0 if (a_present and b_present) else 0.0
        if out[i, 0] != expected_flag:
            out[i, 0] = expected_flag
            repaired += 1

    return out, swapped, repaired


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Re-sort hand slots to the geometric convention."
    )
    parser.add_argument("--dataset", default=str(PROJECT_ROOT / "dataset" / "words"))
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would change without writing.")
    args = parser.parse_args()

    root = Path(args.dataset)
    if not root.is_dir():
        print(f"Dataset directory not found: {root}")
        return 1

    labels = sorted(p for p in root.iterdir() if p.is_dir())
    if not labels:
        print(f"No label folders in {root}")
        return 1

    print(f"{'label':<14}{'files':>7}{'frames':>9}{'2-hand':>9}{'swapped':>9}{'flags':>8}")
    print("-" * 56)

    total_files = total_frames = total_two = total_swapped = total_flags = 0

    for label_dir in labels:
        files = sorted(label_dir.glob("sequence_*.npy"))
        l_frames = l_two = l_swapped = l_flags = 0

        for path in files:
            sequence = np.load(path).astype(np.float32)
            migrated, swapped, repaired = migrate_sequence(sequence)

            two = int(np.sum([
                hand_present(sequence[i, SLOT_A]) and hand_present(sequence[i, SLOT_B])
                for i in range(sequence.shape[0])
            ]))

            l_frames += sequence.shape[0]
            l_two += two
            l_swapped += swapped
            l_flags += repaired

            if not args.dry_run and (swapped or repaired):
                np.save(path, migrated)

        print(f"{label_dir.name:<14}{len(files):>7}{l_frames:>9}{l_two:>9}"
              f"{l_swapped:>9}{l_flags:>8}")

        total_files += len(files)
        total_frames += l_frames
        total_two += l_two
        total_swapped += l_swapped
        total_flags += l_flags

    print("-" * 56)
    print(f"{'TOTAL':<14}{total_files:>7}{total_frames:>9}{total_two:>9}"
          f"{total_swapped:>9}{total_flags:>8}")

    two_pct = total_two / total_frames if total_frames else 0
    print(f"\nTwo-handed frames: {two_pct:.1%}")

    if args.dry_run:
        print("\nDry run — nothing written. Re-run without --dry-run to apply.")
    else:
        print(f"\nRewrote {total_swapped} slot assignments and "
              f"{total_flags} uses_two_hands flags.")
        print("Re-run preprocessing and training so the model learns the new layout.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
