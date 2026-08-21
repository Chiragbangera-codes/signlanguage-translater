from __future__ import annotations

import argparse
import json
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_FILES = {
    "words": ("sign_speak_words_lstm.h5", "label_encoder_words.pkl"),
    "numbers": ("sign_speak_lstm.h5", "label_encoder.pkl"),
}


def export_labels(encoder_path: Path, out_dir: Path) -> list[str]:
    """Writes the label-encoder classes in index order.

    Order is load-bearing: the model emits a probability per class index, and
    the browser maps index -> word through this array. A reordering here
    silently relabels every prediction.
    """
    with open(encoder_path, "rb") as f:
        encoder = pickle.load(f)

    labels = [str(name) for name in encoder.classes_]
    (out_dir / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
    return labels


def convert(model_path: Path, out_dir: Path) -> None:
    try:
        import tensorflowjs
    except ImportError:
        raise SystemExit(
            "tensorflowjs is not installed in this environment.\n\n"
            "On Windows it cannot be installed at all: a transitive dependency\n"
            "needs uvloop, which is Unix-only. Convert on Linux (Kaggle, Colab\n"
            "or WSL) and run this with --labels-only here. See the module\n"
            "docstring for the exact steps."
        )

    command = [
        sys.executable, "-m", "tensorflowjs.converters.converter",
        "--input_format=keras",
        str(model_path),
        str(out_dir),
    ]
    print("Running:", " ".join(command))
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise SystemExit(
            f"tensorflowjs_converter failed (exit {result.returncode}). "
            "If it reports an unsupported layer, that layer has no TF.js "
            "equivalent and the architecture needs adjusting."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a Keras model to TensorFlow.js.")
    parser.add_argument("--mode", choices=sorted(MODEL_FILES), default="words")
    parser.add_argument("--labels-only", action="store_true",
                        help="Write labels.json only. Requires no tensorflowjs, so it works "
                             "on Windows; pair with converting the model on Linux.")
    parser.add_argument("--model-dir", default=str(PROJECT_ROOT / "ml" / "model"))
    parser.add_argument(
        "--output",
        default=None,
        help="Defaults to frontend/public/model/<mode>.",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    model_name, encoder_name = MODEL_FILES[args.mode]
    model_path = model_dir / model_name
    encoder_path = model_dir / encoder_name

    required = [(encoder_path, "label encoder")]
    if not args.labels_only:
        required.append((model_path, "model"))
    for path, what in required:
        if not path.exists():
            print(f"Missing {what}: {path}")
            return 1

    out_dir = Path(args.output) if args.output else (
        PROJECT_ROOT / "frontend" / "public" / "model" / args.mode
    )

    if args.labels_only:
        out_dir.mkdir(parents=True, exist_ok=True)
        labels = export_labels(encoder_path, out_dir)
        print(f"Wrote {out_dir / 'labels.json'} - {len(labels)} classes")
        print(f"  {', '.join(labels)}")
        print("")
        print("Now convert the model on Linux (see the module docstring) and drop")
        print(f"model.json plus the .bin shards next to it in {out_dir}")
        return 0

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model  : {model_path}")
    print(f"Output : {out_dir}\n")

    convert(model_path, out_dir)
    labels = export_labels(encoder_path, out_dir)

    total = sum(f.stat().st_size for f in out_dir.iterdir() if f.is_file())
    print(f"\nExported {len(labels)} classes: {', '.join(labels[:8])}"
          f"{', ...' if len(labels) > 8 else ''}")
    print(f"Total size: {total / 1_000_000:.1f} MB")
    for f in sorted(out_dir.iterdir()):
        print(f"  {f.name:<28} {f.stat().st_size / 1000:>8.0f} KB")

    print(f"\nThe browser will load this from /model/{args.mode}/model.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
