from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_FILES = {
    "words": "sign_speak_words_lstm.h5",
    "numbers": "sign_speak_lstm.h5",
}

SHARD_NAME = "group1-shard1of1.bin"


def _decode(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _attr_json(group, key):
    """Reads a JSON-encoded HDF5 attribute."""
    raw = group.attrs.get(key)
    if raw is None:
        return None
    return json.loads(_decode(raw))


def collect_weights(h5file) -> list[tuple[str, np.ndarray]]:
    weights_group = h5file["model_weights"]
    entries: list[tuple[str, np.ndarray]] = []

    for layer_name in sorted(weights_group.keys()):
        layer = weights_group[layer_name]
        weight_names = layer.attrs.get("weight_names")
        if weight_names is None or len(weight_names) == 0:
            continue

        for raw_name in weight_names:
            name = _decode(raw_name)
            dataset = layer[name]
            tfjs_name = name[:-2] if name.endswith(":0") else name
            entries.append((tfjs_name, np.asarray(dataset, dtype=np.float32)))

    return entries


def convert(model_path: Path, out_dir: Path) -> dict:
    with h5py.File(model_path, "r") as h5file:
        model_config = _attr_json(h5file, "model_config")
        if model_config is None:
            raise SystemExit(
                f"{model_path.name} has no 'model_config' attribute — it may be a "
                "weights-only checkpoint rather than a full model."
            )

        topology = {
            "keras_version": _decode(h5file.attrs.get("keras_version", "2.15.0")),
            "backend": _decode(h5file.attrs.get("backend", "tensorflow")),
            "model_config": model_config,
        }
        training_config = _attr_json(h5file, "training_config")
        if training_config is not None:
            topology["training_config"] = training_config

        entries = collect_weights(h5file)

    if not entries:
        raise SystemExit(f"No weights found in {model_path.name}.")

    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_weights = []
    with open(out_dir / SHARD_NAME, "wb") as shard:
        for name, array in entries:
            shard.write(np.ascontiguousarray(array, dtype=np.float32).tobytes())
            manifest_weights.append({
                "name": name,
                "shape": list(array.shape),
                "dtype": "float32",
            })

    model_json = {
        "format": "layers-model",
        "generatedBy": f"keras v{topology['keras_version']}",
        "convertedBy": "scripts/keras_to_tfjs.py",
        "modelTopology": topology,
        "weightsManifest": [{"paths": [SHARD_NAME], "weights": manifest_weights}],
    }
    (out_dir / "model.json").write_text(json.dumps(model_json), encoding="utf-8")

    return model_json


def verify(out_dir: Path, reference: Path) -> bool:
    ok = True

    produced = (out_dir / SHARD_NAME).read_bytes()
    expected = (reference / SHARD_NAME).read_bytes()

    if produced == expected:
        print(f"  shard bytes      : IDENTICAL ({len(produced):,} bytes)")
    else:
        ok = False
        print(f"  shard bytes      : DIFFER ({len(produced):,} vs {len(expected):,})")
        if len(produced) == len(expected):
            diff = np.frombuffer(produced, np.float32) - np.frombuffer(expected, np.float32)
            print(f"    max abs difference: {np.abs(diff).max()}")

    a = json.loads((out_dir / "model.json").read_text(encoding="utf-8"))
    b = json.loads((reference / "model.json").read_text(encoding="utf-8"))

    for key in ("format", "modelTopology"):
        same = a.get(key) == b.get(key)
        print(f"  {key:<17}: {'match' if same else 'DIFFER'}")
        ok = ok and same

    wa = a["weightsManifest"][0]["weights"]
    wb = b["weightsManifest"][0]["weights"]
    same = wa == wb
    print(f"  weight manifest  : {'match' if same else 'DIFFER'} "
          f"({len(wa)} vs {len(wb)} entries)")
    if not same:
        for x, y in zip(wa, wb):
            if x != y:
                print(f"    {x} != {y}")
    ok = ok and same

    return ok


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert a Keras .h5 to TensorFlow.js layers format."
    )
    parser.add_argument("--mode", choices=sorted(MODEL_FILES),
                        help="Shorthand for the project's own models.")
    parser.add_argument("--model", help="Path to a .h5 model (overrides --mode).")
    parser.add_argument("--output", help="Output directory.")
    parser.add_argument("--verify-against", help="Compare output to a known-good export directory.")
    args = parser.parse_args()

    if args.model:
        model_path = Path(args.model)
    elif args.mode:
        model_path = PROJECT_ROOT / "ml" / "model" / MODEL_FILES[args.mode]
    else:
        parser.error("pass --mode or --model")

    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return 1

    if args.output:
        out_dir = Path(args.output)
    elif args.mode:
        out_dir = PROJECT_ROOT / "frontend" / "public" / "model" / args.mode
    else:
        parser.error("pass --output when using --model")

    print(f"Model  : {model_path}")
    print(f"Output : {out_dir}\n")

    model_json = convert(model_path, out_dir)

    weights = model_json["weightsManifest"][0]["weights"]
    total_params = sum(int(np.prod(w["shape"])) for w in weights)
    shard_bytes = (out_dir / SHARD_NAME).stat().st_size

    for w in weights:
        print(f"  {w['name']:<45} {str(w['shape'])}")
    print(f"\n  {len(weights)} tensors, {total_params:,} parameters, {shard_bytes:,} bytes")

    if total_params * 4 != shard_bytes:
        print(f"  WARNING: {total_params} float32 values should be "
              f"{total_params * 4} bytes, got {shard_bytes}")
        return 1

    if args.verify_against:
        print(f"\nVerifying against {args.verify_against}:")
        if not verify(out_dir, Path(args.verify_against)):
            print("\nVERIFICATION FAILED — do not ship this output.")
            return 1
        print("\nVerification passed: output is byte-identical to the reference.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
