"""Verify the numerical arrays used by the published T800 experiment."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("motion", type=Path)
    args = parser.parse_args()
    expected = json.loads((ROOT / "docs/results/reference-array-hashes.json").read_text())
    failures = []
    with np.load(args.motion, allow_pickle=False) as data:
        for key, spec in expected["arrays"].items():
            if key not in data:
                failures.append(key + ": missing")
                continue
            array = data[key]
            actual = {"shape": list(array.shape), "dtype": str(array.dtype),
                      "sha256": hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()}
            if actual != spec:
                failures.append(key)
    if failures:
        raise SystemExit("Reference differs from published experiment: " + ", ".join(failures))
    print(f"Verified {len(expected['arrays'])} arrays: 13066 frames, 50 Hz")


if __name__ == "__main__":
    main()
