"""Copy an audited T800 checkpoint to a new checkout without changing tensors.

Only FK source_file and execution.source_root are relocated. All other profile
fields and the strict source lock must match before the copy is accepted.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def relocate_metadata(checkpoint, profile, root):
    """Validate first, then replace the two checkout-location fields in memory."""
    execution = checkpoint["execution"]
    for key, expected in profile.items():
        actual = execution.get(key)
        if key == "fk_compile_compat":
            actual = {k: v for k, v in actual.items() if k != "source_file"}
            expected = {k: v for k, v in expected.items() if k != "source_file"}
        if actual != expected:
            raise ValueError(f"Checkpoint profile differs: {key}")
    source_lock = hashlib.sha256((root / "configs/strict-source-lock.json").read_bytes()).hexdigest()
    if execution.get("source_lock_sha256") != source_lock:
        raise ValueError("Checkpoint strict source lock differs")
    execution["fk_compile_compat"]["source_file"] = profile["fk_compile_compat"]["source_file"]
    execution["source_root"] = str(root / "src")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=("anchor-head", "baseline"), default="anchor-head")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Preserving existing file: {args.output}")
    from scripts.train_strict import select_source, verify_sources
    verify_sources()
    select_source("t800")
    if args.profile == "anchor-head":
        from scripts.t800_anchor_head_profile import install, validate_checkpoint
    else:
        from scripts.t800_fk_profile import install, validate_checkpoint
    profile = install()
    import torch
    from scripts.train_t800_anchor_head import tensor_digest
    with torch.serialization.safe_globals([Path, type(Path("."))]):
        checkpoint = torch.load(args.input, map_location="cpu", weights_only=True)
    state_keys = ("wm", "policy", "critic", "wm_opt", "policy_opt", "critic_opt")
    before = {k: tensor_digest(checkpoint[k]) for k in state_keys}
    old_execution = json.loads(json.dumps(checkpoint["execution"]))
    relocate_metadata(checkpoint, profile, ROOT)
    validate_checkpoint(checkpoint, profile)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidentally replacing a checkpoint.
    with args.output.open("xb") as stream:
        torch.save(checkpoint, stream)
    with torch.serialization.safe_globals([Path, type(Path("."))]):
        restored = torch.load(args.output, map_location="cpu", weights_only=True)
    after = {k: tensor_digest(restored[k]) for k in state_keys}
    if after != before:
        raise RuntimeError("Saved model or optimizer tensors changed")
    validate_checkpoint(restored, profile)
    report = {
        "input_sha256": hashlib.file_digest(args.input.open("rb"), "sha256").hexdigest(),
        "output_sha256": hashlib.file_digest(args.output.open("rb"), "sha256").hexdigest(),
        "old_source_file": old_execution["fk_compile_compat"]["source_file"],
        "new_source_file": profile["fk_compile_compat"]["source_file"],
        "model_optimizer_digests": after,
        "model_optimizer_unchanged": before == after,
        "updates": checkpoint["num_updates"],
    }
    args.output.with_suffix(".relocation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
