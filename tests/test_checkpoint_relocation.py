"""Relocation may change paths, but must reject a different execution profile."""
import copy
import hashlib
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "relocate", Path(__file__).resolve().parents[1] / "scripts/relocate_t800_checkpoint.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture_data(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/strict-source-lock.json").write_bytes(b"locked source")
    profile = {"fk_compile_compat": {"source_file": "/new/source.py", "source_sha256": "abc"},
               "head_control": {"profile": "fixed"}}
    execution = copy.deepcopy(profile)
    execution["fk_compile_compat"]["source_file"] = "/old/source.py"
    execution["source_root"] = "/old/src"
    execution["source_lock_sha256"] = hashlib.sha256(b"locked source").hexdigest()
    return {"execution": execution, "policy": {"weight": [1, 2]}}, profile


def test_only_paths_change(tmp_path):
    checkpoint, profile = fixture_data(tmp_path)
    expected = copy.deepcopy(checkpoint)
    expected["execution"]["source_root"] = str(tmp_path / "src")
    expected["execution"]["fk_compile_compat"]["source_file"] = "/new/source.py"
    module.relocate_metadata(checkpoint, profile, tmp_path)
    assert checkpoint == expected


@pytest.mark.parametrize("field", ["source_sha256", "head_control", "source_lock"])
def test_different_profile_is_rejected_before_mutation(tmp_path, field):
    checkpoint, profile = fixture_data(tmp_path)
    if field == "source_sha256":
        checkpoint["execution"]["fk_compile_compat"][field] = "wrong"
    elif field == "head_control":
        checkpoint["execution"][field] = {"profile": "wrong"}
    else:
        checkpoint["execution"]["source_lock_sha256"] = "wrong"
    before = copy.deepcopy(checkpoint)
    with pytest.raises(ValueError):
        module.relocate_metadata(checkpoint, profile, tmp_path)
    assert checkpoint == before
