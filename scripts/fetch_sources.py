"""Restore pinned source snapshots and verify their hashes (standard library only)."""
import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    sources = json.loads((ROOT / "configs/sources.lock.json").read_text())
    for name, source in sources.items():
        destination = ROOT / source["path"]
        if not destination.exists():
            if args.verify_only or "archive" not in source:
                raise FileNotFoundError(destination)
            archive = ROOT / source["archive"]
            archive.parent.mkdir(parents=True, exist_ok=True)
            if not archive.exists():
                repository = source["repository"].removeprefix("https://github.com/")
                url = f"https://codeload.github.com/{repository}/tar.gz/{source['commit']}"
                temporary = archive.with_suffix(".download")
                with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as output:
                    shutil.copyfileobj(response, output)
                if digest(temporary) != source["archive_sha256"]:
                    raise ValueError(f"Downloaded archive hash mismatch: {name}")
                temporary.replace(archive)
            if digest(archive) != source["archive_sha256"]:
                raise ValueError(f"Archive hash mismatch: {name}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
                with tarfile.open(archive) as package:
                    package.extractall(temporary, filter="data")
                contents = list(Path(temporary).iterdir())
                if len(contents) != 1 or not contents[0].is_dir():
                    raise ValueError(f"Unexpected archive layout: {name}")
                shutil.move(str(contents[0]), destination)
            if name == "mjlab":
                project = destination / "pyproject.toml"
                original = project.read_text()
                original = original.replace('  "torch>=2.14.0",', '  "torch>=2.11.0",', 1)
                project.write_text(original)
        for relative, expected in source["files_sha256"].items():
            path = destination / relative
            if not path.is_file() or digest(path) != expected:
                raise ValueError(f"Source changed or missing: {path}; existing files are not overwritten")
        print(f"Verified {name}: {source['commit']}")


if __name__ == "__main__":
    main()
