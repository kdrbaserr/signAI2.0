"""Build or verify a deterministic raw-data manifest without downloading anything."""

import argparse
import hashlib
import json
from pathlib import Path

RAW = Path("data/raw/ptb-xl/1.0.3")
MANIFEST = Path("data/manifests/ptbxl-1.0.3.json")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inspect(root):
    provenance = json.loads((root / "docs/provenance.json").read_text(encoding="utf-8"))
    if provenance["dataset_version"] != "1.0.3" or provenance["raw_directory"] != RAW.as_posix():
        raise ValueError("Unsupported provenance dataset version or raw directory")
    entries = []
    for item in sorted(provenance["files"], key=lambda item: item["relative_path"]):
        name = item["relative_path"]
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or ":" in name or "\\" in name:
            raise ValueError(f"Unsafe manifest path: {name}")
        path = root / RAW / relative
        if not path.is_file():
            raise ValueError(f"Missing artifact: {name}; run dvc pull before model code")
        actual = {"relative_path": name, "bytes": path.stat().st_size, "sha256": digest(path)}
        if any(actual[key] != item[key] for key in actual):
            raise ValueError(f"Raw data differs from pinned provenance: {name}")
        entries.append(actual)
    actual_names = {path.relative_to(root / RAW).as_posix() for path in (root / RAW).rglob("*") if path.is_file()}
    expected_names = {item["relative_path"] for item in entries}
    if actual_names != expected_names or len(entries) != len(expected_names):
        raise ValueError("Raw inventory differs from provenance (extra or duplicate paths)")
    return {"manifest_version": 1, "dataset": "PTB-XL", "dataset_version": "1.0.3",
            "scope": "metadata and three 100 Hz example waveforms; not the full dataset",
            "raw_directory": RAW.as_posix(), "files": entries}


def verify(root):
    expected = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    actual = inspect(root)
    if expected != actual:
        raise ValueError("Snapshot manifest differs from pinned provenance or raw data")
    return {"files": len(actual["files"]), "bytes": sum(item["bytes"] for item in actual["files"]),
            "manifest_sha256": digest(root / MANIFEST), "result": "passed"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["create", "verify"])
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == "create":
        value = inspect(root)
        path = root / MANIFEST
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(verify(root), indent=2))


if __name__ == "__main__":
    main()
