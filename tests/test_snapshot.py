"""A restore must fail on corruption, missing files and unexpected inventory."""

import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("snapshot", Path(__file__).parents[1] / "scripts/ptbxl_snapshot.py")
snapshot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(snapshot)


@pytest.fixture
def root(tmp_path):
    raw = tmp_path / snapshot.RAW
    raw.mkdir(parents=True)
    signal = raw / "signal.dat"
    signal.write_bytes(b"example ECG bytes")
    (tmp_path / "docs").mkdir()
    provenance = {"dataset_version": "1.0.3", "raw_directory": snapshot.RAW.as_posix(),
                  "files": [{"relative_path": "signal.dat", "bytes": signal.stat().st_size,
                             "sha256": snapshot.digest(signal)}]}
    (tmp_path / "docs/provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    manifest = tmp_path / snapshot.MANIFEST
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps(snapshot.inspect(tmp_path)), encoding="utf-8")
    return tmp_path


def test_verified_restore(root):
    assert snapshot.verify(root)["files"] == 1


@pytest.mark.parametrize("change", ["corrupt", "missing", "extra", "manifest"])
def test_reject_changed_restore(root, change):
    signal = root / snapshot.RAW / "signal.dat"
    if change == "corrupt":
        signal.write_bytes(b"changed ECG bytes")
    elif change == "missing":
        signal.unlink()
    elif change == "extra":
        (signal.parent / "unexpected.dat").write_bytes(b"extra")
    else:
        path = root / snapshot.MANIFEST
        value = json.loads(path.read_text(encoding="utf-8"))
        value["files"][0]["sha256"] = "0" * 64
        path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError):
        snapshot.verify(root)
