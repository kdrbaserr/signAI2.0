"""Time-axis semantics, repeatable plots and honest local-waveform errors."""

import json
from pathlib import Path
import shutil

import numpy as np
import pytest

pytest.importorskip("matplotlib", reason="Install the optional inspect extra for plotting tests")

from signai.inspect_ecg import InspectionError, check_time_axis, inspect_record, run_demo
from signai.ptbxl import sha256, write_json
from signai.splits import build_manifests

ROOT = Path(__file__).resolve().parents[1]


def test_time_axis_has_100_samples_per_second_and_ten_second_coverage():
    report = check_time_axis(np.arange(1000) / 100, 1000, 100, 10)
    assert report["sample_interval_seconds"] == 0.01
    assert report["last_sample_seconds"] == 9.99
    assert report["sample_coverage_seconds"] == 10


@pytest.mark.parametrize("axis", [np.arange(1000), np.arange(1000) / 500, np.linspace(0, 10, 1000), np.arange(1000) / 100 + 1])
def test_wrong_scale_index_axis_and_inclusive_endpoint_are_rejected(axis):
    with pytest.raises(InspectionError) as error:
        check_time_axis(axis, 1000, 100, 10)
    assert error.value.code == "E_TIME_SCALE"


@pytest.fixture
def local_root(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl-cache"))
    schema = tmp_path / "docs/signal_schema.json"
    schema.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / "docs/signal_schema.json", schema)
    raw = tmp_path / "data/raw/ptb-xl/1.0.3/records100/00000"
    raw.mkdir(parents=True)
    # Synthetic identity fixtures; source decoding is separately tested in Day 4.
    (raw / "00001_lr.hea").write_bytes(b"synthetic header")
    (raw / "00001_lr.dat").write_bytes(b"synthetic signal source")
    hashes = {"header_sha256": sha256(raw / "00001_lr.hea"), "data_sha256": sha256(raw / "00001_lr.dat")}
    signal_dir = tmp_path / "data/validated/ptb-xl/1.0.3"
    signal_dir.mkdir(parents=True)
    npz = signal_dir / "ptbxl_1_0_3_00001.npz"
    shutil.copyfile(ROOT / "tests/fixtures/signals/valid.npz", npz)
    metadata = json.loads((ROOT / "tests/fixtures/signals/valid.json").read_text(encoding="utf-8"))
    metadata.update(record_id="ptbxl_1_0_3_00001", source={"ecg_id": 1, **hashes})
    write_json(signal_dir / "ptbxl_1_0_3_00001.json", metadata)
    records = [{"ecg_id": index, "patient_id": index, "strat_fold": 1, "split": "train",
                "filename_lr": f"records100/00000/{index:05}_lr", "label_vector": [1, 0, 0, 0, 0], "excluded_reason": None} for index in (1, 2)]
    manifests = build_manifests(records, {"metadata_sha256": "synthetic", "label_schema_sha256": "synthetic"})
    for split, manifest in manifests.items():
        write_json(tmp_path / "splits" / f"{split}.json", manifest)
    write_json(tmp_path / "splits/summary.json", {"manifest_sha256": {split: sha256(tmp_path / "splits" / f"{split}.json") for split in manifests}})
    write_json(tmp_path / "docs/provenance.json", {"raw_directory": "data/raw/ptb-xl/1.0.3", "files": [
        {"relative_path": "records100/00000/00001_lr.hea", "sha256": hashes["header_sha256"]},
        {"relative_path": "records100/00000/00001_lr.dat", "sha256": hashes["data_sha256"]},
    ]})
    return tmp_path


def test_repeated_inspection_reproduces_png_and_metadata(local_root):
    report = run_demo(local_root, 1, local_root / "output")
    assert report["outputs_identical"]
    assert report["time_scale_experiment"]["error_code"] == "E_TIME_SCALE"
    metadata = json.loads((local_root / "output/ecg_00001.json").read_text(encoding="utf-8"))
    assert metadata["signal"]["shape"] == [1000, 12]
    assert metadata["input_files_unchanged"]


def test_metadata_presence_does_not_imply_local_signal(local_root):
    with pytest.raises(InspectionError) as error:
        inspect_record(local_root, 2, local_root / "output")
    assert error.value.code == "E_WAVEFORM_NOT_LOCAL"


def test_wrong_record_binding_is_rejected(local_root):
    path = local_root / "data/validated/ptb-xl/1.0.3/ptbxl_1_0_3_00001.json"
    metadata = json.loads(path.read_text())
    metadata["source"]["ecg_id"] = 99
    write_json(path, metadata)
    with pytest.raises(InspectionError) as error:
        inspect_record(local_root, 1, local_root / "output")
    assert error.value.code == "E_SOURCE_ID"


def test_changed_raw_waveform_is_rejected(local_root):
    (local_root / "data/raw/ptb-xl/1.0.3/records100/00000/00001_lr.dat").write_bytes(b"changed")
    with pytest.raises(InspectionError) as error:
        inspect_record(local_root, 1, local_root / "output")
    assert error.value.code == "E_SOURCE_HASH"
