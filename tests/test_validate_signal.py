"""Signal semantics, malformed archives and read-only validation."""

import io
import json
from pathlib import Path
import runpy
import subprocess
import sys
import zipfile

import numpy as np
import pytest

from signai.ptbxl import sha256
from signai.validate_signal import SignalValidationError, validate_signal

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "docs" / "signal_schema.json"
FIXTURES = ROOT / "tests" / "fixtures" / "signals"
CASES = json.loads((FIXTURES / "index.json").read_text(encoding="utf-8"))["cases"]


def pair(tmp_path, signal=None, **changes):
    signal = np.zeros((1000, 12), dtype=np.float32) if signal is None else signal
    npz = tmp_path / "signal.npz"
    metadata_path = tmp_path / "signal.json"
    np.savez_compressed(npz, signal=signal)
    metadata = json.loads((FIXTURES / "valid.json").read_text(encoding="utf-8"))
    metadata["npz_sha256"] = sha256(npz)
    metadata.update(changes)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    return npz, metadata_path


@pytest.mark.parametrize("case", CASES, ids=[case["name"] for case in CASES])
def test_static_fixtures_have_expected_codes_and_remain_unchanged(case):
    npz = FIXTURES / (case["name"] + ".npz")
    metadata = FIXTURES / (case["name"] + ".json")
    before = sha256(npz), sha256(metadata)
    if case["expected_error_code"] is None:
        result = validate_signal(npz, metadata, SCHEMA)
        assert result["valid"] and result["shape"] == [1000, 12]
    else:
        with pytest.raises(SignalValidationError) as error:
            validate_signal(npz, metadata, SCHEMA)
        assert error.value.code == case["expected_error_code"]
    assert before == (sha256(npz), sha256(metadata))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_float_dtypes_and_amplitude_boundary_are_accepted(tmp_path, dtype):
    npz, metadata = pair(tmp_path, np.full((1000, 12), 20, dtype=dtype))
    result = validate_signal(npz, metadata, SCHEMA)
    assert result["dtype"] == np.dtype(dtype).name
    assert result["duration_seconds"] == 10


@pytest.mark.parametrize("dtype", [np.int16, np.complex64, object])
def test_integer_complex_and_pickle_object_arrays_are_rejected(tmp_path, dtype):
    npz, metadata = pair(tmp_path, np.zeros((1000, 12), dtype=dtype))
    with pytest.raises(SignalValidationError) as error:
        validate_signal(npz, metadata, SCHEMA)
    assert error.value.code == "E_DTYPE"


@pytest.mark.parametrize("changes,code", [
    ({"axis_order": ["channel", "time"]}, "E_AXIS_ORDER"),
    ({"schema_version": "2.0.0"}, "E_SCHEMA"),
    ({"duration_seconds": 9}, "E_DURATION"),
    ({"sampling_rate_hz": True}, "E_SAMPLE_RATE"),
    ({"sampling_rate_hz": float("inf")}, "E_SAMPLE_RATE"),
    ({"sampling_rate_hz": 10**400}, "E_SAMPLE_RATE"),
    ({"channels": ["I"] * 12}, "E_CHANNEL_SET"),
    ({"npz_sha256": "bad"}, "E_CHECKSUM"),
])
def test_metadata_errors_have_specific_codes(tmp_path, changes, code):
    npz, metadata = pair(tmp_path, **changes)
    with pytest.raises(SignalValidationError) as error:
        validate_signal(npz, metadata, SCHEMA)
    assert error.value.code == code


def test_npz_changed_without_matching_json_is_rejected(tmp_path):
    npz, metadata = pair(tmp_path)
    np.savez_compressed(npz, signal=np.ones((1000, 12), dtype=np.float32))
    with pytest.raises(SignalValidationError) as error:
        validate_signal(npz, metadata, SCHEMA)
    assert error.value.code == "E_CHECKSUM"


def test_missing_metadata_field_is_rejected(tmp_path):
    npz, metadata = pair(tmp_path)
    contents = json.loads(metadata.read_text(encoding="utf-8"))
    del contents["channels"]
    metadata.write_text(json.dumps(contents), encoding="utf-8")
    with pytest.raises(SignalValidationError) as error:
        validate_signal(npz, metadata, SCHEMA)
    assert error.value.code == "E_METADATA"


def test_duplicate_json_key_is_not_silently_overwritten(tmp_path):
    npz, metadata = pair(tmp_path)
    contents = metadata.read_text(encoding="utf-8")
    metadata.write_text('{"unit": "uV",' + contents[1:], encoding="utf-8")
    with pytest.raises(SignalValidationError) as error:
        validate_signal(npz, metadata, SCHEMA)
    assert error.value.code == "E_METADATA"


def test_archive_extra_members_are_rejected(tmp_path):
    npz, metadata = pair(tmp_path)
    with zipfile.ZipFile(npz, "a") as archive:
        archive.writestr("other.npy", b"not part of contract")
    with pytest.raises(SignalValidationError) as error:
        validate_signal(npz, metadata, SCHEMA)
    assert error.value.code == "E_ARCHIVE"


def test_uncompressed_size_limit_is_checked_before_loading(tmp_path):
    npz, metadata = pair(tmp_path, np.zeros((100000, 12), dtype=np.float32))
    assert npz.stat().st_size < 1048576  # Highly compressed input still has a large payload.
    with pytest.raises(SignalValidationError) as error:
        validate_signal(npz, metadata, SCHEMA)
    assert error.value.code == "E_SIZE"


def test_untrusted_header_cannot_trigger_huge_array_allocation(tmp_path):
    npz, metadata = pair(tmp_path)
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(header, {
        "descr": "<f4", "fortran_order": False, "shape": (10**9, 12),
    })
    with zipfile.ZipFile(npz, "w") as archive:
        archive.writestr("signal.npy", header.getvalue())
    with pytest.raises(SignalValidationError) as error:
        validate_signal(npz, metadata, SCHEMA)
    assert error.value.code == "E_DURATION"


def test_header_data_length_mismatch_is_rejected(tmp_path):
    npz, metadata = pair(tmp_path)
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(header, {
        "descr": "<f4", "fortran_order": False, "shape": (1000, 12),
    })
    with zipfile.ZipFile(npz, "w") as archive:
        archive.writestr("signal.npy", header.getvalue() + b"\0" * 4)
    with pytest.raises(SignalValidationError) as error:
        validate_signal(npz, metadata, SCHEMA)
    assert error.value.code == "E_ARCHIVE"


def test_cli_returns_error_code_and_nonzero_status():
    result = subprocess.run([
        sys.executable, "-I", "-m", "signai.validate_signal",
        "--npz", str(FIXTURES / "wrong_sample_rate.npz"),
        "--metadata", str(FIXTURES / "wrong_sample_rate.json"), "--schema", str(SCHEMA),
    ], capture_output=True, text=True)
    assert result.returncode == 1
    assert json.loads(result.stdout)["error_code"] == "E_SAMPLE_RATE"


def wfdb_pair(tmp_path):
    digital = (100 + np.arange(12000) % 100).astype("<i2").reshape(1000, 12)
    data = tmp_path / "sample.dat"
    header = tmp_path / "sample.hea"
    data.write_bytes(digital.tobytes())
    lines = ["sample 12 100 1000"]
    for index, channel in enumerate(["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]):
        checksum = int(digital[:, index].astype(np.int64).sum()) % 65536
        lines.append(f"sample.dat 16 1000.0(100)/mV 16 0 {digital[0, index]} {checksum} 0 {channel}")
    header.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return header, data, digital


def test_format16_conversion_applies_gain_and_baseline_without_mutating_source(tmp_path):
    reader = runpy.run_path(str(ROOT / "scripts" / "prepare_signal_fixtures.py"))["read_verified_ptbxl_record"]
    header, data, digital = wfdb_pair(tmp_path)
    before = sha256(header), sha256(data)
    physical = reader(header, data)
    assert physical.shape == (1000, 12)
    assert physical[0, 0] == 0
    assert physical[0, 1] == 0.001
    np.testing.assert_allclose(physical, (digital.astype(np.float64) - 100) / 1000)
    assert before == (sha256(header), sha256(data))


def test_format16_corrupted_digital_samples_fail_channel_checksum(tmp_path):
    reader = runpy.run_path(str(ROOT / "scripts" / "prepare_signal_fixtures.py"))["read_verified_ptbxl_record"]
    header, data, digital = wfdb_pair(tmp_path)
    digital[30, 1] += 1
    data.write_bytes(digital.tobytes())
    with pytest.raises(ValueError, match="per-channel checksum mismatch"):
        reader(header, data)
