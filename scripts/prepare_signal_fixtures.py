"""Generate synthetic contract fixtures and convert the three pinned samples.

The WFDB reader intentionally supports only the verified PTB-XL format-16,
single-file, 100 Hz records. It is not a general WFDB importer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np

from signai.ptbxl import sha256, write_json
from signai.validate_signal import CHANNELS, SignalValidationError, validate_signal


def save_pair(directory: Path, name: str, signal: np.ndarray, overrides=None) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    npz_path, json_path = directory / f"{name}.npz", directory / f"{name}.json"
    np.savez_compressed(npz_path, signal=signal)
    metadata = {
        "schema_version": "1.0.0", "record_id": name, "axis_order": ["time", "channel"],
        "channels": list(CHANNELS), "unit": "mV", "sampling_rate_hz": 100,
        "duration_seconds": 10.0, "npz_sha256": sha256(npz_path),
    }
    metadata.update(overrides or {})
    write_json(json_path, metadata)
    return npz_path, json_path


def synthetic_fixtures(root: Path) -> list[dict]:
    time = np.arange(1000, dtype=np.float64) / 100
    signal = np.stack([0.1 * np.sin(2 * np.pi * (index + 1) * time / 10) for index in range(12)], axis=1).astype(np.float32)
    swapped = list(CHANNELS)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    nan = signal.copy()
    nan[20, 3] = np.nan
    infinity = signal.copy()
    infinity[20, 3] = np.inf
    amplitude = signal.copy()
    amplitude[20, 3] = 20.01
    cases = [
        ("valid", signal, {}, None),
        ("wrong_channel_order", signal[:, [1, 0, *range(2, 12)]], {"channels": swapped}, "E_CHANNEL_ORDER"),
        ("wrong_sample_rate", signal, {"sampling_rate_hz": 500}, "E_SAMPLE_RATE"),
        ("wrong_unit", signal * 1000, {"unit": "uV"}, "E_UNIT"),
        ("missing_channel", signal[:, :11], {"channels": list(CHANNELS[:11])}, "E_CHANNEL_COUNT"),
        ("nan", nan, {}, "E_NONFINITE"),
        ("infinity", infinity, {}, "E_NONFINITE"),
        ("wrong_duration", signal[:900], {}, "E_DURATION"),
        ("excessive_amplitude", amplitude, {}, "E_AMPLITUDE"),
    ]
    directory = root / "tests" / "fixtures" / "signals"
    results = []
    for name, values, metadata, expected in cases:
        npz_path, json_path = save_pair(directory, name, values, metadata)
        try:
            validate_signal(npz_path, json_path, root / "docs" / "signal_schema.json")
            observed = None
        except SignalValidationError as error:
            observed = error.code
        if observed != expected:
            raise AssertionError(f"{name}: expected {expected}, got {observed}")
        results.append({"name": name, "synthetic": True, "expected_error_code": expected})
    write_json(directory / "index.json", {"fixture_type": "synthetic_contract_examples_not_patient_data", "cases": results})
    return results


def read_verified_ptbxl_record(header: Path, data: Path) -> np.ndarray:
    lines = header.read_text(encoding="utf-8").splitlines()
    record = lines[0].split()
    if record[1:4] != ["12", "100", "1000"] or len(lines) != 13:
        raise ValueError("Only fixed-layout 12-lead, 100 Hz, 1000-sample records are supported")
    if data.stat().st_size != 1000 * 12 * 2:
        raise ValueError("Unexpected format-16 byte length")
    digital = np.frombuffer(data.read_bytes(), dtype="<i2").reshape(1000, 12)
    physical = np.empty((1000, 12), dtype=np.float64)
    for index, line in enumerate(lines[1:]):
        fields = line.split()
        if fields[0] != data.name or fields[1] != "16" or fields[-1] != CHANNELS[index]:
            raise ValueError("Unsupported WFDB storage format or channel order")
        calibration = re.fullmatch(r"([0-9.]+)\((-?\d+)\)/mV", fields[2])
        if calibration is None or float(calibration[1]) <= 0:
            raise ValueError("Unsupported gain/baseline/unit")
        if int(digital[0, index]) != int(fields[5]):
            raise ValueError("WFDB initial sample mismatch")
        if int(digital[:, index].astype(np.int64).sum()) % 65536 != int(fields[6]) % 65536:
            raise ValueError("WFDB per-channel checksum mismatch")
        physical[:, index] = (digital[:, index].astype(np.float64) - int(calibration[2])) / float(calibration[1])
        physical[digital[:, index] == -32768, index] = np.nan
    return physical


def real_samples(root: Path) -> list[dict]:
    provenance = json.loads((root / "docs" / "provenance.json").read_text(encoding="utf-8"))
    raw_dir = root / provenance["raw_directory"]
    pinned = {item["relative_path"]: item["sha256"] for item in provenance["files"]}
    before = {name: sha256(raw_dir / name) for name in pinned}
    if before != pinned:
        raise ValueError("Raw source files differ from pinned provenance")
    outputs = root / "data" / "validated" / "ptb-xl" / "1.0.3"
    results = []
    for sample in provenance["sample_records"]:
        base = raw_dir / sample["filename_lr"]
        signal = read_verified_ptbxl_record(base.with_suffix(".hea"), base.with_suffix(".dat"))
        name = f"ptbxl_1_0_3_{sample['ecg_id']:05}"
        npz_path, json_path = save_pair(outputs, name, signal, {"source": {
            "dataset_version": "1.0.3", "ecg_id": sample["ecg_id"],
            "header_sha256": before[sample["filename_lr"] + ".hea"],
            "data_sha256": before[sample["filename_lr"] + ".dat"],
            "converter_sha256": sha256(Path(__file__)),
            "conversion": "WFDB format 16: (digital - baseline) / adc_gain -> mV; separate derivative",
        }})
        result = validate_signal(npz_path, json_path, root / "docs" / "signal_schema.json")
        results.append(result)
    after = {name: sha256(raw_dir / name) for name in pinned}
    if after != before:
        raise AssertionError("Raw source files were modified")
    write_json(outputs / "validation_report.json", {
        "raw_source_hashes_unchanged": True, "validated_real_records": len(results),
        "converter_sha256": sha256(Path(__file__)),
        "raw_source_sha256": before, "results": results,
    })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--synthetic-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if not (root / "pyproject.toml").is_file():
        parser.error("--root must identify the project root")
    fixtures = synthetic_fixtures(root)
    results = [] if args.synthetic_only else real_samples(root)
    print(json.dumps({"synthetic_cases": fixtures, "validated_real_records": len(results)}, indent=2))


if __name__ == "__main__":
    main()
