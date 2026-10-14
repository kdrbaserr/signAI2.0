"""Reproducible ECG plots with metadata-derived time and source identity."""

from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import sys

import numpy as np

from signai.ptbxl import LABEL_ORDER, sha256, write_json
from signai.splits import load_manifests
from signai.validate_signal import SignalValidationError, validate_signal


class InspectionError(ValueError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


def check_time_axis(axis: np.ndarray, sample_count: int, sampling_rate_hz: float, duration_seconds: float) -> dict:
    expected = np.arange(sample_count, dtype=np.float64) / sampling_rate_hz
    if (axis.shape != expected.shape or not np.isfinite(axis).all()
            or not np.allclose(axis, expected, rtol=0, atol=1e-12)
            or not np.isclose(sample_count / sampling_rate_hz, duration_seconds, rtol=0, atol=1e-12)):
        raise InspectionError("E_TIME_SCALE", "Plot time does not match sample count, sampling rate and metadata duration")
    return {
        "sample_count": sample_count, "sampling_rate_hz": sampling_rate_hz,
        "sample_interval_seconds": 1 / sampling_rate_hz,
        "first_sample_seconds": float(axis[0]), "last_sample_seconds": float(axis[-1]),
        "sample_coverage_seconds": sample_count / sampling_rate_hz,
        "plot_xlim_seconds": [0, duration_seconds], "time_axis_checked": True,
    }


def render_plot(signal, axis, channels, ecg_id, split, labels, output):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        figure, axes = plt.subplots(6, 2, figsize=(13, 12), sharex=True)
        for index, panel in enumerate(axes.flat):
            panel.plot(axis, signal[:, index], color="#176d95", linewidth=0.85)
            panel.set_title(channels[index], loc="left", fontweight="bold", fontsize=11)
            panel.set_ylabel("mV")
            panel.set_xlim(0, 10)
            panel.set_xticks(np.arange(0, 11, 1))
            panel.grid(True, color="#e1e7ec", linewidth=0.6)
        for panel in axes[-1]:
            panel.set_xlabel("Time (seconds)")
        title_labels = ", ".join(labels) if labels else "excluded: no diagnostic target"
        figure.suptitle(f"PTB-XL 1.0.3 | ECG {ecg_id} | {split} | {title_labels}", fontsize=16, fontweight="bold", y=0.98)
        figure.text(0.5, 0.955, "100 Hz | 1000 samples | 10 seconds | 12 leads | source signal in mV", ha="center", color="#52616f")
        figure.tight_layout(rect=(0, 0, 1, 0.945), h_pad=1.1)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=140, metadata={"Software": "signai-2.0 ECG inspector"})
        plt.close(figure)


def inspect_record(root: Path, ecg_id: int, output_dir: Path) -> dict:
    if type(ecg_id) is not int or ecg_id <= 0:
        raise InspectionError("E_RECORD_ID", "ecg_id must be a positive integer")
    manifests = load_manifests(root / "splits")
    record = next((record for manifest in manifests.values()
                   for key in ("records", "excluded_records") for record in manifest[key]
                   if record["ecg_id"] == ecg_id), None)
    if record is None:
        raise InspectionError("E_RECORD_ID", f"No metadata record for ecg_id={ecg_id}")
    pair = root / "data" / "validated" / "ptb-xl" / "1.0.3" / f"ptbxl_1_0_3_{ecg_id:05}"
    npz, metadata_path = pair.with_suffix(".npz"), pair.with_suffix(".json")
    if not npz.is_file() or not metadata_path.is_file():
        raise InspectionError("E_WAVEFORM_NOT_LOCAL", f"Metadata exists for ECG {ecg_id}, but its validated signal is not downloaded/converted locally")
    before = {"npz": sha256(npz), "json": sha256(metadata_path)}
    validation = validate_signal(npz, metadata_path, root / "docs" / "signal_schema.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["source"]["ecg_id"] != ecg_id or metadata["record_id"] != pair.name:
        raise InspectionError("E_SOURCE_ID", "Signal source does not match requested record identity")
    provenance = json.loads((root / "docs" / "provenance.json").read_text(encoding="utf-8"))
    pinned = {item["relative_path"]: item["sha256"] for item in provenance["files"]}
    raw_dir = root / provenance["raw_directory"]
    source_checks = {}
    for extension, source_key in ((".hea", "header_sha256"), (".dat", "data_sha256")):
        name = record["filename_lr"] + extension
        if name not in pinned or metadata["source"][source_key] != pinned[name] or sha256(raw_dir / name) != pinned[name]:
            raise InspectionError("E_SOURCE_HASH", "Source waveform differs from pinned provenance")
        source_checks[name] = pinned[name]
    with np.load(npz, allow_pickle=False) as arrays:
        signal = arrays["signal"]
    axis = np.arange(signal.shape[0], dtype=np.float64) / metadata["sampling_rate_hz"]
    time_info = check_time_axis(axis, len(signal), metadata["sampling_rate_hz"], metadata["duration_seconds"])
    labels = [label for label, positive in zip(LABEL_ORDER, record["label_vector"] or [0] * 5) if positive]
    png = output_dir / f"ecg_{ecg_id:05}.png"
    render_plot(signal, axis, metadata["channels"], ecg_id, record["split"], labels, png)
    if before != {"npz": sha256(npz), "json": sha256(metadata_path)}:
        raise InspectionError("E_MUTATION", "Inspector modified a signal input")
    result = {
        "inspection_version": "1.0.0", "ecg_id": ecg_id, "patient_id": record["patient_id"],
        "split": record["split"], "strat_fold": record["strat_fold"],
        "label_order": list(LABEL_ORDER), "label_vector": record["label_vector"], "labels": labels,
        "excluded_reason": record["excluded_reason"], "signal": validation, "time_axis": time_info,
        "source_waveform_sha256": source_checks, "source_manifest_sha256": sha256(root / "splits" / f"{record['split']}.json"),
        "png_sha256": sha256(png), "inspector_code_sha256": sha256(Path(__file__)),
        "packages": {name: version(name) for name in ("numpy", "matplotlib", "pillow")},
        "input_files_unchanged": True,
        "plot_policy": "Raw physical amplitudes; no filtering, normalization or model prediction. Per-lead y axes autoscale; x axes share 0-10 seconds.",
    }
    write_json(output_dir / f"ecg_{ecg_id:05}.json", result)
    return result


def run_demo(root: Path, ecg_id: int, output_dir: Path) -> dict:
    first = inspect_record(root, ecg_id, output_dir)
    paths = [output_dir / f"ecg_{ecg_id:05}.{extension}" for extension in ("png", "json")]
    hashes = {path.name: sha256(path) for path in paths}
    second = inspect_record(root, ecg_id, output_dir)
    if first != second or hashes != {path.name: sha256(path) for path in paths}:
        raise InspectionError("E_REPRODUCIBILITY", "Repeated inspection did not reproduce identical outputs")
    try:
        check_time_axis(np.arange(1000) / 500, 1000, 100, 10)
    except InspectionError as error:
        experiment = {"status": "rejected_as_expected", "error_code": error.code,
                      "metadata_sampling_rate_hz": 100, "incorrect_plot_sampling_rate_hz": 500,
                      "expected_coverage_seconds": 10, "incorrect_coverage_seconds": 2,
                      "incorrect_last_sample_seconds": 999 / 500, "message": str(error)}
    else:
        raise InspectionError("E_TIME_SCALE", "Incorrect plotting scale was not rejected")
    write_json(output_dir / "time_scale_experiment.json", experiment)
    probe_path = root / "reports" / "W01-D05" / "model_probe.json"
    probe = json.loads(probe_path.read_text(encoding="utf-8")) if probe_path.exists() else None
    result = {
        "status": "passed", "ecg_id": ecg_id, "inspection_runs": 2,
        "outputs_identical": True, "output_sha256": hashes, "time_scale_experiment": experiment,
        "model_probe": {"path": "reports/W01-D05/model_probe.json", "sha256": sha256(probe_path),
                        "status": probe["status"], "checkpoint_revision": probe["checkpoint"]["revision"],
                        "lora_support_proven": probe["status"] == "passed" and probe["changed_parameter_tensors"]["lora"] > 0 and probe["changed_parameter_tensors"]["frozen"] == 0} if probe else None,
        "scope": "Local waveform inspection and existing model probe evidence; no new training or final test evaluation.",
    }
    write_json(output_dir / "demo.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--ecg-id", type=int, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--demo", action="store_true", help="Repeat inspection and run the negative time-scale experiment")
    args = parser.parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir or root / "reports" / "W01"
    try:
        result = run_demo(root, args.ecg_id, output_dir) if args.demo else inspect_record(root, args.ecg_id, output_dir)
    except (InspectionError, SignalValidationError) as error:
        print(json.dumps({"status": "failed", "error_code": error.code, "message": str(error)}))
        sys.exit(1)
    print(json.dumps({"status": "passed", "ecg_id": args.ecg_id, "output_dir": str(output_dir),
                      "outputs_identical": result.get("outputs_identical")}, indent=2))


if __name__ == "__main__":
    main()
