"""V1: train-only scaler fitting, deterministic windows, separate aggregation.

No filtering/resampling/clipping or per-request fitting. Split records before
calling the training entrypoint. External z-score leaves MOMENT's RevIN enabled.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from signai.ptbxl import LABEL_ORDER, sha256, write_json
from signai.splits import FOLDS, SplitIntegrityError, load_manifests
from signai.validate_signal import CHANNELS, load_validated_signal, read_json

POLICY = "ptbxl-moment-preprocessing-v1"
DEFAULT_CONFIG = Path("configs/preprocessing-v1.json")
DEFAULT_SCALER = Path("artifacts/preprocessing/v1/scaler.json")


class PreprocessingError(ValueError):
    """Configuration, fitted state or window contract mismatch."""


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "policy_version": POLICY, "input_samples": 1000, "sampling_rate_hz": 100,
        "channels": list(CHANNELS), "window_samples": 512, "stride_samples": 512,
        "tail_policy": "right_pad", "padding_value_after_scaling": 0.0,
        "mask_convention": "1=observed,0=padding",
        "normalization": "train_per_channel_population_zscore", "minimum_scale": 1e-8,
        "output_dtype": "float32", "output_layout": "window,channel,time",
        "aggregation": "valid_sample_weighted_mean_of_sigmoid", "label_order": list(LABEL_ORDER),
        "moment_internal_normalization": "keep_pretrained_revin",
    }
    if config != expected:
        raise PreprocessingError("Unsupported preprocessing config; version and implementation must change together")
    return config


def _train_records(split_directory: Path):
    manifests = load_manifests(split_directory)
    return {record["ecg_id"]: record for record in manifests["train"]["records"]}


def validate_fit_lineage(state: dict, split_directory: Path) -> None:
    """Check canonical membership, never trust a caller's claimed split string."""
    train = _train_records(split_directory)
    if state["train_manifest_sha256"] != sha256(split_directory / "train.json"):
        raise SplitIntegrityError("Scaler train manifest checksum mismatch")
    seen = set()
    for item in state["fit_records"]:
        ecg_id = item["ecg_id"]
        parent = train.get(ecg_id)
        if parent is None or ecg_id in seen:
            raise SplitIntegrityError(f"Forbidden or duplicate scaler fit record: ecg_id={ecg_id}")
        seen.add(ecg_id)
        if any(item[key] != parent[key] for key in ("patient_id", "strat_fold", "split")):
            raise SplitIntegrityError(f"Scaler lineage disagrees with train parent: ecg_id={ecg_id}")
    if not seen:
        raise SplitIntegrityError("Scaler requires at least one train record")
    expected_scope = "full_train" if seen == set(train) else "explicit_train_subset"
    if state["fit_scope"] != expected_scope:
        raise SplitIntegrityError("Scaler coverage claim disagrees with fit record IDs")


def _load_training_signal(directory: Path, parent: dict, schema: Path):
    stem = directory / f"ptbxl_1_0_3_{parent['ecg_id']:05}"
    metadata = read_json(stem.with_suffix(".json"))
    source = metadata.get("source", {})
    if (source.get("ecg_id") != parent["ecg_id"] or source.get("dataset_version") != "1.0.3"
            or metadata["record_id"] != stem.name):
        raise SplitIntegrityError("Validated signal identity disagrees with split record")
    return load_validated_signal(stem.with_suffix(".npz"), stem.with_suffix(".json"), schema)


def fit_scaler(record_ids: list[int], *, split_directory: Path, signal_directory: Path,
               config_path: Path, schema_path: Path) -> dict:
    """Fit raw, unpadded, non-overlapping records in canonical ID order.

    Explicit subsets are allowed for smoke runs and recorded as such. Validate
    ALL IDs before opening any waveform, including the deliberate leakage case.
    """
    config = load_config(config_path)
    train = _train_records(split_directory)
    if (not record_ids or any(type(value) is not int for value in record_ids)
            or len(set(record_ids)) != len(record_ids)):
        raise SplitIntegrityError("Fit record IDs must be nonempty, unique integers")
    forbidden = sorted(set(record_ids) - set(train))
    if forbidden:
        raise SplitIntegrityError(f"Forbidden scaler fit IDs (not included train records): {forbidden}")
    count, mean, m2, lineage = 0, np.zeros(12), np.zeros(12), []
    for ecg_id in sorted(record_ids):
        parent = train[ecg_id]
        signal, report = _load_training_signal(signal_directory, parent, schema_path)
        values = signal.astype(np.float64)
        n = len(values)
        batch_mean = values.mean(axis=0)
        batch_m2 = np.square(values - batch_mean).sum(axis=0)
        delta = batch_mean - mean
        total = count + n
        m2 += batch_m2 + np.square(delta) * count * n / total
        mean += delta * n / total
        count = total
        lineage.append({key: parent[key] for key in ("ecg_id", "patient_id", "strat_fold", "split")} | {
            "npz_sha256": report["npz_sha256"], "metadata_sha256": report["metadata_sha256"],
        })
    scale = np.maximum(np.sqrt(m2 / count), config["minimum_scale"])
    state = {
        "state_version": 1, "policy_version": POLICY, "config_sha256": fingerprint(config),
        "signal_schema_sha256": sha256(schema_path),
        "train_manifest_sha256": sha256(split_directory / "train.json"),
        "fit_scope": "full_train" if set(record_ids) == set(train) else "explicit_train_subset",
        "sample_count_per_channel": count, "mean": mean.tolist(), "scale": scale.tolist(),
        "fit_records": lineage,
    }
    validate_fit_lineage(state, split_directory)
    return state


@dataclass(frozen=True)
class WindowBatch:
    x_enc: np.ndarray
    input_mask: np.ndarray
    starts: tuple[int, ...]
    valid_samples: tuple[int, ...]
    record_id: str
    scaler_sha256: str
    parent: dict | None = None

    def torch_inputs(self, device="cpu"):
        import torch

        return {"x_enc": torch.from_numpy(self.x_enc.copy()).to(device),
                "input_mask": torch.from_numpy(self.input_mask.copy()).to(device)}


class Preprocessor:
    """Frozen fitted state shared by both entry paths; transform never fits."""

    def __init__(self, config_path: Path, scaler_path: Path, schema_path: Path):
        self.config = load_config(config_path)
        self.schema_path = schema_path
        state = json.loads(scaler_path.read_text(encoding="utf-8"))
        if (state.get("state_version") != 1 or state.get("policy_version") != POLICY
                or state.get("config_sha256") != fingerprint(self.config)
                or state.get("signal_schema_sha256") != sha256(schema_path)):
            raise PreprocessingError("Scaler/config/signal schema version mismatch")
        self.mean = np.asarray(state["mean"], dtype=np.float64)
        self.scale = np.asarray(state["scale"], dtype=np.float64)
        if (self.mean.shape != (12,) or self.scale.shape != (12,)
                or not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all()
                or (self.scale < self.config["minimum_scale"]).any()):
            raise PreprocessingError("Invalid fitted channel statistics")
        entries = state.get("fit_records", [])
        ids = [item["ecg_id"] for item in entries]
        if (not ids or len(set(ids)) != len(ids)
                or any(item["split"] != "train" or item["strat_fold"] not in FOLDS["train"] for item in entries)
                or state["sample_count_per_channel"] != 1000 * len(entries)
                or state["fit_scope"] not in ("full_train", "explicit_train_subset")):
            raise PreprocessingError("Invalid scaler fit lineage")
        self.mean.setflags(write=False)
        self.scale.setflags(write=False)
        self._state = state
        self.scaler_sha256 = fingerprint(state)

    def _transform(self, signal: np.ndarray, report: dict, parent=None) -> WindowBatch:
        # Normalize REAL samples first; padding is zero in normalized space.
        normalized = ((signal.astype(np.float64) - self.mean) / self.scale).astype(np.float32)
        starts = tuple(range(0, self.config["input_samples"], self.config["stride_samples"]))
        x = np.zeros((len(starts), 12, self.config["window_samples"]), dtype=np.float32)
        mask = np.zeros((len(starts), self.config["window_samples"]), dtype=np.int64)
        counts = []
        for index, start in enumerate(starts):
            part = normalized[start:start + self.config["window_samples"]]
            counts.append(len(part))
            x[index, :, :len(part)] = part.T
            mask[index, :len(part)] = 1
        return WindowBatch(x, mask, starts, tuple(counts), report["record_id"], self.scaler_sha256, parent)

    def training_input(self, ecg_id: int, *, split_directory: Path, signal_directory: Path,
                       split: str = "train") -> WindowBatch:
        """Select a split parent BEFORE slicing. Final test has no training path."""
        if split not in ("train", "validation"):
            raise SplitIntegrityError("Training entrypoint allows train/validation only")
        validate_fit_lineage(self._state, split_directory)
        manifests = load_manifests(split_directory)
        parent = next((item for item in manifests[split]["records"] if item["ecg_id"] == ecg_id), None)
        if parent is None:
            raise SplitIntegrityError(f"Unknown/excluded {split} parent: ecg_id={ecg_id}")
        signal, report = _load_training_signal(signal_directory, parent, self.schema_path)
        fitted = next((item for item in self._state["fit_records"] if item["ecg_id"] == ecg_id), None)
        if fitted and any(fitted[key] != report[key] for key in ("npz_sha256", "metadata_sha256")):
            raise SplitIntegrityError("Training artifact differs from scaler fit lineage")
        return self._transform(signal, report, parent.copy())

    def inference_input(self, npz_path: Path, metadata_path: Path) -> WindowBatch:
        signal, report = load_validated_signal(npz_path, metadata_path, self.schema_path)
        return self._transform(signal, report)


def aggregate_record_probabilities(logits: np.ndarray, batch: WindowBatch) -> np.ndarray:
    """Per-label sigmoid, then mean weighted by real sample count; no softmax.

    V1 is one 1000-sample record with two non-overlapping windows. Reject missing,
    duplicated/reordered windows or mismatched masks instead of changing weights.
    """
    values = np.asarray(logits, dtype=np.float64)
    if (batch.starts != (0, 512) or batch.valid_samples != (512, 488)
            or batch.input_mask.shape != (2, 512) or batch.x_enc.shape != (2, 12, 512)
            or values.shape != (2, len(LABEL_ORDER)) or not np.isfinite(values).all()):
        raise PreprocessingError("Expected complete V1 record windows and finite [2,5] logits")
    expected_mask = np.arange(512)[None, :] < np.array(batch.valid_samples)[:, None]
    if not np.array_equal(batch.input_mask, expected_mask):
        raise PreprocessingError("Mask disagrees with real sample counts")
    # Stable sigmoid for large positive/negative logits.
    probabilities = np.exp(-np.logaddexp(0.0, -values))
    return np.average(probabilities, axis=0, weights=batch.valid_samples).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--record-ids", type=int, nargs="+",
                           help="Explicit train IDs to fit; subsets are recorded as such")
    selection.add_argument("--all-train", action="store_true", help="Fit every included train record; missing waveforms are errors")
    args = parser.parse_args()
    root = args.root.resolve()
    ids = sorted(_train_records(root / "splits")) if args.all_train else args.record_ids
    state = fit_scaler(ids, split_directory=root / "splits",
                       signal_directory=root / "data/validated/ptb-xl/1.0.3",
                       config_path=root / DEFAULT_CONFIG, schema_path=root / "docs/signal_schema.json")
    destination = root / DEFAULT_SCALER
    write_json(destination, state)
    print(json.dumps({"scaler": str(destination), "fit_scope": state["fit_scope"],
                      "fit_record_ids": ids, "scaler_sha256": fingerprint(state)}, indent=2))


if __name__ == "__main__":
    main()
