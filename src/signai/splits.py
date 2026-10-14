"""Patient-disjoint PTB-XL manifests and guarded dataset access."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from decimal import Decimal, InvalidOperation
from itertools import combinations
import json
from pathlib import Path
from typing import Any

from signai.ptbxl import (
    DATASET_VERSION, LABEL_ORDER, POLICY_VERSION, encode_labels, load_schema,
    safe_relative_path, sha256, write_json,
)

SPLIT_POLICY = "ptbxl-patient-folds-v1"
FOLDS = {"train": tuple(range(1, 9)), "validation": (9,), "test": (10,)}


class SplitIntegrityError(ValueError):
    """Record, patient, fold, window, or usage violates the split contract."""


def positive_id(value: Any, field: str) -> int:
    try:
        number = Decimal(str(value))
    except InvalidOperation as exc:
        raise SplitIntegrityError(f"Invalid {field}: {value!r}") from exc
    if not number.is_finite() or number <= 0 or number != number.to_integral_value():
        raise SplitIntegrityError(f"Invalid {field}: {value!r}")
    return int(number)


def split_for_fold(fold: int) -> str:
    for split, folds in FOLDS.items():
        if fold in folds:
            return split
    raise SplitIntegrityError(f"Invalid strat_fold: {fold}")


def read_records(metadata: Path, schema: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    with metadata.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            ecg_id = positive_id(row["ecg_id"], "ecg_id")
            patient_id = positive_id(row["patient_id"], "patient_id")
            fold = positive_id(row["strat_fold"], "strat_fold")
            vector = encode_labels(row["scp_codes"], schema)
            filename = row["filename_lr"]
            safe_relative_path(filename)
            if not filename.startswith("records100/"):
                raise SplitIntegrityError(f"Expected 100 Hz path: ecg_id={ecg_id}")
            records.append({
                "ecg_id": ecg_id, "patient_id": patient_id, "strat_fold": fold,
                "split": split_for_fold(fold), "filename_lr": filename,
                "label_vector": list(vector) if vector is not None else None,
                "excluded_reason": "no_diagnostic_superclass" if vector is None else None,
            })
    return sorted(records, key=lambda record: record["ecg_id"])


def statistics(records: list[dict[str, Any]], excluded: list[dict[str, Any]]) -> dict[str, Any]:
    support = Counter({label: 0 for label in LABEL_ORDER})
    patient_support = {label: set() for label in LABEL_ORDER}
    for record in records:
        for label, value in zip(LABEL_ORDER, record["label_vector"]):
            if value:
                support[label] += 1
                patient_support[label].add(record["patient_id"])
    return {
        "metadata_records": len(records) + len(excluded),
        "included_records": len(records), "excluded_records": len(excluded),
        "included_patients": len({record["patient_id"] for record in records}),
        "all_patients": len({record["patient_id"] for record in records + excluded}),
        "positive_records_per_label": dict(support),
        "positive_patients_per_label": {label: len(patients) for label, patients in patient_support.items()},
        "multilabel_records": sum(sum(record["label_vector"]) > 1 for record in records),
    }


def build_manifests(records: list[dict[str, Any]], source: dict[str, str]) -> dict[str, dict[str, Any]]:
    manifests = {}
    for split, folds in FOLDS.items():
        selected = sorted((record for record in records if record["split"] == split), key=lambda record: record["ecg_id"])
        included = [record for record in selected if record["label_vector"] is not None]
        excluded = [record for record in selected if record["label_vector"] is None]
        manifests[split] = {
            "manifest_version": "1.0.0", "split_policy_version": SPLIT_POLICY,
            "dataset_version": DATASET_VERSION, "label_policy_version": POLICY_VERSION,
            "label_order": list(LABEL_ORDER), "split": split, "allowed_folds": list(folds),
            "source": source, "statistics": statistics(included, excluded),
            "records": included, "excluded_records": excluded,
        }
    if sum(manifest["statistics"]["metadata_records"] for manifest in manifests.values()) != len(records):
        raise SplitIntegrityError("Some input records have an unknown split assignment")
    validate_manifests(manifests)
    return manifests


def validate_manifests(manifests: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if set(manifests) != set(FOLDS):
        raise SplitIntegrityError("Exactly train, validation and test manifests are required")
    ecg_owner: dict[int, str] = {}
    patient_fold: dict[int, int] = {}
    patient_sets: dict[str, set[int]] = {}
    record_sets: dict[str, set[int]] = {}
    reference_source = manifests["train"]["source"]
    for split, manifest in manifests.items():
        if (manifest["split"] != split or manifest["allowed_folds"] != list(FOLDS[split])
                or manifest["label_order"] != list(LABEL_ORDER)
                or manifest["split_policy_version"] != SPLIT_POLICY
                or manifest["dataset_version"] != DATASET_VERSION
                or manifest["label_policy_version"] != POLICY_VERSION
                or manifest["source"] != reference_source):
            raise SplitIntegrityError(f"Manifest header mismatch: {split}")
        patient_sets[split], record_sets[split] = set(), set()
        for collection in ("records", "excluded_records"):
            for record in manifest[collection]:
                ecg_id, patient_id, fold = (record[key] for key in ("ecg_id", "patient_id", "strat_fold"))
                if any(type(value) is not int or value <= 0 for value in (ecg_id, patient_id, fold)):
                    raise SplitIntegrityError("Manifest identifiers must be positive integers")
                if record["split"] != split or fold not in FOLDS[split]:
                    raise SplitIntegrityError(f"Record assigned to wrong split: ecg_id={ecg_id}")
                if ecg_id in ecg_owner:
                    raise SplitIntegrityError(f"Duplicate record ecg_id={ecg_id}: {ecg_owner[ecg_id]} / {split}")
                ecg_owner[ecg_id] = split
                if patient_id in patient_fold and patient_fold[patient_id] != fold:
                    raise SplitIntegrityError(f"Patient {patient_id} appears in multiple folds: {patient_fold[patient_id]} / {fold}")
                patient_fold[patient_id] = fold
                vector = record["label_vector"]
                if collection == "records":
                    if (not isinstance(vector, list) or len(vector) != len(LABEL_ORDER)
                            or any(type(value) is not int or value not in (0, 1) for value in vector)
                            or not any(vector) or record["excluded_reason"] is not None):
                        raise SplitIntegrityError(f"Invalid included target: ecg_id={ecg_id}")
                elif vector is not None or record["excluded_reason"] != "no_diagnostic_superclass":
                    raise SplitIntegrityError(f"Invalid excluded target: ecg_id={ecg_id}")
                patient_sets[split].add(patient_id)
                record_sets[split].add(ecg_id)
        expected = statistics(manifest["records"], manifest["excluded_records"])
        if manifest["statistics"] != expected:
            raise SplitIntegrityError(f"Statistics mismatch: {split}")
    intersections = {}
    for left, right in combinations(FOLDS, 2):
        patients = patient_sets[left] & patient_sets[right]
        records = record_sets[left] & record_sets[right]
        if patients or records:
            raise SplitIntegrityError(f"Overlap: {left} / {right}")
        intersections[f"{left}__{right}"] = {"patient_intersection": len(patients), "record_intersection": len(records)}
    return {"intersections": intersections, "metadata_records": len(ecg_owner), "all_patients": len(patient_fold)}


def load_manifests(directory: Path) -> dict[str, dict[str, Any]]:
    manifests = {split: json.loads((directory / f"{split}.json").read_text(encoding="utf-8")) for split in FOLDS}
    validate_manifests(manifests)
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    for split in FOLDS:
        if sha256(directory / f"{split}.json") != summary["manifest_sha256"][split]:
            raise SplitIntegrityError(f"Manifest checksum mismatch: {split}")
    return manifests


def records_for_purpose(manifest: dict[str, Any], purpose: str, *, allow_final_test: bool = False) -> list[dict[str, Any]]:
    required = {"fit": "train", "model_selection": "validation", "final_evaluation": "test"}
    if purpose not in required or manifest["split"] != required[purpose]:
        raise SplitIntegrityError(f"Split {manifest['split']} cannot be used for {purpose}")
    if purpose == "final_evaluation" and not allow_final_test:
        raise SplitIntegrityError("Final test is locked; explicit final evaluation authorization is required")
    for record in manifest["records"]:
        if record["split"] != required[purpose] or record["strat_fold"] not in FOLDS[required[purpose]]:
            raise SplitIntegrityError(f"Forbidden record in {purpose}: ecg_id={record['ecg_id']}")
    return manifest["records"]


def load_training_inputs(directory: Path) -> dict[str, list[dict[str, Any]]]:
    """The future training entrypoint receives train and validation, never test."""
    manifests = load_manifests(directory)
    return {
        "train": records_for_purpose(manifests["train"], "fit"),
        "validation": records_for_purpose(manifests["validation"], "model_selection"),
    }


def assign_windows(windows: list[dict[str, Any]], manifests: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach parent identity/split to window metadata; no waveform slicing yet."""
    validate_manifests(manifests)
    parents = {record["ecg_id"]: record for manifest in manifests.values() for record in manifest["records"]}
    assigned, seen = [], set()
    for window in windows:
        ecg_id = window["ecg_id"]
        if type(ecg_id) is not int or ecg_id not in parents:
            raise SplitIntegrityError(f"Unknown or excluded window parent: {ecg_id}")
        parent = parents[ecg_id]
        start, end = window["start_sample"], window["end_sample"]
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= 1000:
            raise SplitIntegrityError("Invalid window boundaries for a 1000-sample record")
        for key in ("split", "patient_id", "strat_fold", "label_vector"):
            if key in window and window[key] != parent[key]:
                raise SplitIntegrityError(f"Window {key} disagrees with parent: ecg_id={ecg_id}")
        window_id = f"{ecg_id}:{start}:{end}"
        if window_id in seen:
            raise SplitIntegrityError(f"Duplicate window: {window_id}")
        seen.add(window_id)
        assigned.append({
            "window_id": window_id, "ecg_id": ecg_id, "start_sample": start, "end_sample": end,
            **{key: parent[key] for key in ("patient_id", "strat_fold", "split", "label_vector")},
        })
    return sorted(assigned, key=lambda window: (window["ecg_id"], window["start_sample"], window["end_sample"]))


def prepare_splits(root: Path) -> dict[str, Any]:
    metadata = root / "data" / "raw" / "ptb-xl" / DATASET_VERSION / "ptbxl_database.csv"
    schema_path = root / "docs" / "label_schema.json"
    provenance = json.loads((root / "docs" / "provenance.json").read_text(encoding="utf-8"))
    expected_metadata = next(item["sha256"] for item in provenance["files"] if item["relative_path"] == "ptbxl_database.csv")
    if sha256(metadata) != expected_metadata or sha256(schema_path) != provenance["label_schema_sha256"]:
        raise SplitIntegrityError("Metadata/schema differs from pinned Day 2 provenance")
    source = {"metadata_sha256": expected_metadata, "label_schema_sha256": sha256(schema_path)}
    manifests = build_manifests(read_records(metadata, load_schema(schema_path)), source)
    integrity = validate_manifests(manifests)
    if integrity["metadata_records"] != provenance["metadata_audit"]["metadata_records"]:
        raise SplitIntegrityError("Metadata coverage differs from provenance")
    destination = root / "splits"
    for split, manifest in manifests.items():
        write_json(destination / f"{split}.json", manifest)
    summary = {
        "split_policy_version": SPLIT_POLICY, "dataset_version": DATASET_VERSION,
        "label_policy_version": POLICY_VERSION, "label_order": list(LABEL_ORDER), "source": source,
        "source_url": f"https://physionet.org/content/ptb-xl/{DATASET_VERSION}/",
        "generator_sha256": sha256(Path(__file__)),
        "statistics": {split: manifest["statistics"] for split, manifest in manifests.items()},
        **integrity,
        "manifest_sha256": {split: sha256(destination / f"{split}.json") for split in FOLDS},
        "window_policy": "Split records first; windows inherit patient_id, ecg_id, fold and split from included parent.",
        "training_input_policy": "fit=train only; model selection=validation only; final test requires explicit opt-in.",
        "negative_experiment": {
            "description": "Set a validation patient's id to a train patient's id, then validate.",
            "expected_exception": "SplitIntegrityError: Patient appears in multiple folds",
            "automated_test": "tests/test_split_integrity.py::test_same_patient_in_two_folds_fails",
        },
    }
    write_json(destination / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--training-inputs", action="store_true", help="Validate training access and report only train/validation counts")
    args = parser.parse_args()
    root = args.root.resolve()
    if not (root / "pyproject.toml").is_file():
        parser.error("--root must identify the project root")
    if args.training_inputs:
        inputs = load_training_inputs(root / "splits")
        print(json.dumps({split: len(records) for split, records in inputs.items()}, indent=2))
    else:
        print(json.dumps(prepare_splits(root), indent=2))


if __name__ == "__main__":
    main()
