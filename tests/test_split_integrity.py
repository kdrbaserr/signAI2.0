"""Negative leakage experiments and guarded train/window access, offline."""

import pytest

from signai.ptbxl import sha256, write_json
from signai.splits import (
    FOLDS, SplitIntegrityError, assign_windows, build_manifests, load_manifests,
    load_training_inputs, positive_id, records_for_purpose, validate_manifests,
)


def record(ecg_id, patient_id, fold, vector):
    split = "train" if fold <= 8 else "validation" if fold == 9 else "test"
    return {
        "ecg_id": ecg_id, "patient_id": patient_id, "strat_fold": fold, "split": split,
        "filename_lr": f"records100/00000/{ecg_id:05}_lr",
        "label_vector": vector,
        "excluded_reason": "no_diagnostic_superclass" if vector is None else None,
    }


@pytest.fixture
def manifests():
    records = [
        record(1, 11, 1, [1, 0, 0, 0, 0]),
        record(2, 11, 1, [0, 1, 0, 0, 1]),
        record(3, 12, 8, None),
        record(4, 13, 9, [0, 0, 1, 0, 0]),
        record(5, 14, 10, [0, 0, 0, 1, 0]),
    ]
    return build_manifests(records, {"metadata_sha256": "fixture", "label_schema_sha256": "fixture"})


def test_all_patient_and_record_intersections_are_zero(manifests):
    report = validate_manifests(manifests)
    assert report["metadata_records"] == 5
    assert report["all_patients"] == 4
    assert all(value == {"patient_intersection": 0, "record_intersection": 0} for value in report["intersections"].values())


def test_same_patient_in_two_folds_fails(manifests):
    manifests["validation"]["records"][0]["patient_id"] = 11
    with pytest.raises(SplitIntegrityError, match="Patient 11 appears in multiple folds"):
        validate_manifests(manifests)


def test_same_patient_in_two_training_folds_also_fails(manifests):
    manifests["train"]["records"][1]["strat_fold"] = 2
    with pytest.raises(SplitIntegrityError, match="Patient 11 appears in multiple folds"):
        validate_manifests(manifests)


def test_excluded_patient_cannot_leak_into_test(manifests):
    manifests["test"]["records"][0]["patient_id"] = 12
    with pytest.raises(SplitIntegrityError, match="Patient 12 appears in multiple folds"):
        validate_manifests(manifests)


def test_duplicate_record_across_splits_fails(manifests):
    manifests["test"]["records"][0]["ecg_id"] = 1
    with pytest.raises(SplitIntegrityError, match="Duplicate record ecg_id=1"):
        validate_manifests(manifests)


def test_fold_cannot_be_reassigned_to_another_split(manifests):
    manifests["train"]["records"][0]["strat_fold"] = 10
    with pytest.raises(SplitIntegrityError, match="wrong split"):
        validate_manifests(manifests)


def test_support_counts_include_exclusion_and_patient_unit(manifests):
    stats = manifests["train"]["statistics"]
    assert stats["metadata_records"] == 3
    assert stats["included_records"] == 2
    assert stats["excluded_records"] == 1
    assert stats["included_patients"] == 1
    assert stats["all_patients"] == 2
    assert stats["positive_records_per_label"] == {"NORM": 1, "MI": 1, "STTC": 0, "CD": 0, "HYP": 1}
    assert stats["multilabel_records"] == 1


def test_false_statistics_are_rejected(manifests):
    manifests["train"]["statistics"]["excluded_records"] = 0
    with pytest.raises(SplitIntegrityError, match="Statistics mismatch"):
        validate_manifests(manifests)


def test_manifest_generation_is_independent_of_row_order(manifests):
    records = [record for manifest in manifests.values() for key in ("records", "excluded_records") for record in manifest[key]]
    assert build_manifests(list(reversed(records)), manifests["train"]["source"]) == manifests


def test_unknown_split_is_not_silently_dropped(manifests):
    invalid = record(9, 19, 1, [1, 0, 0, 0, 0])
    invalid["split"] = "unknown"
    with pytest.raises(SplitIntegrityError, match="unknown split assignment"):
        build_manifests([invalid], manifests["train"]["source"])


@pytest.mark.parametrize("value", ["15709", "15709.0", "1.5709e4"])
def test_equivalent_patient_ids_have_one_identity(value):
    assert positive_id(value, "patient_id") == 15709


@pytest.mark.parametrize("value", ["", "NaN", "Infinity", "-1", "1.5", True])
def test_invalid_patient_ids_fail(value):
    with pytest.raises(SplitIntegrityError, match="Invalid patient_id"):
        positive_id(value, "patient_id")


def test_windows_inherit_parent_patient_fold_and_split(manifests):
    windows = assign_windows([
        {"ecg_id": ecg_id, "start_sample": start, "end_sample": start + 500}
        for ecg_id in (1, 4, 5) for start in (0, 500)
    ], manifests)
    for window in windows:
        parent = next(record for record in manifests[window["split"]]["records"] if record["ecg_id"] == window["ecg_id"])
        for key in ("patient_id", "strat_fold", "split", "label_vector"):
            assert window[key] == parent[key]


def test_window_cannot_disguise_test_parent_as_train(manifests):
    with pytest.raises(SplitIntegrityError, match="Window split disagrees"):
        assign_windows([{"ecg_id": 5, "start_sample": 0, "end_sample": 500, "split": "train"}], manifests)


def test_test_windows_are_rejected_from_training_access(manifests):
    test_windows = assign_windows([{"ecg_id": 5, "start_sample": 0, "end_sample": 500}], manifests)
    with pytest.raises(SplitIntegrityError, match="Forbidden record in fit"):
        records_for_purpose({"split": "train", "records": test_windows}, "fit")


@pytest.mark.parametrize("ecg_id", [3, 999])
def test_window_parent_must_be_included_and_known(ecg_id, manifests):
    with pytest.raises(SplitIntegrityError, match="Unknown or excluded window parent"):
        assign_windows([{"ecg_id": ecg_id, "start_sample": 0, "end_sample": 500}], manifests)


def test_test_and_validation_cannot_be_fit_inputs(manifests):
    for split in ("validation", "test"):
        with pytest.raises(SplitIntegrityError, match="cannot be used for fit"):
            records_for_purpose(manifests[split], "fit")


def test_test_cannot_be_used_for_model_selection(manifests):
    with pytest.raises(SplitIntegrityError, match="cannot be used for model_selection"):
        records_for_purpose(manifests["test"], "model_selection")


def test_final_evaluation_requires_explicit_opt_in(manifests):
    with pytest.raises(SplitIntegrityError, match="Final test is locked"):
        records_for_purpose(manifests["test"], "final_evaluation")
    assert records_for_purpose(manifests["test"], "final_evaluation", allow_final_test=True) == manifests["test"]["records"]


def test_test_record_in_train_container_is_rejected(manifests):
    manifests["train"]["records"].append(manifests["test"]["records"][0])
    with pytest.raises(SplitIntegrityError, match="Forbidden record in fit"):
        records_for_purpose(manifests["train"], "fit")


def write_bundle(directory, manifests):
    for split, manifest in manifests.items():
        write_json(directory / f"{split}.json", manifest)
    write_json(directory / "summary.json", {
        "manifest_sha256": {split: sha256(directory / f"{split}.json") for split in FOLDS},
    })


def test_training_loader_never_returns_final_test(tmp_path, manifests):
    write_bundle(tmp_path, manifests)
    inputs = load_training_inputs(tmp_path)
    assert set(inputs) == {"train", "validation"}
    returned_ids = {record["ecg_id"] for records in inputs.values() for record in records}
    assert returned_ids == {1, 2, 4}
    assert not returned_ids & {record["ecg_id"] for record in manifests["test"]["records"]}


def test_changed_manifest_is_detected_by_hash(tmp_path, manifests):
    write_bundle(tmp_path, manifests)
    path = tmp_path / "train.json"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(SplitIntegrityError, match="checksum mismatch: train"):
        load_manifests(tmp_path)
