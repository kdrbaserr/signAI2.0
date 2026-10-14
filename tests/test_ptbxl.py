"""Task semantics are tested without network access or private/raw metadata."""

import csv
import hashlib
import io
import json
from pathlib import Path

import pytest

from signai.ptbxl import (
    LABEL_ORDER, LabelValidationError, audit_metadata, download,
    encode_labels, load_schema, safe_relative_path, schema_from_statements,
)

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "docs" / "label_schema.json"


@pytest.fixture
def schema():
    return load_schema(SCHEMA_PATH)


def test_fixed_order_and_multilabel_deduplication(schema):
    assert tuple(schema["label_order"]) == LABEL_ORDER
    assert encode_labels({"NORM": 100}, schema) == (1, 0, 0, 0, 0)
    assert encode_labels({"IMI": 100, "ASMI": 50, "LVH": 80}, schema) == (0, 1, 0, 0, 1)


def test_same_metadata_same_vector_regardless_of_code_order(schema):
    first = "{'IMI': 100.0, 'LVH': 80.0, 'SR': 0.0}"
    second = {"SR": 0, "LVH": 80, "IMI": 100}
    assert encode_labels(first, schema) == encode_labels(second, schema)


def test_zero_is_unknown_likelihood_not_negative(schema):
    assert encode_labels({"IMI": 0}, schema) == (0, 1, 0, 0, 0)


@pytest.mark.parametrize("codes", [{}, {"SR": 0}])
def test_empty_diagnostic_target_is_excluded_not_normal(codes, schema):
    assert encode_labels(codes, schema) is None


def test_explicit_normal_is_not_silently_removed(schema):
    assert encode_labels({"NORM": 100, "IMI": 50}, schema) == (1, 1, 0, 0, 0)


def test_unknown_code_fails_even_with_a_valid_diagnosis(schema):
    with pytest.raises(LabelValidationError, match="Unknown SCP code.*NEW_UNKNOWN_SCP"):
        encode_labels({"NORM": 100, "NEW_UNKNOWN_SCP": 100}, schema)


@pytest.mark.parametrize("likelihood", [-1, 101, float("nan"), float("inf"), True, "100"])
def test_invalid_likelihood_fails_even_for_ignored_rhythm(likelihood, schema):
    with pytest.raises(LabelValidationError, match="Likelihood"):
        encode_labels({"SR": likelihood}, schema)


@pytest.mark.parametrize("raw", ["[]", "{'NORM': 100, 'NORM': 0}", "__import__('os').getcwd()"])
def test_invalid_or_duplicate_literal_fails(raw, schema):
    with pytest.raises(LabelValidationError):
        encode_labels(raw, schema)


def test_changed_label_order_is_rejected(tmp_path, schema):
    schema["label_order"] = list(reversed(LABEL_ORDER))
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(schema), encoding="utf-8")
    with pytest.raises(LabelValidationError, match="Label order"):
        load_schema(path)


def test_changed_confidence_policy_requires_new_implementation(tmp_path, schema):
    schema["confidence_policy"]["threshold"] = 50
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(schema), encoding="utf-8")
    with pytest.raises(LabelValidationError, match="confidence policy"):
        load_schema(path)


def write_metadata(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["ecg_id", "patient_id", "strat_fold", "scp_codes", "filename_lr"])
        writer.writeheader()
        writer.writerows(rows)


def test_audit_reports_exclusions_and_is_deterministic(tmp_path, schema):
    rows = [
        {"ecg_id": "3", "patient_id": "3", "strat_fold": "9", "scp_codes": "{'IMI': 100, 'LVH': 80}", "filename_lr": "records100/00000/00003_lr"},
        {"ecg_id": "2", "patient_id": "2", "strat_fold": "10", "scp_codes": "{'SR': 0}", "filename_lr": "records100/00000/00002_lr"},
        {"ecg_id": "1", "patient_id": "1", "strat_fold": "1", "scp_codes": "{'NORM': 100}", "filename_lr": "records100/00000/00001_lr"},
    ]
    path = tmp_path / "metadata.csv"
    write_metadata(path, rows)
    first = audit_metadata(path, schema)
    write_metadata(path, list(reversed(rows)))
    assert first == audit_metadata(path, schema)
    report, records = first
    assert report["metadata_records"] == 3
    assert report["included_records"] == 2
    assert report["excluded_records"] == report["excluded_by_strat_fold"]["10"] == 1
    assert report["multilabel_records"] == 1
    assert report["positive_records_per_label"] == {"NORM": 1, "MI": 1, "STTC": 0, "CD": 0, "HYP": 1}
    assert records[1]["label_vector"] is None


def test_patient_crossing_folds_is_rejected(tmp_path, schema):
    path = tmp_path / "metadata.csv"
    write_metadata(path, [
        {"ecg_id": str(index), "patient_id": "same", "strat_fold": str(index), "scp_codes": "{'NORM': 100}", "filename_lr": "unused"}
        for index in (1, 2)
    ])
    with pytest.raises(LabelValidationError, match="patient appears across folds"):
        audit_metadata(path, schema)


@pytest.mark.parametrize("name", ["../outside", "/absolute", "C:/outside", "records100\\outside"])
def test_download_paths_cannot_escape_raw_directory(name):
    with pytest.raises(ValueError, match="Unsafe"):
        safe_relative_path(name)


def test_checksum_mismatch_does_not_publish_download(tmp_path, monkeypatch):
    monkeypatch.setattr("signai.ptbxl.urlopen", lambda *args, **kwargs: io.BytesIO(b"wrong data"))
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        download(tmp_path, "scp_statements.csv", hashlib.sha256(b"expected data").hexdigest())
    assert not (tmp_path / "scp_statements.csv").exists()
    assert not (tmp_path / "scp_statements.csv.part").exists()


def test_invalid_superclass_in_source_registry_is_rejected(tmp_path):
    source = tmp_path / "registry.csv"
    source.write_text(",diagnostic,diagnostic_class\nNEW,1.0,NOT_A_SUPERCLASS\n", encoding="utf-8")
    with pytest.raises(LabelValidationError, match="diagnostic_class"):
        schema_from_statements(source)
