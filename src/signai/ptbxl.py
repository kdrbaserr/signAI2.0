"""Versioned PTB-XL superclass labels and a small, verified local sample."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import platform
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.request import Request, urlopen

DATASET_VERSION = "1.0.3"
POLICY_VERSION = "ptbxl-superclass-presence-v1"
SOURCE = f"https://physionet.org/files/ptb-xl/{DATASET_VERSION}/"
DOWNLOAD_SOURCE = f"https://physionet-open.s3.amazonaws.com/ptb-xl/{DATASET_VERSION}/"
LABEL_ORDER = ("NORM", "MI", "STTC", "CD", "HYP")


class LabelValidationError(ValueError):
    """Metadata or label policy violates the task contract."""


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_schema(path: Path) -> dict[str, Any]:
    schema = json.loads(path.read_text(encoding="utf-8"))
    if schema.get("label_order") != list(LABEL_ORDER):
        raise LabelValidationError("Label order must be NORM, MI, STTC, CD, HYP")
    if schema.get("dataset_version") != DATASET_VERSION or schema.get("policy_version") != POLICY_VERSION:
        raise LabelValidationError("Unsupported dataset or label policy version")
    if schema.get("confidence_policy") != {
        "mode": "code_presence", "minimum": 0, "maximum": 100,
        "zero_means": "unknown_likelihood_included", "threshold": None,
    }:
        raise LabelValidationError("Unsupported confidence policy")
    if schema.get("unknown_code_policy") != "error" or schema.get("empty_label_policy") != "exclude":
        raise LabelValidationError("Unsupported unknown-code or empty-label policy")
    codes = schema.get("known_scp_codes")
    mapping = schema.get("diagnostic_to_superclass")
    if not isinstance(codes, list) or not codes or any(not isinstance(code, str) for code in codes):
        raise LabelValidationError("Known SCP codes must be a nonempty string list")
    if codes != sorted(set(codes)) or not isinstance(mapping, dict) or not mapping:
        raise LabelValidationError("Invalid SCP code registry or mapping")
    if set(mapping) - set(codes) or set(mapping.values()) - set(LABEL_ORDER):
        raise LabelValidationError("Invalid diagnostic superclass mapping")
    return schema


def parse_scp_codes(raw: str | dict[str, float]) -> dict[str, float]:
    if isinstance(raw, str):
        try:
            expression = ast.parse(raw, mode="eval")
            if not isinstance(expression.body, ast.Dict):
                raise LabelValidationError("scp_codes must be a dictionary")
            keys = [ast.literal_eval(key) for key in expression.body.keys]
            if any(not isinstance(key, str) for key in keys) or len(keys) != len(set(keys)):
                raise LabelValidationError("SCP keys must be unique strings")
            raw = ast.literal_eval(expression)
        except (SyntaxError, ValueError, TypeError) as exc:
            raise LabelValidationError(f"Invalid scp_codes: {exc}") from exc
    if not isinstance(raw, dict):
        raise LabelValidationError("scp_codes must be a dictionary")
    for code, likelihood in raw.items():
        if not isinstance(code, str):
            raise LabelValidationError("SCP keys must be strings")
        if isinstance(likelihood, bool) or not isinstance(likelihood, (int, float)):
            raise LabelValidationError(f"Likelihood for {code} must be numeric")
        if not 0 <= likelihood <= 100 or not math.isfinite(likelihood):
            raise LabelValidationError(f"Likelihood for {code} must be finite and in [0, 100]")
    return raw


def encode_labels(raw: str | dict[str, float], schema: dict[str, Any]) -> tuple[int, ...] | None:
    """Return a fixed-order multi-hot vector, or None for an excluded record.

    Validate all codes before filtering known non-diagnostic statements.
    Zero likelihood means unknown certainty, not an absent statement.
    """
    codes = parse_scp_codes(raw)
    unknown = set(codes) - set(schema["known_scp_codes"])
    if unknown:
        raise LabelValidationError(f"Unknown SCP code(s): {', '.join(sorted(unknown))}")
    mapping = schema["diagnostic_to_superclass"]
    present = {mapping[code] for code in codes if code in mapping}
    if not present:
        return None
    return tuple(int(label in present) for label in schema["label_order"])


def schema_from_statements(path: Path) -> dict[str, Any]:
    codes: list[str] = []
    mapping: dict[str, str] = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            code = row[""]
            if not code or code in codes:
                raise LabelValidationError("Empty or duplicate code in SCP registry")
            codes.append(code)
            if row["diagnostic"] == "1.0":
                superclass = row["diagnostic_class"]
                if superclass not in LABEL_ORDER:
                    raise LabelValidationError(f"Invalid diagnostic_class for {code}")
                mapping[code] = superclass
            elif row["diagnostic"] not in ("", "0.0"):
                raise LabelValidationError(f"Unexpected diagnostic flag for {code}")
    return {
        "schema_version": "1.0.0",
        "policy_version": POLICY_VERSION,
        "dataset": "PTB-XL",
        "dataset_version": DATASET_VERSION,
        "label_order": list(LABEL_ORDER),
        "encoding": "multi_hot",
        "mapping_source": SOURCE + "scp_statements.csv",
        "mapping_source_sha256": sha256(path),
        "confidence_policy": {
            "mode": "code_presence", "minimum": 0, "maximum": 100,
            "zero_means": "unknown_likelihood_included", "threshold": None,
        },
        "unknown_code_policy": "error",
        "known_non_diagnostic_policy": "validate_then_ignore",
        "empty_label_policy": "exclude",
        "normal_label_policy": "explicit_NORM_only_no_inferred_normal",
        "known_scp_codes": sorted(codes),
        "diagnostic_to_superclass": dict(sorted(mapping.items())),
    }


def safe_relative_path(name: str) -> Path:
    relative = PurePosixPath(name)
    if not name or relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name:
        raise ValueError(f"Unsafe source path: {name}")
    return Path(*relative.parts)


def download(raw_dir: Path, name: str, expected: str | None = None) -> dict[str, Any]:
    destination = raw_dir / safe_relative_path(name)
    url = DOWNLOAD_SOURCE + name
    if not destination.exists() or (expected is not None and sha256(destination) != expected):
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".part")
        try:
            request = Request(url, headers={"User-Agent": "signai-2.0-data-contract/0.1"})
            total = 0
            with urlopen(request, timeout=60) as response, temporary.open("wb") as target:
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > 20 * 1024 * 1024:
                        raise ValueError(f"Small-sample download exceeded 20 MiB: {name}")
                    target.write(chunk)
            if expected is not None and sha256(temporary) != expected:
                raise ValueError(f"Upstream SHA256 mismatch: {name}")
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    digest = sha256(destination)
    if expected is not None and digest != expected:
        raise ValueError(f"SHA256 mismatch: {name}")
    return {
        "relative_path": name, "source_url": url, "canonical_url": SOURCE + name,
        "bytes": destination.stat().st_size,
        "sha256": digest, "upstream_sha256": expected,
        "upstream_checksum_verified": expected is not None,
    }


def audit_metadata(path: Path, schema: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    labels = Counter({label: 0 for label in LABEL_ORDER})
    excluded_by_fold = Counter({str(fold): 0 for fold in range(1, 11)})
    confidence: Counter[str] = Counter()
    patient_folds: dict[str, int] = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            ecg_id, fold = int(row["ecg_id"]), int(row["strat_fold"])
            if ecg_id in seen or fold not in range(1, 11):
                raise LabelValidationError(f"Duplicate ecg_id or invalid fold: {ecg_id}")
            seen.add(ecg_id)
            patient = row["patient_id"]
            if not patient or patient_folds.setdefault(patient, fold) != fold:
                raise LabelValidationError(f"Missing patient or patient appears across folds: {ecg_id}")
            try:
                codes = parse_scp_codes(row["scp_codes"])
                vector = encode_labels(codes, schema)
            except LabelValidationError as exc:
                raise LabelValidationError(f"ecg_id={ecg_id}: {exc}") from exc
            confidence.update(str(float(value)) for value in codes.values())
            if vector is None:
                excluded_by_fold[str(fold)] += 1
            else:
                labels.update(label for label, value in zip(LABEL_ORDER, vector) if value)
            rows.append({
                "ecg_id": ecg_id, "strat_fold": fold, "scp_codes": codes,
                "filename_lr": row["filename_lr"], "label_vector": list(vector) if vector is not None else None,
                "excluded_reason": "no_diagnostic_superclass" if vector is None else None,
            })
    rows.sort(key=lambda row: row["ecg_id"])
    excluded = sum(excluded_by_fold.values())
    report = {
        "metadata_records": len(rows), "included_records": len(rows) - excluded,
        "excluded_records": excluded, "excluded_reason": "no_diagnostic_superclass",
        "excluded_by_strat_fold": dict(sorted(excluded_by_fold.items(), key=lambda pair: int(pair[0]))),
        "label_order": list(LABEL_ORDER), "positive_records_per_label": dict(labels),
        "multilabel_records": sum(row["label_vector"] is not None and sum(row["label_vector"]) > 1 for row in rows),
        "observed_likelihood_value_counts": dict(sorted(confidence.items(), key=lambda pair: float(pair[0]))),
        "unknown_scp_codes": [], "patient_fold_consistency": "passed",
    }
    return report, rows


def prepare(root: Path) -> dict[str, Any]:
    raw_dir = root / "data" / "raw" / "ptb-xl" / DATASET_VERSION
    raw_dir.mkdir(parents=True, exist_ok=True)
    provenance_path = root / "docs" / "provenance.json"
    manifest_digest = None
    if provenance_path.exists():
        previous = json.loads(provenance_path.read_text(encoding="utf-8"))
        if previous.get("dataset_version") != DATASET_VERSION:
            raise LabelValidationError("Existing provenance belongs to a different data version")
        manifest_digest = next(item["sha256"] for item in previous["files"] if item["relative_path"] == "SHA256SUMS.txt")
    files = [download(raw_dir, "SHA256SUMS.txt", manifest_digest)]
    # A previous locally pinned digest detects drift, but is not an independent upstream checksum.
    files[0]["upstream_sha256"] = None
    files[0]["upstream_checksum_verified"] = False
    checksums = {}
    for line in (raw_dir / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        checksums[name.removeprefix("*").removeprefix("./")] = digest
    for name in ("ptbxl_database.csv", "scp_statements.csv", "LICENSE.txt", "example_physionet.py"):
        files.append(download(raw_dir, name, checksums[name]))

    schema_path = root / "docs" / "label_schema.json"
    generated = schema_from_statements(raw_dir / "scp_statements.csv")
    if schema_path.exists() and json.loads(schema_path.read_text(encoding="utf-8")) != generated:
        raise LabelValidationError("Label schema drift: review and version the policy; refusing to overwrite")
    if not schema_path.exists():
        write_json(schema_path, generated)
    schema = load_schema(schema_path)
    report, rows = audit_metadata(raw_dir / "ptbxl_database.csv", schema)

    # A demonstrative sample, not a random subset for model evaluation.
    selectors = {
        "normal": lambda row: row["label_vector"] == [1, 0, 0, 0, 0],
        "multilabel": lambda row: row["label_vector"] is not None and sum(row["label_vector"]) > 1,
        "excluded": lambda row: row["label_vector"] is None,
    }
    sample = []
    for role, select in selectors.items():
        row = next((row for row in rows if select(row)), None)
        if row is None:
            raise LabelValidationError(f"No sample candidate for {role}")
        record = row.copy()
        record["sample_role"] = role
        sample.append(record)
        base = record["filename_lr"]
        if not base.startswith("records100/"):
            raise LabelValidationError("Expected 100 Hz waveform path")
        for extension in (".hea", ".dat"):
            name = base + extension
            files.append(download(raw_dir, name, checksums[name]))
        header = (raw_dir / safe_relative_path(base + ".hea")).read_text(encoding="utf-8").splitlines()
        specification = header[0].split()
        if specification[1:4] != ["12", "100", "1000"]:
            raise LabelValidationError(f"Unexpected waveform header: {base}")

    processed = root / "data" / "processed" / "ptb-xl" / DATASET_VERSION / POLICY_VERSION
    processed.mkdir(parents=True, exist_ok=True)
    write_json(processed / "label_audit.json", report)
    write_json(processed / "sample_labels.json", sample)
    label_path = processed / "labels.jsonl"
    with label_path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            if row["label_vector"] is not None:
                stream.write(json.dumps({key: row[key] for key in ("ecg_id", "strat_fold", "label_vector")}, sort_keys=True) + "\n")
    provenance = {
        "provenance_version": "1.0.0", "dataset": "PTB-XL", "dataset_version": DATASET_VERSION,
        "dataset_doi": "10.13026/kfzx-aw45", "source_page": f"https://physionet.org/content/ptb-xl/{DATASET_VERSION}/",
        "download_mirror": DOWNLOAD_SOURCE,
        "license": {"id": "CC-BY-4.0", "url": "https://creativecommons.org/licenses/by/4.0/"},
        "retrieved_or_verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_directory": raw_dir.relative_to(root).as_posix(),
        "label_policy_version": POLICY_VERSION, "label_schema_sha256": sha256(schema_path),
        "generator": {
            "module": "signai.ptbxl", "source_sha256": sha256(Path(__file__)),
            "python_version": platform.python_version(),
        },
        "files": sorted(files, key=lambda item: item["relative_path"]),
        "checksum_trust": "Downloaded SHA256SUMS over HTTPS; manifest itself has no independently pinned upstream digest.",
        "metadata_audit": report,
        "sample_selection": "Lowest ecg_id for each of normal-only, multilabel, and excluded; 100 Hz waveform/header pairs.",
        "sample_records": [{key: row[key] for key in ("ecg_id", "sample_role", "label_vector", "excluded_reason", "filename_lr")} for row in sample],
        "derived_artifacts": [{
            "path": path.relative_to(root).as_posix(), "sha256": sha256(path), "bytes": path.stat().st_size,
        } for path in (label_path, processed / "label_audit.json", processed / "sample_labels.json")],
        "transformations": "Raw source files unchanged. Diagnostic superclass presence mapping; empty targets excluded from labels.jsonl. Only three waveforms downloaded.",
    }
    write_json(root / "docs" / "provenance.json", provenance)
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Project root containing docs/")
    args = parser.parse_args()
    root = args.root.resolve()
    if not (root / "pyproject.toml").is_file():
        parser.error("--root must identify the project root")
    provenance = prepare(root)
    print(json.dumps({"audit": provenance["metadata_audit"], "samples": provenance["sample_records"]}, indent=2))


if __name__ == "__main__":
    main()
