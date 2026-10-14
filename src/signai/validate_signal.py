"""Read-only validation of the versioned NPZ + JSON ECG upload contract."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sys
from typing import Any
import zipfile

import numpy as np

from signai.ptbxl import sha256

CHANNELS = ("I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6")


class SignalValidationError(ValueError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


def reject(code: str, message: str) -> None:
    raise SignalValidationError(code, message)


def unique_json_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            reject("E_METADATA", f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: Path, limit: int = 16384) -> dict[str, Any]:
    try:
        with path.open("rb") as stream:
            payload = stream.read(limit + 1)
        if len(payload) > limit:
            reject("E_SIZE", "JSON exceeds upload size limit")
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=unique_json_pairs)
    except (OSError, UnicodeError, ValueError) as exc:
        if isinstance(exc, SignalValidationError):
            raise
        reject("E_METADATA", f"Cannot read JSON: {exc}")
    if not isinstance(value, dict):
        reject("E_METADATA", "JSON must be an object")
    return value


def load_signal_schema(path: Path) -> dict[str, Any]:
    schema = read_json(path)
    fixed = {
        "schema_version": "1.0.0", "policy_version": "ptbxl-100hz-12lead-mv-v1",
        "npz_array_key": "signal", "axis_order": ["time", "channel"],
        "shape": [1000, 12], "channels": list(CHANNELS), "unit": "mV",
        "sampling_rate_hz": 100, "duration_seconds": 10.0,
        "allowed_dtypes": ["float32", "float64"], "finite_values_only": True,
        "max_abs_amplitude_mv": 20.0, "max_npz_bytes": 1048576,
        "max_uncompressed_bytes": 1048576, "max_json_bytes": 16384,
        "required_metadata": ["schema_version", "record_id", "channels", "unit", "sampling_rate_hz", "duration_seconds", "axis_order", "npz_sha256"],
    }
    if any(schema.get(key) != value for key, value in fixed.items()):
        reject("E_SCHEMA", "Unsupported signal policy; version and implementation must change together")
    return schema


def finite_number(value: Any) -> bool:
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def validate_metadata(metadata: dict[str, Any], schema: dict[str, Any]) -> None:
    if any(key not in metadata for key in schema["required_metadata"]):
        reject("E_METADATA", "Missing required metadata field")
    if metadata["schema_version"] != schema["schema_version"]:
        reject("E_SCHEMA", "Unsupported metadata schema version")
    if not isinstance(metadata["record_id"], str) or not metadata["record_id"].strip():
        reject("E_METADATA", "record_id must be a nonempty string")
    if metadata["axis_order"] != schema["axis_order"]:
        reject("E_AXIS_ORDER", "Expected axis order: time, channel")
    channels = metadata["channels"]
    if not isinstance(channels, list) or len(channels) != 12:
        reject("E_CHANNEL_COUNT", "Exactly 12 channel names are required")
    if any(not isinstance(channel, str) for channel in channels) or set(channels) != set(CHANNELS):
        reject("E_CHANNEL_SET", "Missing, duplicate or unknown channel name")
    if channels != list(CHANNELS):
        reject("E_CHANNEL_ORDER", "Channel order differs from the signal schema")
    if metadata["unit"] != "mV":
        reject("E_UNIT", "Physical amplitudes must be declared in mV")
    if not finite_number(metadata["sampling_rate_hz"]) or metadata["sampling_rate_hz"] != 100:
        reject("E_SAMPLE_RATE", "Only 100 Hz input is supported")
    if not finite_number(metadata["duration_seconds"]) or not math.isclose(metadata["duration_seconds"], 10.0, rel_tol=0, abs_tol=1e-9):
        reject("E_DURATION", "Only a 10-second record is supported")
    if not isinstance(metadata["npz_sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", metadata["npz_sha256"]):
        reject("E_CHECKSUM", "npz_sha256 must be a lowercase SHA-256 hex digest")


def validate_layout(shape: tuple[int, ...], dtype: np.dtype, schema: dict[str, Any]) -> None:
    if len(shape) != 2:
        reject("E_SHAPE", "Expected a two-dimensional time/channel array")
    if shape[1] != 12:
        reject("E_CHANNEL_COUNT", "Signal array must contain 12 channels")
    if shape[0] != 1000:
        reject("E_DURATION", "100 Hz x 10 seconds requires 1000 samples")
    if dtype.name not in schema["allowed_dtypes"]:
        reject("E_DTYPE", "Only real float32/float64 physical amplitudes are supported")


def read_npz(path: Path, schema: dict[str, Any]) -> tuple[np.ndarray, str]:
    try:
        with path.open("rb") as stream:
            payload = stream.read(schema["max_npz_bytes"] + 1)
    except OSError as exc:
        reject("E_ARCHIVE", f"Cannot read NPZ: {exc}")
    if len(payload) > schema["max_npz_bytes"]:
        reject("E_SIZE", "NPZ exceeds upload size limit")
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            members = archive.infolist()
            if len(members) != 1 or members[0].filename != "signal.npy":
                reject("E_ARCHIVE", "NPZ must contain only signal.npy")
            member = members[0]
            if member.file_size > schema["max_uncompressed_bytes"]:
                reject("E_SIZE", "Uncompressed signal exceeds size limit")
            # Validate NPY shape/dtype before numpy can allocate from an untrusted header.
            with archive.open(member) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    reject("E_ARCHIVE", "Only NPY format versions 1.0 and 2.0 are supported")
                validate_layout(shape, dtype, schema)
                if stream.tell() + math.prod(shape) * dtype.itemsize != member.file_size:
                    reject("E_ARCHIVE", "NPY header/data length mismatch")
        with np.load(io.BytesIO(payload), allow_pickle=False) as arrays:
            signal = arrays["signal"]
    except (ValueError, OSError, EOFError, zipfile.BadZipFile, RuntimeError, NotImplementedError) as exc:
        if isinstance(exc, SignalValidationError):
            raise
        reject("E_ARCHIVE", f"Invalid NPZ: {exc}")
    return signal, hashlib.sha256(payload).hexdigest()


def validate_signal(npz_path: Path, metadata_path: Path, schema_path: Path) -> dict[str, Any]:
    """Never writes, rescales, reorders, clips or resamples either input file."""
    schema = load_signal_schema(schema_path)
    metadata = read_json(metadata_path, schema["max_json_bytes"])
    validate_metadata(metadata, schema)
    signal, digest = read_npz(npz_path, schema)
    if digest != metadata["npz_sha256"]:
        reject("E_CHECKSUM", "NPZ bytes differ from metadata digest")
    if not np.isfinite(signal).all():
        reject("E_NONFINITE", "Signal contains NaN or infinity")
    peaks = np.max(np.abs(signal), axis=0)
    if (peaks > schema["max_abs_amplitude_mv"]).any():
        reject("E_AMPLITUDE", "Absolute amplitude exceeds the project's 20 mV guardrail")
    return {
        "valid": True, "record_id": metadata["record_id"],
        "policy_version": schema["policy_version"], "shape": list(signal.shape),
        "dtype": signal.dtype.name, "axis_order": metadata["axis_order"],
        "channels": metadata["channels"], "unit": "mV", "sampling_rate_hz": 100,
        "duration_seconds": signal.shape[0] / 100, "max_abs_mv_per_channel": peaks.tolist(),
        "npz_sha256": digest, "metadata_sha256": sha256(metadata_path),
        "signal_schema_sha256": sha256(schema_path), "input_files_modified": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--schema", type=Path, default=Path("docs/signal_schema.json"))
    args = parser.parse_args()
    try:
        result = validate_signal(args.npz, args.metadata, args.schema)
    except SignalValidationError as error:
        print(json.dumps({"valid": False, "error_code": error.code, "message": str(error)}))
        sys.exit(1)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
