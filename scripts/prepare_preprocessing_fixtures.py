"""Create tiny synthetic goldens with analytic expectations, independent of preprocessing.py."""

import json
from pathlib import Path

import numpy as np

from signai.ptbxl import sha256
from signai.validate_signal import CHANNELS


def write_json(path, value):
    # Hash-pinned JSON must keep identical bytes after cross-platform Git checkout.
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def main():
    root = Path(__file__).resolve().parents[1]
    destination = root / "tests/fixtures/preprocessing"
    destination.mkdir(parents=True, exist_ok=True)
    offsets = np.arange(12, dtype=np.float64) / 8 - 0.5
    units = (np.arange(12, dtype=np.float64) + 1) / 32
    signs = np.where(np.arange(1000) < 500, -1.0, 1.0)
    fit = offsets + signs[:, None] * (2 * units)
    # Different patterns by lead, with exactly representable binary fractions.
    ticks = (np.arange(1000)[:, None] + np.arange(12)[None, :]) % 9 - 4
    signal = offsets + ticks * units
    for ecg_id, name, values in ((1, "fit", fit), (2, "input", signal)):
        path = destination / f"{name}.npz"
        np.savez_compressed(path, signal=values)
        write_json(path.with_suffix(".json"), {
            "schema_version": "1.0.0", "record_id": f"ptbxl_1_0_3_{ecg_id:05}",
            "axis_order": ["time", "channel"], "channels": list(CHANNELS),
            "unit": "mV", "sampling_rate_hz": 100, "duration_seconds": 10.0,
            "npz_sha256": sha256(path), "source": {"dataset_version": "1.0.3", "ecg_id": ecg_id},
        })
    # Analytically: fit mean=offset, population std=2*unit, input z=tick/2.
    expected = np.zeros((2, 12, 512), dtype=np.float32)
    expected[0] = (ticks[:512] / 2).T
    expected[1, :, :488] = (ticks[512:] / 2).T
    mask = np.ones((2, 512), dtype=np.int64)
    mask[1, 488:] = 0
    np.savez_compressed(destination / "expected.npz", x_enc=expected, input_mask=mask,
                        mean=offsets, scale=2 * units)
    write_json(destination / "golden.json", {
        "fixture_type": "synthetic_not_patient_data", "policy_version": "ptbxl-moment-preprocessing-v1",
        "formula": "offset=lead/8-0.5; unit=(lead+1)/32; fit=offset +/- 2*unit; input=offset+((time+lead)%9-4)*unit",
        "starts": [0, 512], "valid_samples": [512, 488], "padding_samples": [0, 24],
        "aggregation_logits": [[0, 0, 0, 0, 0], [float(np.log(3)), float(-np.log(3)), 0, 1000, -1000]],
        "expected_record_probabilities": [0.622, 0.378, 0.5, 0.744, 0.256],
        "sha256": {name: sha256(destination / name) for name in ("fit.npz", "fit.json", "input.npz", "input.json", "expected.npz")},
    })
    print(json.dumps({"destination": str(destination), "synthetic": True}))


if __name__ == "__main__":
    main()
