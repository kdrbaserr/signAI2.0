"""Independent goldens, train/serve parity, split lineage and tail contracts."""

from dataclasses import replace
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

from signai.inference import load_preprocessing_bundle, predict_record
from signai.preprocessing import (
    DEFAULT_CONFIG, PreprocessingError, Preprocessor, aggregate_record_probabilities,
    fingerprint, fit_scaler, load_config, validate_fit_lineage,
)
from signai.ptbxl import sha256, write_json
from signai.splits import SplitIntegrityError, build_manifests
from signai.validate_signal import SignalValidationError

ROOT = Path(__file__).parents[1]
GOLDEN = ROOT / "tests/fixtures/preprocessing"
CONFIG = ROOT / DEFAULT_CONFIG
SCHEMA = ROOT / "docs/signal_schema.json"


def parent(ecg_id, fold, included=True):
    return {"ecg_id": ecg_id, "patient_id": ecg_id + 100, "strat_fold": fold,
            "split": "train" if fold < 9 else "validation" if fold == 9 else "test",
            "filename_lr": f"records100/00000/{ecg_id:05}_lr",
            "label_vector": [1, 0, 0, 0, 0] if included else None,
            "excluded_reason": None if included else "no_diagnostic_superclass"}


@pytest.fixture
def environment(tmp_path):
    splits = tmp_path / "splits"
    manifests = build_manifests([parent(1, 1), parent(2, 2), parent(39, 9), parent(3, 10), parent(4, 8, False)],
                                {"metadata_sha256": "synthetic", "label_schema_sha256": "synthetic"})
    for split, manifest in manifests.items():
        write_json(splits / f"{split}.json", manifest)
    write_json(splits / "summary.json", {"manifest_sha256": {split: sha256(splits / f"{split}.json") for split in manifests}})
    signals = tmp_path / "signals"
    signals.mkdir()
    for ecg_id, name in ((1, "fit"), (2, "input")):
        for suffix in ("npz", "json"):
            shutil.copyfile(GOLDEN / f"{name}.{suffix}", signals / f"ptbxl_1_0_3_{ecg_id:05}.{suffix}")
    return SimpleNamespace(root=tmp_path, splits=splits, signals=signals, scaler=tmp_path / "scaler.json")


def fit(env, ids=None):
    return fit_scaler([1] if ids is None else ids, split_directory=env.splits,
                      signal_directory=env.signals, config_path=CONFIG, schema_path=SCHEMA)


def processor(env):
    state = fit(env)
    write_json(env.scaler, state)
    return Preprocessor(CONFIG, env.scaler, SCHEMA)


def training_batch(pre, env):
    return pre.training_input(2, split_directory=env.splits, signal_directory=env.signals)


def test_fixture_files_match_reviewed_hashes():
    golden = json.loads((GOLDEN / "golden.json").read_text())
    for name, expected in golden["sha256"].items():
        assert sha256(GOLDEN / name) == expected


def test_fit_statistics_have_independent_analytic_expectation(environment):
    state = fit(environment)
    with np.load(GOLDEN / "expected.npz", allow_pickle=False) as expected:
        np.testing.assert_array_equal(state["mean"], expected["mean"])
        np.testing.assert_array_equal(state["scale"], expected["scale"])
    assert state["sample_count_per_channel"] == 1000  # Padding never enters fit.
    assert state["fit_scope"] == "explicit_train_subset"
    assert [item["ecg_id"] for item in state["fit_records"]] == [1]


def test_training_and_inference_equal_independent_full_tensor_golden(environment):
    pre = processor(environment)
    train = training_batch(pre, environment)
    serve = pre.inference_input(GOLDEN / "input.npz", GOLDEN / "input.json")
    with np.load(GOLDEN / "expected.npz", allow_pickle=False) as expected:
        for batch in (train, serve):
            np.testing.assert_array_equal(batch.x_enc, expected["x_enc"])
            np.testing.assert_array_equal(batch.input_mask, expected["input_mask"])
            assert batch.x_enc.dtype == np.float32
            assert batch.input_mask.dtype == np.int64
            assert batch.starts == (0, 512)
            assert batch.valid_samples == (512, 488)
    assert train.parent["split"] == "train"
    assert train.parent["ecg_id"] == 2
    assert train.scaler_sha256 == serve.scaler_sha256


def test_padding_is_zero_after_normalization_and_only_tail_is_masked(environment):
    batch = training_batch(processor(environment), environment)
    assert batch.input_mask.sum(axis=1).tolist() == [512, 488]
    assert np.all(batch.x_enc[1, :, 488:] == 0)
    assert np.all(batch.input_mask[0] == 1)
    assert np.all(batch.input_mask[1, :488] == 1)
    assert np.all(batch.input_mask[1, 488:] == 0)


@pytest.mark.parametrize("ids", [[1, 39], [1, 3], [1, 4], [1, 999]])
def test_forbidden_fit_ids_fail_before_any_waveform_is_read(environment, monkeypatch, ids):
    def must_not_read(*args):
        pytest.fail("Forbidden fit must stop before reading train OR validation signals")
    monkeypatch.setattr("signai.preprocessing._load_training_signal", must_not_read)
    with pytest.raises(SplitIntegrityError, match="Forbidden scaler fit IDs"):
        fit(environment, ids)


def test_validation_cannot_hide_behind_forged_train_label(environment):
    state = fit(environment)
    # Add a validation ID but claim it was train. Canonical membership wins.
    state["fit_records"].append(dict(state["fit_records"][0], ecg_id=39))
    with pytest.raises(SplitIntegrityError, match="Forbidden or duplicate"):
        validate_fit_lineage(state, environment.splits)


@pytest.mark.parametrize("ids", [[], [1, 1], [True]])
def test_invalid_fit_id_lists_are_rejected(environment, ids):
    with pytest.raises(SplitIntegrityError):
        fit(environment, ids)


def test_full_train_fit_is_order_independent_and_records_coverage(environment):
    left, right = fit(environment, [2, 1]), fit(environment, [1, 2])
    assert left == right
    assert left["fit_scope"] == "full_train"
    assert left["sample_count_per_channel"] == 2000


def test_false_full_train_coverage_claim_is_rejected(environment):
    state = fit(environment)
    state["fit_scope"] = "full_train"
    with pytest.raises(SplitIntegrityError, match="coverage"):
        validate_fit_lineage(state, environment.splits)


def test_deterministic_transform_does_not_mutate_sources_or_scaler(environment):
    pre = processor(environment)
    before = {path: sha256(path) for path in [environment.scaler, *environment.signals.iterdir()]}
    expected = training_batch(pre, environment).x_enc.copy()
    for _ in range(3):
        np.testing.assert_array_equal(training_batch(pre, environment).x_enc, expected)
    assert before == {path: sha256(path) for path in before}
    assert fingerprint(pre._state) == pre.scaler_sha256


@pytest.mark.parametrize("entry", ["train", "serve"])
def test_channel_swap_is_rejected_in_both_paths(environment, entry):
    pre = processor(environment)
    metadata = environment.signals / "ptbxl_1_0_3_00002.json"
    value = json.loads(metadata.read_text())
    value["channels"][0], value["channels"][1] = value["channels"][1], value["channels"][0]
    write_json(metadata, value)
    with pytest.raises(SignalValidationError) as error:
        if entry == "train":
            training_batch(pre, environment)
        else:
            pre.inference_input(metadata.with_suffix(".npz"), metadata)
    assert error.value.code == "E_CHANNEL_ORDER"


def test_validation_transforms_without_refitting_and_test_is_blocked(environment):
    pre = processor(environment)
    for suffix in ("npz", "json"):
        shutil.copyfile(GOLDEN / f"input.{suffix}", environment.signals / f"ptbxl_1_0_3_00039.{suffix}")
    path = environment.signals / "ptbxl_1_0_3_00039.json"
    value = json.loads(path.read_text())
    value["record_id"] = path.stem
    value["source"]["ecg_id"] = 39
    write_json(path, value)
    before = pre.scaler_sha256
    val = pre.training_input(39, split_directory=environment.splits, signal_directory=environment.signals, split="validation")
    assert val.parent["split"] == "validation"
    assert pre.scaler_sha256 == before
    with pytest.raises(SplitIntegrityError, match="train/validation only"):
        pre.training_input(3, split_directory=environment.splits, signal_directory=environment.signals, split="test")


def test_fit_artifact_drift_is_rejected(environment):
    pre = processor(environment)
    metadata = environment.signals / "ptbxl_1_0_3_00001.json"
    value = json.loads(metadata.read_text())
    value["comment"] = "changed after fit"
    write_json(metadata, value)
    with pytest.raises(SplitIntegrityError, match="differs from scaler fit lineage"):
        pre.training_input(1, split_directory=environment.splits, signal_directory=environment.signals)


def test_scaler_manifest_binding_and_config_drift_are_rejected(environment):
    state = fit(environment)
    state["train_manifest_sha256"] = "0" * 64
    with pytest.raises(SplitIntegrityError, match="manifest checksum"):
        validate_fit_lineage(state, environment.splits)
    state["config_sha256"] = "0" * 64
    write_json(environment.scaler, state)
    with pytest.raises(PreprocessingError, match="version mismatch"):
        Preprocessor(CONFIG, environment.scaler, SCHEMA)
    config = load_config(CONFIG)
    config["window_samples"] = 256
    path = environment.root / "config.json"
    write_json(path, config)
    with pytest.raises(PreprocessingError, match="Unsupported preprocessing config"):
        load_config(path)


def test_zero_variance_channel_has_finite_transform(environment):
    npz = environment.signals / "ptbxl_1_0_3_00001.npz"
    np.savez_compressed(npz, signal=np.zeros((1000, 12), dtype=np.float32))
    metadata = json.loads(npz.with_suffix(".json").read_text())
    metadata["npz_sha256"] = sha256(npz)
    write_json(npz.with_suffix(".json"), metadata)
    pre = processor(environment)
    np.testing.assert_array_equal(pre.scale, np.full(12, 1e-8))
    assert np.isfinite(training_batch(pre, environment).x_enc).all()


def test_probability_aggregation_matches_hand_calculated_multilabel_golden(environment):
    golden = json.loads((GOLDEN / "golden.json").read_text())
    batch = training_batch(processor(environment), environment)
    result = aggregate_record_probabilities(golden["aggregation_logits"], batch)
    np.testing.assert_allclose(result, golden["expected_record_probabilities"], rtol=0, atol=1e-7)
    assert not np.isclose(result.sum(), 1)  # Independent labels, not softmax.


@pytest.mark.parametrize("change", ["missing", "reordered", "mask", "nan", "shape"])
def test_aggregation_rejects_invalid_window_outputs(environment, change):
    batch = training_batch(processor(environment), environment)
    logits = np.zeros((2, 5))
    if change == "missing":
        batch = replace(batch, starts=(0,), valid_samples=(512,))
    elif change == "reordered":
        batch = replace(batch, starts=(512, 0))
    elif change == "mask":
        batch.input_mask[1, 500] = 1
    elif change == "nan":
        logits[0, 0] = np.nan
    else:
        logits = np.zeros((2, 4))
    with pytest.raises(PreprocessingError):
        aggregate_record_probabilities(logits, batch)


def test_serving_passes_identical_tensors_and_mask_to_model(environment):
    torch = pytest.importorskip("torch")
    pre = processor(environment)
    expected = training_batch(pre, environment).torch_inputs()

    class CaptureModel(torch.nn.Module):
        def forward(self, *, x_enc, input_mask):
            assert not self.training
            assert not torch.is_grad_enabled()
            assert x_enc.dtype == torch.float32 and input_mask.dtype == torch.int64
            assert torch.equal(x_enc, expected["x_enc"])
            assert torch.equal(input_mask, expected["input_mask"])
            return SimpleNamespace(logits=torch.zeros((2, 5)))

    result = predict_record(CaptureModel(), pre, GOLDEN / "input.npz", GOLDEN / "input.json")
    assert result["probabilities"] == [0.5] * 5
    assert result["scaler_sha256"] == pre.scaler_sha256


def test_serialized_adapter_preprocessing_bundle_preserves_model_input(environment):
    pre = processor(environment)
    bundle = environment.root / "adapter"
    bundle.mkdir()
    for source, name in ((CONFIG, "preprocessing.json"), (environment.scaler, "scaler.json"),
                         (SCHEMA, "signal_schema.json")):
        shutil.copyfile(source, bundle / name)
    restored = load_preprocessing_bundle(bundle)
    actual = restored.inference_input(GOLDEN / "input.npz", GOLDEN / "input.json")
    expected = training_batch(pre, environment)
    np.testing.assert_array_equal(actual.x_enc, expected.x_enc)
    np.testing.assert_array_equal(actual.input_mask, expected.input_mask)
    assert actual.scaler_sha256 == expected.scaler_sha256


def test_mask_is_patch_aligned_for_pinned_moment(environment):
    pytest.importorskip("torch")
    masking = pytest.importorskip("momentfm.utils.masking")
    mask = training_batch(processor(environment), environment).torch_inputs()["input_mask"]
    patch_mask = masking.Masking.convert_seq_to_patch_view(mask, 8)
    assert patch_mask.shape == (2, 64)
    assert patch_mask.sum(dim=1).tolist() == [64, 61]
