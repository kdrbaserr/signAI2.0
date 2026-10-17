"""Local serving entrypoint; callers load the model and its preprocessing bundle."""

from pathlib import Path

from signai.preprocessing import Preprocessor, aggregate_record_probabilities
from signai.ptbxl import LABEL_ORDER


def load_preprocessing_bundle(adapter_directory: Path) -> Preprocessor:
    """Read the frozen config, scaler and schema saved with a trained adapter."""
    return Preprocessor(adapter_directory / "preprocessing.json", adapter_directory / "scaler.json",
                        adapter_directory / "signal_schema.json")


def predict_record(model, preprocessor: Preprocessor, npz_path: Path, metadata_path: Path,
                   *, device="cpu") -> dict:
    import torch

    batch = preprocessor.inference_input(npz_path, metadata_path)
    model.eval()
    with torch.inference_mode():
        logits = model(**batch.torch_inputs(device)).logits.detach().cpu().numpy()
    probabilities = aggregate_record_probabilities(logits, batch)
    return {"record_id": batch.record_id, "label_order": list(LABEL_ORDER),
            "probabilities": probabilities.tolist(), "scaler_sha256": batch.scaler_sha256,
            "aggregation": preprocessor.config["aggregation"]}
