"""One real MOMENT + PEFT step; a compatibility probe, not model training."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import random
import re
import time
from typing import Any

import numpy as np
import torch
from torch import nn

from signai.ptbxl import LABEL_ORDER, sha256, write_json
from signai.moment_adapter import MomentClassifier
from signai.splits import load_training_inputs
from signai.validate_signal import validate_signal


class ModelProbeError(ValueError):
    pass


def select_lora_targets(model: nn.Module, suffixes: list[str]) -> list[str]:
    if not suffixes or len(suffixes) != len(set(suffixes)):
        raise ModelProbeError("Target suffixes must be nonempty and unique")
    selected = []
    for suffix in suffixes:
        matches = [
            name for name, module in model.named_modules()
            if name.startswith("moment.encoder.") and ".SelfAttention." in name
            and name.endswith("." + suffix) and isinstance(module, nn.Linear)
        ]
        if not matches:
            raise ModelProbeError(f"Zero Linear target matches for '{suffix}'; optimizer must not start")
        selected.extend(matches)
    return sorted(selected)


def parameter_hashes(parameters) -> dict[str, str]:
    return {
        name: hashlib.sha256(parameter.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        for name, parameter in parameters
    }


def group_hash(hashes: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def assert_trainable_policy(model: nn.Module) -> dict[str, list[str]]:
    groups = {"lora": [], "head": [], "frozen": []}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            groups["frozen"].append(name)
        elif ".lora_" in name:
            groups["lora"].append(name)
        elif ".head.modules_to_save.default." in name:
            groups["head"].append(name)
        else:
            raise ModelProbeError(f"Unexpected trainable base parameter: {name}")
    if not all(groups.values()):
        raise ModelProbeError("Expected nonempty LoRA, head and frozen parameter groups")
    return groups


def run_probe(root: Path, device_name: str = "cuda", target_override: list[str] | None = None) -> dict[str, Any]:
    from momentfm import MOMENTPipeline
    from peft import LoraConfig, get_peft_model

    config_path = root / "docs" / "moment_probe_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not re.fullmatch(r"[0-9a-f]{40}", config["checkpoint_revision"]):
        raise ModelProbeError("Checkpoint revision must be an immutable commit SHA")
    if config["optimizer_steps"] != 1 or config["channels"] != 12 or config["num_labels"] != 5:
        raise ModelProbeError("This probe supports one optimizer step, 12 channels and 5 labels")
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if device_name == "cuda" and not torch.cuda.is_available():
        raise ModelProbeError("CUDA requested but unavailable; use explicit --device cpu for a CPU probe")
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config["seed"])
        torch.cuda.reset_peak_memory_stats(device)

    checkpoint = root / "artifacts" / "moment-probe" / "checkpoint"
    upstream_manifest = json.loads((root / "artifacts" / "moment-probe" / "source" / "checkpoint_manifest.json").read_text(encoding="utf-8-sig"))
    if upstream_manifest["sha"] != config["checkpoint_revision"]:
        raise ModelProbeError("Downloaded checkpoint revision differs from probe configuration")
    weight_info = next(item for item in upstream_manifest["siblings"] if item["rfilename"] == "model.safetensors")
    weight_hash = sha256(checkpoint / "model.safetensors")
    if weight_hash != weight_info["lfs"]["sha256"]:
        raise ModelProbeError("Checkpoint weights fail upstream SHA256 verification")
    checkpoint_config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    # The pinned config blob was obtained from the same checkpoint revision.
    if sha256(checkpoint / "config.json") != sha256(root / "artifacts" / "moment-probe" / "source" / "checkpoint_config.json"):
        raise ModelProbeError("Checkpoint configuration differs from pinned source copy")
    context = config["context_samples"]
    if context != checkpoint_config["seq_len"] or context % checkpoint_config["patch_len"]:
        raise ModelProbeError("Probe context must match checkpoint and be divisible by patch length")

    train = load_training_inputs(root / "splits")["train"]
    record = next((record for record in train if record["ecg_id"] == config["sample_ecg_id"]), None)
    if record is None:
        raise ModelProbeError("Probe sample must be an included training record")
    sample = root / "data" / "validated" / "ptb-xl" / "1.0.3" / f"ptbxl_1_0_3_{record['ecg_id']:05}"
    signal_check = validate_signal(sample.with_suffix(".npz"), sample.with_suffix(".json"), root / "docs" / "signal_schema.json")
    with np.load(sample.with_suffix(".npz"), allow_pickle=False) as arrays:
        signal = arrays["signal"]
    start = config["window_start_sample"]
    if type(start) is not int or not 0 <= start <= signal.shape[0] - context:
        raise ModelProbeError("Window boundaries exceed the source record")
    x = torch.from_numpy(signal[start:start + context].T.copy()).float().unsqueeze(0).to(device)
    mask = torch.ones((1, context), dtype=torch.long, device=device)
    target = torch.tensor([record["label_vector"]], dtype=torch.float32, device=device)

    model_kwargs = {
        "task_name": "classification", "n_channels": 12, "num_class": 5,
        "freeze_encoder": True, "freeze_embedder": True, "freeze_head": False,
        "enable_gradient_checkpointing": False, "head_dropout": 0.0, "patch_dropout": 0.0,
    }
    moment = MOMENTPipeline.from_pretrained(str(checkpoint), model_kwargs=model_kwargs.copy(), local_files_only=True)
    moment.init()
    model = MomentClassifier(moment)
    model.name_or_path = config["model_id"]
    targets = select_lora_targets(model, target_override or config["target_suffixes"])
    reports = root / "reports" / "W01-D05"
    reports.mkdir(parents=True, exist_ok=True)
    module_lines = [
        f"{name or '<root>'}\t{type(module).__name__}\t"
        f"{sum(parameter.numel() for parameter in module.parameters(recurse=False))}\t"
        f"{'LORA_TARGET' if name in targets else ''}"
        for name, module in model.named_modules()
    ]
    (reports / "module_map.txt").write_text("name\ttype\tdirect_parameters\tselection\n" + "\n".join(module_lines) + "\n", encoding="utf-8")
    peft_config = LoraConfig(
        r=config["lora_rank"], lora_alpha=config["lora_alpha"],
        revision=config["checkpoint_revision"], base_model_name_or_path=config["model_id"],
        lora_dropout=config["lora_dropout"], bias="none",
        target_modules=targets, modules_to_save=["moment.head"],
    )
    model = get_peft_model(model, peft_config).to(device)
    groups = assert_trainable_policy(model)
    parameters = dict(model.named_parameters())
    before = {group: parameter_hashes((name, parameters[name]) for name in names) for group, names in groups.items()}
    optimizer = torch.optim.AdamW([parameter for parameter in parameters.values() if parameter.requires_grad], lr=config["learning_rate"], weight_decay=0.0)
    model.train()
    begin = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    outputs = model(x_enc=x, input_mask=mask)
    logits = outputs.logits
    if tuple(logits.shape) != (1, 5) or not torch.isfinite(logits).all():
        raise ModelProbeError("Expected finite logits of shape [1, 5]")
    loss = nn.BCEWithLogitsLoss()(logits, target)
    if not torch.isfinite(loss):
        raise ModelProbeError("Loss is not finite")
    loss.backward()
    gradient_norms = {}
    for group in ("lora", "head"):
        for name in groups[group]:
            gradient = parameters[name].grad
            if gradient is None or not torch.isfinite(gradient).all():
                raise ModelProbeError(f"Missing/nonfinite gradient: {name}")
            gradient_norms[name] = gradient.detach().norm().item()
        if not any(gradient_norms[name] > 0 for name in groups[group]):
            raise ModelProbeError(f"No nonzero gradient in {group}")
    if any(parameters[name].grad is not None for name in groups["frozen"]):
        raise ModelProbeError("Frozen base received a gradient")
    optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - begin
    after = {group: parameter_hashes((name, parameters[name]) for name in names) for group, names in groups.items()}
    changed = {group: [name for name in groups[group] if before[group][name] != after[group][name]] for group in groups}
    if changed["frozen"] or not changed["lora"] or not changed["head"]:
        raise ModelProbeError("Weight update policy failed: base must stay fixed, adapter/head must change")

    try:
        select_lora_targets(MomentClassifier(moment), ["NOT_A_REAL_TARGET"])
    except ModelProbeError as error:
        negative = {"outcome": "rejected_before_optimizer", "error": str(error), "optimizer_steps": 0}
    else:
        raise ModelProbeError("Invalid target was not rejected")
    adapter_dir = root / "artifacts" / "moment-probe" / "smoke-adapter"
    # Base embedding weights are frozen and hash-verified; no vocabulary lookup is needed.
    model.save_pretrained(adapter_dir, safe_serialization=True, save_embedding_layers=False)
    report = {
        "status": "passed", "purpose": "one-step compatibility probe; no quality claim",
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": {"model_id": config["model_id"], "revision": config["checkpoint_revision"],
                       "weights_sha256": weight_hash, "config_sha256": sha256(checkpoint / "config.json"), "config": checkpoint_config},
        "moment_source_revision": config["moment_source_revision"],
        "probe_code_sha256": sha256(Path(__file__)), "probe_config_sha256": sha256(config_path),
        "wrapper_code_sha256": sha256(Path(__file__).with_name("moment_adapter.py")),
        "packages": {package: version(package) for package in ("torch", "momentfm", "peft", "transformers", "numpy", "huggingface-hub")},
        "python_version": platform.python_version(), "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cuda_runtime": torch.version.cuda,
        "input": {"shape": list(x.shape), "ecg_id": record["ecg_id"], "patient_id": record["patient_id"],
                  "split": record["split"], "strat_fold": record["strat_fold"], "start_sample": start,
                  "end_sample": start + context, "source_npz_sha256": signal_check["npz_sha256"], "label_order": list(LABEL_ORDER)},
        "context_samples": context, "internal_patch_samples": checkpoint_config["patch_len"],
        "internal_patch_count": context // checkpoint_config["patch_len"],
        "targets": targets, "lora": {"rank": config["lora_rank"], "alpha": config["lora_alpha"], "modules_to_save": ["moment.head"]},
        "parameter_counts": {group: sum(parameters[name].numel() for name in names) for group, names in groups.items()},
        "changed_parameter_tensors": {group: len(names) for group, names in changed.items()},
        "group_hashes_before": {group: group_hash(hashes) for group, hashes in before.items()},
        "group_hashes_after": {group: group_hash(hashes) for group, hashes in after.items()},
        "trainable_parameter_names": groups["lora"] + groups["head"],
        "gradient_norms": gradient_norms, "logits_shape": list(logits.shape), "logits": logits.detach().cpu().tolist(),
        "loss": loss.item(), "optimizer_steps": 1, "step_seconds": elapsed,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
        "negative_target_experiment": negative,
        "adapter_artifacts": {path.name: sha256(path) for path in sorted(adapter_dir.iterdir()) if path.is_file()},
        "license": config["license"], "license_source": config["license_source"],
        "pretraining_dataset": config["pretraining_dataset"], "ptbxl_pretraining_overlap": config["ptbxl_pretraining_overlap"],
        "integration_note": "A thin wrapper supplies PEFT's dict config while MOMENT keeps its Namespace config.",
        "limitations": ["Single training record, single 512-sample window, single optimizer step", "No model selection or final test evaluation", "Window inherits record label", "PTB-XL pretraining overlap is not verified absent"],
    }
    write_json(reports / "model_probe.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--targets", nargs="+")
    args = parser.parse_args()
    report = run_probe(args.root.resolve(), args.device, args.targets)
    print(json.dumps({key: report[key] for key in ("status", "device", "logits_shape", "loss", "changed_parameter_tensors", "peak_cuda_allocated_bytes", "negative_target_experiment")}, indent=2))


if __name__ == "__main__":
    main()
