"""Shared training and evaluation utilities for COBICount."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple
import csv
import math
import os
import random

import numpy as np
import torch

from .model import COBICount, ModelEMA523, V523BICConfig


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve ``auto``, ``cpu``, ``cuda``, or a concrete torch device string."""
    requested = str(requested).strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return device


def seed_everything(seed: int = 3407, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch.

    ``deterministic=True`` improves repeatability but may reduce throughput and
    does not guarantee bitwise-identical results across hardware or PyTorch
    versions.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    try:
        torch.use_deterministic_algorithms(deterministic, warn_only=True)
    except AttributeError:
        pass


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def dataloader_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def move_batch_to_device(batch: Mapping[str, Any], device: torch.device | str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device, non_blocking=True)
        elif key == "points":
            out[key] = [point.to(device, non_blocking=True) for point in value]
        else:
            out[key] = value
    return out


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    loader: Optional[Iterable[Mapping[str, Any]]],
    device: torch.device | str,
    max_batches: Optional[int] = None,
) -> Dict[str, Any]:
    """Compute count MAE/RMSE and retain per-image predictions."""
    if loader is None:
        return {"mae": float("nan"), "rmse": float("nan"), "n": 0, "pred": [], "gt": [], "names": []}

    model.eval()
    abs_errors: List[float] = []
    sq_errors: List[float] = []
    predictions: List[float] = []
    targets: List[float] = []
    names: List[str] = []

    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch_device = move_batch_to_device(batch, device)
        output = model(
            batch_device["image"],
            valid_mask=batch_device.get("valid_mask"),
            neutral_mask=batch_device.get("neutral_mask"),
        )
        pred = output["pred_count"].detach().view(-1).float()
        gt = torch.as_tensor(batch_device["count"], dtype=torch.float32, device=device).view(-1)
        error = pred - gt
        abs_errors.extend(error.abs().cpu().tolist())
        sq_errors.extend((error**2).cpu().tolist())
        predictions.extend(pred.cpu().tolist())
        targets.extend(gt.cpu().tolist())
        names.extend([str(name) for name in batch.get("name", [""] * len(pred))])

    if not abs_errors:
        return {"mae": float("nan"), "rmse": float("nan"), "n": 0, "pred": [], "gt": [], "names": []}

    return {
        "mae": float(np.mean(abs_errors)),
        "rmse": float(math.sqrt(np.mean(sq_errors))),
        "n": len(abs_errors),
        "pred": predictions,
        "gt": targets,
        "names": names,
    }


def _as_serializable_config(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    return value


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    ema: ModelEMA523,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    epoch: int,
    train_config: Any,
    model_config: V523BICConfig,
    loss_config: Any,
    best_source_val_mae: float,
) -> None:
    """Save a resumable source-only training checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "epoch": int(epoch),
        "model": model.state_dict(),
        "ema": ema.ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "train_config": _as_serializable_config(train_config),
        "model_config": _as_serializable_config(model_config),
        "loss_config": _as_serializable_config(loss_config),
        "best_source_val_mae": float(best_source_val_mae),
        "protocol": "strict_source_only",
    }
    torch.save(payload, path)


def _torch_load(path: str | Path, map_location: torch.device | str) -> Any:
    """Load a trusted local checkpoint across supported PyTorch versions."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_training_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    ema: ModelEMA523,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: Any,
    device: torch.device | str,
) -> Tuple[int, float]:
    checkpoint = _torch_load(path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise TypeError("A resumable training checkpoint must be a dictionary.")

    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"], strict=True)
    if "ema" in checkpoint:
        ema.ema.load_state_dict(checkpoint["ema"], strict=True)
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])

    return int(checkpoint.get("epoch", 0)), float(
        checkpoint.get("best_source_val_mae", checkpoint.get("best_rsoc_mae", float("inf")))
    )


def load_model_for_inference(
    checkpoint_path: str | Path,
    device: torch.device | str,
    *,
    prefer_ema: bool = True,
) -> COBICount:
    """Load a released or locally trained checkpoint for evaluation/inference."""
    checkpoint = _torch_load(checkpoint_path, map_location=device)

    model_config_dict: Dict[str, Any] = {}
    state_dict: Any = checkpoint
    if isinstance(checkpoint, dict):
        model_config_dict = checkpoint.get("model_config", checkpoint.get("model_cfg", {})) or {}
        state_dict = (
            checkpoint.get("ema") if prefer_ema else None
        ) or checkpoint.get("model") or checkpoint.get("state_dict") or checkpoint

    model_config = V523BICConfig(**model_config_dict) if model_config_dict else V523BICConfig()
    model = COBICount(model_config).to(device)
    incompat = model.load_state_dict(state_dict, strict=False)
    if incompat.missing_keys or incompat.unexpected_keys:
        missing = ", ".join(incompat.missing_keys) or "none"
        unexpected = ", ".join(incompat.unexpected_keys) or "none"
        raise RuntimeError(
            "Checkpoint architecture mismatch. "
            f"Missing keys: {missing}. Unexpected keys: {unexpected}."
        )
    model.eval()
    return model


def save_history_csv(history: Mapping[str, List[float]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(history.keys())
    lengths = {len(history[key]) for key in keys}
    if len(lengths) > 1:
        raise ValueError("All history columns must have the same length.")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(keys)
        for row in zip(*(history[key] for key in keys)):
            writer.writerow(row)


def load_history_csv(path: str | Path, keys: Iterable[str]) -> Dict[str, List[float]]:
    history = {key: [] for key in keys}
    path = Path(path)
    if not path.is_file():
        return history
    with path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for key in history:
                if key in row and row[key] not in (None, ""):
                    history[key].append(float(row[key]))
    return history


def find_latest_epoch_checkpoint(output_dir: str | Path) -> Optional[Path]:
    candidates = sorted(Path(output_dir).glob("epoch_*.pth"))
    if not candidates:
        return None

    def epoch_number(path: Path) -> int:
        try:
            return int(path.stem.split("_")[1])
        except (IndexError, ValueError):
            return -1

    return max(candidates, key=epoch_number)


__all__ = [
    "resolve_device",
    "seed_everything",
    "seed_worker",
    "dataloader_generator",
    "move_batch_to_device",
    "evaluate",
    "save_checkpoint",
    "load_training_checkpoint",
    "load_model_for_inference",
    "save_history_csv",
    "load_history_csv",
    "find_latest_epoch_checkpoint",
]
