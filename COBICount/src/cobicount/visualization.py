"""Visualization and result-export utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional
import csv
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .engine import move_batch_to_device


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    array = tensor.detach().float().cpu().numpy().transpose(1, 2, 0)
    return np.clip(array * std + mean, 0.0, 1.0)


def normalize_gray(tensor: torch.Tensor) -> np.ndarray:
    value = tensor.detach().float().cpu()
    if value.dim() == 4:
        value = value[0]
    if value.dim() == 3:
        value = value[0]
    array = np.nan_to_num(value.numpy(), nan=0.0, posinf=0.0, neginf=0.0)
    minimum, maximum = float(array.min()), float(array.max())
    if maximum - minimum < 1e-8:
        return np.zeros_like(array, dtype=np.float32)
    return ((array - minimum) / (maximum - minimum + 1e-8)).astype(np.float32)


def overlay_points(
    image: np.ndarray,
    points: np.ndarray,
    color: tuple[float, float, float] = (0.0, 1.0, 0.0),
    radius: int = 3,
) -> np.ndarray:
    output = image.copy()
    height, width = output.shape[:2]
    for point in points:
        x = int(round(float(point[0])))
        y = int(round(float(point[1])))
        for yy in range(max(0, y - radius), min(height, y + radius + 1)):
            for xx in range(max(0, x - radius), min(width, x + radius + 1)):
                if (xx - x) ** 2 + (yy - y) ** 2 <= radius**2:
                    output[yy, xx] = np.asarray(color, dtype=np.float32)
    return np.clip(output, 0.0, 1.0)


def coarse_points_to_image(
    points: Optional[torch.Tensor], stride: int, height: int, width: int
) -> np.ndarray:
    if points is None or points.numel() == 0:
        return np.zeros((0, 2), dtype=np.float32)
    array = points.detach().float().cpu().numpy().astype(np.float32)
    array[:, 0] *= float(stride)
    array[:, 1] *= float(stride)
    array[:, 0] = np.clip(array[:, 0], 0, width - 1)
    array[:, 1] = np.clip(array[:, 1], 0, height - 1)
    return array


def save_debug_grid(
    model: torch.nn.Module,
    loader: Optional[Iterable[Mapping[str, Any]]],
    output_dir: str | Path,
    device: torch.device | str,
    *,
    prefix: str,
    max_items: int = 2,
) -> None:
    if loader is None or max_items <= 0:
        return
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    saved = 0

    with torch.inference_mode():
        for batch in loader:
            batch_device = move_batch_to_device(batch, device)
            output = model(
                batch_device["image"],
                valid_mask=batch_device.get("valid_mask"),
                neutral_mask=batch_device.get("neutral_mask"),
            )
            batch_size = batch_device["image"].shape[0]
            for index in range(batch_size):
                if saved >= max_items:
                    return
                image = denormalize_image(batch_device["image"][index])
                height, width = image.shape[:2]
                gt_points = (
                    batch_device["points"][index].detach().cpu().numpy()
                    if batch_device["points"][index].numel()
                    else np.zeros((0, 2), dtype=np.float32)
                )
                pred_points = coarse_points_to_image(
                    output["pred_points_c"][index], int(output.get("stride", 4)), height, width
                )
                image_gt = overlay_points(image, gt_points, color=(0.0, 1.0, 0.0), radius=3)
                image_pred = overlay_points(image, pred_points, color=(1.0, 0.25, 0.05), radius=3)
                audit_counts = output.get("audit_counts", [{} for _ in range(batch_size)])
                audit = audit_counts[index] if index < len(audit_counts) else {}
                gt_count = float(batch_device["count"][index].detach().cpu().item())
                pred_count = float(output["pred_count"][index].detach().cpu().item())
                name = str(batch.get("name", [f"item{saved}"])[index])

                panels = [
                    (image, "Image"),
                    (normalize_gray(batch_device["density"][index]), "GT density"),
                    (normalize_gray(output["origin_score_density"][index]), "Origin score"),
                    (normalize_gray(output["rc_score_density"][index]), "Final score"),
                    (image_pred, "Predicted diagnostic points"),
                    (normalize_gray(output["scale_small"][index]), "Nominal small route"),
                    (normalize_gray(output["scale_mid"][index]), "Nominal mid route"),
                    (normalize_gray(output["scale_large"][index]), "Nominal large route"),
                    (normalize_gray(output["center_core"][index]), "Center gate"),
                    (normalize_gray(output["object_origin"][index]), "Source support gate"),
                    (normalize_gray(output["component_quality"][index]), "Auxiliary quality gate"),
                    (normalize_gray(output["reliability"][index]), "Auxiliary reliability gate"),
                    (normalize_gray(output["background_like"][index]), "Background-like risk"),
                    (normalize_gray(output["line_coherence"][index]), "Line-like risk"),
                    (normalize_gray(output["hollow_veto"][index]), "Broad off-center risk"),
                    (normalize_gray(output["anti_grid_void"][index]), "Alternating-structure risk"),
                    (normalize_gray(output["audit_vehicle_like"][index]), "Audit vehicle-like"),
                    (normalize_gray(output["audit_building_like"][index]), "Audit building-like"),
                    (normalize_gray(output["audit_ship_like"][index]), "Audit ship-like"),
                    (normalize_gray(output["audit_background_like"][index]), "Audit background-like"),
                    (image_gt, "Ground-truth points"),
                ]

                columns = 5
                rows = math.ceil(len(panels) / columns)
                figure = plt.figure(figsize=(24, 4.8 * rows))
                figure.suptitle(
                    f"{name} | GT={gt_count:.1f}, Pred={pred_count:.2f} | audit={audit}",
                    fontsize=10,
                )
                for panel_index, (panel, title) in enumerate(panels, start=1):
                    axis = figure.add_subplot(rows, columns, panel_index)
                    axis.imshow(panel, cmap="magma" if panel.ndim == 2 else None)
                    axis.set_title(title, fontsize=8)
                    axis.axis("off")
                figure.tight_layout()
                safe_name = name.replace("/", "_").replace("\\", "_")
                figure.savefig(output_dir / f"{prefix}_{saved:03d}_{safe_name}.png", dpi=150)
                plt.close(figure)
                saved += 1


def save_eval_csv(stats: Mapping[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["name", "gt", "pred", "abs_error"])
        for name, gt, pred in zip(stats.get("names", []), stats.get("gt", []), stats.get("pred", [])):
            writer.writerow([name, float(gt), float(pred), abs(float(pred) - float(gt))])


def save_error_histogram(stats: Mapping[str, Any], path: str | Path, title: str) -> None:
    errors = [abs(float(pred) - float(gt)) for pred, gt in zip(stats.get("pred", []), stats.get("gt", []))]
    if not errors:
        return
    figure = plt.figure(figsize=(9, 6))
    axis = figure.add_subplot(1, 1, 1)
    axis.hist(errors, bins=min(40, max(8, len(errors))))
    axis.set_title(title)
    axis.set_xlabel("|Pred - GT|")
    axis.set_ylabel("Frequency")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def save_training_curve(history: Mapping[str, list[float]], path: str | Path) -> None:
    epochs = history.get("epoch", [])
    if not epochs:
        return

    def column(key: str) -> list[float]:
        return history.get(key, [float("nan")] * len(epochs))

    figure = plt.figure(figsize=(16, 10))
    axis = figure.add_subplot(2, 2, 1)
    axis.plot(epochs, column("train_loss"), label="train_loss")
    axis.set_title("Training loss")
    axis.set_xlabel("Epoch")
    axis.legend()

    axis = figure.add_subplot(2, 2, 2)
    axis.plot(epochs, column("source_val_mae"), label="source_val_mae")
    axis.plot(epochs, column("source_val_rmse"), label="source_val_rmse")
    axis.set_title("Source validation")
    axis.set_xlabel("Epoch")
    axis.legend()

    axis = figure.add_subplot(2, 2, 3)
    for key in ["score", "count", "log_count", "center", "origin", "reliability", "quality"]:
        axis.plot(epochs, column(key), label=key)
    axis.set_title("Main losses")
    axis.set_xlabel("Epoch")
    axis.legend(fontsize=8)

    axis = figure.add_subplot(2, 2, 4)
    for key in [
        "bg_veto",
        "line_veto",
        "hollow_veto",
        "antigrid_veto",
        "hard_negative",
        "scale_entropy",
        "scale_balance",
        "audit",
    ]:
        axis.plot(epochs, column(key), label=key)
    axis.set_title("Auxiliary losses")
    axis.set_xlabel("Epoch")
    axis.legend(fontsize=8)

    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


__all__ = [
    "denormalize_image",
    "normalize_gray",
    "overlay_points",
    "coarse_points_to_image",
    "save_debug_grid",
    "save_eval_csv",
    "save_error_histogram",
    "save_training_curve",
]
