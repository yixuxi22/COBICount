"""Strict source-only training entry point for COBICount."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional
import argparse
import json
import math
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from cobicount.data import RSOCBuildingDataset, redcount_collate_fn
from cobicount.engine import (
    dataloader_generator,
    evaluate,
    find_latest_epoch_checkpoint,
    load_history_csv,
    load_training_checkpoint,
    move_batch_to_device,
    resolve_device,
    save_checkpoint,
    save_history_csv,
    seed_everything,
    seed_worker,
)
from cobicount.losses import V523BICLossConfig, compute_v523_bic_losses
from cobicount.model import COBICount, ModelEMA523, V523BICConfig
from cobicount.visualization import save_debug_grid, save_training_curve


@dataclass
class TrainConfig:
    rsoc_root: str
    output_dir: str = "runs/cobicount_rsoc_building"
    train_split: str = "train_data"
    val_split: str = "test_data"
    image_size: int = 512
    patch_size: int = 512
    density_sigma: float = 3.0
    epochs: int = 80
    batch_size: int = 6
    num_workers: int = 4
    learning_rate: float = 1.8e-4
    weight_decay: float = 1e-4
    amp: bool = True
    grad_clip: float = 5.0
    ema_decay: float = 0.999
    seed: int = 3407
    device: str = "auto"
    deterministic: bool = False
    eval_every: int = 5
    save_every: int = 5
    visualize_every: int = 5
    num_visualizations: int = 2
    resume: str = ""


def parse_args(argv: Optional[List[str]] = None) -> TrainConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Train COBICount under a strict source-only protocol. This command "
            "does not import or construct DOTA/DIOR target-domain datasets."
        )
    )
    parser.add_argument("--rsoc-root", required=True, help="RSOC Building root directory.")
    parser.add_argument("--output-dir", default="runs/cobicount_rsoc_building")
    parser.add_argument("--train-split", default="train_data")
    parser.add_argument(
        "--val-split",
        default="test_data",
        help=(
            "Source-domain split used only for checkpoint selection. The default "
            "matches the original RSOC directory layout; use a disjoint validation "
            "split when available."
        ),
    )
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--density-sigma", type=float, default=3.0)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1.8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--no-amp", action="store_true", help="Disable automatic mixed precision.")
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--visualize-every", type=int, default=5)
    parser.add_argument("--num-visualizations", type=int, default=2)
    parser.add_argument(
        "--resume",
        default="",
        help="Checkpoint path, 'auto' for latest epoch checkpoint, or empty to start fresh.",
    )
    args = parser.parse_args(argv)
    return TrainConfig(
        rsoc_root=args.rsoc_root,
        output_dir=args.output_dir,
        train_split=args.train_split,
        val_split=args.val_split,
        image_size=args.image_size,
        patch_size=args.patch_size,
        density_sigma=args.density_sigma,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        amp=not args.no_amp,
        grad_clip=args.grad_clip,
        ema_decay=args.ema_decay,
        seed=args.seed,
        device=args.device,
        deterministic=args.deterministic,
        eval_every=args.eval_every,
        save_every=args.save_every,
        visualize_every=args.visualize_every,
        num_visualizations=args.num_visualizations,
        resume=args.resume,
    )


def _validate_config(config: TrainConfig) -> None:
    if config.epochs < 1:
        raise ValueError("--epochs must be at least 1.")
    if config.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if config.image_size < 32 or config.patch_size < 32:
        raise ValueError("Image and patch sizes must be at least 32 pixels.")
    if config.eval_every < 1 or config.save_every < 1 or config.visualize_every < 1:
        raise ValueError("Evaluation/save/visualization intervals must be positive.")


def build_source_loaders(config: TrainConfig, device: torch.device) -> tuple[DataLoader, DataLoader]:
    """Build only source-domain training and validation loaders."""
    root = Path(config.rsoc_root).expanduser().resolve()
    train_images = root / config.train_split / "images"
    train_ground_truth = root / config.train_split / "ground_truth"
    val_images = root / config.val_split / "images"
    val_ground_truth = root / config.val_split / "ground_truth"

    for path in [train_images, train_ground_truth, val_images, val_ground_truth]:
        if not path.is_dir():
            raise FileNotFoundError(f"Required source-domain directory not found: {path}")

    train_dataset = RSOCBuildingDataset(
        str(train_images),
        str(train_ground_truth),
        img_size=config.image_size,
        sigma=config.density_sigma,
        mode="train",
        train_use_patch=True,
        patch_size=config.patch_size,
        seed=config.seed,
    )
    val_dataset = RSOCBuildingDataset(
        str(val_images),
        str(val_ground_truth),
        img_size=config.image_size,
        sigma=config.density_sigma,
        mode="val",
        train_use_patch=False,
        keep_original_size=False,
        seed=config.seed,
    )

    common = {
        "num_workers": config.num_workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": redcount_collate_fn,
        "worker_init_fn": seed_worker if config.num_workers > 0 else None,
        "persistent_workers": config.num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
        generator=dataloader_generator(config.seed),
        **common,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=max(1, config.batch_size // 2),
        shuffle=False,
        drop_last=False,
        generator=dataloader_generator(config.seed + 1),
        **common,
    )
    return train_loader, val_loader


def _make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _autocast(enabled: bool):
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)


def _resolve_resume_path(config: TrainConfig) -> Optional[Path]:
    if not config.resume:
        return None
    if config.resume.lower() == "auto":
        latest = find_latest_epoch_checkpoint(config.output_dir)
        if latest is None:
            best = Path(config.output_dir) / "best_source_val.pth"
            return best if best.is_file() else None
        return latest
    path = Path(config.resume).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")
    return path


def main(argv: Optional[List[str]] = None) -> None:
    config = parse_args(argv)
    _validate_config(config)
    output_dir = Path(config.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(config.device)
    seed_everything(config.seed, deterministic=config.deterministic)

    (output_dir / "run_config.json").write_text(
        json.dumps(asdict(config), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("[Protocol] strict source-only training")
    print("[Protocol] target-domain datasets are not imported by this command")
    print(f"[Device] {device}")
    print(f"[Output] {output_dir}")

    train_loader, val_loader = build_source_loaders(config, device)
    print(f"[Data] source train={len(train_loader.dataset)}, source val={len(val_loader.dataset)}")

    model_config = V523BICConfig()
    loss_config = V523BICLossConfig()
    model = COBICount(model_config).to(device)
    ema = ModelEMA523(model, decay=config.ema_decay)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    amp_enabled = bool(config.amp and device.type == "cuda")
    scaler = _make_grad_scaler(amp_enabled)

    history_keys = [
        "epoch",
        "train_loss",
        "score",
        "count",
        "log_count",
        "center",
        "origin",
        "reliability",
        "quality",
        "bg_veto",
        "line_veto",
        "hollow_veto",
        "antigrid_veto",
        "hard_negative",
        "scale_entropy",
        "scale_balance",
        "audit",
        "count_weight",
        "source_val_mae",
        "source_val_rmse",
    ]
    history_path = output_dir / "metrics.csv"
    history: Dict[str, List[float]] = {key: [] for key in history_keys}
    best_source_val = float("inf")
    start_epoch = 0

    resume_path = _resolve_resume_path(config)
    if resume_path is not None:
        print(f"[Resume] {resume_path}")
        start_epoch, best_source_val = load_training_checkpoint(
            resume_path,
            model=model,
            ema=ema,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
        )
        history = load_history_csv(history_path, history_keys)
        if history["epoch"] and int(history["epoch"][-1]) != start_epoch:
            print(
                "[Warning] metrics.csv and checkpoint epoch differ; continuing from "
                f"checkpoint epoch {start_epoch}."
            )

    for epoch in range(start_epoch + 1, config.epochs + 1):
        start_time = time.time()
        model.train()
        meters: Dict[str, List[float]] = {}

        for iteration, batch in enumerate(train_loader):
            batch_device = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(amp_enabled):
                output = model(
                    batch_device["image"],
                    valid_mask=batch_device.get("valid_mask"),
                    neutral_mask=batch_device.get("neutral_mask"),
                )
                losses = compute_v523_bic_losses(
                    output, batch_device, epoch, model_config, loss_config
                )
                total_loss = losses["total"]

            if not torch.isfinite(total_loss):
                print(f"[Warning] non-finite loss skipped at epoch={epoch}, iteration={iteration}")
                continue

            scaler.scale(total_loss).backward()
            if config.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)

            for key, value in losses.items():
                meters.setdefault(key, []).append(float(value.detach().cpu().item()))
            if iteration % 50 == 0 or iteration == len(train_loader) - 1:
                print(
                    f"[Epoch {epoch:03d}/{config.epochs:03d}] "
                    f"iteration={iteration:04d}/{len(train_loader):04d} "
                    f"loss={meters['total'][-1]:.4f} count={meters['count'][-1]:.4f}"
                )

        train_loss = float(np.mean(meters.get("total", [float("nan")])))
        source_metrics = {"mae": float("nan"), "rmse": float("nan")}
        should_evaluate = epoch == 1 or epoch % config.eval_every == 0 or epoch == config.epochs
        if should_evaluate:
            source_metrics = evaluate(ema.ema, val_loader, device)
            if source_metrics["mae"] < best_source_val:
                best_source_val = float(source_metrics["mae"])
                save_checkpoint(
                    output_dir / "best_source_val.pth",
                    model=model,
                    ema=ema,
                    optimizer=optimizer,
                    scaler=scaler,
                    epoch=epoch,
                    train_config=config,
                    model_config=model_config,
                    loss_config=loss_config,
                    best_source_val_mae=best_source_val,
                )
                print(f"[Best] source validation MAE={best_source_val:.4f}")

        if epoch % config.save_every == 0 or epoch == config.epochs:
            save_checkpoint(
                output_dir / f"epoch_{epoch:03d}.pth",
                model=model,
                ema=ema,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                train_config=config,
                model_config=model_config,
                loss_config=loss_config,
                best_source_val_mae=best_source_val,
            )

        if epoch == 1 or epoch % config.visualize_every == 0 or epoch == config.epochs:
            save_debug_grid(
                ema.ema,
                val_loader,
                output_dir / "visualizations" / f"epoch_{epoch:03d}",
                device,
                prefix="source_val",
                max_items=config.num_visualizations,
            )

        history["epoch"].append(float(epoch))
        history["train_loss"].append(train_loss)
        for key in [
            "score",
            "count",
            "log_count",
            "center",
            "origin",
            "reliability",
            "quality",
            "bg_veto",
            "line_veto",
            "hollow_veto",
            "antigrid_veto",
            "hard_negative",
            "scale_entropy",
            "scale_balance",
            "audit",
            "count_weight",
        ]:
            history[key].append(float(np.mean(meters.get(key, [float("nan")]))))
        history["source_val_mae"].append(float(source_metrics["mae"]))
        history["source_val_rmse"].append(float(source_metrics["rmse"]))
        save_history_csv(history, history_path)
        save_training_curve(history, output_dir / "training_curves.png")

        elapsed = time.time() - start_time
        print(
            f"[Epoch {epoch:03d}] train_loss={train_loss:.4f} "
            f"source_val_mae={source_metrics['mae']:.4f} elapsed={elapsed:.1f}s"
        )

    print(f"[Done] best source validation MAE={best_source_val:.4f}")
    print("[Next] Run cobicount-evaluate only after training is complete and the checkpoint is frozen.")


if __name__ == "__main__":
    main()
