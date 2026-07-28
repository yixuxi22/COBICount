"""Post-training evaluation entry point for COBICount."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence
import argparse
import json

import torch
from torch.utils.data import DataLoader

from cobicount.data import (
    DOTAFullImageEvalDataset,
    DOTAProxyPositiveEvalDataset,
    RSOCBuildingDataset,
    redcount_collate_fn,
)
from cobicount.engine import (
    dataloader_generator,
    evaluate,
    load_model_for_inference,
    resolve_device,
    seed_everything,
    seed_worker,
)
from cobicount.visualization import save_debug_grid, save_error_histogram, save_eval_csv


def parse_class_filter(value: str) -> Optional[List[str]]:
    if not value or not value.strip():
        return None
    return [item.strip() for item in value.replace(";", ",").split(",") if item.strip()]


def read_file_list(path: Optional[str]) -> Optional[List[str]]:
    if not path:
        return None
    file_path = Path(path).expanduser()
    if not file_path.is_file():
        raise FileNotFoundError(f"File list not found: {file_path}")
    return [line.strip() for line in file_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a frozen COBICount checkpoint. Target-domain datasets are "
            "loaded only by this post-training command."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True, choices=["rsoc", "dota-proxy", "dota-full"])
    parser.add_argument("--output-dir", default="outputs/evaluation")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--num-visualizations", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=None)

    parser.add_argument("--rsoc-root", default=None)
    parser.add_argument("--rsoc-split", default="test_data")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--density-sigma", type=float, default=3.0)

    parser.add_argument("--dota-image-root", default=None)
    parser.add_argument("--dota-annotation-root", default=None)
    parser.add_argument("--file-list", default=None)
    parser.add_argument("--class-filter", default="")
    parser.add_argument("--proxy-crop-size", type=int, default=512)
    parser.add_argument("--proxy-crops-per-image", type=int, default=4)
    parser.add_argument("--max-side", type=int, default=1440)
    parser.add_argument("--keep-empty", action="store_true")
    parser.add_argument(
        "--use-raw-model",
        action="store_true",
        help="Load the raw model weights instead of EMA weights when both are present.",
    )
    return parser


def build_dataset(args: argparse.Namespace):
    if args.dataset == "rsoc":
        if not args.rsoc_root:
            raise ValueError("--rsoc-root is required for --dataset rsoc.")
        root = Path(args.rsoc_root).expanduser().resolve() / args.rsoc_split
        return RSOCBuildingDataset(
            str(root / "images"),
            str(root / "ground_truth"),
            img_size=args.image_size,
            sigma=args.density_sigma,
            mode="val",
            train_use_patch=False,
            keep_original_size=False,
            seed=args.seed,
        )

    if not args.dota_image_root or not args.dota_annotation_root:
        raise ValueError(
            "--dota-image-root and --dota-annotation-root are required for DOTA evaluation."
        )
    class_filter = parse_class_filter(args.class_filter)
    file_list = read_file_list(args.file_list)

    if args.dataset == "dota-proxy":
        return DOTAProxyPositiveEvalDataset(
            args.dota_image_root,
            args.dota_annotation_root,
            file_list=file_list,
            crop_size=args.proxy_crop_size,
            crops_per_image=args.proxy_crops_per_image,
            sigma=args.density_sigma,
            class_filter=class_filter,
            seed=args.seed,
            keep_empty=args.keep_empty,
        )

    return DOTAFullImageEvalDataset(
        args.dota_image_root,
        args.dota_annotation_root,
        file_list=file_list,
        max_side=args.max_side,
        sigma=args.density_sigma,
        class_filter=class_filter,
        keep_empty=args.keep_empty,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    device = resolve_device(args.device)
    seed_everything(args.seed, deterministic=True)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(args)
    batch_size = 1 if args.dataset == "dota-full" else max(1, args.batch_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=redcount_collate_fn,
        worker_init_fn=seed_worker if args.num_workers > 0 else None,
        generator=dataloader_generator(args.seed),
        persistent_workers=args.num_workers > 0,
    )

    model = load_model_for_inference(
        args.checkpoint, device, prefer_ema=not args.use_raw_model
    )
    print(f"[Evaluation] dataset={args.dataset}, samples={len(dataset)}, device={device}")
    stats = evaluate(model, loader, device, max_batches=args.max_batches)
    print(f"[Result] MAE={stats['mae']:.4f}, RMSE={stats['rmse']:.4f}, N={stats['n']}")

    save_eval_csv(stats, output_dir / "predictions.csv")
    save_error_histogram(
        stats,
        output_dir / "absolute_error_histogram.png",
        f"{args.dataset}: MAE={stats['mae']:.3f}, RMSE={stats['rmse']:.3f}",
    )
    save_debug_grid(
        model,
        loader,
        output_dir / "visualizations",
        device,
        prefix=args.dataset.replace("-", "_"),
        max_items=args.num_visualizations,
    )
    summary = {
        "dataset": args.dataset,
        "checkpoint": str(Path(args.checkpoint).expanduser()),
        "mae": stats["mae"],
        "rmse": stats["rmse"],
        "num_samples": stats["n"],
        "class_filter": parse_class_filter(args.class_filter),
        "protocol": "post_training_frozen_checkpoint",
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
