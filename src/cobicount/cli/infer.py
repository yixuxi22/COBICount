"""Single-image inference for COBICount."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

from cobicount.data import build_valid_and_neutral_masks, pil_to_tensor, resize_keep_aspect
from cobicount.engine import load_model_for_inference, resolve_device
from cobicount.visualization import coarse_points_to_image, normalize_gray, overlay_points


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run COBICount on one unlabeled image.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", default="outputs/inference")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-side", type=int, default=1440)
    parser.add_argument("--use-raw-model", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    image_path = Path(args.image).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    model = load_model_for_inference(
        args.checkpoint, device, prefer_ema=not args.use_raw_model
    )

    original = Image.open(image_path).convert("RGB")
    original_width, original_height = original.size
    resized, _ = resize_keep_aspect(original, args.max_side, multiple=32)
    valid, neutral = build_valid_and_neutral_masks(resized)
    image_tensor = pil_to_tensor(resized).unsqueeze(0).to(device)
    valid_tensor = valid.unsqueeze(0).to(device)
    neutral_tensor = neutral.unsqueeze(0).to(device)

    with torch.inference_mode():
        output = model(image_tensor, valid_mask=valid_tensor, neutral_mask=neutral_tensor)

    resized_width, resized_height = resized.size
    points_resized = coarse_points_to_image(
        output["pred_points_c"][0], int(output.get("stride", 4)), resized_height, resized_width
    )
    scale_x = original_width / float(resized_width)
    scale_y = original_height / float(resized_height)
    points_original = points_resized.copy()
    if len(points_original):
        points_original[:, 0] *= scale_x
        points_original[:, 1] *= scale_y

    resized_array = np.asarray(resized).astype(np.float32) / 255.0
    overlay = overlay_points(resized_array, points_resized, color=(1.0, 0.25, 0.05), radius=3)
    plt.imsave(output_dir / "diagnostic_points.png", overlay)
    plt.imsave(output_dir / "score_map.png", normalize_gray(output["score_density"][0]), cmap="magma")

    result = {
        "image": str(image_path),
        "checkpoint": str(Path(args.checkpoint).expanduser()),
        "predicted_count": float(output["pred_count"][0].detach().cpu().item()),
        "num_diagnostic_points": int(len(points_original)),
        "diagnostic_points_xy_original": points_original.tolist(),
        "note": (
            "The continuous predicted count is the score-map integral and is not "
            "constrained to equal the number of diagnostic points."
        ),
    }
    (output_dir / "prediction.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
