# -*- coding: utf-8 -*-
from __future__ import annotations

"""Dataset utilities for COBICount.

The training CLI imports only ``RSOCBuildingDataset`` and therefore cannot
construct target-domain loaders. DOTA datasets are exposed exclusively for
post-training evaluation and visualization. Dataset files are not distributed
with this repository; users must obtain them from their official sources and
follow the respective licenses.
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from collections import OrderedDict
import json
import os
import random

import numpy as np
from PIL import Image, ImageEnhance, ImageFile
import scipy.io as sio
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def pil_to_numpy(img: Image.Image) -> np.ndarray:
    arr = np.asarray(img.convert("RGB")).astype(np.float32) / 255.0
    return arr


def pil_to_tensor(img: Image.Image, normalize: bool = True) -> torch.Tensor:
    arr = pil_to_numpy(img)
    if normalize:
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(arr.transpose(2, 0, 1)).float()


def points_to_density(points_xy: np.ndarray, h: int, w: int, sigma: float = 3.0, normalize_kernel: bool = True) -> np.ndarray:
    den = np.zeros((h, w), dtype=np.float32)
    if points_xy is None or len(points_xy) == 0:
        return den
    radius = max(1, int(round(3.0 * float(sigma))))
    xs = np.arange(-radius, radius + 1, dtype=np.float32)
    ys = np.arange(-radius, radius + 1, dtype=np.float32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    k = np.exp(-(xx * xx + yy * yy) / (2.0 * float(sigma) * float(sigma))).astype(np.float32)
    if normalize_kernel:
        k /= max(float(k.sum()), 1e-8)
    else:
        k /= max(float(k.max()), 1e-8)
    for x, y in points_xy.astype(np.float32):
        cx = int(round(float(x)))
        cy = int(round(float(y)))
        if cx < 0 or cx >= w or cy < 0 or cy >= h:
            continue
        x0 = max(0, cx - radius); x1 = min(w, cx + radius + 1)
        y0 = max(0, cy - radius); y1 = min(h, cy + radius + 1)
        kx0 = x0 - (cx - radius); kx1 = kx0 + (x1 - x0)
        ky0 = y0 - (cy - radius); ky1 = ky0 + (y1 - y0)
        den[y0:y1, x0:x1] += k[ky0:ky1, kx0:kx1]
    return den


def build_valid_and_neutral_masks(img: Image.Image, valid_thr: float = 0.035, neutral_dilate: int = 9) -> Tuple[torch.Tensor, torch.Tensor]:
    arr = pil_to_numpy(img)
    intensity = arr.mean(axis=2)
    valid = (intensity > float(valid_thr)).astype(np.float32)
    valid_t = torch.from_numpy(valid).unsqueeze(0).unsqueeze(0)
    invalid = 1.0 - valid_t
    if neutral_dilate and neutral_dilate > 1:
        neutral = F.max_pool2d(invalid, kernel_size=int(neutral_dilate) | 1, stride=1, padding=(int(neutral_dilate) | 1) // 2)
    else:
        neutral = invalid
    neutral = (neutral > 0).float()[0]
    valid = valid_t[0] * (1.0 - neutral)
    return valid.float(), neutral.float()


def resize_keep_aspect(img: Image.Image, max_side: int, multiple: int = 32) -> Tuple[Image.Image, float]:
    w, h = img.size
    if max(w, h) <= 0:
        return img, 1.0
    scale = float(max_side) / float(max(w, h))
    if abs(scale - 1.0) < 1e-6:
        new_w, new_h = w, h
    else:
        new_w = max(multiple, int(round(w * scale)))
        new_h = max(multiple, int(round(h * scale)))
    new_w = int(np.ceil(new_w / multiple) * multiple)
    new_h = int(np.ceil(new_h / multiple) * multiple)
    if (new_w, new_h) == (w, h):
        return img.convert("RGB"), 1.0
    return img.convert("RGB").resize((new_w, new_h), Image.BILINEAR), new_w / float(w)


def pad_to_size(img: Image.Image, size: int) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    if w == size and h == size:
        return img
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(img, (0, 0))
    return canvas


def crop_with_points(img: Image.Image, points_xy: np.ndarray, left: int, top: int, size: int) -> Tuple[Image.Image, np.ndarray]:
    w, h = img.size
    left = int(max(0, min(left, max(0, w - size))))
    top = int(max(0, min(top, max(0, h - size))))
    right = min(w, left + size)
    bottom = min(h, top + size)
    patch = img.crop((left, top, right, bottom)).convert("RGB")
    patch = pad_to_size(patch, size)
    if points_xy is None or len(points_xy) == 0:
        return patch, np.zeros((0, 2), dtype=np.float32)
    pts = points_xy.astype(np.float32).copy()
    inside = (pts[:, 0] >= left) & (pts[:, 0] < right) & (pts[:, 1] >= top) & (pts[:, 1] < bottom)
    pts = pts[inside]
    if len(pts) > 0:
        pts[:, 0] -= float(left)
        pts[:, 1] -= float(top)
    return patch, pts.astype(np.float32)


def _augment_image_and_points(img: Image.Image, pts: np.ndarray, rng: random.Random) -> Tuple[Image.Image, np.ndarray]:
    pts = pts.astype(np.float32).copy()
    w, h = img.size
    if rng.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if len(pts):
            pts[:, 0] = (w - 1) - pts[:, 0]
    if rng.random() < 0.5:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        if len(pts):
            pts[:, 1] = (h - 1) - pts[:, 1]
    if rng.random() < 0.25:
        # conservative color jitter for remote-sensing images
        img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.85, 1.15))
        img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.85, 1.15))
        img = ImageEnhance.Color(img).enhance(rng.uniform(0.90, 1.10))
    return img, pts


def read_rsoc_points_from_mat(mat_path: str) -> np.ndarray:
    mat = sio.loadmat(mat_path)
    if "center" not in mat:
        raise KeyError(f"{mat_path} does not contain key 'center'")
    center = mat["center"]
    pts = None
    if isinstance(center, np.ndarray) and center.dtype == object and center.size > 0:
        first = center.flat[0]
        if isinstance(first, np.ndarray) and first.ndim == 2 and first.shape[1] >= 2:
            pts = first[:, :2].astype(np.float32)
    elif isinstance(center, np.ndarray) and center.ndim == 2 and center.shape[1] >= 2:
        pts = center[:, :2].astype(np.float32)
    if pts is None:
        raise RuntimeError(f"Unable to parse RSOC mat format: {mat_path}")
    return pts.astype(np.float32)


class RSOCBuildingDataset(Dataset):
    def __init__(self,
                 img_root: str,
                 gt_root: str,
                 img_size: int = 512,
                 sigma: float = 3.0,
                 mode: str = "train",
                 normalize: bool = True,
                 seed: int = 3407,
                 train_use_patch: bool = True,
                 patch_size: int = 512,
                 point_crop_prob: float = 0.85,
                 center_jitter: int = 96,
                 keep_original_size: bool = False,
                 valid_thr: float = 0.035,
                 neutral_dilate: int = 9,
                 file_list: Optional[Sequence[str]] = None):
        self.img_root = img_root
        self.gt_root = gt_root
        self.img_size = int(img_size)
        self.sigma = float(sigma)
        self.mode = mode
        self.normalize = normalize
        self.seed = int(seed)
        self.train_use_patch = bool(train_use_patch)
        self.patch_size = int(patch_size)
        self.point_crop_prob = float(point_crop_prob)
        self.center_jitter = int(center_jitter)
        self.keep_original_size = bool(keep_original_size)
        self.valid_thr = float(valid_thr)
        self.neutral_dilate = int(neutral_dilate)
        if file_list is None:
            files = sorted([f for f in os.listdir(img_root) if f.lower().endswith(IMAGE_EXTS)])
        else:
            files = list(file_list)
        usable = []
        for f in files:
            stem = os.path.splitext(f)[0]
            # Try both GT_IMG_xxx.mat and IMG_xxx.mat formats
            mat_path1 = os.path.join(gt_root, "GT_" + stem + ".mat")
            mat_path2 = os.path.join(gt_root, stem + ".mat")
            if os.path.isfile(mat_path1):
                usable.append(f)
            elif os.path.isfile(mat_path2):
                usable.append(f)
        self.file_list = usable
        if len(self.file_list) == 0:
            raise RuntimeError(f"No usable RSOC image/mat pairs in {img_root} and {gt_root}")

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        name = self.file_list[idx]
        img_path = os.path.join(self.img_root, name)
        stem = os.path.splitext(name)[0]
        # Try both GT_IMG_xxx.mat and IMG_xxx.mat formats
        gt_path1 = os.path.join(self.gt_root, "GT_" + stem + ".mat")
        gt_path2 = os.path.join(self.gt_root, stem + ".mat")
        if os.path.isfile(gt_path1):
            gt_path = gt_path1
        else:
            gt_path = gt_path2
        img = Image.open(img_path).convert("RGB")
        pts = read_rsoc_points_from_mat(gt_path)
        rng = random.Random(self.seed + idx * 9973 + (0 if self.mode != "train" else random.randint(0, 10_000_000)))

        if self.mode == "train" and self.train_use_patch:
            size = self.patch_size
            w, h = img.size
            if len(pts) > 0 and rng.random() < self.point_crop_prob:
                p = pts[rng.randrange(len(pts))]
                left = int(round(float(p[0]) - size / 2 + rng.randint(-self.center_jitter, self.center_jitter)))
                top = int(round(float(p[1]) - size / 2 + rng.randint(-self.center_jitter, self.center_jitter)))
            else:
                left = rng.randint(0, max(0, w - size)) if w > size else 0
                top = rng.randint(0, max(0, h - size)) if h > size else 0
            img, pts = crop_with_points(img, pts, left, top, size)
            img, pts = _augment_image_and_points(img, pts, rng)
        else:
            if self.keep_original_size:
                img, scale = resize_keep_aspect(img, self.img_size, multiple=32)
            else:
                old_w, old_h = img.size
                img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
                scale = self.img_size / float(max(old_w, 1))
                pts = pts.copy()
                pts[:, 0] *= self.img_size / float(max(old_w, 1))
                pts[:, 1] *= self.img_size / float(max(old_h, 1))
            if self.keep_original_size:
                pts = pts.copy() * float(scale)

        h, w = img.size[1], img.size[0]
        density = points_to_density(pts, h, w, self.sigma, normalize_kernel=True)
        valid, neutral = build_valid_and_neutral_masks(img, self.valid_thr, self.neutral_dilate)
        return {
            "image": pil_to_tensor(img, normalize=self.normalize),
            "density": torch.from_numpy(density).unsqueeze(0).float(),
            "points": torch.from_numpy(pts.astype(np.float32)),
            "count": float(len(pts)),
            "valid_mask": valid.float(),
            "neutral_mask": neutral.float(),
            "name": name,
            "domain": "rsoc",
            "species": ["building"],
            "known_species": 1,
            "unknown_species": 0,
        }


def _category_ok(cat: str, class_filter: Optional[Sequence[str]]) -> bool:
    if class_filter is None or len(class_filter) == 0:
        return True
    low = cat.lower().replace("_", " ").strip()
    filters = [c.lower().replace("_", " ").strip() for c in class_filter]
    return any(f == low or f in low for f in filters)


def _poly_centroid(poly: Sequence[float]) -> Optional[Tuple[float, float]]:
    arr = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    if arr.shape[0] == 0:
        return None
    return float(arr[:, 0].mean()), float(arr[:, 1].mean())


def _extract_obj_point_and_cat(obj: Any) -> Optional[Tuple[float, float, str]]:
    cat = "unknown"
    if isinstance(obj, dict):
        for k in ["category", "class", "label", "name", "type", "classTitle"]:
            if k in obj:
                cat = str(obj[k])
                break
        if "poly" in obj:
            cen = _poly_centroid(obj["poly"])
            if cen is not None:
                return cen[0], cen[1], cat
        if "points" in obj:
            pts_data = obj["points"]
            # Handle DOTA format: {"exterior": [[x,y],...], "interior": []}
            if isinstance(pts_data, dict) and "exterior" in pts_data:
                cen = _poly_centroid(pts_data["exterior"])
            else:
                cen = _poly_centroid(pts_data)
            if cen is not None:
                return cen[0], cen[1], cat
        if "polygon" in obj:
            cen = _poly_centroid(obj["polygon"])
            if cen is not None:
                return cen[0], cen[1], cat
        if "bbox" in obj:
            b = np.asarray(obj["bbox"], dtype=np.float32).reshape(-1)
            if len(b) >= 4:
                # support both xywh and xyxy approximately
                x0, y0, a, b4 = b[:4]
                if a > x0 and b4 > y0:
                    return float((x0 + a) / 2.0), float((y0 + b4) / 2.0), cat
                return float(x0 + a / 2.0), float(y0 + b4 / 2.0), cat
        if "x" in obj and "y" in obj:
            return float(obj["x"]), float(obj["y"]), cat
    elif isinstance(obj, (list, tuple)) and len(obj) >= 2:
        return float(obj[0]), float(obj[1]), cat
    return None


def read_dota_points(ann_path: str, class_filter: Optional[Sequence[str]] = None) -> Tuple[np.ndarray, List[str]]:
    pts: List[Tuple[float, float]] = []
    cats: List[str] = []
    if not os.path.isfile(ann_path):
        return np.zeros((0, 2), dtype=np.float32), []
    ext = os.path.splitext(ann_path)[1].lower()
    try:
        if ext == ".json":
            with open(ann_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            objs = data.get("objects", data.get("annotations", data.get("labels", []))) if isinstance(data, dict) else data
            if isinstance(objs, dict):
                objs = list(objs.values())
            for obj in objs or []:
                item = _extract_obj_point_and_cat(obj)
                if item is None:
                    continue
                x, y, cat = item
                if _category_ok(cat, class_filter):
                    pts.append((x, y)); cats.append(cat)
        else:
            with open(ann_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 9:
                        continue
                    try:
                        nums = [float(v) for v in parts[:8]]
                    except Exception:
                        continue
                    cat = " ".join(parts[8:-1]) if len(parts) > 9 else parts[8]
                    if _category_ok(cat, class_filter):
                        cen = _poly_centroid(nums)
                        if cen is not None:
                            pts.append(cen); cats.append(cat)
    except Exception:
        return np.zeros((0, 2), dtype=np.float32), []
    return np.asarray(pts, dtype=np.float32).reshape(-1, 2), cats


def find_ann_for_image(ann_root: str, img_name: str) -> Optional[str]:
    stem = os.path.splitext(os.path.basename(img_name))[0]
    # Try multiple formats: stem.json, stem.txt, stem.png.json, stem.png.txt
    for ext in [".json", ".txt", ".png.json", ".png.txt"]:
        p = os.path.join(ann_root, stem + ext)
        if os.path.isfile(p):
            return p
    # Fallback: try with full image name (e.g., P0000.png -> P0000.png.json)
    full_name = os.path.basename(img_name)
    for ext in [".json", ".txt"]:
        p = os.path.join(ann_root, full_name + ext)
        if os.path.isfile(p):
            return p
    return None


class DOTAProxyPositiveEvalDataset(Dataset):
    def __init__(self,
                 img_root: str,
                 ann_root: str,
                 file_list: Optional[Sequence[str]] = None,
                 crop_size: int = 512,
                 crops_per_image: int = 2,
                 sigma: float = 3.0,
                 class_filter: Optional[Sequence[str]] = None,
                 seed: int = 3407,
                 normalize: bool = True,
                 valid_thr: float = 0.035,
                 neutral_dilate: int = 9,
                 keep_empty: bool = False):
        self.img_root = img_root
        self.ann_root = ann_root
        self.crop_size = int(crop_size)
        self.crops_per_image = int(crops_per_image)
        self.sigma = float(sigma)
        self.class_filter = list(class_filter) if class_filter else None
        self.seed = int(seed)
        self.normalize = normalize
        self.valid_thr = float(valid_thr)
        self.neutral_dilate = int(neutral_dilate)
        files = sorted([f for f in os.listdir(img_root) if f.lower().endswith(IMAGE_EXTS)]) if file_list is None else list(file_list)
        self.entries: List[Tuple[str, int]] = []
        self.cache: Dict[str, Tuple[np.ndarray, List[str]]] = {}
        for f in files:
            ann = find_ann_for_image(ann_root, f)
            pts, cats = read_dota_points(ann, self.class_filter) if ann else (np.zeros((0, 2), dtype=np.float32), [])
            if keep_empty or len(pts) > 0:
                self.cache[f] = (pts, cats)
                for ci in range(max(1, self.crops_per_image)):
                    self.entries.append((f, ci))
        if len(self.entries) == 0:
            raise RuntimeError(f"No DOTA positive entries found under {img_root} / {ann_root}. Check annotations or class_filter.")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        name, crop_id = self.entries[idx]
        img = Image.open(os.path.join(self.img_root, name)).convert("RGB")
        pts, cats = self.cache.get(name, (np.zeros((0, 2), dtype=np.float32), []))
        rng = random.Random(self.seed + idx * 7919 + crop_id * 123)
        size = self.crop_size
        w, h = img.size
        if len(pts) > 0:
            p = pts[rng.randrange(len(pts))]
            left = int(round(float(p[0]) - size / 2 + rng.randint(-size // 6, size // 6)))
            top = int(round(float(p[1]) - size / 2 + rng.randint(-size // 6, size // 6)))
        else:
            left = rng.randint(0, max(0, w - size)) if w > size else 0
            top = rng.randint(0, max(0, h - size)) if h > size else 0
        patch, crop_pts = crop_with_points(img, pts, left, top, size)
        h2, w2 = patch.size[1], patch.size[0]
        density = points_to_density(crop_pts, h2, w2, self.sigma, normalize_kernel=True)
        valid, neutral = build_valid_and_neutral_masks(patch, self.valid_thr, self.neutral_dilate)
        # categories inside crop
        in_cats: List[str] = []
        if len(pts) > 0:
            inside = (pts[:, 0] >= left) & (pts[:, 0] < left + size) & (pts[:, 1] >= top) & (pts[:, 1] < top + size)
            in_cats = [c for c, ok in zip(cats, inside.tolist()) if ok]
        species = sorted(list(OrderedDict.fromkeys([c if c else "unknown" for c in in_cats]))) or ["unknown"]
        return {
            "image": pil_to_tensor(patch, normalize=self.normalize),
            "density": torch.from_numpy(density).unsqueeze(0).float(),
            "points": torch.from_numpy(crop_pts.astype(np.float32)),
            "count": float(len(crop_pts)),
            "valid_mask": valid.float(),
            "neutral_mask": neutral.float(),
            "name": f"{name}_crop{crop_id:02d}",
            "domain": "dota_proxy",
            "species": species,
            "known_species": 0,
            "unknown_species": len(species),
            "crop_xy": (left, top),
        }


class DOTAFullImageEvalDataset(Dataset):
    def __init__(self,
                 img_root: str,
                 ann_root: str,
                 file_list: Optional[Sequence[str]] = None,
                 max_side: int = 1440,
                 sigma: float = 3.0,
                 class_filter: Optional[Sequence[str]] = None,
                 normalize: bool = True,
                 valid_thr: float = 0.035,
                 neutral_dilate: int = 9,
                 keep_empty: bool = False):
        self.img_root = img_root
        self.ann_root = ann_root
        self.max_side = int(max_side)
        self.sigma = float(sigma)
        self.class_filter = list(class_filter) if class_filter else None
        self.normalize = normalize
        self.valid_thr = float(valid_thr)
        self.neutral_dilate = int(neutral_dilate)
        files = sorted([f for f in os.listdir(img_root) if f.lower().endswith(IMAGE_EXTS)]) if file_list is None else list(file_list)
        self.files: List[str] = []
        self.cache: Dict[str, Tuple[np.ndarray, List[str]]] = {}
        for f in files:
            ann = find_ann_for_image(ann_root, f)
            pts, cats = read_dota_points(ann, self.class_filter) if ann else (np.zeros((0, 2), dtype=np.float32), [])
            if keep_empty or len(pts) > 0:
                self.files.append(f)
                self.cache[f] = (pts, cats)
        if len(self.files) == 0:
            raise RuntimeError(f"No DOTA full-image entries found under {img_root} / {ann_root}.")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        name = self.files[idx]
        img0 = Image.open(os.path.join(self.img_root, name)).convert("RGB")
        pts0, cats = self.cache.get(name, (np.zeros((0, 2), dtype=np.float32), []))
        old_w, old_h = img0.size
        img, scale = resize_keep_aspect(img0, self.max_side, multiple=32)
        pts = pts0.astype(np.float32).copy()
        if len(pts):
            pts[:, 0] *= float(img.size[0]) / max(1.0, float(old_w))
            pts[:, 1] *= float(img.size[1]) / max(1.0, float(old_h))
        h, w = img.size[1], img.size[0]
        density = points_to_density(pts, h, w, self.sigma, normalize_kernel=True)
        valid, neutral = build_valid_and_neutral_masks(img, self.valid_thr, self.neutral_dilate)
        species = sorted(list(OrderedDict.fromkeys([c if c else "unknown" for c in cats]))) or ["unknown"]
        return {
            "image": pil_to_tensor(img, normalize=self.normalize),
            "density": torch.from_numpy(density).unsqueeze(0).float(),
            "points": torch.from_numpy(pts.astype(np.float32)),
            "count": float(len(pts)),
            "valid_mask": valid.float(),
            "neutral_mask": neutral.float(),
            "name": name,
            "domain": "dota_full",
            "species": species,
            "known_species": 0,
            "unknown_species": len(species),
            "scale": scale,
            "original_size": (old_w, old_h),
        }


def redcount_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    # This collate keeps variable-length points and metadata as lists.
    out: Dict[str, Any] = {}
    tensor_keys = ["image", "density", "valid_mask", "neutral_mask"]
    for k in tensor_keys:
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    out["points"] = [b["points"].float() for b in batch]
    out["count"] = torch.tensor([float(b["count"]) for b in batch], dtype=torch.float32)
    for k in ["name", "domain", "species", "known_species", "unknown_species", "crop_xy", "scale", "original_size"]:
        vals = [b.get(k, None) for b in batch]
        if any(v is not None for v in vals):
            out[k] = vals
    return out


__all__ = [
    "RSOCBuildingDataset",
    "DOTAProxyPositiveEvalDataset",
    "DOTAFullImageEvalDataset",
    "redcount_collate_fn",
    "pil_to_tensor",
    "resize_keep_aspect",
    "build_valid_and_neutral_masks",
    "read_dota_points",
]
