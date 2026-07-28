# -*- coding: utf-8 -*-
from __future__ import annotations

"""Source-derived training losses for COBICount."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np
import torch
import torch.nn.functional as F


def _sanitize(x: torch.Tensor, nan: float = 0.0, posinf: float = 1e4, neginf: float = -1e4) -> torch.Tensor:
    return torch.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)


def _safe_mean(x: torch.Tensor, default: float = 0.0) -> torch.Tensor:
    if x.numel() == 0:
        return torch.tensor(default, device=x.device, dtype=torch.float32)
    return _sanitize(x).mean()


def _points_to_gaussian(points_xy: np.ndarray, h: int, w: int, sigma: float, normalize_sum: bool = False) -> np.ndarray:
    den = np.zeros((h, w), dtype=np.float32)
    if points_xy is None or len(points_xy) == 0:
        return den
    radius = max(1, int(round(3.0 * float(sigma))))
    xs = np.arange(-radius, radius + 1, dtype=np.float32)
    ys = np.arange(-radius, radius + 1, dtype=np.float32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    k = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma)).astype(np.float32)
    if normalize_sum:
        k /= max(float(k.sum()), 1e-8)
    else:
        k /= max(float(k.max()), 1e-8)
    for x, y in points_xy:
        cx = int(round(float(x))); cy = int(round(float(y)))
        if cx < 0 or cx >= w or cy < 0 or cy >= h:
            continue
        x0 = max(0, cx - radius); x1 = min(w, cx + radius + 1)
        y0 = max(0, cy - radius); y1 = min(h, cy + radius + 1)
        kx0 = x0 - (cx - radius); kx1 = kx0 + (x1 - x0)
        ky0 = y0 - (cy - radius); ky1 = ky0 + (y1 - y0)
        if normalize_sum:
            den[y0:y1, x0:x1] += k[ky0:ky1, kx0:kx1]
        else:
            den[y0:y1, x0:x1] = np.maximum(den[y0:y1, x0:x1], k[ky0:ky1, kx0:kx1])
    return den


def _build_point_target(points_list: List[torch.Tensor], h: int, w: int, stride: int, sigma: float, device: torch.device, normalize_sum: bool = False) -> torch.Tensor:
    arrs = []
    for pts in points_list:
        if pts.numel() == 0:
            pts_np = np.zeros((0, 2), dtype=np.float32)
        else:
            pts_np = pts.detach().cpu().numpy().astype(np.float32).copy()
            pts_np[:, 0] /= float(stride)
            pts_np[:, 1] /= float(stride)
        arrs.append(torch.from_numpy(_points_to_gaussian(pts_np, h, w, sigma=sigma, normalize_sum=normalize_sum)).unsqueeze(0))
    return torch.stack(arrs, dim=0).to(device=device, dtype=torch.float32)


def _resize_mask(mask: Optional[torch.Tensor], size, device, default: float = 1.0) -> torch.Tensor:
    if mask is None:
        return torch.full((1, 1, size[0], size[1]), default, device=device)
    m = mask.to(device)
    if m.dim() == 3:
        m = m.unsqueeze(1)
    return F.interpolate(m.float(), size=size, mode="nearest")


def _dilate(x: torch.Tensor, k: int) -> torch.Tensor:
    k = int(k) | 1
    return (F.max_pool2d(x, kernel_size=k, stride=1, padding=k // 2) > 0).float()


def _weighted_bce_logits(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor, pos_weight: float = 1.0) -> torch.Tensor:
    logits = _sanitize(logits, posinf=16.0, neginf=-16.0)
    target = target.clamp(0.0, 1.0)
    w = weight * (1.0 + (float(pos_weight) - 1.0) * target)
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return _safe_mean(loss * w)


def _hard_negative_mean(score: torch.Tensor, pos_mask: torch.Tensor, valid: torch.Tensor, topk: int) -> torch.Tensor:
    vals = []
    det = score.detach() * (1.0 - pos_mask) * valid
    raw = score * (1.0 - pos_mask) * valid
    for bi in range(score.shape[0]):
        flat_det = det[bi, 0].flatten()
        flat_raw = raw[bi, 0].flatten()
        if flat_det.numel() == 0:
            continue
        k = min(int(topk), flat_det.numel())
        if k <= 0:
            continue
        _, idx = torch.topk(flat_det, k=k, largest=True, sorted=False)
        vals.append(flat_raw[idx].mean())
    if len(vals) == 0:
        return torch.tensor(0.0, device=score.device)
    return torch.stack(vals).mean()


@dataclass
class V523BICLossConfig:
    point_sigma: float = 1.15
    support_sigma: float = 2.35
    bg_dilate_kernel: int = 9
    hard_neg_topk: int = 768
    count_clip: float = 1e5

    # main losses
    w_score: float = 1.25
    w_count: float = 0.72
    w_log_count: float = 0.20
    w_center: float = 0.55
    w_origin: float = 0.65
    w_reliability: float = 0.40
    w_component_quality: float = 0.35

    # bias-isolation losses
    w_bg_veto: float = 0.38
    w_line_veto: float = 0.22
    w_hollow_veto: float = 0.25
    w_antigrid_veto: float = 0.30
    w_hard_negative: float = 0.18
    w_scale_entropy: float = 0.018
    w_scale_balance: float = 0.020

    # audit supervision on source only: building-like near source points, background outside support
    w_audit: float = 0.16
    pos_weight_center: float = 8.0
    pos_weight_object: float = 5.0

    # curriculum
    warm_count_weight_epoch: int = 8
    audit_start_epoch: int = 10


def compute_v523_bic_losses(
    out: Dict[str, Any],
    batch: Dict[str, Any],
    epoch: int,
    model_cfg: Any,
    loss_cfg: Optional[V523BICLossConfig] = None,
) -> Dict[str, torch.Tensor]:
    loss_cfg = loss_cfg or V523BICLossConfig()
    device = out["score_density_c"].device
    h, w = out["score_density_c"].shape[-2:]
    stride = int(getattr(model_cfg, "stride", 4))
    points = batch["points"]
    gt_count = torch.as_tensor(batch["count"], dtype=torch.float32, device=device).view(-1, 1).clamp(0.0, loss_cfg.count_clip)

    point_peak = _build_point_target(points, h, w, stride, loss_cfg.point_sigma, device, normalize_sum=False)
    point_mass = _build_point_target(points, h, w, stride, loss_cfg.point_sigma, device, normalize_sum=True)
    support = _build_point_target(points, h, w, stride, loss_cfg.support_sigma, device, normalize_sum=False)
    point_bin = (point_peak > 0.05).float()
    support_bin = (support > 0.05).float()
    fg = _dilate(point_bin, loss_cfg.bg_dilate_kernel)

    valid_c = out.get("valid_c")
    if valid_c is None:
        valid_c = _resize_mask(batch.get("valid_mask", None), (h, w), device, 1.0)
    neutral_c = out.get("neutral_c")
    if neutral_c is None:
        neutral_c = _resize_mask(batch.get("neutral_mask", None), (h, w), device, 0.0)
    effective_valid = (valid_c * (1.0 - neutral_c)).clamp(0.0, 1.0)

    score = _sanitize(out["score_density_c"]).clamp_min(0.0)
    # Density loss is relative: prevents high-count images from dominating all learning.
    denom = torch.clamp(gt_count.view(-1, 1, 1, 1), min=1.0)
    score_loss = F.smooth_l1_loss(score / denom, point_mass / denom, reduction="none")
    score_loss = _safe_mean(score_loss * (1.0 + 4.0 * support_bin) * effective_valid)

    pred_count = _sanitize(out["pred_count"]).view(-1, 1).clamp(0.0, loss_cfg.count_clip)
    count_l1 = torch.abs(pred_count - gt_count) / torch.clamp(gt_count + 1.0, min=1.0)
    count_loss = count_l1.mean()
    log_count_loss = torch.abs(torch.log1p(pred_count) - torch.log1p(gt_count)).mean()
    cw = min(1.0, float(epoch) / max(1, int(loss_cfg.warm_count_weight_epoch)))

    center_logits = torch.logit(out["center_core_c"].clamp(1e-4, 1.0 - 1e-4))
    center_loss = _weighted_bce_logits(center_logits, point_peak.clamp(0, 1), effective_valid, loss_cfg.pos_weight_center)

    obj = out["object_origin_c"].clamp(1e-4, 1 - 1e-4)
    bg = out["background_origin_c"].clamp(1e-4, 1 - 1e-4)
    # Use float32 to avoid autocast issues with binary_cross_entropy
    with torch.autocast(device_type=device.type, enabled=False):
        origin_loss = F.binary_cross_entropy(obj.float(), support.clamp(0, 1).float(), reduction="none")
    origin_loss = origin_loss * (1.0 + (loss_cfg.pos_weight_object - 1.0) * support_bin)
    origin_loss = _safe_mean(origin_loss * effective_valid)
    bg_target = ((1.0 - fg) * effective_valid).detach()
    with torch.autocast(device_type=device.type, enabled=False):
        bg_loss = F.binary_cross_entropy(bg.float(), bg_target.float(), reduction="none")
    bg_loss = _safe_mean(bg_loss * effective_valid)
    origin_loss = origin_loss + 0.55 * bg_loss

    with torch.autocast(device_type=device.type, enabled=False):
        reliability_loss = F.binary_cross_entropy(out["reliability_c"].clamp(1e-4, 1 - 1e-4).float(), support.clamp(0, 1).float(), reduction="none")
    reliability_loss = _safe_mean(reliability_loss * (1.0 + 2.0 * support_bin) * effective_valid)
    with torch.autocast(device_type=device.type, enabled=False):
        comp_q_loss = F.binary_cross_entropy(out["component_quality_c"].clamp(1e-4, 1 - 1e-4).float(), support.clamp(0, 1).float(), reduction="none")
    comp_q_loss = _safe_mean(comp_q_loss * (1.0 + 2.0 * support_bin) * effective_valid)

    # Bias isolation: outside object support, responses that coincide with road-block/hollow/line evidence are expensive.
    bg_veto_loss = _safe_mean(score * out["background_like_c"].detach() * (1.0 - fg) * effective_valid)
    line_veto_loss = _safe_mean(score * out["line_coherence_c"].detach() * (1.0 - support_bin) * effective_valid)
    hollow_veto_loss = _safe_mean(score * out["hollow_veto_c"].detach() * (1.0 - support_bin) * effective_valid)
    antigrid_veto_loss = _safe_mean(score * out["anti_grid_void_c"].detach() * (1.0 - support_bin) * effective_valid)
    hard_negative_loss = _hard_negative_mean(score, fg, effective_valid, loss_cfg.hard_neg_topk)

    scale_prob = out["scale_prob"].clamp(1e-6, 1.0)
    scale_entropy = -(scale_prob * torch.log(scale_prob)).sum(dim=1, keepdim=True)
    # Penalize excessive certainty everywhere, but not at object centers.
    scale_entropy_loss = _safe_mean((1.10 - scale_entropy) * (1.0 - support_bin) * effective_valid)
    scale_balance = scale_prob.mean(dim=(-2, -1))
    scale_balance_loss = ((scale_balance.mean(dim=0) - torch.tensor([0.36, 0.34, 0.30], device=device)) ** 2).mean()

    # Source audit: on RSOC source, target points should be building-like; background outside support should be background-like.
    audit_loss = torch.tensor(0.0, device=device)
    audit_prob = out.get("audit_prob", None)
    if audit_prob is not None and epoch >= int(loss_cfg.audit_start_epoch):
        # 0 vehicle, 1 building, 2 ship, 3 background
        build_prob = audit_prob[:, 1:2].clamp(1e-4, 1.0)
        bg_prob = audit_prob[:, 3:4].clamp(1e-4, 1.0)
        audit_loss = -_safe_mean(torch.log(build_prob) * support_bin * effective_valid) - 0.35 * _safe_mean(torch.log(bg_prob) * (1.0 - fg) * effective_valid)

    total = (
        loss_cfg.w_score * score_loss
        + cw * loss_cfg.w_count * count_loss
        + cw * loss_cfg.w_log_count * log_count_loss
        + loss_cfg.w_center * center_loss
        + loss_cfg.w_origin * origin_loss
        + loss_cfg.w_reliability * reliability_loss
        + loss_cfg.w_component_quality * comp_q_loss
        + loss_cfg.w_bg_veto * bg_veto_loss
        + loss_cfg.w_line_veto * line_veto_loss
        + loss_cfg.w_hollow_veto * hollow_veto_loss
        + loss_cfg.w_antigrid_veto * antigrid_veto_loss
        + loss_cfg.w_hard_negative * hard_negative_loss
        + loss_cfg.w_scale_entropy * scale_entropy_loss
        + loss_cfg.w_scale_balance * scale_balance_loss
        + loss_cfg.w_audit * audit_loss
    )
    return {
        "total": total,
        "score": score_loss.detach(),
        "count": count_loss.detach(),
        "log_count": log_count_loss.detach(),
        "center": center_loss.detach(),
        "origin": origin_loss.detach(),
        "reliability": reliability_loss.detach(),
        "quality": comp_q_loss.detach(),
        "bg_veto": bg_veto_loss.detach(),
        "line_veto": line_veto_loss.detach(),
        "hollow_veto": hollow_veto_loss.detach(),
        "antigrid_veto": antigrid_veto_loss.detach(),
        "hard_negative": hard_negative_loss.detach(),
        "scale_entropy": scale_entropy_loss.detach(),
        "scale_balance": scale_balance_loss.detach(),
        "audit": audit_loss.detach(),
        "count_weight": torch.tensor(cw, device=device),
    }


__all__ = ["V523BICLossConfig", "compute_v523_bic_losses"]
