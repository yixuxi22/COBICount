# -*- coding: utf-8 -*-
from __future__ import annotations

"""COBICount model definition.

The public class name ``COBICount`` is an alias of ``REDCountV523BIC``.
The legacy class name is retained so previously trained checkpoints remain
compatible with the cleaned open-source release.

The model factorizes the final score into Candidate Evidence (CE), Component
Acceptance (CA), Bias Isolation (BI), and an effective-valid mask. The audit
channels are auxiliary diagnostic responses; they are not calibrated target
category probabilities and do not gate the final count.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def _sanitize(x: torch.Tensor, nan: float = 0.0, posinf: float = 1e4, neginf: float = -1e4) -> torch.Tensor:
    return torch.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)


def mass_preserve_resize(x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    h0, w0 = x.shape[-2:]
    h1, w1 = size
    if (h0, w0) == (h1, w1):
        return x
    y = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
    return y * float(h0 * w0) / float(max(1, h1 * w1))


def interp_like(x: torch.Tensor, hw: Tuple[int, int], mode: str = "bilinear") -> torch.Tensor:
    if x.dim() == 3:
        x = x.unsqueeze(1)
    if mode == "nearest":
        return F.interpolate(x, size=hw, mode="nearest")
    return F.interpolate(x, size=hw, mode=mode, align_corners=False)


def norm01(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = _sanitize(x)
    mn = x.amin(dim=(-2, -1), keepdim=True)
    mx = x.amax(dim=(-2, -1), keepdim=True)
    return (x - mn) / (mx - mn + eps)


class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: Optional[int] = None, dilation: int = 1):
        super().__init__()
        if p is None:
            p = dilation * (k // 2)
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, dilation: int = 1):
        super().__init__()
        self.conv1 = ConvBNAct(in_ch, out_ch, 3, stride, dilation=dilation)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.skip = nn.Identity() if in_ch == out_ch and stride == 1 else nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv2(self.conv1(x)) + self.skip(x))


class TinyFPNBackbone(nn.Module):
    """A compact ResNet-FPN backbone, stable on small source-only datasets."""

    def __init__(self, base: int = 48, fpn_ch: int = 128):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNAct(3, base, 3, 2),
            ConvBNAct(base, base, 3, 1),
            ConvBNAct(base, base, 3, 1),
        )
        self.l2 = nn.Sequential(ResidualBlock(base, base * 2, stride=2), ResidualBlock(base * 2, base * 2))       # stride 4
        self.l3 = nn.Sequential(ResidualBlock(base * 2, base * 4, stride=2), ResidualBlock(base * 4, base * 4))   # stride 8
        self.l4 = nn.Sequential(ResidualBlock(base * 4, base * 6, stride=2, dilation=1), ResidualBlock(base * 6, base * 6, dilation=2))  # stride 16
        self.l5 = nn.Sequential(ResidualBlock(base * 6, base * 8, stride=2, dilation=1), ResidualBlock(base * 8, base * 8, dilation=2))  # stride 32

        self.p2 = nn.Conv2d(base * 2, fpn_ch, 1)
        self.p3 = nn.Conv2d(base * 4, fpn_ch, 1)
        self.p4 = nn.Conv2d(base * 6, fpn_ch, 1)
        self.p5 = nn.Conv2d(base * 8, fpn_ch, 1)
        self.s2 = ConvBNAct(fpn_ch, fpn_ch, 3, 1)
        self.s3 = ConvBNAct(fpn_ch, fpn_ch, 3, 1)
        self.s4 = ConvBNAct(fpn_ch, fpn_ch, 3, 1)
        self.s5 = ConvBNAct(fpn_ch, fpn_ch, 3, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        c2 = self.l2(x)
        c3 = self.l3(c2)
        c4 = self.l4(c3)
        c5 = self.l5(c4)
        p5 = self.p5(c5)
        p4 = self.p4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="bilinear", align_corners=False)
        p3 = self.p3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="bilinear", align_corners=False)
        p2 = self.p2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="bilinear", align_corners=False)
        return self.s2(p2), self.s3(p3), self.s4(p4), self.s5(p5)


class Head(nn.Module):
    def __init__(self, in_ch: int, out_ch: int = 1, hidden: Optional[int] = None, bias_init: float = 0.0):
        super().__init__()
        hidden = hidden or max(16, in_ch // 2)
        # 1x1 lightweight heads keep V5.19-style many-map interpretability without excessive cost.
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, out_ch, 1, bias=True),
        )
        nn.init.constant_(self.net[-1].bias, bias_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class V523BICConfig:
    # geometry
    stride: int = 4
    backbone_base: int = 32
    fpn_channels: int = 96
    count_clip: float = 1e5

    # branch temperatures / gains
    density_softplus_beta: float = 1.0
    score_temperature: float = 0.60
    reliability_gain: float = 1.00
    component_quality_gain: float = 1.10

    # bias isolation
    background_veto_gain: float = 0.72
    line_veto_gain: float = 0.58
    hollow_veto_gain: float = 0.48
    anti_grid_veto_gain: float = 0.60
    neutral_veto_gain: float = 1.00

    # candidate field smoothing
    primitive_pool: int = 5
    high_recall_pool: int = 5
    center_kernel: int = 5
    line_kernel: int = 9
    lowfreq_kernel: int = 25

    # point extraction for visualization
    point_threshold_rel: float = 0.28
    point_threshold_abs: float = 0.020
    min_center_distance_c: float = 3.2
    max_points_cap: int = 1024

    # model behavior
    use_mass_preserve_hr: bool = True


class REDCountV523BIC(nn.Module):
    """Bias-Isolated Component Counter."""

    AUDIT_NAMES = ("vehicle_like", "building_like", "ship_like", "background_like")

    def __init__(self, cfg: Optional[V523BICConfig] = None):
        super().__init__()
        cfg = cfg or V523BICConfig()
        self.cfg = cfg
        c = cfg.fpn_channels
        self.backbone = TinyFPNBackbone(base=cfg.backbone_base, fpn_ch=c)
        self.fuse = nn.Sequential(
            ConvBNAct(c * 4, c, 3, 1),
            ConvBNAct(c, c, 3, 1),
        )

        # V5.2E-like high recall substrate and V5.3/V5.4-like bias judges.
        self.primitive_head = Head(c, 1, bias_init=-0.10)
        self.birth_head = Head(c, 1, bias_init=-0.20)
        self.high_recall_head = Head(c, 1, bias_init=-0.05)

        self.scale_head = Head(c, 3, bias_init=0.0)       # small/mid/large gate logits
        self.small_score_head = Head(c, 1, bias_init=-0.35)
        self.mid_score_head = Head(c, 1, bias_init=-0.20)
        self.large_score_head = Head(c, 1, bias_init=-0.15)

        self.object_origin_head = Head(c, 1, bias_init=-0.15)
        self.background_origin_head = Head(c, 1, bias_init=-0.05)
        self.line_coherence_head = Head(c, 1, bias_init=-0.25)
        self.hollow_veto_head = Head(c, 1, bias_init=-0.25)
        self.anti_grid_void_head = Head(c, 1, bias_init=-0.20)
        self.component_quality_head = Head(c, 1, bias_init=-0.05)
        self.reliability_head = Head(c, 1, bias_init=0.0)
        self.center_core_head = Head(c, 1, bias_init=-0.10)
        self.audit_head = Head(c, 4, bias_init=0.0)

        # fixed derivative filters for interpretable primitive diagnostics
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3) / 8.0
        sobel_y = sobel_x.transpose(-1, -2)
        lap = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x, persistent=False)
        self.register_buffer("sobel_y", sobel_y, persistent=False)
        self.register_buffer("lap_kernel", lap, persistent=False)

    def _coarse_feature(self, x: torch.Tensor) -> torch.Tensor:
        p2, p3, p4, p5 = self.backbone(x)
        hw = p2.shape[-2:]
        p3u = F.interpolate(p3, size=hw, mode="bilinear", align_corners=False)
        p4u = F.interpolate(p4, size=hw, mode="bilinear", align_corners=False)
        p5u = F.interpolate(p5, size=hw, mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([p2, p3u, p4u, p5u], dim=1))

    def _valid_neutral_from_image(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x is normalized ImageNet; convert roughly back to intensity.
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        img = (x * std + mean).clamp(0.0, 1.0)
        gray = img.mean(dim=1, keepdim=True)
        valid = (gray > 0.035).float()
        neutral = 1.0 - valid
        neutral = F.max_pool2d(neutral, kernel_size=9, stride=1, padding=4)
        valid = valid * (1.0 - neutral)
        return valid, neutral.clamp(0.0, 1.0)

    def _line_and_blob_priors(self, primitive: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gx = F.conv2d(primitive, self.sobel_x, padding=1)
        gy = F.conv2d(primitive, self.sobel_y, padding=1)
        grad = torch.sqrt(gx * gx + gy * gy + 1e-6)
        lap = F.conv2d(primitive, self.lap_kernel, padding=1)
        k = int(self.cfg.lowfreq_kernel) | 1
        low = F.avg_pool2d(primitive, kernel_size=k, stride=1, padding=k // 2)
        return norm01(grad), norm01(lap.abs()), norm01(low)

    def _density_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        # softplus is more stable than sigmoid for count integral.
        return F.softplus(_sanitize(logits, posinf=10.0, neginf=-10.0), beta=float(self.cfg.density_softplus_beta))

    @torch.no_grad()
    def _extract_points(self, score: torch.Tensor, valid: Optional[torch.Tensor] = None, budget: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        if score.dim() == 3:
            score = score.unsqueeze(1)
        score = _sanitize(score).clamp_min(0.0)
        if valid is not None:
            score = score * valid.clamp(0.0, 1.0)
        b, _, h, w = score.shape
        min_d2 = float(self.cfg.min_center_distance_c) ** 2
        pts_all: List[torch.Tensor] = []
        for bi in range(b):
            s = score[bi, 0].detach().clone()
            smax = float(s.max().item()) if s.numel() > 0 else 0.0
            thr = max(float(self.cfg.point_threshold_abs), float(self.cfg.point_threshold_rel) * smax)
            if smax <= 0.0:
                pts_all.append(torch.zeros((0, 2), device=score.device, dtype=torch.float32))
                continue
            if budget is not None:
                raw = float(torch.nan_to_num(budget[bi].view(-1)[0].detach(), nan=0.0, posinf=0.0, neginf=0.0).item())
                max_pts = min(int(self.cfg.max_points_cap), max(1, int(round(raw * 1.55 + 16))))
            else:
                max_pts = int(self.cfg.max_points_cap)
            pts: List[torch.Tensor] = []
            for _ in range(max_pts):
                val, idx = torch.max(s.reshape(-1), dim=0)
                if float(val.item()) < thr:
                    break
                y = torch.div(idx, w, rounding_mode="floor").float()
                x = (idx % w).float()
                accept = True
                if len(pts) > 0:
                    prev = torch.stack(pts, dim=0)
                    d2 = (prev[:, 0] - x) ** 2 + (prev[:, 1] - y) ** 2
                    accept = bool(torch.all(d2 >= min_d2).item())
                # NMS erase
                xi = int(x.item()); yi = int(y.item())
                rad = max(1, int(round(self.cfg.min_center_distance_c)))
                x0 = max(0, xi - rad); x1 = min(w, xi + rad + 1)
                y0 = max(0, yi - rad); y1 = min(h, yi + rad + 1)
                s[y0:y1, x0:x1] = 0.0
                if accept:
                    pts.append(torch.stack([x, y]))
            pts_all.append(torch.stack(pts, dim=0) if pts else torch.zeros((0, 2), device=score.device, dtype=torch.float32))
        return pts_all

    @torch.no_grad()
    def _component_label_map(self, support: torch.Tensor, pred_points: List[torch.Tensor]) -> torch.Tensor:
        b, _, h, w = support.shape
        label = torch.zeros((b, 1, h, w), device=support.device, dtype=torch.float32)
        yy, xx = torch.meshgrid(
            torch.arange(h, device=support.device, dtype=torch.float32),
            torch.arange(w, device=support.device, dtype=torch.float32),
            indexing="ij",
        )
        for bi in range(b):
            pts = pred_points[bi]
            if pts.numel() == 0:
                continue
            best = None
            lab = None
            for pi, p in enumerate(pts):
                d = (xx - p[0]) ** 2 + (yy - p[1]) ** 2
                if best is None:
                    best = d; lab = torch.full_like(d, float(pi + 1))
                else:
                    m = d < best
                    best = torch.where(m, d, best)
                    lab = torch.where(m, torch.full_like(lab, float(pi + 1)), lab)
            label[bi, 0] = lab * (support[bi, 0] > 0.10).float()
        return label

    @torch.no_grad()
    def audit_counts_from_points(self, out: Dict[str, Any]) -> List[Dict[str, int]]:
        pts_list: List[torch.Tensor] = out.get("pred_points_c", [])
        audit_maps = out.get("audit_prob", None)
        if audit_maps is None or not pts_list:
            return []
        b, c, h, w = audit_maps.shape
        counts: List[Dict[str, int]] = []
        for bi in range(b):
            d = {name: 0 for name in self.AUDIT_NAMES}
            pts = pts_list[bi]
            for p in pts:
                x = int(max(0, min(w - 1, round(float(p[0].item())))))
                y = int(max(0, min(h - 1, round(float(p[1].item())))))
                ci = int(torch.argmax(audit_maps[bi, :, y, x]).item())
                d[self.AUDIT_NAMES[ci]] += 1
            counts.append(d)
        return counts

    def forward(self, x: torch.Tensor, valid_mask: Optional[torch.Tensor] = None, neutral_mask: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        x = _sanitize(x, posinf=10.0, neginf=-10.0)
        b, _, H, W = x.shape
        feat = self._coarse_feature(x)
        h, w = feat.shape[-2:]

        valid_img, neutral_img = self._valid_neutral_from_image(x)
        if valid_mask is not None:
            valid_img = valid_mask.float()
            if valid_img.dim() == 3:
                valid_img = valid_img.unsqueeze(1)
        if neutral_mask is not None:
            neutral_img = neutral_mask.float()
            if neutral_img.dim() == 3:
                neutral_img = neutral_img.unsqueeze(1)
        valid_c = interp_like(valid_img, (h, w), "nearest").clamp(0.0, 1.0)
        neutral_c = interp_like(neutral_img, (h, w), "nearest").clamp(0.0, 1.0)
        effective_valid = (valid_c * (1.0 - neutral_c)).clamp(0.0, 1.0)

        primitive = torch.sigmoid(_sanitize(self.primitive_head(feat), posinf=10.0, neginf=-10.0)) * effective_valid
        grad_prior, lap_prior, low_prior = self._line_and_blob_priors(primitive)
        birth = torch.sigmoid(_sanitize(self.birth_head(feat), posinf=10.0, neginf=-10.0))
        birth = birth * (1.0 - 0.35 * grad_prior).clamp(0.0, 1.0) * effective_valid
        high_recall = torch.sigmoid(_sanitize(self.high_recall_head(feat), posinf=10.0, neginf=-10.0))
        k_hr = int(self.cfg.high_recall_pool) | 1
        high_recall = F.max_pool2d(high_recall * (0.50 + 0.50 * primitive), k_hr, stride=1, padding=k_hr // 2) * effective_valid

        scale_logits = _sanitize(self.scale_head(feat), posinf=10.0, neginf=-10.0)
        scale_prob = torch.softmax(scale_logits, dim=1)
        small_den_raw = self._density_from_logits(self.small_score_head(feat)) * effective_valid
        mid_den_raw = self._density_from_logits(self.mid_score_head(feat)) * effective_valid
        large_den_raw = self._density_from_logits(self.large_score_head(feat)) * effective_valid
        multi_den = (scale_prob[:, 0:1] * small_den_raw + scale_prob[:, 1:2] * mid_den_raw + scale_prob[:, 2:3] * large_den_raw)

        obj_origin = torch.sigmoid(_sanitize(self.object_origin_head(feat), posinf=10.0, neginf=-10.0))
        bg_origin = torch.sigmoid(_sanitize(self.background_origin_head(feat), posinf=10.0, neginf=-10.0))
        line_net = torch.sigmoid(_sanitize(self.line_coherence_head(feat), posinf=10.0, neginf=-10.0))
        hollow_net = torch.sigmoid(_sanitize(self.hollow_veto_head(feat), posinf=10.0, neginf=-10.0))
        anti_grid_net = torch.sigmoid(_sanitize(self.anti_grid_void_head(feat), posinf=10.0, neginf=-10.0))
        comp_quality = torch.sigmoid(_sanitize(self.component_quality_head(feat), posinf=10.0, neginf=-10.0))
        reliability = torch.sigmoid(_sanitize(self.reliability_head(feat), posinf=10.0, neginf=-10.0))
        center_core = torch.sigmoid(_sanitize(self.center_core_head(feat), posinf=10.0, neginf=-10.0))

        line_coherence = torch.clamp(0.55 * line_net + 0.45 * grad_prior, 0.0, 1.0)
        hollow_veto = torch.clamp(0.60 * hollow_net + 0.40 * low_prior * (1.0 - center_core), 0.0, 1.0)
        anti_grid_void = torch.clamp(0.65 * anti_grid_net + 0.35 * lap_prior * (1.0 - center_core), 0.0, 1.0)
        background_like = torch.clamp(bg_origin * (0.50 + 0.50 * hollow_veto) + 0.25 * neutral_c, 0.0, 1.0)

        veto = (
            1.0
            - self.cfg.background_veto_gain * background_like
            - self.cfg.line_veto_gain * line_coherence
            - self.cfg.hollow_veto_gain * hollow_veto
            - self.cfg.anti_grid_veto_gain * anti_grid_void
            - self.cfg.neutral_veto_gain * neutral_c
        ).clamp(0.0, 1.0)
        component_accept = (center_core * obj_origin * (0.35 + 0.65 * comp_quality) * (0.35 + 0.65 * reliability)).clamp(0.0, 1.0)
        candidate = (0.34 * multi_den + 0.24 * primitive + 0.22 * birth + 0.20 * high_recall).clamp_min(0.0)
        rc_score = candidate * component_accept * veto * effective_valid

        # Mass calibration: keep count integral in coarse grid; loss handles numeric scale.
        score_density_c = _sanitize(rc_score).clamp(0.0, self.cfg.count_clip)
        pred_count = score_density_c.flatten(1).sum(dim=1, keepdim=True).clamp(0.0, self.cfg.count_clip)
        pred_known_count = pred_count
        pred_unknown_count = pred_count

        # High-resolution mass-preserving maps for visualization.
        if self.cfg.use_mass_preserve_hr:
            score_density_hr = mass_preserve_resize(score_density_c, (H, W))
            origen_score_hr = mass_preserve_resize(multi_den * obj_origin * reliability * effective_valid, (H, W))
        else:
            score_density_hr = interp_like(score_density_c, (H, W))
            origen_score_hr = interp_like(multi_den * obj_origin * reliability * effective_valid, (H, W))

        audit_logits = _sanitize(self.audit_head(feat), posinf=10.0, neginf=-10.0)
        # analytically bias audit channels so labels remain interpretable early in training.
        audit_logits = audit_logits.clone()
        audit_logits[:, 0:1] = audit_logits[:, 0:1] + 0.75 * scale_prob[:, 0:1] + 0.20 * scale_prob[:, 1:2] + 0.20 * center_core  # vehicle-like
        audit_logits[:, 1:2] = audit_logits[:, 1:2] + 0.80 * scale_prob[:, 2:3] + 0.35 * hollow_veto                 # building-like
        audit_logits[:, 2:3] = audit_logits[:, 2:3] + 0.50 * line_coherence + 0.20 * scale_prob[:, 1:2]                 # ship-like / elongated
        audit_logits[:, 3:4] = audit_logits[:, 3:4] + 1.10 * background_like + 0.35 * anti_grid_void                   # background-like
        audit_prob = torch.softmax(audit_logits, dim=1)

        if self.training:
            pts_c = [torch.zeros((0, 2), device=x.device, dtype=torch.float32) for _ in range(b)]
            comp_label = torch.zeros((b, 1, h, w), device=x.device, dtype=torch.float32)
        else:
            with torch.no_grad():
                pts_c = self._extract_points(score_density_c.detach(), effective_valid.detach(), pred_count.detach())
                comp_label = self._component_label_map((score_density_c.detach() > 0.02).float(), pts_c)

        out: Dict[str, Any] = {
            "score_density_c": score_density_c,
            "score_density": score_density_hr,
            "pred_count": pred_count,
            "pred_known_count": pred_known_count,
            "pred_unknown_count": pred_unknown_count,
            "origen_score_density_c": multi_den * obj_origin * reliability * effective_valid,
            "origen_score_density": origen_score_hr,
            "origin_score_density_c": multi_den * obj_origin * reliability * effective_valid,
            "origin_score_density": origen_score_hr,
            "rc_score_density_c": score_density_c,
            "rc_score_density": score_density_hr,
            "primitive_mass_v52e_c": primitive,
            "birth_consensus_c": birth,
            "high_recall_c": high_recall,
            "scale_logits": scale_logits,
            "scale_prob": scale_prob,
            "scale_small_c": small_den_raw * scale_prob[:, 0:1],
            "scale_mid_c": mid_den_raw * scale_prob[:, 1:2],
            "scale_large_c": large_den_raw * scale_prob[:, 2:3],
            "object_origin_c": obj_origin,
            "background_origin_c": bg_origin,
            "origin_margin_c": (obj_origin - bg_origin).clamp(-1.0, 1.0),
            "line_coherence_c": line_coherence,
            "hollow_veto_c": hollow_veto,
            "anti_grid_void_c": anti_grid_void,
            "component_quality_c": comp_quality,
            "reliability_c": reliability,
            "source_reliability_c": reliability,
            "center_core_c": center_core,
            "background_like_c": background_like,
            "audit_logits": audit_logits,
            "audit_prob": audit_prob,
            "audit_vehicle_like_c": audit_prob[:, 0:1],
            "audit_building_like_c": audit_prob[:, 1:2],
            "audit_ship_like_c": audit_prob[:, 2:3],
            "audit_background_like_c": audit_prob[:, 3:4],
            "valid_c": effective_valid,
            "neutral_c": neutral_c,
            "component_label_map": comp_label,
            "pred_points_c": pts_c,
            "stride": self.cfg.stride,
        }
        # high-res aliases for visualization convenience
        for k in [
            "primitive_mass_v52e", "birth_consensus", "high_recall", "scale_small", "scale_mid", "scale_large",
            "object_origin", "background_origin", "origin_margin", "line_coherence", "hollow_veto", "anti_grid_void",
            "component_quality", "reliability", "source_reliability", "center_core", "background_like",
            "audit_vehicle_like", "audit_building_like", "audit_ship_like", "audit_background_like",
        ]:
            kc = k + "_c"
            if kc in out:
                out[k] = interp_like(out[kc], (H, W), "bilinear")
        out["audit_counts"] = self.audit_counts_from_points(out) if not self.training else []
        return out


class ModelEMA523:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        import copy
        self.ema = copy.deepcopy(model).eval()
        self.decay = float(decay)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if k in msd:
                v.copy_(v * self.decay + msd[k].detach() * (1.0 - self.decay))


# Public paper-facing alias.
COBICount = REDCountV523BIC

__all__ = ["COBICount", "REDCountV523BIC", "V523BICConfig", "ModelEMA523", "mass_preserve_resize"]
