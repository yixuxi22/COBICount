"""Fast CPU smoke tests for the public COBICount API."""

from __future__ import annotations

import torch

from cobicount.losses import V523BICLossConfig, compute_v523_bic_losses
from cobicount.model import COBICount, V523BICConfig, mass_preserve_resize


def tiny_config() -> V523BICConfig:
    return V523BICConfig(backbone_base=8, fpn_channels=16, max_points_cap=32)


def test_mass_preserve_resize() -> None:
    source = torch.rand(2, 1, 8, 8)
    resized = mass_preserve_resize(source, (32, 32))
    assert resized.shape == (2, 1, 32, 32)
    assert torch.allclose(source.sum(dim=(-2, -1)), resized.sum(dim=(-2, -1)), rtol=1e-4, atol=1e-4)


def test_model_forward_cpu() -> None:
    model = COBICount(tiny_config()).eval()
    image = torch.randn(2, 3, 64, 64)
    with torch.inference_mode():
        output = model(image)
    assert output["score_density_c"].shape == (2, 1, 16, 16)
    assert output["pred_count"].shape == (2, 1)
    assert torch.isfinite(output["pred_count"]).all()
    assert len(output["pred_points_c"]) == 2


def test_loss_backward_cpu() -> None:
    config = tiny_config()
    model = COBICount(config).train()
    image = torch.randn(2, 3, 64, 64)
    points = [torch.tensor([[20.0, 20.0]]), torch.tensor([[40.0, 36.0]])]
    batch = {
        "image": image,
        "points": points,
        "count": torch.tensor([1.0, 1.0]),
        "valid_mask": torch.ones(2, 1, 64, 64),
        "neutral_mask": torch.zeros(2, 1, 64, 64),
    }
    output = model(image, valid_mask=batch["valid_mask"], neutral_mask=batch["neutral_mask"])
    losses = compute_v523_bic_losses(output, batch, epoch=10, model_cfg=config, loss_cfg=V523BICLossConfig())
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
