"""Protocol-level checks for the cleaned training entry point."""

from pathlib import Path


def test_training_entry_point_has_no_target_dataset_imports() -> None:
    source = (Path(__file__).parents[1] / "src" / "cobicount" / "cli" / "train.py").read_text(encoding="utf-8")
    assert "from cobicount.data import RSOCBuildingDataset" in source
    assert "DOTAProxyPositiveEvalDataset" not in source
    assert "DOTAFullImageEvalDataset" not in source
