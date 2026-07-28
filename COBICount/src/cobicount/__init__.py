"""COBICount: source-only remote sensing object counting."""

from .model import COBICount, ModelEMA523, REDCountV523BIC, V523BICConfig
from .losses import V523BICLossConfig, compute_v523_bic_losses

__version__ = "0.1.0"

__all__ = [
    "COBICount",
    "REDCountV523BIC",
    "V523BICConfig",
    "ModelEMA523",
    "V523BICLossConfig",
    "compute_v523_bic_losses",
    "__version__",
]
