from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch


_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_MODEL_PATH = Path(__file__).with_name("nn_model.pt")


def _load_base_model() -> torch.nn.Module:
    """
    Load a PyTorch model from `nn_model.pt`.

    Supports:
    - torch.jit (Script / Trace) modules
    - pickled `nn.Module` instances
    """
    if not _MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {_MODEL_PATH}")

    # Try TorchScript first (common for deployment).
    try:
        base = torch.jit.load(str(_MODEL_PATH), map_location=_DEVICE)
        if isinstance(base, torch.nn.Module):
            base.eval()
            return base
    except Exception:
        pass

    # Fallback: plain torch.load (e.g., pickled nn.Module).
    obj: Any = torch.load(str(_MODEL_PATH), map_location=_DEVICE)
    if isinstance(obj, torch.nn.Module):
        obj.eval()
        return obj

    raise TypeError(
        f"Unsupported object loaded from {_MODEL_PATH!s}: {type(obj)!r}. "
        "Expected a TorchScript module or an nn.Module instance."
    )


class _WrappedModel:
    """
    Thin wrapper with a stable `.predict(x)` API.

    - `x` is a NumPy array of shape (N, 6)
    - Returns a scalar score; app clamps it to [1, 5]
    """

    def __init__(self) -> None:
        self._base = _load_base_model().to(_DEVICE)

    @torch.no_grad()
    def predict(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != 6:
            raise ValueError(f"Expected input shape (N, 6), got {x.shape!r}")

        t = torch.from_numpy(x).to(_DEVICE)
        out = self._base(t)

        # Unpack simple (output, hidden) tuples, etc.
        if isinstance(out, (list, tuple)) and len(out) > 0:
            out = out[0]

        if not isinstance(out, torch.Tensor):
            raise TypeError(f"Model output must be a Tensor, got {type(out)!r}")

        # Reduce to a single scalar score.
        if out.ndim == 0:
            score = out.item()
        elif out.ndim == 1:
            # One value per sample -> average.
            score = out.float().mean().item()
        elif out.ndim >= 2:
            n_classes = out.shape[1]
            if n_classes == 5:
                # Treat as logits for 5-point rating classes.
                probs = torch.softmax(out.float(), dim=1)
                class_ids = torch.arange(1, 6, device=probs.device).view(1, 5)
                per_sample_score = (probs * class_ids).sum(dim=1)
                score = per_sample_score.mean().item()
            elif n_classes == 1:
                score = out.float().mean().item()
            else:
                # Generic fallback: global mean.
                score = out.float().mean().item()
        else:
            score = out.float().mean().item()

        return float(score)


# This is what `app.py` imports.
model = _WrappedModel()

