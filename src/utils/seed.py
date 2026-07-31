"""Global reproducibility helpers."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic_cuda: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducible runs.

    CUDA determinism is best-effort: it makes cuDNN pick deterministic
    kernels, which can be slower but removes run-to-run variance in results.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
