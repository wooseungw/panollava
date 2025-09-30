from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


def resolve_module_dtype(
    cache: Dict[str, torch.dtype],
    cache_key: str,
    module: Optional[nn.Module],
    default: Optional[torch.dtype] = None,
) -> Optional[torch.dtype]:
    """Cache-aware dtype resolution for modules with optional parameters/buffers."""
    if cache_key in cache:
        return cache[cache_key]
    if module is None:
        return default
    dtype: Optional[torch.dtype] = None
    try:
        param = next(module.parameters())
        dtype = param.dtype
    except (StopIteration, AttributeError):
        dtype = None
    if dtype is None:
        try:
            buf = next(module.buffers())
            dtype = buf.dtype
        except (StopIteration, AttributeError):
            dtype = None
    if dtype is None:
        dtype = default
    if dtype is not None:
        cache[cache_key] = dtype
    return dtype
