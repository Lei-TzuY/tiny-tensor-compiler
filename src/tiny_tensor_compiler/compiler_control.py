from __future__ import annotations

import math


def normalize_compiler_timeout(timeout: float | None) -> float | None:
    """Validate an optional wall-clock bound for one external compiler process."""
    if timeout is None:
        return None
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise TypeError("compiler_timeout must be a positive finite number of seconds or None")
    normalized = float(timeout)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError("compiler_timeout must be a positive finite number of seconds")
    return normalized
