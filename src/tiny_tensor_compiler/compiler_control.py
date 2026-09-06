from __future__ import annotations

import math
import time


def normalize_compiler_timeout(timeout: float | None) -> float | None:
    """Validate an optional wall-clock bound for one external compiler process."""
    return _normalize_positive_seconds(timeout, name="compiler_timeout")


def normalize_compile_deadline(deadline: float | None) -> float | None:
    """Validate an optional total wall-clock budget for one native build attempt."""
    return _normalize_positive_seconds(deadline, name="compile_deadline")


def start_compile_deadline(duration: float | None) -> float | None:
    """Convert a normalized duration to one absolute monotonic deadline."""
    if duration is None:
        return None
    return time.monotonic() + duration


def remaining_compile_deadline(deadline_at: float | None) -> float | None:
    """Return the remaining duration for an absolute monotonic deadline."""
    if deadline_at is None:
        return None
    return deadline_at - time.monotonic()


def _normalize_positive_seconds(value: float | None, *, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a positive finite number of seconds or None")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{name} must be a positive finite number of seconds")
    return normalized
