from __future__ import annotations

import math
import time
from dataclasses import dataclass


def normalize_compiler_timeout(timeout: float | None) -> float | None:
    """Validate an optional wall-clock bound for one external compiler process."""
    return _normalize_positive_timeout(timeout, "compiler_timeout")


def normalize_compilation_timeout(timeout: float | None) -> float | None:
    """Validate an optional wall-clock bound for one native compilation transaction."""
    return _normalize_positive_timeout(timeout, "compilation_timeout")


@dataclass(frozen=True)
class CompilationDeadline:
    """One monotonic relative deadline shared across native artifact acquisition phases."""

    timeout: float
    expires_at: float

    @classmethod
    def start(cls, timeout: float | None) -> CompilationDeadline | None:
        if timeout is None:
            return None
        return cls(timeout=timeout, expires_at=time.monotonic() + timeout)

    def remaining(self) -> float:
        return self.expires_at - time.monotonic()


def _normalize_positive_timeout(timeout: float | None, name: str) -> float | None:
    if timeout is None:
        return None
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise TypeError(f"{name} must be a positive finite number of seconds or None")
    normalized = float(timeout)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{name} must be a positive finite number of seconds")
    return normalized
