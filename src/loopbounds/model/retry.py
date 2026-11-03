"""Retries and detectability (design doc §3.4).

Per tool attempt: failure w.p. f; a failure is detectable w.p. d. Detectable
failures are retried up to R times by the tool layer; silent failures pass
through as corrupted content.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _geom(fd: float, R: int) -> float:
    """(1 - (fd)^{R+1}) / (1 - fd), with the fd -> 1 limit handled."""
    if np.isclose(fd, 1.0):
        return float(R + 1)
    return (1.0 - fd ** (R + 1)) / (1.0 - fd)


def p_clean(f: float, d: float, R: int) -> float:
    """Eq. 9: P_clean = (1-f)·(1-(fd)^{R+1})/(1-fd)."""
    return (1.0 - f) * _geom(f * d, R)


def p_silent(f: float, d: float, R: int) -> float:
    """Eq. 10: P_silent = f(1-d)·(1-(fd)^{R+1})/(1-fd)."""
    return f * (1.0 - d) * _geom(f * d, R)


def p_exhaust(f: float, d: float, R: int) -> float:
    """Eq. 11: P_exhaust = (fd)^{R+1}."""
    return (f * d) ** (R + 1)


def expected_attempts(f: float, d: float, R: int) -> float:
    """Eq. 12: E[A] = (1-(fd)^{R+1})/(1-fd)."""
    return _geom(f * d, R)


def clean_ceiling(f: float, d: float) -> float:
    """Eq. 13: lim_{R→∞} P_clean = (1-f)/(1-fd) (< 1 whenever d < 1 and f > 0)."""
    return (1.0 - f) / (1.0 - f * d)


@dataclass(frozen=True)
class Delivery:
    clean: float
    silent: float
    exhaust: float
    attempts: float


def delivery(f: float, d: float, R: int) -> Delivery:
    return Delivery(p_clean(f, d, R), p_silent(f, d, R), p_exhaust(f, d, R), expected_attempts(f, d, R))
