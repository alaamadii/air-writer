"""Timestamp-aware One Euro filtering with no OpenCV dependency."""

from __future__ import annotations

import math
from dataclasses import dataclass


def _smoothing_factor(delta_seconds: float, cutoff_hz: float) -> float:
    if delta_seconds <= 0 or cutoff_hz <= 0:
        raise ValueError("delta_seconds and cutoff_hz must be positive")
    time_constant = 1.0 / (2.0 * math.pi * cutoff_hz)
    return 1.0 / (1.0 + time_constant / delta_seconds)


@dataclass
class _LowPassFilter:
    value: float | None = None

    def apply(self, value: float, alpha: float) -> float:
        if self.value is None:
            self.value = value
        else:
            self.value = alpha * value + (1.0 - alpha) * self.value
        return self.value

    def reset(self) -> None:
        self.value = None


class OneEuroFilter:
    """Adaptive low-pass filter for a scalar sampled at millisecond timestamps.

    ``min_cutoff`` controls low-speed smoothing. ``beta`` raises the cutoff as
    speed increases to reduce lag. ``derivative_cutoff`` smooths the estimated
    speed. A long sample gap starts a fresh filter instead of bridging it.
    """

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.05,
        derivative_cutoff: float = 1.0,
        max_gap_ms: int = 250,
    ) -> None:
        if min_cutoff <= 0 or beta < 0 or derivative_cutoff <= 0:
            raise ValueError("cutoffs must be positive and beta must be non-negative")
        if max_gap_ms <= 0:
            raise ValueError("max_gap_ms must be positive")

        self.min_cutoff = min_cutoff
        self.beta = beta
        self.derivative_cutoff = derivative_cutoff
        self.max_gap_ms = max_gap_ms
        self._value_filter = _LowPassFilter()
        self._derivative_filter = _LowPassFilter()
        self._last_raw: float | None = None
        self._last_timestamp_ms: int | None = None

    def filter(self, value: float, timestamp_ms: int) -> float:
        """Return the filtered value for a strictly newer timestamp."""
        if not math.isfinite(value):
            raise ValueError("value must be finite")

        if self._last_timestamp_ms is None:
            return self._initialize(value, timestamp_ms)

        elapsed_ms = timestamp_ms - self._last_timestamp_ms
        if elapsed_ms <= 0:
            raise ValueError("timestamps must be strictly increasing")
        if elapsed_ms > self.max_gap_ms:
            self.reset()
            return self._initialize(value, timestamp_ms)

        delta_seconds = elapsed_ms / 1000.0
        assert self._last_raw is not None
        derivative = (value - self._last_raw) / delta_seconds
        derivative_alpha = _smoothing_factor(delta_seconds, self.derivative_cutoff)
        smoothed_derivative = self._derivative_filter.apply(derivative, derivative_alpha)
        cutoff = self.min_cutoff + self.beta * abs(smoothed_derivative)
        value_alpha = _smoothing_factor(delta_seconds, cutoff)
        filtered = self._value_filter.apply(value, value_alpha)

        self._last_raw = value
        self._last_timestamp_ms = timestamp_ms
        return filtered

    def _initialize(self, value: float, timestamp_ms: int) -> float:
        self._last_raw = value
        self._last_timestamp_ms = timestamp_ms
        self._derivative_filter.apply(0.0, 1.0)
        return self._value_filter.apply(value, 1.0)

    def reset(self) -> None:
        """Forget all samples so the next value is returned without lag."""
        self._value_filter.reset()
        self._derivative_filter.reset()
        self._last_raw = None
        self._last_timestamp_ms = None


class PointOneEuroFilter:
    """Apply matching One Euro filters to normalized x/y coordinates."""

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.05,
        derivative_cutoff: float = 1.0,
        max_gap_ms: int = 250,
    ) -> None:
        settings = (min_cutoff, beta, derivative_cutoff, max_gap_ms)
        self._x = OneEuroFilter(*settings)
        self._y = OneEuroFilter(*settings)

    def filter(self, point: tuple[float, float], timestamp_ms: int) -> tuple[float, float]:
        x, y = point
        return self._x.filter(x, timestamp_ms), self._y.filter(y, timestamp_ms)

    def reset(self) -> None:
        self._x.reset()
        self._y.reset()
