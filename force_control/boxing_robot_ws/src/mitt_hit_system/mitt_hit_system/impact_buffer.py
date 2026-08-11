"""Bounded pre-contact and post-contact wrench buffering."""

from collections import deque
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class BufferedWrenchSample:
    timestamp_ns: int
    raw_wrench: tuple[float, float, float, float, float, float]
    filtered_wrench: tuple[float, float, float, float, float, float]


@dataclass(frozen=True)
class CapturedImpact:
    hit_id: int
    samples: tuple[BufferedWrenchSample, ...]


class ImpactBuffer:
    def __init__(
        self,
        *,
        sample_period_ms: float,
        pre_buffer_ms: float = 50.0,
        post_buffer_ms: float = 100.0,
        maximum_samples: int = 1000,
    ) -> None:
        values = (sample_period_ms, pre_buffer_ms, post_buffer_ms)
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise ValueError("buffer timing values must be finite and non-negative")
        if sample_period_ms <= 0.0:
            raise ValueError("sample_period_ms must be positive")
        if maximum_samples <= 0:
            raise ValueError("maximum_samples must be positive")

        self._pre_sample_count = math.ceil(pre_buffer_ms / sample_period_ms)
        self._post_sample_count = math.ceil(post_buffer_ms / sample_period_ms)
        if self._pre_sample_count + self._post_sample_count + 1 > maximum_samples:
            raise ValueError("configured pre/post buffers exceed maximum_samples")
        self._maximum_samples = maximum_samples
        self._history: deque[BufferedWrenchSample] = deque(
            maxlen=max(1, self._pre_sample_count + 1)
        )
        self._hit_id: int | None = None
        self._capture: list[BufferedWrenchSample] = []
        self._post_remaining: int | None = None

    @property
    def capturing(self) -> bool:
        return self._hit_id is not None

    def append(self, sample: BufferedWrenchSample) -> CapturedImpact | None:
        self._history.append(sample)
        if not self.capturing:
            return None
        if not self._capture or self._capture[-1] is not sample:
            self._capture.append(sample)
        if len(self._capture) > self._maximum_samples:
            raise OverflowError("impact capture exceeded maximum_samples")
        if self._post_remaining is None:
            return None
        self._post_remaining -= 1
        if self._post_remaining <= 0:
            return self.finish_now()
        return None

    def start_capture(self, hit_id: int) -> None:
        if hit_id <= 0:
            raise ValueError("hit_id must be positive")
        if self.capturing:
            raise RuntimeError("an impact capture is already active")
        self._hit_id = hit_id
        self._capture = list(self._history)
        self._post_remaining = None

    def mark_contact_complete(self) -> CapturedImpact | None:
        if not self.capturing:
            raise RuntimeError("no impact capture is active")
        if self._post_remaining is not None:
            raise RuntimeError("contact was already marked complete")
        self._post_remaining = self._post_sample_count
        if self._post_remaining == 0:
            return self.finish_now()
        return None

    def finish_now(self) -> CapturedImpact:
        if self._hit_id is None:
            raise RuntimeError("no impact capture is active")
        event = CapturedImpact(self._hit_id, tuple(self._capture))
        self._hit_id = None
        self._capture = []
        self._post_remaining = None
        return event

    def cancel(self) -> None:
        self._hit_id = None
        self._capture = []
        self._post_remaining = None
