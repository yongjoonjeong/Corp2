"""Bounded session override for the Doosan collision sensitivity setting."""

from dataclasses import dataclass
import time
from typing import Any, Callable


@dataclass(frozen=True)
class CollisionSensitivityConfig:
    enabled: bool = False
    training_percent: int = 40
    restore_percent: int = 40
    service_timeout_sec: float = 0.2

    def validate(self) -> None:
        if not self.enabled:
            return
        for name, value in (
            ("training_percent", self.training_percent),
            ("restore_percent", self.restore_percent),
        ):
            if not 1 <= int(value) <= 100:
                raise ValueError(f"collision {name} must be between 1 and 100")
        if self.service_timeout_sec <= 0.0:
            raise ValueError("collision sensitivity service timeout must be positive")


class CollisionSensitivityController:
    """Apply a less-sensitive training value and restore the normal value."""

    def __init__(
        self,
        config: CollisionSensitivityConfig,
        client: Any | None = None,
        *,
        request_factory: Callable[[], Any] | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self._client = client
        self._request_factory = request_factory
        self.training_override_active = False

    def service_ready(self) -> bool:
        return (
            not self.config.enabled
            or self._client is not None
            and self._client.service_is_ready()
        )

    def apply_training(self) -> tuple[bool, str]:
        if not self.config.enabled:
            return True, "collision sensitivity override disabled"
        success, detail = self._change(self.config.training_percent)
        if success:
            self.training_override_active = True
            return True, (
                "collision sensitivity set to "
                f"{self.config.training_percent}% for training"
            )
        return False, detail

    def restore(self) -> tuple[bool, str]:
        if not self.config.enabled or not self.training_override_active:
            return True, "collision sensitivity restore not required"
        success, detail = self._change(self.config.restore_percent)
        if success:
            self.training_override_active = False
            return True, (
                "collision sensitivity restored to "
                f"{self.config.restore_percent}%"
            )
        return False, detail

    def _change(self, sensitivity: int) -> tuple[bool, str]:
        if self._client is None or self._request_factory is None:
            return False, "collision sensitivity service is not configured"
        request = self._request_factory()
        request.sensitivity = int(sensitivity)
        try:
            future = self._client.call_async(request)
            deadline = time.monotonic() + self.config.service_timeout_sec
            while not future.done():
                if time.monotonic() >= deadline:
                    future.cancel()
                    return False, "collision sensitivity service timeout"
                time.sleep(0.001)
            response = future.result()
        except Exception as error:
            return False, f"collision sensitivity service failed: {error}"
        if response is None or not bool(response.success):
            return False, "collision sensitivity service returned success=false"
        return True, "collision sensitivity changed"
