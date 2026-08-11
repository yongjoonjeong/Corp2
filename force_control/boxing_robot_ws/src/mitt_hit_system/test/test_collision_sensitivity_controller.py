from types import SimpleNamespace

import pytest

from mitt_hit_system.collision_sensitivity_controller import (
    CollisionSensitivityConfig,
    CollisionSensitivityController,
)


class Future:
    def __init__(self, response):
        self._response = response

    def done(self):
        return True

    def result(self):
        return self._response


class Client:
    def __init__(self, successes):
        self.successes = iter(successes)
        self.requests = []

    def service_is_ready(self):
        return True

    def call_async(self, request):
        self.requests.append(request)
        return Future(SimpleNamespace(success=next(self.successes)))


def make_controller(successes=(True, True)):
    client = Client(successes)
    controller = CollisionSensitivityController(
        CollisionSensitivityConfig(True, 30, 40, 0.2),
        client,
        request_factory=lambda: SimpleNamespace(sensitivity=0),
    )
    return controller, client


def test_training_value_is_applied_and_normal_value_is_restored():
    controller, client = make_controller()

    assert controller.apply_training()[0]
    assert controller.training_override_active
    assert controller.restore()[0]

    assert [request.sensitivity for request in client.requests] == [30, 40]
    assert not controller.training_override_active


def test_failed_restore_keeps_override_marked_active_for_retry():
    controller, client = make_controller((True, False, True))

    assert controller.apply_training()[0]
    assert not controller.restore()[0]
    assert controller.training_override_active
    assert controller.restore()[0]
    assert [request.sensitivity for request in client.requests] == [30, 40, 40]


@pytest.mark.parametrize("value", [0, 101])
def test_sensitivity_outside_service_range_is_rejected(value):
    with pytest.raises(ValueError):
        CollisionSensitivityConfig(True, value, 40, 0.2).validate()
