import numpy as np
import pytest


class FakeSource:
    """A 'video' whose frame at t is a solid image of value int(t), for checking timing."""

    def __init__(self, duration: float) -> None:
        self.duration = duration

    def frames_at(self, times: list[float]) -> list[np.ndarray]:
        return [np.full((4, 4, 3), int(t), dtype=np.uint8) for t in times]


@pytest.fixture
def fake_source():
    return FakeSource
