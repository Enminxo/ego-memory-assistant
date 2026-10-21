"""Where frames come from.

Offline, a source is a video file and the pipeline asks for the frame at a
given second. A live camera (Raspberry Pi, webcam) cannot be asked for the
past: it needs a sampler thread that grabs frames on the wall clock, as in
egoagent's `perception/sampler.py`. Both end up producing the same `Clip`
objects (see clips.py), so everything downstream is shared.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import av
import numpy as np


class FrameSource(Protocol):
    """Anything that can return the RGB frame shown at second `t`."""

    duration: float

    def frames_at(self, times: list[float]) -> list[np.ndarray]: ...


class VideoFileSource:
    """Decode frames at given timestamps from a video file with PyAV."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        with av.open(str(self.path)) as container:
            stream = container.streams.video[0]
            if stream.duration is not None:
                self.duration = float(stream.duration * stream.time_base)
            else:
                self.duration = container.duration / av.time_base

    def frames_at(self, times: list[float]) -> list[np.ndarray]:
        """Return one RGB uint8 frame per requested time, in order.

        Seeks to the keyframe before each time and decodes forward to the
        first frame at or after it. Times past the end return the last frame.
        """
        frames = []
        with av.open(str(self.path)) as container:
            stream = container.streams.video[0]
            for t in times:
                container.seek(int(t / stream.time_base), stream=stream, backward=True)
                chosen = None
                for frame in container.decode(stream):
                    chosen = frame
                    if frame.time is not None and frame.time >= t:
                        break
                if chosen is None:
                    raise ValueError(f"no frame decoded at t={t:.2f}s in {self.path}")
                frames.append(chosen.to_ndarray(format="rgb24"))
        return frames
