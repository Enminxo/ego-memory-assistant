"""Cutting a stream into the clips the describer turns into diary entries.

The stream is split into non-overlapping clips of `clip_sec` seconds, and each
clip is represented by `n_frames` frames spread evenly and centred inside it
(for 8 frames over 15 s: 0.94 s, 2.81 s, ..., 14.06 s). Anything shorter than
the gap between two frames can fall between them and leave no trace in the
diary: that blind spot is clip_sec / n_frames seconds.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np

from egomem.sources import FrameSource


@dataclass
class Clip:
    index: int
    t0: float
    t1: float
    times: list[float] = field(default_factory=list)
    frames: list[np.ndarray] = field(default_factory=list)


def frame_offsets(clip_sec: float, n_frames: int) -> list[float]:
    """Sampling instants inside one clip, evenly spread and centred."""
    return [(i + 0.5) * clip_sec / n_frames for i in range(n_frames)]


def clip_bounds(duration: float, clip_sec: float, min_sec: float = 1.0) -> list[tuple[float, float]]:
    """(t0, t1) of every clip; the last one may be shorter, but not below min_sec."""
    bounds = []
    t0 = 0.0
    while t0 < duration:
        t1 = min(t0 + clip_sec, duration)
        if t1 - t0 >= min_sec:
            bounds.append((t0, t1))
        t0 += clip_sec
    return bounds


def iter_clips(source: FrameSource, clip_sec: float, n_frames: int,
               skip: set[int] | None = None) -> Iterator[Clip]:
    """Yield the clips of a whole source, decoding only their sampled frames.

    `skip` holds clip indices already processed, so an interrupted run can
    resume without decoding them again.
    """
    for index, (t0, t1) in enumerate(clip_bounds(source.duration, clip_sec)):
        if skip and index in skip:
            continue
        times = [t0 + off for off in frame_offsets(t1 - t0, n_frames)]
        yield Clip(index=index, t0=t0, t1=t1, times=times, frames=source.frames_at(times))
