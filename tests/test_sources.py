import shutil
import subprocess

import numpy as np
import pytest

from egomem.sources import VideoFileSource

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


def make_video(path, seconds: float) -> object:
    # testsrc draws a running counter, so frames at different times differ.
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
                    "-i", "testsrc=size=160x120:rate=10", "-t", str(seconds),
                    "-pix_fmt", "yuv420p", str(path)], check=True)
    return path


@needs_ffmpeg
def test_segment_reads_the_same_frames_as_the_full_file(tmp_path):
    video = make_video(tmp_path / "v.mp4", 6)
    full = VideoFileSource(video)
    segment = VideoFileSource(video, start=2.0, end=5.0)

    assert segment.duration == pytest.approx(3.0, abs=0.11)
    [from_segment] = segment.frames_at([1.0])
    [from_full] = full.frames_at([3.0])
    np.testing.assert_array_equal(from_segment, from_full)
    [other] = full.frames_at([1.0])
    assert not np.array_equal(from_segment, other)


@needs_ffmpeg
def test_segment_end_is_clamped_to_the_file(tmp_path):
    video = make_video(tmp_path / "v.mp4", 4)
    assert VideoFileSource(video, start=1.0, end=100.0).duration == pytest.approx(3.0, abs=0.11)


@needs_ffmpeg
def test_empty_segment_is_rejected(tmp_path):
    video = make_video(tmp_path / "v.mp4", 2)
    with pytest.raises(ValueError):
        VideoFileSource(video, start=5.0)
