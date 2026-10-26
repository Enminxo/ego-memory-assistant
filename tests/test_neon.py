import json
import shutil
import subprocess

import numpy as np
import pytest

from egomem.neon import import_recording, scene_resample_indices
from egomem.scenarios import load_scenario
from egomem.sources import VideoFileSource

S = 1_000_000_000  # nanoseconds per second


def test_resample_halves_a_steady_30fps_stream():
    ts = np.arange(31) * S // 30  # 1 s at 30 fps
    idx = scene_resample_indices(ts, fps=15)
    assert len(idx) == 16  # ticks at 0, 1/15, ..., 1 s
    assert list(idx[:4]) == [0, 2, 4, 6] and idx[-1] == 30


def test_a_gap_freezes_the_picture_and_later_frames_keep_their_time():
    # 10 fps for 1 s, nothing for 1 s, then 10 fps again
    ts = np.concatenate([np.arange(10), np.arange(20, 30)]) * S // 10
    idx = scene_resample_indices(ts, fps=10)
    assert len(idx) == 30
    assert list(idx[9:20]) == [9] * 11  # 0.9 s to 1.9 s: the last frame before the gap
    assert idx[20] == 10 and idx[29] == 19  # the frame taken at 2.0 s is shown at 2.0 s


def test_no_frame_is_shown_before_it_was_taken():
    rng = np.random.default_rng(0)
    ts = np.cumsum(rng.uniform(0.02, 0.05, 300) * S).astype(np.int64)
    idx = scene_resample_indices(ts, fps=15)
    ticks = ts[0] + np.round(np.arange(len(idx)) * S / 15).astype(np.int64)
    assert np.all(ts[idx] <= ticks)
    later = idx + 1 < len(ts)
    assert np.all(ts[idx[later] + 1] > ticks[later])  # and it is the latest one taken


def test_resample_rejects_empty_or_unsorted():
    with pytest.raises(ValueError):
        scene_resample_indices(np.array([], dtype=np.int64), 15)
    with pytest.raises(ValueError):
        scene_resample_indices(np.array([2, 1]) * S, 15)


def write_fake_recording(rec_dir, t_start_ns):
    """A recording in Neon's native layout: scene mp4, its .time file and info.json.

    Frame i is a flat grey of value 4 * i, so the imported video can be checked
    for which frame it shows when. Frames 0-29 are 1 s at 30 fps; frames 30-59
    start 1 s later, as if the scene camera had stalled.
    """
    rec_dir.mkdir()
    w, h, n = 160, 120, 60
    frames = np.repeat(np.arange(n, dtype=np.uint8) * 4, w * h * 3).tobytes()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                    "-s", f"{w}x{h}", "-r", "30", "-i", "-", "-c:v", "libx264", "-qp", "0",
                    "-pix_fmt", "yuv444p", str(rec_dir / "Neon Scene Camera v1 ps1.mp4")],
                   input=frames, check=True)
    ts = t_start_ns + np.concatenate([np.arange(30) * S // 30, 2 * S + np.arange(30) * S // 30])
    ts.astype("<i8").tofile(rec_dir / "Neon Scene Camera v1 ps1.time")
    (rec_dir / "info.json").write_text(json.dumps(
        {"recording_id": "fake-rec", "start_time": int(t_start_ns), "duration": int(3 * S)}))


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_import_writes_a_constant_rate_video_and_its_scenario(tmp_path):
    pytest.importorskip("pupil_labs.neon_recording")
    t_start = 1_761_000_000 * S
    write_fake_recording(tmp_path / "rec", t_start)

    scenario = import_recording(tmp_path / "rec", "mine", "My kitchen",
                                tmp_path / "scenarios", root=tmp_path, fps=10, width=160)
    assert load_scenario(tmp_path / "scenarios/mine.json") == scenario
    assert scenario.video == "data/scenes/mine.mp4" and not scenario.live
    assert scenario.source["neon_recording_id"] == "fake-rec"
    assert scenario.source["scene_start_unix_ns"] == t_start

    source = VideoFileSource(tmp_path / scenario.video)
    assert source.duration == pytest.approx(3.0, abs=0.11)  # 2.97 s of frames, 30 ticks at 10 fps
    shown = [float(f.mean()) / 4 for f in source.frames_at([0.5, 1.5, 2.5])]
    # 0.5 s: frame 15; 1.5 s: inside the stall, still frame 29; 2.5 s: frame 30 + 15
    assert shown == pytest.approx([15, 29, 45], abs=1.5)
