from egomem.clips import clip_bounds, frame_offsets, iter_clips


def test_offsets_are_centred_and_evenly_spread():
    assert frame_offsets(15, 2) == [3.75, 11.25]
    offs = frame_offsets(15, 8)
    assert offs[0] == 15 / 16 and offs[-1] == 15 - 15 / 16


def test_bounds_cover_the_stream_and_drop_tiny_tail():
    assert clip_bounds(45, 15) == [(0, 15), (15, 30), (30, 45)]
    assert clip_bounds(50, 15)[-1] == (45, 50)
    assert clip_bounds(45.5, 15)[-1] == (30, 45)  # 0.5 s tail is below min_sec


def test_iter_clips_samples_inside_each_clip(fake_source):
    clips = list(iter_clips(fake_source(30), clip_sec=15, n_frames=3))
    assert [c.index for c in clips] == [0, 1]
    for clip in clips:
        assert len(clip.frames) == 3
        assert all(clip.t0 <= t < clip.t1 for t in clip.times)


def test_iter_clips_skips_done_indices(fake_source):
    clips = list(iter_clips(fake_source(45), clip_sec=15, n_frames=2, skip={0, 2}))
    assert [c.index for c in clips] == [1]
