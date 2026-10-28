import time

import numpy as np
import pytest

from egomem.neon_stream import (H264Depacketizer, SceneStream, StaleStream, StreamParser,
                                capture_time, parse_sdp, parse_sender_reports, ws_url)

pytest.importorskip("websockets")  # the fake phone needs it: uv sync --extra neon
from fake_companion import (FakeCompanion, encode_flat_frames, interleave, rtp_packets,
                            sender_report)

# 20 flat frames, a keyframe every 10: values 20, 30, ..., 210
VALUES = [20 + 10 * i for i in range(20)]


@pytest.fixture(scope="module")
def h264():
    return encode_flat_frames(VALUES, keyint=10)


def units(depacketizer, packets):
    return [u for u in (depacketizer.push(p) for p in packets) if u is not None]


def frame_packets(frames, max_payload=200):
    """RTP packets of each frame, numbered and timed as the phone does."""
    out, seq = [], 1000
    for i, nals in enumerate(frames):
        packets = rtp_packets(nals, seq, 3000 * i, max_payload)
        out.append(packets)
        seq += len(packets)
    return out


def test_parser_splits_rtsp_answers_from_interleaved_packets():
    parser = StreamParser()
    answer = b"RTSP/1.0 200 OK\r\nCSeq: 1\r\nContent-Length: 3\r\n\r\nabc"
    data = answer + interleave(0, b"\x80" * 20) + interleave(1, b"rtcp")
    items = []
    for i in range(0, len(data), 7):  # arriving in arbitrary pieces
        items += parser.feed(data[i:i + 7])
    assert items[0].status == 200 and items[0].body == "abc"
    assert items[1:] == [(0, b"\x80" * 20), (1, b"rtcp")]


def test_sdp_video_track_is_found_when_audio_comes_first():
    sdp = ("v=0\r\nm=audio 5010 RTP/AVP 97\r\na=control:trackID=0\r\n"
           "m=video 5006 RTP/AVP 96\r\n"
           "a=fmtp:96 packetization-mode=1;sprop-parameter-sets=Z0KAKNoBkAl6AbQoTUA=,aM4G8g==;\r\n"
           "a=control:trackID=1\r\n")
    sprop, control = parse_sdp(sdp)
    assert control == "trackID=1"
    assert [nal[0] & 0x1F for nal in sprop] == [7, 8]  # SPS, PPS


def test_sender_report_gives_the_capture_time_of_every_frame():
    (report,) = parse_sender_reports(sender_report(90_000, 1_793_011_268.5))
    assert report[0] == 90_000 and report[1] == pytest.approx(1_793_011_268.5, abs=1e-6)
    assert capture_time(90_000 + 45_000, report) == pytest.approx(report[1] + 0.5)
    assert capture_time(5, (2**32 - 85, 100.0)) == pytest.approx(100.0 + 90 / 90_000)  # wraps
    assert capture_time(5, None) is None


def test_whole_frames_come_out_from_the_first_keyframe(h264):
    _, frames = h264
    packets = frame_packets(frames[3:])  # joined mid-stream: frames 3-9 cannot be decoded
    out = units(H264Depacketizer(), [p for f in packets for p in f])
    assert len(out) == 10 and out[0].key and not out[1].key
    assert max(len(p) for f in packets for p in f) <= 200 + 14  # the keyframe went as FU-A


def test_a_lost_packet_holds_frames_back_until_the_next_keyframe(h264):
    _, frames = h264
    stream = frames[5:] + frames  # joined mid-stream; keyframes at positions 5, 15 and 25
    packets = frame_packets(stream)
    depacketizer = H264Depacketizer()
    out = units(depacketizer, [p for f in packets[:7] for p in f])   # 5 and 6 come out
    out += units(depacketizer, packets[7][1:])                       # 7 loses its first packet
    out += units(depacketizer, [p for f in packets[8:] for p in f])  # 8-14 depend on it
    assert depacketizer.lost == 1
    assert [u.rtp_ts // 3000 for u in out] == [5, 6] + list(range(15, 35))


def test_scene_stream_decodes_the_phones_video(h264):
    with FakeCompanion(*h264) as phone:
        stream = SceneStream("127.0.0.1", phone.port)
        try:
            rgb, taken = stream.latest(timeout=5)
            assert rgb.shape == (48, 64, 3)
            # the picture is one of the frames sent, not a broken mix of them
            assert min(abs(rgb.mean() - v) for v in VALUES) < 3
            time.sleep(0.6)
            assert stream.fps > 20 and stream.latest()[1] is not None  # sender reports arrived
            assert stream.latest_jpeg(width=32)[:2] == b"\xff\xd8"
        finally:
            stream.close()
        assert phone.requests[:3] == ["DESCRIBE", "SETUP", "PLAY"]
        assert phone.requests[-1] == "TEARDOWN"  # the phone is told the session is over


def test_scene_stream_reconnects_when_the_phone_hangs_up(h264):
    with FakeCompanion(*h264, close_after=25) as phone:
        stream = SceneStream("127.0.0.1", phone.port, retry_sec=0.2)
        try:
            stream.latest(timeout=5)
            deadline = time.monotonic() + 5
            while stream.reconnects == 0 and time.monotonic() < deadline:
                time.sleep(0.05)
            n = stream.frames_read
            time.sleep(0.5)
            assert stream.reconnects >= 1 and phone.connections >= 2
            assert stream.frames_read > n  # frames flow again after reconnecting
        finally:
            stream.close()


def test_a_stream_that_stops_is_stale(h264):
    with FakeCompanion(*h264, close_after=12) as phone:
        # nothing listens after the first connection closes, so no new frames arrive
        stream = SceneStream("127.0.0.1", phone.port, stale_sec=0.5, retry_sec=60)
        try:
            stream.latest(timeout=5)
            time.sleep(1.0)
            with pytest.raises(StaleStream):
                stream.latest_rgb()
        finally:
            stream.close()


def test_ws_url_brackets_ipv6():
    assert ws_url("100.64.0.1") == "ws://100.64.0.1:8686/rtsp-over-websocket"
    assert ws_url("fd7a:115c::51", 9) == "ws://[fd7a:115c::51]:9/rtsp-over-websocket"


def test_a_slow_link_is_not_cut_while_data_still_flows(h264):
    sprop, frames = h264
    # 0.6 s of frames that cannot be decoded (no keyframe yet), then the stream as usual
    late_key = frames[1:10] * 4 + frames
    with FakeCompanion(sprop, late_key) as phone:
        stream = SceneStream("127.0.0.1", phone.port, stall_sec=0.3)
        try:
            stream.latest(timeout=5)
            assert stream.reconnects == 0 and phone.connections == 1
        finally:
            stream.close()
