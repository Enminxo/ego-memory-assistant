"""Does the diary lose information when the glasses' photos are smaller?

Two tests, written down before any run: this header is the pre-registration,
and its commit comes before the first diary.

A. Neon footage, no annotation.
   The five videos uploaded from the Neon glasses keep their originals at
   1600x1200 (data/uploads/*/). Each 15 s window gives the same 8 frames as
   the live diary. They are described at three widths, 1600 (native), 720
   (what the demo relay sends) and 360, each with describer seeds 0 and 1:
   six diaries over the same 31 windows. Only the size changes: same frames,
   same prompt, same model and settings (configs/demo.yaml).
   Facts are then pulled out of the six descriptions of each window (objects,
   colours, counts, brands, text read, actions). Every fact that not all six
   state is judged by hand, blind to which size wrote it, as true, false or
   "can't tell" from the 8 frames at 1600 px.

   Primary endpoint: true facts per window that the 720 description misses
   while the 1600 description of the same seed states them, averaged over
   windows and both seeds, minus the same count between the two seeds at 1600
   (what changing only the seed already loses). 95% bootstrap CI over windows.
   Reading, fixed now:
     - upper bound below 1 fact per window: 720 loses at most one true fact
       per window beyond seed noise, enough for this footage;
     - lower bound above 0: 720 loses facts beyond noise; report which kinds.
   Check on the knob: the same count at 360 must be at least the one at 720.
   If 360 loses no more than 720, the 720 loss is not attributed to size.
   Secondary: false facts per window at each size; true facts missed by kind
   (count, text or brand, colour, object, action).

B. Ego4D with annotation, scaled down instead of up (full-size Ego4D needs
   credentials, and ours have expired).
   QAEgo4D test_close: 148 clips, 500 four-option questions, already described
   at 540 px tall by gemma4:31b, seed 0 (tag g31b, 2 Nov 2026). The same clips
   are described again with every frame halved to 270 px tall, with the same
   frames, prompt and settings (tag g31b_half), and scored with egomem-eval
   with option-order seeds 0 and 1, as g31b was.

   Primary endpoint: accuracy(g31b_half) - accuracy(g31b), averaged over the
   two option-order seeds, 95% CI from a bootstrap over the 148 clips.
   Reading, fixed now:
     - lower bound above -3 points: halving costs less than 3 points;
     - upper bound below 0: the smaller frames lose accuracy.
   Before the 270 run the 540 pipeline is checked: windows of the first clip
   are described again at 540 (tag g31b_repro) and must match g31b word for
   word in at least 90% of them. Otherwise the 540 arm is described again
   too, so that both arms come from the same conditions.
   Known limitation: one describer seed per arm, so describer noise is only
   bounded by test A's seed comparison.

    uv run python experiments/resolution_compare/run.py look
    uv run python experiments/resolution_compare/run.py neon --width 720 --seed 0
    uv run python experiments/resolution_compare/run.py ego4d --scale 0.5 --tag g31b_half
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import numpy as np
from PIL import Image

from egomem.clips import clip_bounds, frame_offsets
from egomem.config import load_config, make_describer
from egomem.describer import VLMDescriber
from egomem.memory import MemoryEntry, TextualMemory, fmt_time
from egomem.scenarios import load_scenario

REPO_DATA = Path.home() / "projects/ego-memory-assistant/data"   # the main checkout's data/
NEON_VIDEOS: dict[str, Path] = {
    "cerrar": REPO_DATA / "uploads/upload-video-cerrar/Video_cerrar.mp4",
    "cerrar-ventana": REPO_DATA / "uploads/upload-video-cerrar-ventana/Video_cerrar ventana.mp4",
    "impresora3d": REPO_DATA / "uploads/upload-video-impresora3d/video_impresora3d.mp4",
    "maquina": REPO_DATA / "uploads/upload-video-maquina/Video_maquina.mp4",
    "vitaminas": REPO_DATA / "uploads/upload-video-vitaminas/Video_vitaminas.mp4",
}
NEON_WIDTHS = (1600, 720, 360)
OUT = Path.home() / "data/ego-memory-assistant/resolution_compare"
EVAL_ROOT = Path.home() / "data/ego-memory-assistant/describer_eval"
CONFIG = Path("configs/demo.yaml")
EXPERIMENT = "resolution_compare"
# The settings of the g31b run (memories/*/g31b/run.json); test B must match them.
G31B_SETTINGS = {"model": "gemma4:31b", "clip_sec": 15, "n_frames": 8, "max_tokens": 160,
                 "max_words": 60, "temperature": 0.2}


class SegmentSource:
    """Frames of one segment of a video file, as the g31b run read them.

    A copy of VideoFileSource at commit c188fe5 (the code that wrote g31b):
    `start` and `end` restrict the source to an Ego4D NLQ clip inside its full
    video, and times passed to `frames_at` are relative to `start`. The current
    VideoFileSource lost this, so test B cannot use it.
    """

    def __init__(self, path: Path, start: float = 0.0, end: float | None = None) -> None:
        self.path = Path(path)
        self.start = float(start)
        with av.open(str(self.path)) as container:
            stream = container.streams.video[0]
            if stream.duration is not None:
                file_duration = float(stream.duration * stream.time_base)
            else:
                file_duration = container.duration / av.time_base
        stop = file_duration if end is None else min(float(end), file_duration)
        if stop <= self.start:
            raise ValueError(f"empty segment [{self.start}, {stop}] in {self.path}")
        self.duration = stop - self.start

    def frames_at(self, times: list[float]) -> list[np.ndarray]:
        frames = []
        with av.open(str(self.path)) as container:
            stream = container.streams.video[0]
            for rel_t in times:
                t = self.start + rel_t
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


def resize_to_width(frame: np.ndarray, width: int) -> np.ndarray:
    """Shrink a frame to `width`, keeping its shape; the height is rounded to
    an even number, as the relay's ffmpeg `scale=W:-2` does. Never enlarges."""
    height, current = frame.shape[:2]
    if width >= current:
        return frame
    new_height = round(height * width / current / 2) * 2
    image = Image.fromarray(frame).resize((width, new_height), Image.Resampling.BICUBIC)
    return np.asarray(image)


def window_frames(source: SegmentSource, index: int, clip_sec: float,
                  n_frames: int) -> tuple[float, float, list[np.ndarray]]:
    """The (t0, t1, frames) of window `index`, sampled exactly as the diary does."""
    t0, t1 = clip_bounds(source.duration, clip_sec)[index]
    times = [t0 + off for off in frame_offsets(t1 - t0, n_frames)]
    return t0, t1, source.frames_at(times)


def prepare_out(out: Path, meta: dict) -> TextualMemory:
    """Open a diary this experiment writes; refuse a folder written by anything else.

    The Ego4D diaries live next to earlier runs (g31b, e4b, ...), and this
    check is what stops a mistyped tag from appending to one of them.
    """
    run_file = out / "run.json"
    if out.exists() and any(out.iterdir()):
        if not run_file.exists():
            raise SystemExit(f"{out} exists and was not written by {EXPERIMENT}; refusing")
        previous = json.loads(run_file.read_text())
        if previous.get("experiment") != EXPERIMENT:
            raise SystemExit(f"{out} belongs to another run; refusing to write into it")
        if previous.get("frames") != meta.get("frames"):
            raise SystemExit(f"{out} was written with frames {previous.get('frames')}, "
                             f"not {meta.get('frames')}; use another tag")
    out.mkdir(parents=True, exist_ok=True)
    run_file.write_text(json.dumps(meta, indent=1))
    return TextualMemory.load(out / "memory.jsonl")


def describe_source(describer: VLMDescriber, source: SegmentSource, memory: TextualMemory,
                    size: dict, clip_sec: float, n_frames: int, label: str,
                    limit: int | None = None) -> int:
    """Describe every window not yet in `memory`, with frames resized as `size` says.

    `size` is {"width": W} (an absolute width) or {"scale": s} (a fraction of
    each frame's own width). Returns how many windows were described.
    """
    done = {e.index for e in memory.entries()}
    bounds = clip_bounds(source.duration, clip_sec)
    written = 0
    for index in range(len(bounds)):
        if index in done:
            continue
        if limit is not None and written >= limit:
            break
        t0, t1, frames = window_frames(source, index, clip_sec, n_frames)
        if "width" in size:
            width = int(size["width"])
        else:
            width = round(frames[0].shape[1] * float(size["scale"]))
        frames = [resize_to_width(f, width) for f in frames]
        result = describer.describe(frames, t0, t1)
        memory.append(MemoryEntry(index=index, t0=round(t0, 2), t1=round(t1, 2),
                                  text=result.text, latency_s=round(result.latency_s, 2),
                                  n_frames=len(frames)))
        written += 1
        h, w = frames[0].shape[:2]
        print(f"{label} [{index + 1:3d}/{len(bounds)}] {fmt_time(t0)} {w}x{h} "
              f"({result.latency_s:4.1f}s) {result.text}", flush=True)
    return written


def describer_config(seed: int) -> dict:
    config = load_config(CONFIG)
    config["describer"]["seed"] = seed
    return config


def run_neon(width: int, seed: int, videos: list[str], limit: int | None) -> None:
    config = describer_config(seed)
    d = config["describer"]
    describer = make_describer(config)
    for name in videos:
        out = OUT / "neon" / f"w{width}_s{seed}" / name
        meta = {"experiment": EXPERIMENT, "test": "A", "video": str(NEON_VIDEOS[name]),
                "frames": {"width": width}, "describer": d, "base_url": config["base_url"]}
        memory = prepare_out(out, meta)
        source = SegmentSource(NEON_VIDEOS[name])
        describe_source(describer, source, memory, {"width": width}, d["clip_sec"],
                        d["n_frames"], f"neon w{width} s{seed} {name}", limit)


def run_ego4d(scale: float, seed: int, tag: str, ids: list[str] | None,
              limit: int | None) -> None:
    config = describer_config(seed)
    d = config["describer"]
    mismatched = {k: (d.get(k), v) for k, v in G31B_SETTINGS.items() if d.get(k) != v}
    if mismatched:
        raise SystemExit(f"describer settings differ from the g31b run: {mismatched}")
    describer = make_describer(config)
    paths = sorted((EVAL_ROOT / "scenarios").glob("*.json"))
    if ids is not None:
        paths = [p for p in paths if p.stem in ids]
    for path in paths:
        scenario = load_scenario(path)
        start, end = scenario.source["segment"]
        out = EVAL_ROOT / "memories" / scenario.id / tag
        meta = {"experiment": EXPERIMENT, "test": "B", "video": scenario.video,
                "segment": [start, end], "frames": {"scale": scale},
                "describer": d, "base_url": config["base_url"]}
        memory = prepare_out(out, meta)
        source = SegmentSource(Path(scenario.video), start=start, end=end)
        describe_source(describer, source, memory, {"scale": scale}, d["clip_sec"],
                        d["n_frames"], f"{tag} {scenario.id[:8]}", limit)


def look(out_dir: Path) -> None:
    """Save one frame at every size of both tests, to look at before running.

    Each version is enlarged back to the same display size, so what is lost is
    visible side by side.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    d = load_config(CONFIG)["describer"]
    source = SegmentSource(NEON_VIDEOS["impresora3d"])
    _, _, frames = window_frames(source, 1, d["clip_sec"], d["n_frames"])
    frame = frames[4]
    tiles = [Image.fromarray(resize_to_width(frame, w)).resize((800, 600), Image.Resampling.NEAREST)
             for w in NEON_WIDTHS]
    sheet = Image.new("RGB", (800 * len(tiles), 600))
    for i, tile in enumerate(tiles):
        sheet.paste(tile, (800 * i, 0))
    sheet.save(out_dir / "neon_1600_720_360.jpg", quality=92)
    print("neon frame", frame.shape, "->", [resize_to_width(frame, w).shape for w in NEON_WIDTHS])

    path = sorted((EVAL_ROOT / "scenarios").glob("*.json"))[0]
    scenario = load_scenario(path)
    start, end = scenario.source["segment"]
    ego = SegmentSource(Path(scenario.video), start=start, end=end)
    _, _, frames = window_frames(ego, 0, d["clip_sec"], d["n_frames"])
    frame = frames[4]
    half = resize_to_width(frame, round(frame.shape[1] * 0.5))
    shown = [Image.fromarray(f).resize((frame.shape[1], frame.shape[0]), Image.Resampling.NEAREST)
             for f in (frame, half)]
    sheet = Image.new("RGB", (frame.shape[1] * 2, frame.shape[0]))
    for i, tile in enumerate(shown):
        sheet.paste(tile, (frame.shape[1] * i, 0))
    sheet.save(out_dir / "ego4d_540_270.jpg", quality=92)
    print("ego4d frame", scenario.id[:8], frame.shape, "->", half.shape)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p_look = sub.add_parser("look", help="save one frame at every size")
    p_look.add_argument("--out", type=Path, default=OUT / "look")
    p_neon = sub.add_parser("neon", help="test A: describe the Neon videos at one width")
    p_neon.add_argument("--width", type=int, required=True, choices=NEON_WIDTHS)
    p_neon.add_argument("--seed", type=int, required=True)
    p_neon.add_argument("--videos", nargs="+", default=list(NEON_VIDEOS), choices=list(NEON_VIDEOS))
    p_neon.add_argument("--limit", type=int, help="at most N new windows per video")
    p_ego = sub.add_parser("ego4d", help="test B: describe QAEgo4D test_close at a scale")
    p_ego.add_argument("--scale", type=float, required=True)
    p_ego.add_argument("--seed", type=int, default=0)
    p_ego.add_argument("--tag", required=True)
    p_ego.add_argument("--ids", nargs="+", help="only these clip ids")
    p_ego.add_argument("--limit", type=int, help="at most N new windows per clip")
    args = parser.parse_args()

    if args.command == "look":
        look(args.out)
    elif args.command == "neon":
        run_neon(args.width, args.seed, args.videos, args.limit)
    else:
        run_ego4d(args.scale, args.seed, args.tag, args.ids, args.limit)


if __name__ == "__main__":
    main()
