"""Scenarios: a video plus the questions someone would ask about it.

A scenario is one JSON file in `scenarios/`. The format is the same whether
the video comes from a public dataset or from a recording of our own, so the
build, evaluation and app code never need to know the difference:

    {
      "id": "gas", "title": "...", "category": "appliances",
      "video": "data/scenes/gas.mp4",
      "questions": [{"id": ..., "question": ..., "answer": ...,
                     "wrong_answers": [...], "t0": ..., "t1": ..., "category": ...}],
      "demo_questions": [{"text": "¿He apagado el gas?", "expected": "no"}],
      "source": {...}
    }

All times are seconds from the start of the scenario's own video.

`make_ego4d_scenes` builds scenarios from Ego4D NLQ clips and their QAEgo4D
questions, from a spec file such as `scripts/demo_scenes.yaml`.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

CATEGORIES = ("objects", "medication", "appliances", "people", "other")

_MEDICATION = re.compile(r"\b(drugs?|medicines?|medications?|pills?|capsules?|vitamins?)\b", re.I)
# Ego4D annotators often type "l" for "I" ("what did l take"), hence [il].
_STATE_QUESTION = re.compile(r"^\s*(did|have) [il]\b", re.I)
_DEVICE_OR_STATE = re.compile(
    r"\b(gas|stove|cooker|oven|tap|faucet|vent|lights?|doors?|windows?|blinds|fridge|"
    r"lock|unlock|turn|switch|off|on|open|close)\b", re.I)
_PEOPLE = re.compile(r"\b(who|whom|talk|talked|person|man|woman|people)\b", re.I)
_OBJECT = re.compile(
    r"^\s*(where|in what (location|place|room)|what did [il] (put|take|pick|leave|drop|throw))", re.I)


def categorize(question: str) -> str:
    """Assign one of CATEGORIES with keyword rules.

    A starting point, not a classifier: the rules are checked in order and the
    first match wins. Defining the categories properly is objective 1 of the
    project.
    """
    if _MEDICATION.search(question):
        return "medication"
    if _STATE_QUESTION.search(question) and _DEVICE_OR_STATE.search(question):
        return "appliances"
    if _PEOPLE.search(question):
        return "people"
    if _OBJECT.search(question):
        return "objects"
    return "other"


@dataclass
class Question:
    id: str
    question: str
    answer: str | None = None
    wrong_answers: list[str] = field(default_factory=list)
    t0: float | None = None
    t1: float | None = None
    category: str = "other"


@dataclass
class Scenario:
    id: str
    title: str
    category: str
    # None for a live session (egomem-live): the frames are described as they
    # arrive and only the diary and its evidence frames are kept.
    video: str | None = None
    questions: list[Question] = field(default_factory=list)
    demo_questions: list[dict] = field(default_factory=list)
    source: dict = field(default_factory=dict)
    live: bool = False

    def video_path(self, root: Path) -> Path | None:
        return (root / self.video).resolve() if self.video is not None else None

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=1, ensure_ascii=False) + "\n",
                        encoding="utf-8")


def load_scenario(path: Path) -> Scenario:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    raw["questions"] = [Question(**q) for q in raw.get("questions", [])]
    return Scenario(**raw)


def load_scenarios(directory: Path) -> dict[str, Scenario]:
    scenarios = [load_scenario(p) for p in sorted(Path(directory).glob("*.json"))]
    return {s.id: s for s in scenarios}


# --- Building scenarios from Ego4D -------------------------------------------

def load_nlq_clips(annotation_files: list[Path]) -> dict[str, dict]:
    """clip_uid -> video_uid and the clip's [start, end] inside its video."""
    clips = {}
    for path in annotation_files:
        for video in json.loads(Path(path).read_text())["videos"]:
            for clip in video["clips"]:
                clips[clip["clip_uid"]] = {"video_uid": video["video_uid"],
                                           "start": clip["video_start_sec"],
                                           "end": clip["video_end_sec"]}
    return clips


def load_qaego4d(directory: Path) -> list[dict]:
    """All QAEgo4D questions; in these files `video_id` is the NLQ clip_uid."""
    rows = []
    for split in ("train", "val", "test_close"):
        path = Path(directory) / f"annotations.QaEgo4D_{split}.json"
        if path.exists():
            rows += json.loads(path.read_text())
    return rows


def cut_video(src: Path, start: float, end: float, dst: Path) -> None:
    """Re-encode [start, end] of src so the scene starts exactly at t=0.

    Re-encoding rather than stream-copying: a copy can only cut at keyframes,
    which would shift every timestamp in the scene. `faststart` lets a browser
    seek before the whole file has downloaded.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.3f}", "-i", str(src),
           "-t", f"{end - start:.3f}", "-c:v", "libx264", "-preset", "veryfast",
           "-crf", "23", "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", str(dst)]
    subprocess.run(cmd, check=True)


def make_ego4d_scenes(spec_path: Path, video_dir: Path, nlq_files: list[Path],
                      qaego4d_dir: Path, out_dir: Path, scenes_dir: Path) -> list[Scenario]:
    spec = yaml.safe_load(Path(spec_path).read_text(encoding="utf-8"))
    clips = load_nlq_clips(nlq_files)
    qa_rows = load_qaego4d(qaego4d_dir)
    scenarios = []
    for scene in spec["scenes"]:
        clip_uid = scene["clip_uid"]
        if clip_uid not in clips:
            raise KeyError(f"clip {clip_uid} not found in the NLQ annotations")
        clip = clips[clip_uid]
        rows = [r for r in qa_rows if r["video_id"] == clip_uid]
        # Join check: QAEgo4D windows are in video time and must fall inside
        # the clip, or the clip/question mapping is wrong.
        for r in rows:
            if not (clip["start"] - 1 <= r["video_start_sec"] <= clip["end"] + 1):
                raise ValueError(f"question {r['sample_id']} lies outside clip {clip_uid}")
        questions = [Question(id=r["sample_id"], question=r["question"].strip(),
                              answer=r["answer"], wrong_answers=r.get("wrong_answers", []),
                              t0=round(r["video_start_sec"] - clip["start"], 2),
                              t1=round(r["video_end_sec"] - clip["start"], 2),
                              category=categorize(r["question"]))
                     for r in sorted(rows, key=lambda r: r["video_start_sec"])]
        video_rel = Path("data/scenes") / f"{scene['id']}.mp4"
        cut_video(Path(video_dir) / f"{clip['video_uid']}.mp4", clip["start"], clip["end"],
                  Path(out_dir) / video_rel)
        scenario = Scenario(
            id=scene["id"], title=scene["title"], category=scene["category"],
            video=str(video_rel), questions=questions,
            demo_questions=scene.get("demo_questions", []),
            source={"dataset": "ego4d", "clip_uid": clip_uid, **clip})
        scenario.save(Path(scenes_dir) / f"{scene['id']}.json")
        scenarios.append(scenario)
        print(f"{scene['id']}: {len(questions)} questions, "
              f"{clip['end'] - clip['start']:.0f}s of video")
    return scenarios


def main() -> None:
    parser = argparse.ArgumentParser(description="Build scenarios from Ego4D NLQ clips.")
    parser.add_argument("--spec", type=Path, required=True, help="YAML listing the scenes")
    parser.add_argument("--ego4d-videos", type=Path, required=True,
                        help="directory with <video_uid>.mp4 (e.g. nlq/v2/video_540ss)")
    parser.add_argument("--nlq", type=Path, nargs="+", required=True,
                        help="Ego4D nlq_train.json / nlq_val.json")
    parser.add_argument("--qaego4d", type=Path, required=True,
                        help="directory with annotations.QaEgo4D_*.json")
    parser.add_argument("--root", type=Path, default=Path("."), help="repo root")
    parser.add_argument("--scenarios", type=Path, default=Path("scenarios"))
    args = parser.parse_args()
    make_ego4d_scenes(args.spec, args.ego4d_videos, args.nlq, args.qaego4d,
                      args.root, args.scenarios)


if __name__ == "__main__":
    main()
