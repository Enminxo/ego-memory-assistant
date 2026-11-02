"""Ego4D narrations and noun taxonomy, shared by narration_diary.py and object_precision.py."""

from __future__ import annotations

import ast
import csv
import json
import re
from bisect import bisect_left, bisect_right
from pathlib import Path

# Pre-registered stop words for the primary A0 endpoint (see README.md).
STOP_WORDS: tuple[str, ...] = (
    "left hand", "right hand", "hands", "hand", "fingers", "finger", "palm", "thumb",
    "person", "people", "man", "men", "woman", "women", "lady", "they", "himself",
    "herself", "shoulder", "floor", "ground", "surface", "land", "compound",
)


class Narrations:
    """Timestamped narrations of each video, sorted by time (video seconds)."""

    def __init__(self, by_video: dict[str, list[tuple[float, str]]]) -> None:
        self.by_video = by_video
        self._times = {v: [t for t, _ in rows] for v, rows in by_video.items()}

    @classmethod
    def load(cls, path: Path, pass_name: str = "narration_pass_1") -> "Narrations":
        raw = json.loads(Path(path).read_text())
        by_video: dict[str, list[tuple[float, str]]] = {}
        for video_uid, passes in raw.items():
            rows = (passes.get(pass_name) or {}).get("narrations", [])
            by_video[video_uid] = sorted((float(r["timestamp_sec"]), r["narration_text"])
                                         for r in rows)
        return cls(by_video)

    def between(self, video_uid: str, t0: float, t1: float) -> list[str]:
        """Narration texts with t0 <= time < t1."""
        times = self._times.get(video_uid, [])
        rows = self.by_video.get(video_uid, [])
        return [rows[i][1] for i in range(bisect_left(times, t0), bisect_right(times, t1))
                if rows[i][0] < t1]


def to_diary_text(narration: str) -> str:
    """'#C C opens a shelf' -> 'I opens a shelf'; '#O man X ...' -> 'man X ...'.

    The verb is left as written: the reader only needs the facts, and rewriting
    verbs would add a step that can go wrong.
    """
    text = re.sub(r"#unsure", "", narration, flags=re.I)
    text = re.sub(r"#[CO]\b", "", text)
    text = re.sub(r"\bC\b", "I", text)
    return " ".join(text.split()).strip(" .")


class Taxonomy:
    """Ego4D noun taxonomy: words and phrases mapped to labels."""

    def __init__(self, phrase_to_labels: dict[str, set[str]]) -> None:
        self.phrase_to_labels = phrase_to_labels
        # Longest phrases first, so "door handle" is consumed before "door".
        phrases = sorted(phrase_to_labels, key=len, reverse=True)
        self.pattern = re.compile(
            r"\b(" + "|".join(re.escape(p) for p in phrases) + r")(?:e?s)?\b")

    @classmethod
    def load(cls, path: Path) -> "Taxonomy":
        phrase_to_labels: dict[str, set[str]] = {}
        with Path(path).open(newline="") as fh:
            for row in csv.DictReader(fh):
                words = ast.literal_eval(row["group"])
                for w in words:
                    phrase_to_labels.setdefault(w.lower().strip(), set()).add(row["label"])
        return cls(phrase_to_labels)

    def labels(self, text: str, stop_words: tuple[str, ...] = ()) -> set[str]:
        """Taxonomy labels named in `text`, after removing `stop_words`."""
        t = " " + text.lower() + " "
        for w in stop_words:
            t = re.sub(r"\b" + re.escape(w) + r"\b", " ", t)
        found: set[str] = set()
        for m in self.pattern.finditer(t):
            found |= self.phrase_to_labels[m.group(1)]
        return found
