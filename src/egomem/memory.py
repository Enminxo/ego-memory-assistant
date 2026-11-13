"""The textual memory: an append-only, timestamped diary of the stream.

Each entry is the description of one clip. The diary is the only thing the
reader ever sees, which is what makes the design cheap (a few kB per minute of
video) and, if frames are not kept, privacy preserving.

Entries are streamed to a JSONL file as they are written, so a run survives a
crash and the diary can be read back, compared across models, or evaluated
offline.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image


def fmt_time(t: float) -> str:
    """Seconds to mm:ss, the format used in the diary and in citations."""
    total = int(round(t))
    return f"{total // 60:02d}:{total % 60:02d}"


@dataclass(frozen=True)
class MemoryEntry:
    index: int
    t0: float
    t1: float
    text: str
    latency_s: float
    n_frames: int
    # Names the face greeting recognized during the clip (live sessions). Kept
    # apart from `text`, which is only what the describer wrote.
    people: tuple[str, ...] = ()

    def render(self) -> str:
        line = f"[{fmt_time(self.t0)}-{fmt_time(self.t1)}] {self.text}"
        if self.people:
            line += f" (Recognized by face: {', '.join(self.people)}.)"
        return line


class TextualMemory:
    """Ordered diary entries, optionally mirrored to a JSONL file."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self._entries: list[MemoryEntry] = []
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, entry: MemoryEntry) -> None:
        self._entries.append(entry)
        self._entries.sort(key=lambda e: e.index)
        if self.path is not None:
            row = asdict(entry)
            if not row["people"]:
                del row["people"]  # diaries without face recognition keep their format
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    def entries(self) -> list[MemoryEntry]:
        return list(self._entries)

    def until(self, t: float) -> list[MemoryEntry]:
        """Entries of clips that had finished by second t.

        A question asked at t can only use what has already been written: this
        is what keeps a replayed demo honest about being online.
        """
        return [e for e in self._entries if e.t1 <= t + 1e-6]

    def __len__(self) -> int:
        return len(self._entries)

    @staticmethod
    def render(entries: list[MemoryEntry]) -> str:
        return "\n".join(e.render() for e in entries)

    @classmethod
    def load(cls, path: Path) -> "TextualMemory":
        """Rebuild a memory from its JSONL, keeping the file as its mirror."""
        memory = cls(path=None)
        path = Path(path)
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    row["people"] = tuple(row.get("people", ()))
                    memory._entries.append(MemoryEntry(**row))
        memory._entries.sort(key=lambda e: e.index)
        memory.path = path
        return memory


class EvidenceStore:
    """The frames each entry was written from, kept as JPEGs.

    Optional on purpose. Keeping them lets a person or caregiver check an
    answer against what the camera saw, but it gives up the "only text is
    stored" property of the design.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)

    def path(self, index: int, i: int) -> Path:
        return self.directory / f"clip{index:04d}_{i}.jpg"

    def save(self, index: int, frames: list[np.ndarray], width: int = 480) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(frames):
            image = Image.fromarray(frame)
            if image.width > width:
                image = image.resize((width, round(image.height * width / image.width)))
            image.save(self.path(index, i), quality=85)

    def count(self, index: int) -> int:
        return len(list(self.directory.glob(f"clip{index:04d}_*.jpg")))
