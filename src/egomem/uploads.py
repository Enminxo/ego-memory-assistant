"""Recordings uploaded from the web page: turned into a scene, then described.

    page ─▶ POST /api/uploads ─▶ data/uploads/<id>/ ─▶ worker thread:
            convert (Neon native recording, or any video) ─▶ scene
            describe it clip by clip (egomem-build) ─▶ the diary grows on the page

The demo scenes were described beforehand; an uploaded one is described right
after it arrives, and the page shows its diary growing. One worker thread
handles the uploads in order, so two of them never compete for the GPU.

Two kinds of upload are accepted:

- A Neon recording in native format (the folder copied from the phone, or the
  "Native Recording Data" zip from Pupil Cloud). It goes through
  neon.import_recording, which places every frame at the time Neon stamped
  it, so the diary's times are the real ones.
- Any other video. It is re-encoded so a browser can play it and seek in it;
  its times are the container's own.

Jobs live in memory. If the app stops halfway, the partial diary stays on disk
and `egomem-build <id>` resumes it.
"""

from __future__ import annotations

import queue
import re
import shutil
import subprocess
import threading
import unicodedata
import zipfile
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO

from egomem.clips import clip_bounds
from egomem.scenarios import Scenario
from egomem.sources import VideoFileSource

VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
# The scene video of a native recording; pupil_labs.neon_recording looks for the same name.
NEON_SCENE_VIDEO = "Neon Scene Camera v1*.mp4"


def safe_relative(name: str) -> Path:
    """A file name sent by the browser, as a relative path that stays inside its folder.

    A folder picked in the browser arrives as one file per entry, named
    "folder/sub/file"; "..", "." and leading slashes are dropped.
    """
    parts = [p for p in re.split(r"[\\/]+", name) if p not in ("", ".", "..")]
    if not parts:
        raise ValueError(f"unusable file name {name!r}")
    return Path(*parts)


def save_upload(files: list[tuple[str, BinaryIO]], dest: Path) -> list[Path]:
    paths = []
    for name, data in files:
        path = dest / safe_relative(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as out:
            shutil.copyfileobj(data, out, 1 << 20)
        paths.append(path)
    return paths


def extract_zip(path: Path, dest: Path) -> None:
    """Unpack a zip, refusing it whole if any entry would land outside dest."""
    dest = dest.resolve()
    with zipfile.ZipFile(path) as archive:
        for member in archive.namelist():
            if not (dest / member).resolve().is_relative_to(dest):
                raise ValueError(f"{path.name} has an entry outside its folder: {member}")
        archive.extractall(dest)


def find_recording(directory: Path) -> tuple[str, Path]:
    """("neon", folder) for a native recording, else ("video", the largest video file).

    A native recording is a folder with info.json next to the scene video.
    Pupil Cloud's other exports also carry an info.json, but not that video,
    so they are taken as a plain video.
    """
    folders = sorted({p.parent for p in directory.rglob("info.json")
                      if any(p.parent.glob(NEON_SCENE_VIDEO))})
    if len(folders) > 1:
        raise ValueError(f"the upload holds {len(folders)} Neon recordings; upload one at a time")
    if folders:
        return "neon", folders[0]
    videos = [p for p in directory.rglob("*")
              if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES]
    if not videos:
        raise ValueError("no Neon recording and no video file in the upload")
    return "video", max(videos, key=lambda p: p.stat().st_size)


def reencode_video(src: Path, dst: Path, width: int = 1280) -> None:
    """Rewrite any video as one every browser plays and can seek in.

    H.264 in yuv420p, at most `width` pixels wide, with `faststart` so the
    page can seek before the whole file has arrived. The audio is kept if
    there is any. A phone's rotation flag is applied by ffmpeg.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-map", "0:v:0", "-map", "0:a:0?",
           "-vf", f"scale='min({width},iw)':-2", "-c:v", "libx264", "-preset", "veryfast",
           "-crf", "23", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "96k",
           "-movflags", "+faststart", str(dst)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        lines = result.stderr.strip().splitlines()
        raise RuntimeError(f"ffmpeg could not convert {src.name}: "
                           f"{lines[-1] if lines else 'exit ' + str(result.returncode)}")


def convert_upload(upload_dir: Path, scene_id: str, title: str, scenarios_dir: Path,
                   root: Path = Path("."), width: int = 1280) -> Scenario:
    """Turn what was uploaded into data/scenes/<id>.mp4 and scenarios/<id>.json.

    The scenario file is written last, once the video exists: the app lists
    every scenario file, and a tab must not appear before its video can play.
    """
    for archive in sorted(upload_dir.rglob("*.zip")):
        extract_zip(archive, archive.with_suffix(""))
    kind, path = find_recording(upload_dir)
    upload = {"upload": str(upload_dir.relative_to(root)), "file": str(path.relative_to(upload_dir))}
    if kind == "neon":
        from egomem.neon import import_recording

        try:
            scenario = import_recording(path, scene_id, title, scenarios_dir, root=root,
                                        width=width)
        except ModuleNotFoundError as exc:
            raise RuntimeError("reading a Neon recording needs the Pupil Labs packages: "
                               "uv sync --extra neon") from exc
        scenario.source.update(upload)
    else:
        video_rel = Path("data/scenes") / f"{scene_id}.mp4"
        reencode_video(path, root / video_rel, width)
        scenario = Scenario(id=scene_id, title=title, category="other", video=str(video_rel),
                            source=upload)
    scenario.save(Path(scenarios_dir) / f"{scene_id}.json")
    return scenario


def slugify(text: str) -> str:
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", ascii_text.lower()).strip("-")[:40] or "recording"


@dataclass
class Job:
    id: str
    title: str
    stage: str = "queued"  # queued, converting, describing, done, error
    total_clips: int | None = None
    error: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


class UploadJobs:
    """The uploads, converted and described one at a time by a worker thread.

    `build` writes the diary of a scene id; the app passes egomem-build's
    build(), the tests a fake one.
    """

    def __init__(self, scenarios_dir: Path, uploads_dir: Path, root: Path,
                 build: Callable[[str], object], clip_sec: float, width: int = 1280) -> None:
        self.scenarios_dir = Path(scenarios_dir)
        self.uploads_dir = Path(uploads_dir)
        self.root = Path(root)
        self.build = build
        self.clip_sec = clip_sec
        self.width = width
        self.jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._queue: queue.Queue[Job] = queue.Queue()
        threading.Thread(target=self._work, name="uploads", daemon=True).start()

    def _new_id(self, title: str) -> str:
        base = f"upload-{slugify(title)}"
        scene_id, n = base, 1
        while (scene_id in self.jobs or (self.scenarios_dir / f"{scene_id}.json").exists()
               or (self.uploads_dir / scene_id).exists()):
            n += 1
            scene_id = f"{base}-{n}"
        return scene_id

    def submit(self, title: str, files: list[tuple[str, BinaryIO]]) -> Job:
        """Save the files and queue them; the slow part runs in the worker."""
        if not files:
            raise ValueError("no files in the upload")
        names = [safe_relative(name) for name, _ in files]
        if not any(n.suffix.lower() in VIDEO_SUFFIXES | {".zip"} or n.name == "info.json"
                   for n in names):
            raise ValueError("upload a Neon recording folder, a .zip, or a video file "
                             f"({', '.join(sorted(VIDEO_SUFFIXES))})")
        title = title.strip() or names[0].parts[0].removesuffix(names[0].suffix)
        with self._lock:
            job = Job(id=self._new_id(title), title=title)
            self.jobs[job.id] = job
        try:
            save_upload(files, self.uploads_dir / job.id)
        except BaseException:
            shutil.rmtree(self.uploads_dir / job.id, ignore_errors=True)
            with self._lock:
                del self.jobs[job.id]
            raise
        self._queue.put(job)
        return job

    def _work(self) -> None:
        while True:
            job = self._queue.get()
            try:
                self._run(job)
            except Exception as exc:  # shown on the page, and the worker goes on
                job.stage, job.error = "error", str(exc) or repr(exc)
            finally:
                self._queue.task_done()

    def _run(self, job: Job) -> None:
        job.stage = "converting"
        scenario = convert_upload(self.uploads_dir / job.id, job.id, job.title,
                                  self.scenarios_dir, self.root, self.width)
        duration = VideoFileSource(scenario.video_path(self.root)).duration
        job.total_clips = len(clip_bounds(duration, self.clip_sec))
        job.stage = "describing"
        self.build(job.id)
        job.stage = "done"

    def wait_idle(self) -> None:
        """Block until every queued upload has been handled (for tests)."""
        self._queue.join()
