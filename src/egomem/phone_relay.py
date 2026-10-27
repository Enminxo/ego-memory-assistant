"""The phone side of the live link: runs in Termux on the Neon Companion phone.

The Companion app streams the scene camera as RTSP. Over the office Wi-Fi the
whole video (about 5 Mbit/s) did not fit: on 2026-10-26 13 of 22 live clips
were lost. Inside the phone there is no Wi-Fi in the way: read at 127.0.0.1,
the same plain RTSP lost 0 packets in 10 minutes (2026-10-27). So the video is
decoded here, and only what the computers need leaves the phone:

    Companion ─RTSP 127.0.0.1─▶ ffmpeg ─┬─▶ diary/: n photos per clip ──POST──▶ egomem-app
                                        └─▶ face/latest.jpg, 3 per s ◀──GET─── facewatch
              phone speaker ◀── termux-tts-speak ◀── POST /say ◀─────────────── facewatch

The diary's photos are sent in order and retried until they arrive, never
skipped: if the Wi-Fi drops, they wait on the phone and catch up afterwards
(they take about 0.7 Mbit/s, a seventh of the video). Each photo's place in
the diary comes from when it was taken on the phone's clock, so a late photo
still lands in its own clip. The face photo is only ever the latest one: a
face seen two seconds ago is of no use, so nothing queues up for it.

Only the standard library and Termux's ffmpeg, so that it runs as one file:

    pkg install python ffmpeg termux-api
    python phone_relay.py --bind 100.64.0.1      # the phone's Tailscale address

    POST /diary/start  {"port": 8090, "token": "...", "clip_sec": 15, "n_frames": 8}
                       photos go to http://<caller>:<port>/api/phone/photo
    POST /diary/stop
    GET  /latest.jpg   the newest face photo; X-Frame-Seq changes with each one
         /latest.jpg?after=<seq>   waits up to 2 s for a newer one, else 304
    POST /say          {"text": "Hola, Ana"}
    GET  /status
"""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

DEFAULT_PORT = 8770
RTSP_URL = "rtsp://127.0.0.1:8086/?camera=world"
FACE_STALE_SEC = 5.0     # an older face photo is not handed out
RETRY_MAX_SEC = 5.0      # longest wait between two tries of the same photo
RESTART_SEC = 2.0        # wait before restarting an ffmpeg that died


@dataclass
class DiarySession:
    """Where the diary's photos go, and how they are numbered."""

    url: str
    token: str
    clip_sec: float
    n_frames: int
    width: int | None = None
    origin: float | None = None  # phone time at which slot 0 starts, set by the first photo
    last_k: int = -1
    sent: int = 0

    @property
    def period(self) -> float:
        return self.clip_sec / self.n_frames

    def place(self, taken: float) -> int:
        """The photo's number in the session (clip k // n_frames, slot k % n_frames).

        The first photo is put in the middle of slot 0, so later ones can come
        up to half a period early or late and keep their own slot. Two photos
        never share a slot: the second moves to the next one. After a gap
        (ffmpeg restarted) the numbers jump with the clock, and the clip left
        with holes is dropped by the app instead of the diary shrinking.
        """
        if self.origin is None:
            self.origin = taken - self.period / 2
        k = max(math.floor((taken - self.origin) / self.period), self.last_k + 1)
        self.last_k = k
        return k


def ffmpeg_command(ffmpeg: str, rtsp_url: str, diary_dir: Path, face_path: Path,
                   clip_sec: float, n_frames: int, diary_width: int | None,
                   face_fps: float, face_width: int) -> list[str]:
    """One decoder, two outputs: the diary's photos and the latest face photo.

    Both are written to a temporary name and renamed when complete
    (atomic_writing), so a file that exists is a whole JPEG.
    """
    diary = f"fps={n_frames}/{clip_sec:g}" + (f",scale={diary_width}:-2" if diary_width else "")
    graph = (f"[0:v]split=2[d][f];[d]{diary}[diary];"
             f"[f]fps={face_fps:g},scale={face_width}:-2[face]")
    return [ffmpeg, "-hide_banner", "-loglevel", "warning", "-rtsp_transport", "udp",
            "-i", rtsp_url, "-filter_complex", graph,
            "-map", "[diary]", "-q:v", "3", "-f", "image2", "-atomic_writing", "1",
            str(diary_dir / "f_%06d.jpg"),
            "-map", "[face]", "-q:v", "5", "-f", "image2", "-update", "1",
            "-atomic_writing", "1", str(face_path)]


def termux_speak(stream: str, lang: str, region: str) -> Callable[[str], None]:
    """Say a text through the phone's speaker with Termux:API.

    The ALARM stream is the default because it sounds even with the phone in
    silent mode, as the Companion phone was on 2026-10-27.
    """
    def speak(text: str) -> None:
        subprocess.run(["termux-tts-speak", "-s", stream, "-l", lang, "-n", region, text],
                       check=True, timeout=60)
    return speak


class Relay:
    """ffmpeg kept running, the diary's uploads, and the voice, each in a thread.

    `ffmpeg=None` starts no decoder (the tests drop photos into the folders).
    """

    def __init__(self, workdir: Path, speak: Callable[[str], None], ffmpeg: str | None = "ffmpeg",
                 rtsp_url: str = RTSP_URL, face_fps: float = 3.0, face_width: int = 960) -> None:
        self.workdir = Path(workdir)
        self.diary_dir = self.workdir / "diary"
        self.face_path = self.workdir / "face" / "latest.jpg"
        self.speak = speak
        self.ffmpeg = ffmpeg
        self.rtsp_url = rtsp_url
        self.face_fps = face_fps
        self.face_width = face_width
        self.diary_params: tuple[float, int, int | None] = (15.0, 8, None)
        self.session: DiarySession | None = None
        self.ffmpeg_runs = 0
        self.ffmpeg_error: str | None = None
        self.upload_error: str | None = None
        self.voice_error: str | None = None
        self.voice: queue.Queue[str | None] = queue.Queue()
        self.lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._restart = threading.Event()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self.diary_dir.mkdir(parents=True, exist_ok=True)
        self.face_path.parent.mkdir(parents=True, exist_ok=True)

    def start(self) -> None:
        targets = [self._upload_loop, self._speak_loop]
        if self.ffmpeg is not None:
            targets.append(self._supervise_ffmpeg)
        for target in targets:
            thread = threading.Thread(target=target, name=target.__name__, daemon=True)
            thread.start()
            self._threads.append(thread)

    def close(self) -> None:
        self._stop.set()
        self.voice.put(None)
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
        for thread in self._threads:
            thread.join(timeout=5)

    # --- ffmpeg ---------------------------------------------------------------

    def _supervise_ffmpeg(self) -> None:
        """Keep ffmpeg running: restart it when it dies, or when the diary's rate changes.

        Each run writes into its own folder, so the numbering of a new run
        never overwrites photos still waiting to be sent.
        """
        while not self._stop.is_set():
            run_dir = self.diary_dir / f"{self.ffmpeg_runs:04d}"
            run_dir.mkdir(parents=True, exist_ok=True)
            clip_sec, n_frames, width = self.diary_params
            command = ffmpeg_command(self.ffmpeg, self.rtsp_url, run_dir, self.face_path,
                                     clip_sec, n_frames, width, self.face_fps, self.face_width)
            with open(self.workdir / "ffmpeg.log", "ab") as log:
                self._proc = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                              stdout=log, stderr=log)
            self.ffmpeg_runs += 1
            code = self._proc.wait()
            if self._stop.is_set():
                break
            if self._restart.is_set():  # asked for: a new rate, not a failure
                self._restart.clear()
                continue
            self.ffmpeg_error = (f"ffmpeg exited with code {code} at {time.strftime('%H:%M:%S')};"
                                 f" see {self.workdir / 'ffmpeg.log'}")
            self._stop.wait(RESTART_SEC)

    def _restart_ffmpeg(self) -> None:
        self._restart.set()
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()

    # --- the diary --------------------------------------------------------------

    def photos(self) -> list[Path]:
        """Diary photos waiting on the phone, oldest first."""
        return sorted(self.diary_dir.glob("*/f_*.jpg"))

    def diary_start(self, url: str, token: str, clip_sec: float, n_frames: int,
                    width: int | None = None) -> None:
        with self.lock:
            self.session = DiarySession(url, token, clip_sec, n_frames, width)
            self.upload_error = None
            for photo in self.photos():  # taken before this session began
                photo.unlink(missing_ok=True)
            if (clip_sec, n_frames, width) != self.diary_params:
                self.diary_params = (clip_sec, n_frames, width)
                self._restart_ffmpeg()

    def diary_stop(self) -> None:
        with self.lock:
            self.session = None

    def _send(self, session: DiarySession, data: bytes, k: int, taken: float,
              pending: int) -> int:
        """POST one photo; the HTTP status, or 0 if the app could not be reached."""
        query = urllib.parse.urlencode({
            "token": session.token, "clip": k // session.n_frames, "slot": k % session.n_frames,
            "taken": f"{taken:.3f}", "pending": pending})
        request = urllib.request.Request(f"{session.url}?{query}", data=data, method="POST",
                                         headers={"Content-Type": "image/jpeg"})
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return response.status
        except urllib.error.HTTPError as exc:
            self.upload_error = f"the app answered {exc.code}: {exc.reason}"
            return exc.code
        except OSError as exc:  # no route, refused, timed out
            self.upload_error = repr(exc)
            return 0

    def _upload_loop(self) -> None:
        """Send the diary's photos in order, each until it arrives.

        A photo is numbered once, when it is first picked, and keeps that
        number through every retry. Outside a session photos are thrown away.
        """
        current: tuple[DiarySession, Path, int, float] | None = None
        retry = 0.0
        while not self._stop.is_set():
            with self.lock:
                session = self.session
            photos = self.photos()
            if session is None:
                for photo in photos:
                    photo.unlink(missing_ok=True)
                current = None
                self._stop.wait(0.1)
                continue
            if current is None or current[0] is not session:
                current = None
                try:
                    taken = photos[0].stat().st_mtime
                except (IndexError, FileNotFoundError):  # nothing new yet
                    self._stop.wait(0.1)
                    continue
                current = (session, photos[0], session.place(taken), taken)
            _, photo, k, taken = current
            try:
                data = photo.read_bytes()
            except FileNotFoundError:  # thrown away by a new session meanwhile
                current = None
                continue
            status = self._send(session, data, k, taken, pending=len(photos))
            if status == 200:
                photo.unlink(missing_ok=True)
                session.sent += 1
                current, retry = None, 0.0
                self.upload_error = None
            elif status == 409:  # the app has no such session any more
                with self.lock:
                    if self.session is session:
                        self.session = None
                current, retry = None, 0.0
            else:
                retry = min(max(retry * 2, 0.5), RETRY_MAX_SEC)
                self._stop.wait(retry)

    # --- faces and voice -------------------------------------------------------

    def latest_face(self) -> tuple[bytes, int] | None:
        """The newest face photo and a number that changes with each one; None if stale."""
        try:
            with self.face_path.open("rb") as handle:
                stat = os.fstat(handle.fileno())
                data = handle.read()
        except FileNotFoundError:
            return None
        if time.time() - stat.st_mtime > FACE_STALE_SEC:
            return None
        return data, stat.st_mtime_ns

    def wait_for_face(self, after: int, timeout: float = 2.0) -> tuple[bytes, int] | None:
        """A face photo newer than number `after`, waiting up to `timeout` s for one.

        Returns the same photo again if none came, so the caller can answer 304
        and nothing crosses the Wi-Fi twice.
        """
        deadline = time.monotonic() + timeout
        while True:
            face = self.latest_face()
            if (face is not None and face[1] != after) or time.monotonic() > deadline:
                return face
            self._stop.wait(0.02)

    def _speak_loop(self) -> None:
        """Say the texts one after another, so two greetings never talk over each other."""
        while (text := self.voice.get()) is not None:
            try:
                self.speak(text)
                self.voice_error = None
            except Exception as exc:  # a voice that fails must not stop the relay
                self.voice_error = repr(exc)

    def status(self) -> dict:
        with self.lock:
            session = self.session
        face = self.latest_face()
        return {
            "diary": None if session is None else {
                "clip_sec": session.clip_sec, "n_frames": session.n_frames,
                "sent": session.sent, "pending": len(self.photos())},
            "face": face is not None,
            "ffmpeg_runs": self.ffmpeg_runs, "ffmpeg_error": self.ffmpeg_error,
            "upload_error": self.upload_error, "voice_error": self.voice_error,
        }


def make_handler(relay: Relay) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _reply(self, code: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _body(self) -> dict:
            length = int(self.headers.get("Content-Length", "0"))
            return json.loads(self.rfile.read(length) or b"{}")

        def do_GET(self) -> None:
            url = urllib.parse.urlparse(self.path)
            query = urllib.parse.parse_qs(url.query)
            if url.path == "/status":
                self._reply(200, relay.status())
            elif url.path == "/latest.jpg":
                after = int(query["after"][0]) if "after" in query else None
                face = relay.latest_face() if after is None else relay.wait_for_face(after)
                if face is None:
                    self._reply(503, {"error": "no recent picture from the glasses"})
                    return
                data, seq = face
                if seq == after:  # nothing newer than what the caller already has
                    self.send_response(304)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("X-Frame-Seq", str(seq))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)
            else:
                self._reply(404, {"error": f"no {url.path} here"})

        def do_POST(self) -> None:
            path = urllib.parse.urlparse(self.path).path
            try:
                body = self._body()
                if path == "/diary/start":
                    host = self.client_address[0]
                    netloc = f"[{host}]" if ":" in host else host
                    relay.diary_start(f"http://{netloc}:{int(body['port'])}/api/phone/photo",
                                      str(body["token"]), float(body["clip_sec"]),
                                      int(body["n_frames"]), body.get("width"))
                elif path == "/diary/stop":
                    relay.diary_stop()
                elif path == "/say":
                    relay.voice.put(str(body["text"]))
                else:
                    self._reply(404, {"error": f"no {path} here"})
                    return
            except (KeyError, ValueError) as exc:
                self._reply(400, {"error": repr(exc)})
                return
            self._reply(200, relay.status())

        def log_message(self, format: str, *args: object) -> None:
            pass  # one line per photo and per face picture would bury the errors

    return Handler


def serve(relay: Relay, host: str, port: int) -> ThreadingHTTPServer:
    """Answer HTTP in a thread; returns the server (shutdown() to stop it)."""
    server = ThreadingHTTPServer((host, port), make_handler(relay))
    threading.Thread(target=server.serve_forever, name="http", daemon=True).start()
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="Relay the Neon glasses from the phone.")
    parser.add_argument("--bind", required=True,
                        help="address to listen on: the phone's Tailscale IP, not the Wi-Fi's")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--workdir", type=Path, default=Path.home() / "egorelay")
    parser.add_argument("--rtsp", default=RTSP_URL)
    parser.add_argument("--face-fps", type=float, default=3.0)
    parser.add_argument("--face-width", type=int, default=960)
    parser.add_argument("--voice-stream", default="ALARM",
                        help="ALARM sounds even in silent mode; MUSIC follows the media volume")
    parser.add_argument("--lang", default="es")
    parser.add_argument("--region", default="ES")
    args = parser.parse_args()

    relay = Relay(args.workdir, termux_speak(args.voice_stream, args.lang, args.region),
                  rtsp_url=args.rtsp, face_fps=args.face_fps, face_width=args.face_width)
    relay.start()
    server = serve(relay, args.bind, args.port)
    print(f"relay on http://{args.bind}:{args.port}  (Ctrl+C to stop)", flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("stopping")
    finally:
        server.shutdown()
        relay.close()


if __name__ == "__main__":
    main()
