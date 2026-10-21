"""Minimal client for any OpenAI-compatible chat server.

One client serves both roles of the pipeline (describer and reader), and the
same code runs against Ollama on a workstation, llama-server on a Raspberry Pi
or vLLM on a GPU server: only `base_url` and `model` change.

Stdlib only, so the client has no dependency to install on small devices.
"""

from __future__ import annotations

import base64
import io
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

import numpy as np
from PIL import Image

JPEG_QUALITY = 90


@dataclass
class ChatResult:
    text: str
    latency_s: float
    prompt_tokens: int | None
    completion_tokens: int | None


def image_part(frame_rgb: np.ndarray) -> dict:
    """Encode one RGB frame as an OpenAI `image_url` content part."""
    buf = io.BytesIO()
    Image.fromarray(frame_rgb).save(buf, format="JPEG", quality=JPEG_QUALITY)
    data = base64.b64encode(buf.getvalue()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}}


class ChatClient:
    """POST /chat/completions and return the text with its timing."""

    def __init__(self, base_url: str, model: str, reasoning_effort: str | None = "none",
                 timeout: float = 600.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        # Gemma 4 thinks by default, and the trace eats the whole token budget:
        # measured on Ollama 0.22, a 160-token describer call came back with an
        # empty answer. "none" switches it off. Servers that do not know the
        # field ignore it; set it to None to leave it out entirely.
        self.reasoning_effort = reasoning_effort
        self.timeout = timeout

    def chat(self, messages: list[dict], max_tokens: int = 256,
             temperature: float = 0.0, seed: int | None = None,
             response_format: dict | None = None) -> ChatResult:
        body: dict = {"model": self.model, "messages": messages,
                      "max_tokens": max_tokens, "temperature": temperature}
        if self.reasoning_effort is not None:
            body["reasoning_effort"] = self.reasoning_effort
        if seed is not None:
            body["seed"] = seed
        if response_format is not None:
            # Constrained decoding: the server only lets the model emit text
            # that matches the schema (Ollama, llama-server and vLLM all do).
            body["response_format"] = response_format
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read())
        except urllib.error.URLError as exc:
            raise ConnectionError(f"chat server at {self.base_url} failed: {exc}") from exc
        latency = time.perf_counter() - t0
        usage = payload.get("usage") or {}
        text = payload["choices"][0]["message"].get("content") or ""
        return ChatResult(text=text.strip(), latency_s=latency,
                          prompt_tokens=usage.get("prompt_tokens"),
                          completion_tokens=usage.get("completion_tokens"))
