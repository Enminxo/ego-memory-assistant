"""Describer: turns the frames of one clip into one diary entry.

`Describer` is the seam where other backends plug in. `VLMDescriber` talks to
any OpenAI-compatible server; a describer running on the Hailo-10H of a
Raspberry Pi (egoagent, `reasoning/descriptor.py`) would implement the same
`describe` method.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from egomem.llm import ChatClient, ChatResult, image_part
from egomem.memory import fmt_time
from egomem.prompts import describer_prompt

_SENTENCE_END = (".", "!", "?")


def trim_incomplete_sentence(text: str) -> str:
    """Drop a trailing fragment left by hitting the token cap.

    A fragment that names an object without saying what happened to it is
    worse than nothing for a reader that trusts the diary. Kept only if
    trimming would leave nothing.
    """
    stripped = text.rstrip()
    if stripped.endswith(_SENTENCE_END):
        return stripped
    cut = max(stripped.rfind(end) for end in _SENTENCE_END)
    return stripped[:cut + 1] if cut > 0 else stripped


class Describer(Protocol):
    def describe(self, frames: list[np.ndarray], t0: float, t1: float) -> ChatResult: ...


class VLMDescriber:
    """Describe clips with a vision-language model behind a chat server."""

    def __init__(self, client: ChatClient, max_tokens: int = 160, max_words: int = 60,
                 temperature: float = 0.2, seed: int | None = None) -> None:
        self.client = client
        self.max_tokens = max_tokens
        self.max_words = max_words
        self.temperature = temperature
        self.seed = seed

    def describe(self, frames: list[np.ndarray], t0: float, t1: float) -> ChatResult:
        # Each clip is described on its own, without the previous entry as
        # context: the paper's ablation found that feeding it back does not
        # help, and it lets one mistake propagate through the diary.
        prompt = describer_prompt(fmt_time(t0), fmt_time(t1), len(frames), self.max_words)
        content = [{"type": "text", "text": prompt}] + [image_part(f) for f in frames]
        result = self.client.chat([{"role": "user", "content": content}],
                                  max_tokens=self.max_tokens,
                                  temperature=self.temperature, seed=self.seed)
        result.text = trim_incomplete_sentence(result.text) or "(no description)"
        return result
