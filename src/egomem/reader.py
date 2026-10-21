"""Reader: answers questions from the diary alone.

Two modes. `answer` is the open-ended reply the person sees in the app, with
the moments it cites. `choose` is the closed protocol of QAEgo4D-Closed (pick
one of four options), which is what the evaluation scores.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from egomem.llm import ChatClient
from egomem.memory import MemoryEntry, TextualMemory
from egomem.prompts import LETTERS, READER_CHOICE_SYSTEM, READER_SYSTEM, reader_prompt

# A citation is mm:ss, optionally a range "mm:ss-mm:ss"; a range counts as its start.
_CITATION = re.compile(r"\b(\d{1,2}):([0-5]\d)\b(?:\s*[-–]\s*\d{1,2}:[0-5]\d\b)?")
_CHOICE = re.compile(r"\b([A-D])\b")


@dataclass
class Answer:
    text: str
    citations: list[float] = field(default_factory=list)
    latency_s: float = 0.0
    n_entries: int = 0


def parse_citations(text: str) -> list[float]:
    """Every mm:ss in the answer, as seconds, in order of appearance."""
    return [int(m) * 60 + int(s) for m, s in _CITATION.findall(text)]


def parse_choice(text: str, n_options: int = 4) -> int | None:
    """Index of the option letter the model answered, or None if there is none.

    Matches capital letters only, so the article "a" in "on a table" is never
    read as option A.
    """
    match = _CHOICE.search(text.strip())
    if match is None:
        return None
    index = LETTERS.index(match.group(1))
    return index if index < n_options else None


class Reader:
    def __init__(self, client: ChatClient, max_tokens: int = 200,
                 temperature: float = 0.0) -> None:
        self.client = client
        self.max_tokens = max_tokens
        self.temperature = temperature

    def _messages(self, entries: list[MemoryEntry], question: str,
                  options: list[str] | None) -> list[dict]:
        system = READER_SYSTEM if options is None else READER_CHOICE_SYSTEM
        return [{"role": "system", "content": system},
                {"role": "user", "content": reader_prompt(
                    TextualMemory.render(entries), question, options)}]

    def answer(self, entries: list[MemoryEntry], question: str) -> Answer:
        result = self.client.chat(self._messages(entries, question, None),
                                  max_tokens=self.max_tokens, temperature=self.temperature)
        return Answer(text=result.text, citations=parse_citations(result.text),
                      latency_s=result.latency_s, n_entries=len(entries))

    def choose(self, entries: list[MemoryEntry], question: str,
               options: list[str]) -> tuple[int | None, str]:
        """Index of the chosen option, and the raw reply.

        The reply is constrained to {"answer": <one of the letters>}, so the
        reader cannot decline: it picks the letter it finds most likely, like
        scoring the first-token logits of A-D. Without the constraint it
        refuses on questions whose object the diary never mentions, and every
        refusal would count as a miss only for the arm that has a diary.
        """
        letters = list(LETTERS[:len(options)])
        schema = {"type": "object", "properties": {"answer": {"type": "string", "enum": letters}},
                  "required": ["answer"], "additionalProperties": False}
        result = self.client.chat(
            self._messages(entries, question, options), max_tokens=16,
            temperature=self.temperature,
            response_format={"type": "json_schema",
                             "json_schema": {"name": "choice", "strict": True, "schema": schema}})
        try:
            letter = json.loads(result.text)["answer"]
            return letters.index(letter), result.text
        except (ValueError, KeyError, TypeError):
            return parse_choice(result.text, len(options)), result.text
