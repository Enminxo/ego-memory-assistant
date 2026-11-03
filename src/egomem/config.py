"""Loading the YAML run configuration (see configs/demo.yaml)."""

from __future__ import annotations

from pathlib import Path

import yaml

from egomem.describer import VLMDescriber
from egomem.llm import ChatClient
from egomem.reader import Reader


def load_config(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def memory_dir(config: dict, scenario_id: str, tag: str) -> Path:
    """Where one run of the describer over one scenario is stored."""
    return Path(config["data_dir"]) / "memories" / scenario_id / tag


def make_describer(config: dict) -> VLMDescriber:
    d = config["describer"]
    return VLMDescriber(ChatClient(config["base_url"], d["model"]),
                        max_tokens=d["max_tokens"], max_words=d["max_words"],
                        temperature=d["temperature"], seed=d.get("seed"),
                        prompt_variant=d.get("prompt_variant", "default"))


def make_reader(config: dict) -> Reader:
    r = config["reader"]
    return Reader(ChatClient(config["base_url"], r["model"]),
                  max_tokens=r["max_tokens"], temperature=r["temperature"])
