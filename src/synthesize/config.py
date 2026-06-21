"""
Strata Synthesizer config management.

Configs live at ~/.strata/configs/{name}.json and control every aspect of a
synthesis run.  Two modes are supported:

  standard  — from-scratch generation driven by a single user_prompt.
  gfs       — Grounding-From-Source: ingest real files, generate examples
               grounded in that content.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from src.config import CONFIGS_DIR

VALID_TASKS = ("sft", "chat", "dpo", "rl")


# --------------------------------------------------------------------------- #
# standard config                                                              #
# --------------------------------------------------------------------------- #

@dataclass
class SynthConfig:
    name: str
    description: str
    user_prompt: str
    model: str
    task: Literal["sft", "chat", "dpo", "rl"]
    count: int
    temperature: float
    batch_size: int
    mode: str = "standard"

    @classmethod
    def load(cls, name: str) -> "SynthConfig":
        path = _config_path(name)
        if not path.exists():
            raise FileNotFoundError(
                f"Synthesis config '{name}' not found. "
                f"Run 'uv run main.py synthesize config list' to see available configs."
            )
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        data.setdefault("mode", "standard")
        return cls(**data)

    def save(self) -> None:
        CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
        with _config_path(self.name).open("w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2)

    def validate(self) -> None:
        if self.task not in VALID_TASKS:
            raise ValueError(
                f"Config task '{self.task}' is invalid. "
                f"Must be one of: {', '.join(VALID_TASKS)}."
            )
        if self.count < 1:
            raise ValueError("Config 'count' must be at least 1.")
        if self.batch_size < 1:
            raise ValueError("Config 'batch_size' must be at least 1.")
        if not self.model.strip():
            raise ValueError("Config 'model' must not be empty.")
        if not self.user_prompt.strip():
            raise ValueError("Config 'user_prompt' must not be empty.")


# --------------------------------------------------------------------------- #
# GFS config                                                                   #
# --------------------------------------------------------------------------- #

@dataclass
class GFSConfig:
    """
    Grounding-From-Source config.

    count      = examples generated per prompt per source unit.
    prompts    = list of user prompts; each drives a separate pass over every
                 source unit.
    source     = path to a file or directory.
    glob       = optional glob pattern for directory walks (e.g. "**/*.py").
                 Ignored when source is a single file.
    max_source_chars = source content is truncated to this length before being
                 sent to the model (default 8000).
    """
    name: str
    description: str
    prompts: list
    source: str
    model: str
    task: Literal["sft", "chat", "dpo", "rl"]
    count: int
    temperature: float
    batch_size: int
    mode: str = "gfs"
    glob: str = ""
    max_source_chars: int = 8000

    @classmethod
    def load(cls, name: str) -> "GFSConfig":
        path = _config_path(name)
        if not path.exists():
            raise FileNotFoundError(
                f"Synthesis config '{name}' not found. "
                f"Run 'uv run main.py synthesize config list' to see available configs."
            )
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        data.setdefault("mode", "gfs")
        data.setdefault("glob", "")
        data.setdefault("max_source_chars", 8000)
        return cls(**data)

    def save(self) -> None:
        CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
        with _config_path(self.name).open("w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2)

    def validate(self) -> None:
        if self.task not in VALID_TASKS:
            raise ValueError(
                f"Config task '{self.task}' is invalid. "
                f"Must be one of: {', '.join(VALID_TASKS)}."
            )
        if not self.prompts:
            raise ValueError("GFS config 'prompts' must contain at least one prompt.")
        for i, p in enumerate(self.prompts):
            if not str(p).strip():
                raise ValueError(f"GFS config prompt at index {i} is empty.")
        if self.count < 1:
            raise ValueError("Config 'count' must be at least 1.")
        if self.batch_size < 1:
            raise ValueError("Config 'batch_size' must be at least 1.")
        if not self.model.strip():
            raise ValueError("Config 'model' must not be empty.")
        if not self.source.strip():
            raise ValueError("GFS config 'source' must not be empty.")
        if self.max_source_chars < 100:
            raise ValueError("Config 'max_source_chars' must be at least 100.")


# --------------------------------------------------------------------------- #
# config directory helpers                                                     #
# --------------------------------------------------------------------------- #

def _config_path(name: str) -> Path:
    return CONFIGS_DIR / f"{name}.json"


def load_any_config(name: str) -> "SynthConfig | GFSConfig":
    """Load a config by name, returning the correct type based on its 'mode' field."""
    path = _config_path(name)
    if not path.exists():
        raise FileNotFoundError(
            f"Synthesis config '{name}' not found. "
            f"Run 'uv run main.py synthesize config list' to see available configs."
        )
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    mode = data.get("mode", "standard")
    if mode == "gfs":
        return GFSConfig.load(name)
    return SynthConfig.load(name)


def list_configs() -> list[dict]:
    """Return a list of {name, mode} dicts for all saved configs."""
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for p in sorted(CONFIGS_DIR.glob("*.json")):
        try:
            with p.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            results.append({
                "name": p.stem,
                "mode": data.get("mode", "standard"),
                "task": data.get("task", "?"),
                "model": data.get("model", "?"),
                "count": data.get("count", "?"),
            })
        except Exception:
            results.append({"name": p.stem, "mode": "?", "task": "?", "model": "?", "count": "?"})
    return results


def delete_config(name: str) -> None:
    path = _config_path(name)
    if not path.exists():
        raise FileNotFoundError(f"Config '{name}' does not exist.")
    path.unlink()
