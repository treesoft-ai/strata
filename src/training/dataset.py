"""
Strata training dataset loader.

Loads a proprietary Strata JSONL file and auto-detects the task type from
the fields present in each row:

  SFT / LoRA   — {"prompt": "...", "completion": "..."}
  Chat / SFT   — {"messages": [{"role": "...", "content": "..."}, ...]}
  DPO          — {"prompt": "...", "chosen": "...", "rejected": "..."}

A file may contain rows of different task types; the detected type for the
whole file is determined by the first valid row.  Mixed files are rejected.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


TASK_SFT = "sft"
TASK_CHAT = "chat"
TASK_DPO = "dpo"
TASK_RL = "rl"


def _detect_row_type(row: dict) -> str | None:
    """Return the task type for a single row, or None if unrecognisable."""
    keys = set(row.keys())
    if "prompt" in keys and "chosen" in keys and "rejected" in keys:
        return TASK_DPO
    if "messages" in keys and isinstance(row["messages"], list):
        return TASK_CHAT
    if "prompt" in keys and "completion" in keys:
        return TASK_SFT
    if "prompt" in keys and "completion" not in keys and "chosen" not in keys and "messages" not in keys:
        return TASK_RL
    return None


def load_dataset(path: str | Path) -> dict[str, Any]:
    """
    Load a Strata JSONL file.

    Returns a dict:
        {
            "task":  "sft" | "chat" | "dpo",
            "rows":  [<raw row dicts>],
            "count": <int>,
        }

    Raises ValueError on empty files, unrecognisable rows, or mixed task types.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    if path.suffix.lower() != ".jsonl":
        raise ValueError(f"Dataset must be a .jsonl file, got: {path.suffix}")

    rows: list[dict] = []
    detected_task: str | None = None
    skipped = 0

    with path.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {lineno}: {exc}") from exc

            row_type = _detect_row_type(row)
            if row_type is None:
                skipped += 1
                continue

            if detected_task is None:
                detected_task = row_type
            elif row_type != detected_task:
                raise ValueError(
                    f"Mixed task types detected on line {lineno}: "
                    f"expected '{detected_task}', got '{row_type}'. "
                    "All rows in a Strata dataset must share the same format."
                )
            rows.append(row)

    if not rows:
        raise ValueError(
            "Dataset is empty or contains no recognisable rows. "
            "Expected rows with fields: "
            "(prompt + completion), (messages), (prompt + chosen + rejected), or (prompt only for RL)."
        )

    return {
        "task": detected_task,
        "rows": rows,
        "count": len(rows),
        "skipped": skipped,
    }


def to_hf_dataset(strata_dataset: dict[str, Any]):
    """
    Convert a loaded Strata dataset dict into a HuggingFace Dataset object.
    Requires the `datasets` package.
    """
    try:
        from datasets import Dataset
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required for training. "
            "Install it with: pip install datasets"
        ) from exc

    task = strata_dataset["task"]
    rows = strata_dataset["rows"]

    if task == TASK_SFT:
        records = [{"prompt": r["prompt"], "completion": r["completion"]} for r in rows]
    elif task == TASK_CHAT:
        records = [{"messages": r["messages"]} for r in rows]
    elif task == TASK_DPO:
        records = [
            {
                "prompt": r["prompt"],
                "chosen": r["chosen"],
                "rejected": r["rejected"],
            }
            for r in rows
        ]
    elif task == TASK_RL:
        records = [{"prompt": r["prompt"]} for r in rows]
    else:
        raise ValueError(f"Unknown task type: {task}")

    return Dataset.from_list(records)
