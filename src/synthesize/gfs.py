"""
Strata Synthesizer — Grounding-From-Source (GFS) generation.

Source ingestion rules:
  Single file:
    .json   → if top-level value is an array, each item is a source unit
              (JSON-serialised); if it's an object/scalar, the whole file
              is one source unit.
    .jsonl  → each line is a source unit (JSON-serialised).
    other   → whole file text is one source unit.

  Directory (recursive walk, optional glob filter):
    Each file found = one source unit (read as UTF-8 text, errors replaced).
    JSON array explosion does NOT happen in directory mode — each file is
    treated as opaque text.

Source units are truncated to max_source_chars before being sent to the model.
"""

from __future__ import annotations

import json
import pathlib
import time
from typing import Callable

from src.synthesize.config import GFSConfig
import src.synthesize.openrouter as _openrouter
import src.synthesize.agentrouter as _agentrouter
from src.training.dataset import _detect_row_type


def _chat_completion(*, model: str, system: str, user: str, temperature: float) -> str:
    """Route to the correct provider based on model slug prefix."""
    if model.startswith("agentrouter/"):
        return _agentrouter.chat_completion(
            model=model[len("agentrouter/"):],
            system=system,
            user=user,
            temperature=temperature,
        )
    return _openrouter.chat_completion(
        model=model,
        system=system,
        user=user,
        temperature=temperature,
    )


# --------------------------------------------------------------------------- #
# source ingestion                                                             #
# --------------------------------------------------------------------------- #

def _read_file_text(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _load_source_units(config: GFSConfig) -> list[str]:
    source = pathlib.Path(config.source).expanduser()

    if not source.exists():
        raise FileNotFoundError(f"GFS source not found: {source}")

    units: list[str] = []

    if source.is_file():
        suffix = source.suffix.lower()
        if suffix == ".json":
            with source.open("r", encoding="utf-8", errors="replace") as fh:
                data = json.load(fh)
            if isinstance(data, list):
                for item in data:
                    units.append(json.dumps(item, ensure_ascii=False))
            else:
                units.append(json.dumps(data, ensure_ascii=False))
        elif suffix == ".jsonl":
            for line in source.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    units.append(json.dumps(item, ensure_ascii=False))
                except json.JSONDecodeError:
                    units.append(line)
        else:
            units.append(_read_file_text(source))

    elif source.is_dir():
        glob_pattern = config.glob.strip() if config.glob.strip() else "**/*"
        files = sorted(
            p for p in source.glob(glob_pattern)
            if p.is_file()
        )
        if not files:
            raise ValueError(
                f"No files found under '{source}' matching glob '{glob_pattern}'."
            )
        for f in files:
            units.append(_read_file_text(f))
    else:
        raise ValueError(f"GFS source is neither a file nor a directory: {source}")

    # truncate each unit
    max_c = config.max_source_chars
    truncated = []
    for u in units:
        if len(u) > max_c:
            u = u[:max_c] + f"\n... [truncated at {max_c} chars]"
        truncated.append(u)

    return truncated


# --------------------------------------------------------------------------- #
# generation                                                                   #
# --------------------------------------------------------------------------- #

_TASK_SCHEMA_HINT = {
    "sft":  'each line: {"prompt": "...", "completion": "..."}',
    "chat": 'each line: {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}',
    "dpo":  'each line: {"prompt": "...", "chosen": "...", "rejected": "..."}',
    "rl":   'each line: {"prompt": "..."}',
}


def _build_user_message(
    prompt: str,
    source_unit: str,
    task: str,
    count: int,
) -> str:
    schema = _TASK_SCHEMA_HINT[task]
    return (
        f"{prompt}\n\n"
        "────────────────────────────────────────────────────────\n"
        "Source material\n"
        "────────────────────────────────────────────────────────\n"
        f"{source_unit}\n\n"
        "────────────────────────────────────────────────────────\n"
        "Task\n"
        "────────────────────────────────────────────────────────\n"
        f"Task type: {task.upper()}\n"
        f"Schema: {schema}\n\n"
        f"Generate exactly {count} examples grounded in the source material above.\n"
        "Base every example strictly on information present in the source — "
        "do not hallucinate facts that are not there.\n"
        "Output only raw JSONL — one JSON object per line, nothing else."
    )


def _parse_batch(text: str, expected_task: str) -> tuple[list[dict], int]:
    valid: list[dict] = []
    skipped = 0
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("```"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            skipped += 1
            continue
        if not isinstance(row, dict):
            skipped += 1
            continue
        if _detect_row_type(row) != expected_task:
            skipped += 1
            continue
        valid.append(row)
    return valid, skipped


def synthesize_gfs(
    config: GFSConfig,
    output_path: pathlib.Path,
    progress_callback: Callable[[int, int, int, int, int], None] | None = None,
) -> dict:
    """
    Run a GFS synthesis job.

    progress_callback(source_idx, total_sources, prompt_idx, total_prompts,
                      examples_written, total_skipped)

    Returns:
        {"generated": int, "skipped": int, "sources": int, "output": Path, "elapsed_s": float}
    """
    config.validate()

    source_units = _load_source_units(config)
    total_sources = len(source_units)
    total_prompts = len(config.prompts)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    generated = 0
    total_skipped = 0
    t0 = time.time()
    system_prompt = _build_gfs_system_prompt(getattr(config, "taip_ignore", []))

    with output_path.open("w", encoding="utf-8") as out_fh:
        for src_idx, source_unit in enumerate(source_units):
            for prm_idx, prompt in enumerate(config.prompts):
                needed = config.count
                written_this_pass = 0

                while written_this_pass < needed:
                    batch_size = min(config.batch_size, needed - written_this_pass)
                    user_msg = _build_user_message(
                        prompt, source_unit, config.task, batch_size
                    )
                    text = _chat_completion(
                        model=config.model,
                        system=system_prompt,
                        user=user_msg,
                        temperature=config.temperature,
                    )
                    rows, skipped = _parse_batch(text, config.task)
                    total_skipped += skipped

                    to_write = rows[: needed - written_this_pass]
                    for row in to_write:
                        out_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    out_fh.flush()

                    written_this_pass += len(to_write)
                    generated += len(to_write)

                    if progress_callback:
                        progress_callback(
                            src_idx + 1, total_sources,
                            prm_idx + 1, total_prompts,
                            generated, total_skipped,
                        )

    elapsed = time.time() - t0
    return {
        "generated": generated,
        "skipped": total_skipped,
        "sources": total_sources,
        "output": output_path,
        "elapsed_s": elapsed,
    }


# --------------------------------------------------------------------------- #
# system prompt (GFS variant)                                                  #
# --------------------------------------------------------------------------- #

# Imported lazily to avoid circular imports at module level — defined here
# so it lives next to the generation logic.
from src.synthesize.generator import _build_system_prompt

_GFS_SUFFIX = """

────────────────────────────────────────────────────────
GFS — Grounding-From-Source mode
────────────────────────────────────────────────────────
You will be given a piece of source material (a document, JSON object, code \
file, or any other text). Every example you generate MUST be grounded in \
that source material:
• Extract facts, entities, and relationships directly from the source.
• Do NOT invent information that is absent from the source.
• Do NOT refer to the source itself in your output (e.g. do NOT write \
"According to this document…"); the examples must stand on their own.
• If the source is code, generate examples that teach correct usage, \
explain what the code does, or test understanding of it — never output \
fabricated function signatures or APIs that don't exist in the source.\
"""


def _build_gfs_system_prompt(taip_ignore: list) -> str:
    return _build_system_prompt(taip_ignore) + _GFS_SUFFIX
