"""
Strata Synthesizer — core generation loop.

Calls an OpenRouter model in batches, parses the returned JSONL, validates
each row against the configured task schema, and streams valid rows to the
output .jsonl file.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
from typing import Callable

from src.synthesize.config import SynthConfig
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
# system prompt                                                                #
# --------------------------------------------------------------------------- #

_PROMPT_HEADER = """\
You are a data architect specialising in generating high-quality synthetic \
training datasets for fine-tuning large language models.

You will be given instructions describing what kind of data to produce. \
Your job is to generate examples in the Strata JSONL format — exactly one \
JSON object per line, with absolutely no extra text, markdown code fences, \
headings, or explanations anywhere in your response.

────────────────────────────────────────────────────────
Strata JSONL — supported task schemas
────────────────────────────────────────────────────────

SFT (Supervised Fine-Tuning)
  {"prompt": "...", "completion": "..."}

Chat
  {"messages": [{"role": "user", "content": "..."}, \
{"role": "assistant", "content": "..."}]}

DPO (Direct Preference Optimisation)
  {"prompt": "...", "chosen": "...", "rejected": "..."}

RL (Reinforcement Learning — prompt only)
  {"prompt": "..."}\
"""

# Individual TAIP rules keyed by canonical name.
# taip_ignore in a config lists keys to omit (e.g. ["ebd"]).
_TAIP_RULES: dict[str, str] = {
    "ebd": """\
RULE 1 — Explain-Before-Doing
Applies to any multi-step task (coding, debugging, research, analysis). \
Does NOT apply to simple single-step factual Q&A.
Structure: (1) Diagnosis — what was observed or concluded. \
(2) Plan — what will be done about it. (3) Action — the actual work.
Length scales with complexity: a one-line fix gets a one-line diagnosis; \
a multi-file refactor gets a real explanation. No padding.
No fixed openers or template phrases — vary phrasing naturally every time.
Violation: jumping straight to action with no narration, or narration that \
contains no actual diagnosis ("Okay let's take a look..." says nothing).\
""",
    "aup": """\
RULE 2 — Acknowledge User Progress
Applies when the user has reported doing something ("I already did X", \
"I tried Y and it failed"). The completion must acknowledge the specific \
stated progress before moving forward — not a generic "Got it", but \
something that reflects what was actually said.
Violation: ignoring the update and responding as if the user hadn't spoken.\
""",
    "cc": """\
RULE 3 — Calibrated Confidence
By default respond directly and confidently on well-established facts — \
no reflexive hedging. When confidence is genuinely low (uncertain facts, \
ambiguous specs, contested tradeoffs), state the confidence level explicitly \
rather than hiding uncertainty behind confident phrasing or drowning the \
answer in hedges.
Violation: stating uncertain things as fact, or hedging on settled points.\
""",
}

_PROMPT_CONTRASTIVE = """\
CONTRASTIVE PAIRS (preferred generation strategy)
Where the task type allows (SFT, Chat, DPO), prefer generating contrastive \
pairs: same scenario, one TAIP-compliant completion and one violating \
completion. This teaches the structure of the rule, not just surface style. \
DPO rows are ideal for this — "chosen" should be TAIP-compliant, \
"rejected" should illustrate a specific TAIP violation.\
"""

_PROMPT_OUTPUT_RULES = """\
────────────────────────────────────────────────────────
Output rules (strictly enforced)
────────────────────────────────────────────────────────
• Output ONLY raw JSONL lines — one complete JSON object per line.
• Every line must be valid, parseable JSON.
• Every line must match the task schema requested by the user.
• Do NOT mix task schemas in the same response.
• Do NOT include blank lines, comments, or any non-JSON text.
• Every example must be unique, diverse, and genuinely useful for training.
• Vary content, length, complexity, and style across examples.
• Maintain factual accuracy and avoid harmful or misleading content.\
"""


def _build_system_prompt(taip_ignore: list[str]) -> str:
    """Build the system prompt, omitting any TAIP rules listed in taip_ignore."""
    ignore = {k.lower().strip() for k in taip_ignore}
    active_rules = [text for key, text in _TAIP_RULES.items() if key not in ignore]

    parts = [_PROMPT_HEADER]

    if active_rules:
        taip_block = (
            "\n\n────────────────────────────────────────────────────────\n"
            "TAIP — TreeSoft AI Persona (behavioral spec for completions)\n"
            "────────────────────────────────────────────────────────\n\n"
            "Every completion you generate must conform to the TAIP behavioral rules below. "
            "These rules govern how the AI character in each example thinks and responds — "
            "they are not about your own output format (which must remain plain JSONL).\n\n"
            + "\n\n".join(active_rules)
            + "\n\n"
            + _PROMPT_CONTRASTIVE
        )
        parts.append(taip_block)

    parts.append("\n\n" + _PROMPT_OUTPUT_RULES)
    return "".join(parts)


def _task_user_message(config: SynthConfig, batch_size: int, generated: int) -> str:
    """Build the user turn asking for the next batch of examples."""
    task_hint = {
        "sft":  'each line: {"prompt": "...", "completion": "..."}',
        "chat": 'each line: {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}',
        "dpo":  'each line: {"prompt": "...", "chosen": "...", "rejected": "..."}',
        "rl":   'each line: {"prompt": "..."}',
    }[config.task]

    return (
        f"{config.user_prompt}\n\n"
        f"Task type: {config.task.upper()}\n"
        f"Schema: {task_hint}\n\n"
        f"Generate exactly {batch_size} new, unique examples. "
        f"({generated} already generated so far — do not repeat them.)\n\n"
        "Output only raw JSONL — one JSON object per line, nothing else."
    )


def _parse_batch(text: str, expected_task: str) -> tuple[list[dict], int]:
    """
    Parse raw model output into valid rows matching expected_task.

    Returns (valid_rows, skipped_count).
    """
    valid: list[dict] = []
    skipped = 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # strip markdown fences models sometimes sneak in
        if line.startswith("```"):
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


# --------------------------------------------------------------------------- #
# writer thread                                                                #
# --------------------------------------------------------------------------- #

_WRITER_SENTINEL = object()


class _WriterThread(threading.Thread):
    """
    Dedicated writer thread that drains a queue and appends each row to the
    output file immediately, flushing after every line so data is durable
    against process crashes. fsync is called once at the end to commit
    everything to disk.

    If a write fails the exception is stored in self.error and the thread
    stops; the coordinator checks this and re-raises in the main thread.
    """

    def __init__(self, output_path: Path, mode: str = "w") -> None:
        super().__init__(name="strata-writer", daemon=False)
        self.output_path = output_path
        self.mode = mode
        self.queue: queue.Queue = queue.Queue()
        self.error: BaseException | None = None

    def run(self) -> None:
        try:
            with self.output_path.open(self.mode, encoding="utf-8") as fh:
                while True:
                    item = self.queue.get()
                    if item is _WRITER_SENTINEL:
                        fh.flush()
                        try:
                            os.fsync(fh.fileno())
                        except OSError:
                            pass  # some fs/platforms don't support fsync
                        break
                    fh.write(json.dumps(item, ensure_ascii=False) + "\n")
                    fh.flush()
        except Exception as exc:
            self.error = exc
            # drain so producers never block on a dead writer
            while True:
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    break

    def put(self, row: dict) -> None:
        """Enqueue a single row. Raises immediately if the writer has died."""
        if self.error is not None:
            raise self.error
        self.queue.put(row)

    def stop(self) -> None:
        """Signal the writer to finalise and exit, then block until it does."""
        self.queue.put(_WRITER_SENTINEL)
        self.join()
        if self.error is not None:
            raise self.error


# --------------------------------------------------------------------------- #
# public entry point                                                           #
# --------------------------------------------------------------------------- #

def _fetch_batch(config: SynthConfig, batch_size: int, already_generated: int, system_prompt: str) -> tuple[list[dict], int]:
    """Fetch and parse one batch. Returns (valid_rows, skipped_count)."""
    user_msg = _task_user_message(config, batch_size, already_generated)
    text = _chat_completion(
        model=config.model,
        system=system_prompt,
        user=user_msg,
        temperature=config.temperature,
    )
    return _parse_batch(text, config.task)


def synthesize(
    config: SynthConfig,
    output_path: Path,
    progress_callback: Callable[[int, int, int], None] | None = None,
    workers: int = 3,
    resume_from: int = 0,
) -> dict:
    """
    Run a synthesis job.

    Each valid row is handed off to a dedicated writer thread the moment it
    arrives and is flushed to disk immediately, so progress is never lost if
    the process is interrupted. A final fsync is issued when the job completes.

    progress_callback(generated, target, total_skipped) is called after each
    batch completes.

    workers controls how many API calls run concurrently (default 3).

    resume_from indicates how many rows already exist in output_path. When > 0,
    the writer opens in append mode and the generator only produces the
    remaining target - resume_from rows.

    Returns a summary dict:
        {"generated": int, "skipped": int, "output": Path, "elapsed_s": float}
    """
    config.validate()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    target = config.count
    generated = resume_from
    total_skipped = 0
    t0 = time.time()

    # Pre-compute batch sizes for the remaining rows only
    batches: list[int] = []
    remaining = max(0, target - resume_from)
    while remaining > 0:
        batches.append(min(config.batch_size, remaining))
        remaining -= config.batch_size

    batch_iter = iter(batches)
    system_prompt = _build_system_prompt(getattr(config, "taip_ignore", []))
    writer = _WriterThread(output_path, mode="a" if resume_from > 0 else "w")
    writer.start()

    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            pending: set = set()

            def _submit_next() -> None:
                try:
                    batch_size = next(batch_iter)
                    pending.add(executor.submit(_fetch_batch, config, batch_size, generated, system_prompt))
                except StopIteration:
                    pass

            # Seed the pool with up to `workers` concurrent requests
            for _ in range(workers):
                _submit_next()

            while pending:
                # Poll with a short timeout so we can react quickly if the
                # writer thread dies between future completions.
                done, _ = wait(pending, return_when=FIRST_COMPLETED, timeout=1.0)

                if writer.error is not None:
                    raise writer.error

                for future in done:
                    pending.discard(future)
                    rows, skipped = future.result()
                    total_skipped += skipped

                    if generated < target:
                        to_write = rows[: target - generated]
                        for row in to_write:
                            writer.put(row)  # raises immediately if writer died
                        generated += len(to_write)

                        if progress_callback:
                            progress_callback(generated, target, total_skipped)

                        if generated < target:
                            _submit_next()

    finally:
        writer.stop()  # flush, fsync, join — raises if the writer failed

    elapsed = time.time() - t0
    return {
        "generated": generated,
        "skipped": total_skipped,
        "output": output_path,
        "elapsed_s": elapsed,
    }
