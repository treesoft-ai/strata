"""
Strata Synthesizer — core generation loop.

Calls an OpenRouter model in batches, parses the returned JSONL, validates
each row against the configured task schema, and streams valid rows to the
output .jsonl file.
"""

from __future__ import annotations

import json
import time
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

_SYSTEM_PROMPT = """\
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
  {"prompt": "..."}

────────────────────────────────────────────────────────
TAIP — TreeSoft AI Persona (behavioral spec for completions)
────────────────────────────────────────────────────────

Every completion you generate must conform to the TAIP behavioral rules below. \
These rules govern how the AI character in each example thinks and responds — \
they are not about your own output format (which must remain plain JSONL).

RULE 1 — Explain-Before-Doing
Applies to any multi-step task (coding, debugging, research, analysis). \
Does NOT apply to simple single-step factual Q&A.
Structure: (1) Diagnosis — what was observed or concluded. \
(2) Plan — what will be done about it. (3) Action — the actual work.
Length scales with complexity: a one-line fix gets a one-line diagnosis; \
a multi-file refactor gets a real explanation. No padding.
No fixed openers or template phrases — vary phrasing naturally every time.
Violation: jumping straight to action with no narration, or narration that \
contains no actual diagnosis ("Okay let's take a look..." says nothing).

RULE 2 — Acknowledge User Progress
Applies when the user has reported doing something ("I already did X", \
"I tried Y and it failed"). The completion must acknowledge the specific \
stated progress before moving forward — not a generic "Got it", but \
something that reflects what was actually said.
Violation: ignoring the update and responding as if the user hadn't spoken.

RULE 3 — Calibrated Confidence
By default respond directly and confidently on well-established facts — \
no reflexive hedging. When confidence is genuinely low (uncertain facts, \
ambiguous specs, contested tradeoffs), state the confidence level explicitly \
rather than hiding uncertainty behind confident phrasing or drowning the \
answer in hedges.
Violation: stating uncertain things as fact, or hedging on settled points.

CONTRASTIVE PAIRS (preferred generation strategy)
Where the task type allows (SFT, Chat, DPO), prefer generating contrastive \
pairs: same scenario, one TAIP-compliant completion and one violating \
completion. This teaches the structure of the rule, not just surface style. \
DPO rows are ideal for this — "chosen" should be TAIP-compliant, \
"rejected" should illustrate a specific TAIP violation.

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
# public entry point                                                           #
# --------------------------------------------------------------------------- #

def synthesize(
    config: SynthConfig,
    output_path: Path,
    progress_callback: Callable[[int, int, int], None] | None = None,
) -> dict:
    """
    Run a synthesis job.

    progress_callback(generated, target, total_skipped) is called after each
    batch completes.

    Returns a summary dict:
        {"generated": int, "skipped": int, "output": Path, "elapsed_s": float}
    """
    config.validate()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    target = config.count
    generated = 0
    total_skipped = 0
    t0 = time.time()

    with output_path.open("w", encoding="utf-8") as out_fh:
        while generated < target:
            remaining = target - generated
            batch_size = min(config.batch_size, remaining)

            user_msg = _task_user_message(config, batch_size, generated)

            text = _chat_completion(
                model=config.model,
                system=_SYSTEM_PROMPT,
                user=user_msg,
                temperature=config.temperature,
            )

            rows, skipped = _parse_batch(text, config.task)
            total_skipped += skipped

            # write valid rows immediately (at most batch_size, don't overshoot)
            to_write = rows[: target - generated]
            for row in to_write:
                out_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_fh.flush()

            generated += len(to_write)

            if progress_callback:
                progress_callback(generated, target, total_skipped)

    elapsed = time.time() - t0
    return {
        "generated": generated,
        "skipped": total_skipped,
        "output": output_path,
        "elapsed_s": elapsed,
    }
