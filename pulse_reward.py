"""
Pulse GRPO reward function.

Scores model responses on two signals:
  1. Reasoning chain presence  (0.0 – 0.4)
  2. Answer correctness        (0.0 – 0.6)

Total reward range: 0.0 – 1.0

Usage:
    uv run main.py train qwen3.5-0.8b --data pulse-rl.jsonl \
        --method grpo --reward-fn pulse_reward.py \
        --num-generations 8 --use-lora
"""

from __future__ import annotations

import re


# --------------------------------------------------------------------------- #
# answer extraction                                                            #
# --------------------------------------------------------------------------- #

def _extract_embedded_answer(prompt: str) -> str | None:
    """Pull the ground-truth answer from the [answer:...] tag in the prompt."""
    m = re.search(r'\[answer:(.+?)\]', prompt, re.IGNORECASE)
    if not m:
        return None
    return m.group(1).strip().lower()


def _extract_model_answer(response: str) -> str | None:
    """
    Extract the model's final answer from its response.

    Looks for explicit answer markers first, then falls back to the last
    number or capitalised token in the response.
    """
    # Explicit answer markers (most reliable)
    patterns = [
        r'(?:the\s+)?answer\s+is[:\s]+(.+?)(?:\.|$)',
        r'(?:therefore|thus|so)[,\s]+(?:the\s+answer\s+is\s+)?(.+?)(?:\.|$)',
        r'=\s*([^\s=]+)\s*$',
        r'\*\*(.+?)\*\*\s*$',       # bold final answer
        r'\\boxed\{(.+?)\}',         # LaTeX box (common in math)
    ]
    for pat in patterns:
        m = re.search(pat, response, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).strip().lower()

    # Fall back: last number in the response
    numbers = re.findall(r'-?\d+(?:\.\d+)?', response)
    if numbers:
        return numbers[-1].strip().lower()

    # Fall back: last non-trivial word
    words = re.findall(r'\b[a-zA-Z]{2,}\b', response)
    if words:
        return words[-1].strip().lower()

    return None


def _answers_match(expected: str, got: str) -> bool:
    """Flexible answer comparison."""
    expected = expected.strip().lower()
    got = got.strip().lower()

    if expected == got:
        return True

    # Numeric comparison with tolerance
    try:
        return abs(float(expected) - float(got)) < 1e-6
    except ValueError:
        pass

    # Strip punctuation and recompare
    clean = lambda s: re.sub(r'[^\w\s]', '', s).strip()
    if clean(expected) == clean(got):
        return True

    # Allow "yes"/"no" variants
    yes_variants = {"yes", "true", "correct", "always", "definitely"}
    no_variants  = {"no", "false", "incorrect", "never", "not necessarily"}
    if expected in yes_variants and got in yes_variants:
        return True
    if expected in no_variants and got in no_variants:
        return True

    return False


# --------------------------------------------------------------------------- #
# reasoning chain scorer                                                       #
# --------------------------------------------------------------------------- #

_REASONING_MARKERS = [
    r'\bfirst\b', r'\bstep\b', r'\bthen\b', r'\bnext\b', r'\bsince\b',
    r'\bbecause\b', r'\btherefore\b', r'\bthus\b', r'\bso\b',
    r'\bif\b.{0,40}\bthen\b', r'\blet\b', r'\bwe (have|get|know|can)\b',
    r'=\s*\d',         # arithmetic working
    r'\d+\s*[+\-×÷*/]\s*\d+',
]

def _reasoning_score(response: str) -> float:
    """
    Return 0.0–0.4 based on how much reasoning structure the response shows.

    A response that jumps straight to an answer with no working gets 0.0.
    A response with clear multi-step working and an explicit conclusion gets 0.4.
    """
    if not response.strip():
        return 0.0

    lines = [l.strip() for l in response.splitlines() if l.strip()]

    # Penalise one-line answers heavily
    if len(lines) == 1:
        return 0.05

    # Count distinct reasoning markers
    text = response.lower()
    hits = sum(1 for pat in _REASONING_MARKERS if re.search(pat, text))

    # Score: 0 hits → 0.05, 3+ hits → 0.4
    if hits == 0:
        return 0.05
    if hits == 1:
        return 0.15
    if hits == 2:
        return 0.25
    if hits == 3:
        return 0.33
    return 0.4


# --------------------------------------------------------------------------- #
# public entry point                                                           #
# --------------------------------------------------------------------------- #

def reward(prompt: str, response: str) -> float:
    """
    Score a (prompt, response) pair for Pulse GRPO training.

    Returns a float in [0.0, 1.0].
    """
    if not response or not response.strip():
        return 0.0

    r_score = _reasoning_score(response)

    # If the prompt has no embedded answer tag, reward reasoning only
    expected = _extract_embedded_answer(prompt)
    if expected is None:
        return r_score

    got = _extract_model_answer(response)
    if got is None:
        return r_score  # showed reasoning but couldn't extract answer

    correctness = 0.6 if _answers_match(expected, got) else 0.0
    return r_score + correctness
