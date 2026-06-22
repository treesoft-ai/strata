"""
AgentRouter API client for Strata Synthesizer.

Uses the agentrouter/ model slug prefix. The prefix is stripped before the
request is sent, so "agentrouter/claude-sonnet-4-6" becomes "claude-sonnet-4-6"
on the wire.

AgentRouter requires client-identification headers matching a recognised client
(Codex CLI) to authorise requests.
"""

from __future__ import annotations

import json
import random
import time
import urllib.request
import urllib.error
from pathlib import Path

from src.config import AGENTROUTER_KEY_FILE

_CHAT_URL = "https://agentrouter.org/v1/chat/completions"
_USER_AGENT = "codex_cli_rs/0.101.0 (Mac OS 26.0.1; arm64) Apple_Terminal/464"
_ORIGINATOR = "codex_cli_rs"
_VERSION = "0.101.0"


# --------------------------------------------------------------------------- #
# key management                                                               #
# --------------------------------------------------------------------------- #

def save_key(api_key: str) -> None:
    AGENTROUTER_KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with AGENTROUTER_KEY_FILE.open("w", encoding="utf-8") as fh:
        json.dump({"api_key": api_key}, fh)


def load_key() -> str:
    if not AGENTROUTER_KEY_FILE.exists():
        raise FileNotFoundError(
            "AgentRouter API key not set. "
            "Run 'uv run main.py synthesize key set agentrouter {api_key}' to store it."
        )
    with AGENTROUTER_KEY_FILE.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    key = data.get("api_key", "").strip()
    if not key:
        raise ValueError(
            "AgentRouter API key is empty. "
            "Run 'uv run main.py synthesize key set agentrouter {api_key}' to update it."
        )
    return key


def key_exists() -> bool:
    try:
        load_key()
        return True
    except (FileNotFoundError, ValueError):
        return False


# --------------------------------------------------------------------------- #
# chat completion                                                              #
# --------------------------------------------------------------------------- #

_TIMEOUT = 240          # seconds per request
_RETRY_DELAY = 5        # seconds for first retry, capped at _RETRY_MAX_DELAY
_RETRY_MAX_DELAY = 20   # maximum seconds between retries


def chat_completion(
    *,
    model: str,
    system: str,
    user: str,
    temperature: float = 0.9,
) -> str:
    """
    Call AgentRouter chat completions and return the assistant message content.
    Retries up to _MAX_RETRIES times on transient network/timeout errors.
    Raises RuntimeError on API errors or exhausted retries.
    """
    api_key = load_key()

    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }).encode("utf-8")

    last_error: Exception | None = None
    attempt = 0

    while True:
        attempt += 1
        req = urllib.request.Request(
            _CHAT_URL,
            data=payload,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
                "User-Agent": _USER_AGENT,
                "Originator": _ORIGINATOR,
                "Version": _VERSION,
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            try:
                err = json.loads(raw)
                msg = err.get("error", {}).get("message", raw)
            except Exception:
                msg = raw
            # 429 rate limit is transient — retry forever with backoff + jitter
            if exc.code == 429:
                base = min(_RETRY_DELAY * attempt, _RETRY_MAX_DELAY)
                time.sleep(base + random.uniform(0, base * 0.5))
                continue
            # all other HTTP errors are not retryable
            raise RuntimeError(f"AgentRouter API error {exc.code}: {msg}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            base = min(_RETRY_DELAY * attempt, _RETRY_MAX_DELAY)
            time.sleep(base + random.uniform(0, base * 0.5))
            continue

        if not raw.strip():
            base = min(_RETRY_DELAY * attempt, _RETRY_MAX_DELAY)
            time.sleep(base + random.uniform(0, base * 0.5))
            continue

        try:
            body = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"AgentRouter returned non-JSON response: {raw[:200]}") from exc

        try:
            return body["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise RuntimeError(f"Unexpected AgentRouter response shape: {body}") from exc

