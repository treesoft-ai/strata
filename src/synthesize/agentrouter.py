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

def chat_completion(
    *,
    model: str,
    system: str,
    user: str,
    temperature: float = 0.9,
) -> str:
    """
    Call AgentRouter chat completions and return the assistant message content.
    Raises RuntimeError on API errors.
    """
    api_key = load_key()

    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
    }).encode("utf-8")

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
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            err = json.loads(raw)
            msg = err.get("error", {}).get("message", raw)
        except Exception:
            msg = raw
        raise RuntimeError(f"AgentRouter API error {exc.code}: {msg}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Network error calling AgentRouter: {exc.reason}") from exc

    if not raw.strip():
        raise RuntimeError("AgentRouter returned an empty response body.")

    try:
        body = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"AgentRouter returned non-JSON response: {raw[:200]}") from exc

    try:
        return body["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise RuntimeError(f"Unexpected AgentRouter response shape: {body}") from exc
