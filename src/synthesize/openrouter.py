"""
OpenRouter API client for Strata Synthesizer.

Handles key storage/retrieval and chat completion calls.
"""

from __future__ import annotations

import json
import urllib.request
import urllib.error
from pathlib import Path

from src.config import OPENROUTER_KEY_FILE

_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
_REFERER = "https://github.com/strata"


# --------------------------------------------------------------------------- #
# key management                                                               #
# --------------------------------------------------------------------------- #

def save_key(api_key: str) -> None:
    OPENROUTER_KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OPENROUTER_KEY_FILE.open("w", encoding="utf-8") as fh:
        json.dump({"api_key": api_key}, fh)


def load_key() -> str:
    if not OPENROUTER_KEY_FILE.exists():
        raise FileNotFoundError(
            "OpenRouter API key not set. "
            "Run 'uv run main.py synthesize key set {api_key}' to store it."
        )
    with OPENROUTER_KEY_FILE.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    key = data.get("api_key", "").strip()
    if not key:
        raise ValueError(
            "OpenRouter API key is empty. "
            "Run 'uv run main.py synthesize key set {api_key}' to update it."
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
    Call OpenRouter chat completions and return the assistant message content.
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
            "HTTP-Referer": _REFERER,
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            err = json.loads(raw)
            msg = err.get("error", {}).get("message", raw)
        except Exception:
            msg = raw
        raise RuntimeError(f"OpenRouter API error {exc.code}: {msg}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Network error calling OpenRouter: {exc.reason}") from exc

    try:
        return body["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise RuntimeError(f"Unexpected OpenRouter response shape: {body}") from exc
