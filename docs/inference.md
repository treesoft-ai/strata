# Running Inference

## Basic usage

```bash
uv run main.py run {model_name} [prompt] [options]
```

The prompt is optional. If omitted, Strata will prompt you interactively.

```bash
# Inline prompt
uv run main.py run mistral-7b "Explain gradient descent in one paragraph."

# Interactive prompt
uv run main.py run mistral-7b
```

---

## Flags

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--max-tokens N` | | `256` | Maximum tokens to generate |
| `--temperature F` | | `0.7` | Sampling temperature (0.0 = greedy, higher = more random) |
| `--verbose` | `-v` | off | Show detailed per-chunk diagnostics and token trace |
| `--silent` | `-s` | off | Minimal output — token count and time only |
| `--bare` | `-b` | off | Disable background temperature and RAM tracking |
| `--log` | | off | Save the run to a log file in `~/.strata/logs/` |
| `--log-file {path}` | | | Save log to a specific path (implies `--log`) |

Flags can be combined freely:

```bash
uv run main.py run my-model "What is RLHF?" --max-tokens 512 --temperature 0.9 --log
uv run main.py run my-model "Hello" --silent --bare
uv run main.py run my-model "Debug this" --verbose --log-file ./debug.json
```

---

## Output modes

**Default** — streams the generated text, then prints a summary block:

```
* Strata / Run

  [generated text streams here...]

  Model:          mistral-7b
  Device:         cuda
  Prompt tokens:  12
  Output tokens:  87
  Time:           3.4s
  TPS overall:    25.6
  TPS peak:       31.2
  TPS lowest:     19.8
  Temp avg:       61.3°C
  RAM delta:      +142 MB
```

**`--silent`** — only shows token count and elapsed time, no text streaming header.

**`--verbose`** — shows the full summary plus a per-chunk token trace (each generated token's speed, elapsed time, temperature).

**`--bare`** — disables the background temperature/RAM tracker thread entirely. Useful when monitoring overhead is not desired or when psutil is unavailable.

---

## Logging

Use `--log` to persist a full run record to `~/.strata/logs/`. The log is a JSON file named by timestamp.

Log contents:

```json
{
  "model": "mistral-7b",
  "device": "cuda",
  "prompt": "...",
  "response": "...",
  "args": { "max_tokens": 256, "temperature": 0.7 },
  "stats": {
    "time_s": 3.4,
    "prompt_tokens": 12,
    "response_tokens": 87,
    "tps_overall": 25.6,
    "tps_peak": 31.2,
    "tps_lowest": 19.8,
    "temp_avg_c": 61.3,
    "ram_delta_mb": 142.0
  },
  "chunk_trace": [
    { "chunk": "The", "tokens": 1, "tps": 31.2, "elapsed_s": 0.03 },
    ...
  ],
  "timestamp": "2026-06-21T14:30:00"
}
```

---

## Viewing logs

```bash
uv run main.py list logs          # list all saved logs
uv run main.py view {filename}    # open a log in your editor
```

On Windows, logs open in Notepad. On Linux, Strata tries `nano`, then `vim`, then `vi`.
