# Strata Synthesizer

The Synthesizer generates synthetic training datasets in Strata JSONL format using AI models from [OpenRouter](https://openrouter.ai). It is config-driven — you cannot run it without a config.

---

## Setup

### 1. Get an OpenRouter API key

Create an account at [openrouter.ai](https://openrouter.ai) and generate an API key.

### 2. Store the key in Strata

```bash
uv run main.py synthesize key set sk-or-your-key-here
```

The key is saved to `~/.strata/openrouter.json`. To verify it was saved:

```bash
uv run main.py synthesize key show
# API key: sk-or-v1-...****
```

### 3. Create a config

Configs live in `~/.strata/configs/{name}.json`. There is no default — you must create your own. See the sections below.

---

## Running the Synthesizer

```bash
uv run main.py synthesize {config_name}
```

Output is saved automatically to `~/.strata/datasets/{timestamp}_{id}.jsonl`. The resulting file is a valid Strata JSONL dataset ready to be passed directly to `train`.

---

## Config management

```bash
uv run main.py synthesize config list              # list all configs
uv run main.py synthesize config show {name}       # print full config details
uv run main.py synthesize config rm {name}         # delete a config
```

---

## Two modes

Strata Synthesizer supports two modes, selected by the `mode` field in the config:

| Mode | How it works |
|------|-------------|
| `standard` | Generates examples from scratch based on a user prompt |
| `gfs` | Grounds generation in real source files — no hallucinated facts |

---

## Standard mode

Generates examples from scratch. The AI is given a prompt describing what to produce and outputs JSONL directly.

### Config schema

```json
{
  "name": "my-config",
  "description": "Short description of what this generates",
  "mode": "standard",
  "user_prompt": "Generate diverse Q&A pairs covering Python programming concepts.",
  "model": "inclusionai/ling-2.6-flash",
  "task": "sft",
  "count": 100,
  "temperature": 0.9,
  "batch_size": 10
}
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Config name — must match the filename (`name.json`) |
| `description` | string | Human-readable description |
| `mode` | string | `"standard"` |
| `user_prompt` | string | The instruction given to the AI describing what data to generate |
| `model` | string | OpenRouter model ID (e.g. `"inclusionai/ling-2.6-flash"`) |
| `task` | string | Output task type: `sft`, `chat`, `dpo`, or `rl` |
| `count` | integer | Total number of examples to generate |
| `temperature` | float | Sampling temperature (0.0–2.0). Higher = more varied output |
| `batch_size` | integer | Examples requested per API call. Larger = fewer calls but bigger responses |

### Example configs

**Python SFT dataset:**
```json
{
  "name": "python-sft",
  "description": "Python programming Q&A pairs",
  "mode": "standard",
  "user_prompt": "Generate high-quality question and answer pairs covering Python programming. Include topics like data structures, standard library modules, OOP, error handling, and performance. Each question should be specific and practical. Each answer should be accurate, clear, and complete.",
  "model": "inclusionai/ling-2.6-flash",
  "task": "sft",
  "count": 200,
  "temperature": 0.85,
  "batch_size": 10
}
```

**DPO preference pairs (TAIP-aligned):**
```json
{
  "name": "taip-dpo",
  "description": "TAIP-compliant vs violating response pairs for preference training",
  "mode": "standard",
  "user_prompt": "Generate DPO training pairs where 'chosen' is a TAIP-compliant response (explains reasoning before acting, acknowledges user context, calibrated confidence) and 'rejected' is a TAIP-violating response (jumps to action, ignores user update, or over-hedges). Cover software engineering tasks: debugging, code review, architecture decisions.",
  "model": "inclusionai/ling-2.6-flash",
  "task": "dpo",
  "count": 100,
  "temperature": 0.9,
  "batch_size": 5
}
```

**RL prompt-only dataset:**
```json
{
  "name": "reasoning-rl",
  "description": "Reasoning prompts for RL training",
  "mode": "standard",
  "user_prompt": "Generate diverse reasoning prompts that require multi-step thinking. Include math word problems, logic puzzles, code debugging scenarios, and causal reasoning questions. Each prompt should be self-contained and have a clear correct answer.",
  "model": "inclusionai/ling-2.6-flash",
  "task": "rl",
  "count": 300,
  "temperature": 1.0,
  "batch_size": 15
}
```

---

## GFS mode — Grounding-From-Source

GFS generates examples grounded in real source material you provide. The AI reads each source unit and creates examples based strictly on what's in it — no hallucinated facts.

**Total examples produced** = `count × len(prompts) × number_of_source_units`

### Config schema

```json
{
  "name": "my-gfs-config",
  "description": "Short description",
  "mode": "gfs",
  "source": "/path/to/data",
  "glob": "**/*.py",
  "prompts": [
    "Generate a factual Q&A pair about this content.",
    "Generate a fill-in-the-blank exercise based on this content."
  ],
  "model": "inclusionai/ling-2.6-flash",
  "task": "sft",
  "count": 2,
  "temperature": 0.8,
  "batch_size": 5,
  "max_source_chars": 8000
}
```

### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | yes | Config name |
| `description` | string | yes | Human-readable description |
| `mode` | string | yes | `"gfs"` |
| `source` | string | yes | Path to a file or directory |
| `glob` | string | no | Glob filter for directory walks (e.g. `"**/*.py"`). Ignored for single files. Default: all files. |
| `prompts` | list of strings | yes | One prompt per generation pass. Each prompt runs over every source unit independently. |
| `model` | string | yes | OpenRouter model ID |
| `task` | string | yes | `sft`, `chat`, `dpo`, or `rl` |
| `count` | integer | yes | Examples to generate **per prompt per source unit** |
| `temperature` | float | yes | Sampling temperature |
| `batch_size` | integer | yes | Examples requested per API call |
| `max_source_chars` | integer | no | Truncate source units to this length. Default: `8000`. |

### Source ingestion rules

**Single file:**

| File type | How it's split into source units |
|-----------|----------------------------------|
| `.json` | If top-level is an array → each element is one source unit (JSON-serialised). If object/scalar → whole file is one unit. |
| `.jsonl` | Each line is one source unit (JSON-serialised). |
| Any other (`.txt`, `.md`, `.py`, `.csv`, etc.) | Whole file is one source unit. |

**Directory:**

Strata walks the directory recursively. Every file found becomes one source unit (read as plain text). JSON files in a directory are treated as plain text — array explosion only happens when you point directly at a single JSON file.

Apply a glob filter to select only relevant files:

```json
"source": "/path/to/repo",
"glob": "**/*.py"
```

### Example configs

**Wikipedia events → SFT Q&A:**

You've scraped world events into `events.json` as an array of objects like `{"title": "...", "summary": "..."}`.

```json
{
  "name": "wiki-events",
  "description": "Q&A pairs from scraped Wikipedia events",
  "mode": "gfs",
  "source": "/home/user/data/events.json",
  "glob": "",
  "prompts": [
    "Generate a factual question and answer pair about this event. The answer must be directly supported by the source.",
    "Generate a true/false question about this event with a brief explanation of why it is true or false.",
    "Generate a fill-in-the-blank sentence based on a key fact in this event."
  ],
  "model": "inclusionai/ling-2.6-flash",
  "task": "sft",
  "count": 1,
  "temperature": 0.7,
  "batch_size": 5,
  "max_source_chars": 4000
}
```

With 50 events in the array and 3 prompts, this produces **150 examples** (50 × 3 × 1).

---

**GitHub repo → code understanding pairs:**

You've cloned a repo to `/home/user/repos/mylib`.

```json
{
  "name": "mylib-code",
  "description": "Code understanding pairs from mylib source",
  "mode": "gfs",
  "source": "/home/user/repos/mylib",
  "glob": "**/*.py",
  "prompts": [
    "Generate a Q&A pair that tests understanding of what this code does and why.",
    "Generate a Q&A pair about the API this code exposes: its inputs, outputs, and any side effects."
  ],
  "model": "inclusionai/ling-2.6-flash",
  "task": "sft",
  "count": 2,
  "temperature": 0.75,
  "batch_size": 5,
  "max_source_chars": 8000
}
```

With 30 `.py` files and 2 prompts, this produces **120 examples** (30 × 2 × 2).

---

**Markdown docs → chat format:**

```json
{
  "name": "docs-chat",
  "description": "Multi-turn conversations grounded in documentation",
  "mode": "gfs",
  "source": "/home/user/project/docs",
  "glob": "**/*.md",
  "prompts": [
    "Generate a multi-turn conversation where a developer asks about the feature described in this documentation and an assistant answers accurately based on the doc."
  ],
  "model": "inclusionai/ling-2.6-flash",
  "task": "chat",
  "count": 3,
  "temperature": 0.85,
  "batch_size": 3,
  "max_source_chars": 6000
}
```

---

## Saving a config

Write the JSON manually to `~/.strata/configs/{name}.json`. The filename must match the `name` field exactly.

```bash
# Windows
notepad %USERPROFILE%\.strata\configs\my-config.json

# Linux / macOS
nano ~/.strata/configs/my-config.json
```

Then verify it loaded correctly:

```bash
uv run main.py synthesize config show my-config
```

---

## Full workflow example

```bash
# 1. Store API key
uv run main.py synthesize key set sk-or-v1-xxxx

# 2. Create a config
# Write ~/.strata/configs/python-sft.json manually

# 3. Verify the config looks right
uv run main.py synthesize config show python-sft

# 4. Run synthesis
uv run main.py synthesize python-sft

# 5. Check the output
uv run main.py list logs     # datasets go to ~/.strata/datasets/, not logs
# or just ls ~/.strata/datasets/

# 6. Train on the generated dataset
uv run main.py train my-model --data ~/.strata/datasets/20260621_143000_abc123.jsonl
```

---

## Choosing a model

Any model available on OpenRouter works. Some options:

| Model slug | Notes |
|-----------|-------|
| `inclusionai/ling-2.6-flash` | Default — fast, capable |
| `openai/gpt-4o` | High quality, higher cost |
| `openai/gpt-4o-mini` | Good quality, low cost |
| `anthropic/claude-3.5-sonnet` | Excellent instruction following |
| `meta-llama/llama-3.1-70b-instruct` | Strong open-weight option |
| `google/gemini-flash-1.5` | Fast and cheap |

See [openrouter.ai/models](https://openrouter.ai/models) for the full list and pricing.

---

## The system prompt

The Synthesizer uses a fixed system prompt that:

1. Instructs the model to act as a **data architect** generating Strata JSONL
2. Explains all four Strata task schemas (`sft`, `chat`, `dpo`, `rl`)
3. Enforces strict plain-JSONL output — no markdown fences, no commentary
4. Embeds the **TAIP behavioral spec** so generated completions follow TreeSoft's AI persona rules (explain-before-doing, acknowledge user progress, calibrated confidence)
5. In GFS mode, adds a grounding section that instructs the model to base everything on the provided source material and never invent absent facts

You cannot change the system prompt via a config. The `user_prompt` (standard mode) or `prompts` (GFS mode) are the user turn — they control what gets generated within the constraints the system prompt enforces.

---

## Tips

- **`batch_size`** controls how many examples are requested per API call. Larger batches are more efficient (fewer calls) but produce bigger responses that are more likely to contain malformed lines. `5`–`10` is a safe range for most models.
- **Invalid lines** from model output (malformed JSON, wrong schema) are skipped and counted in the summary. A high skip count usually means the model is adding markdown fences or commentary — try a stronger model or lower temperature.
- **`max_source_chars`** (GFS only) should be set with the model's context window in mind. The source is injected into the user message alongside the prompt and the task instruction. Leave headroom.
- **`temperature`** around `0.8`–`0.95` works well for diverse SFT/chat data. For DPO pairs or code, `0.7`–`0.8` reduces noise. For RL prompts, `1.0`+ gives more variety.
- GFS configs with large directories can take a long time. Start with a small `count` (e.g. `1`) and a limited `glob` to test before scaling up.
