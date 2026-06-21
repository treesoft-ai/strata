# Getting Started

## Requirements

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (recommended runner)
- CUDA-capable GPU optional but recommended for training

## Installation

```bash
git clone <repo>
cd Strata
uv sync          # installs all dependencies from requirements.txt
```

## Running any command

All commands are invoked through `main.py` via `uv run`:

```bash
uv run main.py <command> [args] [flags]
```

Get top-level help at any time:

```bash
uv run main.py
uv run main.py --help
```

Get help for a specific command:

```bash
uv run main.py <command> --help
```

## Storage layout

Strata stores everything in `~/.strata/`. The directory is created automatically on first run.

```
~/.strata/
├── models/          Downloaded GGUF and Transformers models
├── trained/         Fine-tuned model outputs
│   └── {run_name}/
│       ├── model/          Final saved weights
│       └── checkpoints/    Per-epoch checkpoints
├── logs/            Inference run logs (JSON)
├── configs/         Synthesizer configs (JSON)
├── datasets/        Synthesizer output datasets (JSONL)
└── openrouter.json  OpenRouter API key (Synthesizer only)
```

## Environment variables

| Variable | Purpose |
|----------|---------|
| `HF_TOKEN` | HuggingFace token for gated/private model downloads (optional) |

Set it in your shell profile or prefix commands:

```bash
HF_TOKEN=hf_xxx uv run main.py download meta-llama/Llama-2-7b-hf
```
