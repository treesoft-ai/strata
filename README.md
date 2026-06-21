# Strata

> A unified local model management, inference execution, and fine-tuning CLI utility with real-time performance and hardware diagnostics.

---

## Overview

Strata is an internal research utility and framework designed to download, manage, execute inference on, and fine-tune large language models locally. It abstracts away the complexity of supporting different model formats, providing seamless execution for both GGUF formats (via `llama-cpp-python`) and Hugging Face Transformers formats (via PyTorch). Beyond inference, Strata offers deep real-time diagnostic capabilities — monitoring process and system memory utilization, token generation metrics, device utilization, and hardware temperatures during execution. Its training subsystem supports LoRA, QLoRA, full fine-tuning, and DPO preference alignment directly from the CLI.

---

## Features

- **Multi-Format Model Support**: Run models in both GGUF and PyTorch/Safetensors Hugging Face Transformers formats using a unified inference interface.
- **Automated Downloads**: Download models directly from Hugging Face by providing full URLs or simple repository shorthands (e.g., `owner/repo/file.gguf` or `owner/repo`).
- **Fine-Tuning & Training**: Fine-tune any local Transformers model with LoRA, QLoRA, full parameter training, or DPO preference alignment — configurable entirely from CLI flags.
- **Strata JSONL Dataset Format**: A unified proprietary dataset format that auto-detects task type (SFT, chat, or DPO) from row fields, with validation and mixed-type rejection.
- **Checkpoint Resumption**: Training saves a checkpoint after each epoch and automatically resumes from the latest one on rerun.
- **Real-Time Performance Logging**: Track metrics such as token generation speed (overall, peak, and lowest tokens-per-second) and execution time.
- **Hardware Diagnostic Monitoring**: Monitor process RAM consumption, system-wide memory load, compute device utilization, and hardware temperature in real-time via a background tracker thread.
- **Model Lifecycle Management**: Easily list, rename, and delete models locally from the persistent user models directory.
- **Detailed Run Logs**: Save comprehensive execution logs to a persistent JSON directory, capturing full prompts, token counts, hardware statistics, and chunk-by-chunk token traces.

---

## Project Structure

```
Strata/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── downloader.py
│   │   ├── gguf.py
│   │   ├── manager.py
│   │   └── transformers.py
│   └── training/
│       ├── __init__.py
│       ├── config.py
│       ├── dataset.py
│       └── trainer.py
├── .gitignore
├── LICENSE
├── main.py
├── README.md
└── requirements.txt
```

---

## Commands

```
download    Download a model from Hugging Face
run         Run inference on a model
train       Fine-tune a model on a Strata JSONL dataset
list        List models or logs
rm          Remove a model
rename      Rename a model
view        View a log file
```

Run `uv run main.py <command> --help` for details on any command.

---

## Training

Strata supports fine-tuning local Transformers models via the `train` command.

**Methods**

| Flag | Method | Description |
| --- | --- | --- |
| `--method lora` | LoRA | Parameter-efficient fine-tune (default) |
| `--method qlora` | QLoRA | 4-bit quantised LoRA, lowest VRAM |
| `--method full` | Full | All weights trained |
| `--method dpo` | DPO | Preference alignment from chosen/rejected pairs |

**Dataset format (Strata JSONL)**

One JSON object per line. Task type is auto-detected from the fields present.

```
SFT:   {"prompt": "...", "completion": "..."}
Chat:  {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
DPO:   {"prompt": "...", "chosen": "...", "rejected": "..."}
```

**Example usage**

```
uv run main.py train my-model --data train.jsonl
uv run main.py train my-model --data train.jsonl --method qlora --epochs 5 --lr 1e-4
uv run main.py train my-model --data prefs.jsonl --method dpo --dpo-beta 0.05
uv run main.py train my-model --data train.jsonl --method lora --lora-r 32 --run-name my-run
```

Trained models are saved to `~/.strata/trained/{run_name}/model/`.

---

## Storage Layout

```
~/.strata/
├── models/      Downloaded models (GGUF and Transformers)
├── trained/     Fine-tuned model outputs and checkpoints
└── logs/        Inference run logs (JSON)
```

---

## License

This project is licensed under the [TreeSoft Proprietary License](LICENSE).

---

## Maintainers

| Name        | Role              | GitHub                         |
| ----------- | ----------------- | ------------------------------ |
| Alexutzu    | Lead Engineer     | [@alexutzusoft](https://github.com/alexutzusoft) |

---
