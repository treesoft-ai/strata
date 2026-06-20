# Strata

> A unified local model management and inference execution CLI utility with real-time performance and hardware diagnostics.

---

## Overview

Strata is an internal research utility and framework designed to download, manage, and execute inference on large language models locally. It abstracts away the complexity of supporting different model formats, providing seamless execution for both GGUF formats (via `llama-cpp-python`) and Hugging Face Transformers formats (via PyTorch). Beyond simple inference, Strata offers deep, real-time diagnostic capabilities by monitoring process and system memory utilization, token generation metrics (TPS), device utilization, and hardware temperatures (supporting Nvidia GPUs, Linux sysfs thermal zones, and Windows PowerShell temperature counters) during prompt execution.

---

## Features

- **Multi-Format Model Support**: Run models in both GGUF and PyTorch/Safetensors Hugging Face Transformers formats using a unified inference interface.
- **Automated Downloads**: Download models directly from Hugging Face by providing full URLs or simple repository shorthands (e.g., `owner/repo/file.gguf` or `owner/repo`).
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
│   └── models/
│       ├── __init__.py
│       ├── base.py
│       ├── downloader.py
│       ├── gguf.py
│       ├── manager.py
│       └── transformers.py
├── .gitignore
├── LICENSE
├── main.py
├── README.md
└── requirements.txt
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
