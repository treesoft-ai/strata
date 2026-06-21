# Model Management

## Downloading models

```bash
uv run main.py download {link}
```

Strata accepts several link formats:

| Format | Example |
|--------|---------|
| Full HuggingFace URL | `https://huggingface.co/TheBloke/Mistral-7B-GGUF/resolve/main/mistral-7b.Q4_K_M.gguf` |
| `owner/repo/file.gguf` | `TheBloke/Mistral-7B-GGUF/mistral-7b.Q4_K_M.gguf` |
| `owner/repo` (full Transformers snapshot) | `mistralai/Mistral-7B-v0.1` |
| Single word (HF shorthand) | `gpt2` |

GGUF files are downloaded as single files. Transformers repos are downloaded as full snapshots (all weights, tokenizer, config).

Models are saved to `~/.strata/models/{model_name}/`.

### Gated models

Set the `HF_TOKEN` environment variable before downloading gated models (e.g. Llama 2, Gemma):

```bash
HF_TOKEN=hf_xxx uv run main.py download meta-llama/Llama-2-7b-hf
```

---

## Listing models

```bash
uv run main.py list models
```

Prints every model in `~/.strata/models/` with its detected type (`gguf` or `transformers`).

---

## Renaming a model

```bash
uv run main.py rename {old_name} {new_name}
```

Renames the model directory in `~/.strata/models/`.

---

## Removing a model

```bash
uv run main.py rm {model_name}
```

Permanently deletes the model directory from `~/.strata/models/`. There is no confirmation prompt — double-check the name first with `list models`.

---

## Model type detection

Strata auto-detects model type when loading:

- **GGUF** — presence of any `*.gguf` file in the model directory
- **Transformers** — presence of `config.json` plus `*.safetensors` or `*.bin` files

Both types share the same inference interface (`run` command) and the same management commands (`list`, `rm`, `rename`).
