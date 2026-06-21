# Training

Strata fine-tunes local Transformers models via the `train` command. GGUF models are not supported for training — download the Transformers version of the model instead.

## Basic usage

```bash
uv run main.py train {model_name} --data {file.jsonl} [options]
```

`model_name` must match a directory in `~/.strata/models/`.

---

## Training methods

| Flag | Method | Description | Best for |
|------|--------|-------------|----------|
| `--method lora` | LoRA | Parameter-efficient adapter training (default) | Most use cases, balances quality and VRAM |
| `--method qlora` | QLoRA | 4-bit quantised LoRA | Low VRAM machines (8 GB or less) |
| `--method full` | Full | All model weights trained | Maximum quality, requires lots of VRAM |
| `--method dpo` | DPO | Direct Preference Optimization | Preference alignment from chosen/rejected pairs |
| `--method grpo` | GRPO | Group Relative Policy Optimization (RL) | RL fine-tuning with a reward function |
| `--method ppo` | PPO | Proximal Policy Optimization (RL) | RL fine-tuning with a reward model |

---

## All flags

### Required

| Flag | Description |
|------|-------------|
| `--data {path}` | Path to a Strata JSONL dataset |

### Method

| Flag | Default | Description |
|------|---------|-------------|
| `--method {name}` | `lora` | Training method (see table above) |

### LoRA / QLoRA

| Flag | Default | Description |
|------|---------|-------------|
| `--lora-r N` | `16` | LoRA rank — higher rank = more parameters, more capacity |
| `--lora-alpha N` | `32` | LoRA alpha — scaling factor (usually 2× rank) |
| `--lora-dropout F` | `0.05` | Dropout on LoRA layers |
| `--lora-modules m,m,...` | `q_proj,v_proj` | Comma-separated target module names |

### Training loop

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs N` | `3` | Number of full passes over the dataset |
| `--batch-size N` | `2` | Per-device batch size |
| `--grad-accum N` | `4` | Gradient accumulation steps (effective batch = batch-size × grad-accum) |
| `--lr F` | `2e-4` | Learning rate |
| `--warmup N` | `10` | Warmup steps |
| `--max-seq N` | `512` | Maximum sequence length (tokens) |
| `--weight-decay F` | `0.01` | Weight decay (L2 regularisation) |
| `--seed N` | `42` | Random seed for reproducibility |

### DPO

| Flag | Default | Description |
|------|---------|-------------|
| `--dpo-beta F` | `0.1` | KL penalty coefficient — lower = stronger preference signal |

### GRPO / PPO

| Flag | Default | Description |
|------|---------|-------------|
| `--grpo-beta F` | `0.04` | GRPO KL penalty |
| `--ppo-beta F` | `0.1` | PPO KL penalty |
| `--reward-model {path}` | | HuggingFace reward model path (PPO) |
| `--reward-fn {path}` | | Python script with a `reward(prompt, response) -> float` function (GRPO / PPO) |
| `--num-generations N` | `4` | GRPO: number of completions sampled per prompt |
| `--max-new-tokens N` | `256` | Max tokens to generate per RL step |
| `--use-lora` | off | Apply LoRA adapters during GRPO/PPO training |

### Output

| Flag | Default | Description |
|------|---------|-------------|
| `--run-name {name}` | `{model}_{timestamp}` | Name for this training run — determines output directory |
| `--no-resume` | off | Start fresh; do not resume from an existing checkpoint |

---

## Examples

**Basic LoRA fine-tune:**
```bash
uv run main.py train mistral-7b --data sft.jsonl
```

**QLoRA on limited VRAM:**
```bash
uv run main.py train mistral-7b --data sft.jsonl --method qlora --epochs 5 --batch-size 1
```

**Full fine-tune with custom learning rate:**
```bash
uv run main.py train mistral-7b --data sft.jsonl --method full --lr 5e-5 --epochs 2
```

**DPO preference alignment:**
```bash
uv run main.py train mistral-7b --data dpo.jsonl --method dpo --dpo-beta 0.05
```

**GRPO with a reward function:**
```bash
uv run main.py train mistral-7b --data rl.jsonl --method grpo --reward-fn reward.py --num-generations 8
```

**Named run (useful for tracking experiments):**
```bash
uv run main.py train mistral-7b --data sft.jsonl --method lora --run-name experiment-lr2e4
```

---

## Output

Trained models are saved to:

```
~/.strata/trained/{run_name}/model/
~/.strata/trained/{run_name}/checkpoints/epoch-1/
~/.strata/trained/{run_name}/checkpoints/epoch-2/
...
```

The `model/` directory contains the final weights after all epochs. Checkpoints are saved after each epoch. If training is interrupted, re-running the same command (with the same `--run-name` or default name) will automatically resume from the latest checkpoint unless `--no-resume` is passed.

---

## Writing a reward function (GRPO / PPO)

Create a Python file with a `reward` function:

```python
# reward.py
def reward(prompt: str, response: str) -> float:
    """Return a scalar reward for this (prompt, response) pair."""
    # example: reward longer responses
    return min(len(response.split()) / 100.0, 1.0)
```

Pass it with `--reward-fn reward.py`.

---

## Dataset compatibility by method

| Method | Required task type |
|--------|-------------------|
| `lora`, `qlora`, `full` | `sft` or `chat` |
| `dpo` | `dpo` |
| `grpo`, `ppo` | `rl` |

Strata auto-detects the task type from the dataset and validates compatibility with the chosen method. Mismatches produce a clear error before training begins.
