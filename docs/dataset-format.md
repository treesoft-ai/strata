# Strata JSONL Dataset Format

Strata uses a single unified dataset file format: **JSONL** (one JSON object per line). The task type is auto-detected from the fields present in each row — you never declare it explicitly.

---

## Task types

### SFT — Supervised Fine-Tuning

Used with `--method lora`, `--method qlora`, `--method full`.

```jsonl
{"prompt": "What is the capital of France?", "completion": "Paris."}
{"prompt": "Write a Python function to reverse a string.", "completion": "def reverse(s):\n    return s[::-1]"}
```

Required fields: `prompt`, `completion`.

---

### Chat

Used with `--method lora`, `--method qlora`, `--method full` when the data is in multi-turn conversation format.

```jsonl
{"messages": [{"role": "user", "content": "What is 2 + 2?"}, {"role": "assistant", "content": "4."}]}
{"messages": [{"role": "user", "content": "Tell me a joke."}, {"role": "assistant", "content": "Why did the chicken cross the road? To get to the other side."}]}
```

Required fields: `messages` (a list of `{"role": ..., "content": ...}` objects). Valid roles are `user` and `assistant`.

---

### DPO — Direct Preference Optimization

Used with `--method dpo`.

```jsonl
{"prompt": "Summarize this article.", "chosen": "A concise, accurate summary.", "rejected": "A vague or incorrect summary."}
```

Required fields: `prompt`, `chosen`, `rejected`.

`chosen` is the preferred (better) response. `rejected` is the dispreferred response. The model learns to prefer `chosen` over `rejected` for the given `prompt`.

---

### RL — Reinforcement Learning

Used with `--method grpo` and `--method ppo`.

```jsonl
{"prompt": "Write a haiku about autumn."}
{"prompt": "Explain what a neural network is."}
```

Required fields: `prompt` only. The model generates responses during training; a reward model or reward function scores them.

---

## Format rules

- **One JSON object per line.** Empty lines are skipped.
- **All rows in a file must share the same task type.** Strata rejects mixed files with a clear error message identifying the offending line.
- **Unrecognisable rows are skipped** and counted in the summary. A row is unrecognisable if it has none of the field combinations above.
- **File must end in `.jsonl`.**
- **File must not be empty** (after skipping blank lines and unrecognisable rows).

---

## Quick detection rules

Strata determines task type from these field combinations:

| Fields present | Detected task |
|----------------|---------------|
| `prompt` + `chosen` + `rejected` | `dpo` |
| `messages` (list) | `chat` |
| `prompt` + `completion` | `sft` |
| `prompt` only | `rl` |
| anything else | skipped |

DPO takes priority over SFT if all three fields (`prompt`, `chosen`, `rejected`) are present.

---

## Example files

**sft.jsonl**
```jsonl
{"prompt": "What is backpropagation?", "completion": "Backpropagation is an algorithm for training neural networks by computing gradients of the loss with respect to each weight using the chain rule."}
{"prompt": "What is a transformer?", "completion": "A transformer is a neural network architecture that uses self-attention mechanisms to process sequences in parallel, introduced in the paper 'Attention Is All You Need'."}
```

**chat.jsonl**
```jsonl
{"messages": [{"role": "user", "content": "How do I sort a list in Python?"}, {"role": "assistant", "content": "Use the built-in sorted() function or the list's .sort() method. sorted() returns a new list; .sort() sorts in place."}]}
```

**dpo.jsonl**
```jsonl
{"prompt": "Explain what a mutex is.", "chosen": "A mutex (mutual exclusion lock) is a synchronization primitive that prevents multiple threads from accessing a shared resource simultaneously. Only the thread that holds the mutex can access the resource.", "rejected": "A mutex is a thing that helps with threads or something in programming."}
```

**rl.jsonl**
```jsonl
{"prompt": "Write a function that checks if a number is prime."}
{"prompt": "Explain the concept of overfitting in machine learning."}
```
