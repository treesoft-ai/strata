"""
Strata training engine.

Dispatches to the correct training strategy based on TrainingConfig.method:

  lora   — LoRA fine-tune via PEFT + TRL SFTTrainer
  qlora  — QLoRA (4-bit base + LoRA adapters) via PEFT + TRL SFTTrainer
  full   — Full fine-tune via TRL SFTTrainer (no PEFT)
  dpo    — DPO preference training via TRL DPOTrainer

Progress callback signature:
    progress_callback(step: int, total_steps: int, loss: float) -> None

The callback is called once per optimiser step so the CLI can overwrite
the current line with live progress.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional

from .config import TrainingConfig
from .dataset import TASK_DPO, TASK_CHAT, TASK_SFT, load_dataset, to_hf_dataset


# ------------------------------------------------------------------ #
# helpers
# ------------------------------------------------------------------ #

def _find_latest_checkpoint(output_dir: str) -> Optional[str]:
    """Return path to the latest checkpoint directory, or None."""
    ckpt_dir = Path(output_dir) / "checkpoints"
    if not ckpt_dir.exists():
        return None
    # HuggingFace Trainer saves checkpoints as checkpoint-<step>
    checkpoints = sorted(
        [p for p in ckpt_dir.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")],
        key=lambda p: int(p.name.split("-")[-1]),
    )
    return str(checkpoints[-1]) if checkpoints else None


def _make_lora_config(cfg: TrainingConfig):
    from peft import LoraConfig, TaskType
    return LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def _load_base_model_and_tokenizer(cfg: TrainingConfig):
    """Load the base model and tokenizer, applying quantisation if QLoRA."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_path = str(Path.home() / ".strata" / "models" / cfg.model_name)
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model '{cfg.model_name}' not found in ~/.strata/models. "
            "Run 'uv run main.py list models' to see available models."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = None
    if cfg.method == "qlora":
        import bitsandbytes  # noqa: F401 — ensure installed
        compute_dtype = getattr(torch, cfg.bnb_4bit_compute_dtype, torch.float16)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=cfg.load_in_4bit,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    model_kwargs: dict = {
        "trust_remote_code": True,
        "quantization_config": bnb_config,
    }

    use_cuda = torch.cuda.is_available()
    if use_cuda and cfg.method != "qlora":
        model_kwargs["torch_dtype"] = torch.float16

    if use_cuda:
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

    # enable gradient checkpointing for memory efficiency
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    return model, tokenizer


def _build_training_args(cfg: TrainingConfig, total_steps_hint: int):
    """Build a HuggingFace TrainingArguments object."""
    import torch
    from transformers import TrainingArguments

    use_cuda = torch.cuda.is_available()
    ckpt_dir = str(Path(cfg.output_dir) / "checkpoints")

    # auto-select mixed precision
    fp16 = cfg.fp16
    bf16 = cfg.bf16
    if use_cuda and not fp16 and not bf16 and cfg.method != "qlora":
        # prefer bf16 on Ampere+, fall back to fp16
        if torch.cuda.is_bf16_supported():
            bf16 = True
        else:
            fp16 = True

    return TrainingArguments(
        output_dir=ckpt_dir,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_steps=cfg.warmup_steps,
        weight_decay=cfg.weight_decay,
        fp16=fp16,
        bf16=bf16,
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=cfg.epochs,
        seed=cfg.seed,
        report_to="none",           # no wandb / tensorboard
        disable_tqdm=True,          # TTI handles progress
        dataloader_num_workers=0,
    )


# ------------------------------------------------------------------ #
# progress callback wiring
# ------------------------------------------------------------------ #

def _make_hf_callback(
    progress_callback: Optional[Callable],
    total_steps: int,
):
    """Return a HuggingFace TrainerCallback that forwards progress."""
    if progress_callback is None:
        return None

    from transformers import TrainerCallback

    class _StrataCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is None:
                return
            loss = logs.get("loss") or logs.get("train_loss")
            if loss is not None:
                progress_callback(
                    step=state.global_step,
                    total_steps=total_steps,
                    loss=float(loss),
                )

    return _StrataCallback()


# ------------------------------------------------------------------ #
# public API
# ------------------------------------------------------------------ #

def run_training(
    cfg: TrainingConfig,
    progress_callback: Optional[Callable[[int, int, float], None]] = None,
) -> str:
    """
    Execute training according to cfg.

    Returns the path to the final saved model directory.
    """
    strata_ds = load_dataset(cfg.data_path)

    # task override or auto-detect
    task = cfg.task or strata_ds["task"]

    if cfg.method == "dpo" and task != TASK_DPO:
        raise ValueError(
            "Method 'dpo' requires a DPO dataset (rows with prompt/chosen/rejected). "
            f"Detected task type: '{task}'."
        )
    if task == TASK_DPO and cfg.method != "dpo":
        raise ValueError(
            "Dataset contains DPO rows (prompt/chosen/rejected) but method is "
            f"'{cfg.method}'. Use --method dpo."
        )

    hf_ds = to_hf_dataset(strata_ds)

    # total optimiser steps for the progress bar
    steps_per_epoch = max(
        1, (strata_ds["count"] // (cfg.batch_size * cfg.gradient_accumulation_steps))
    )
    total_steps = steps_per_epoch * cfg.epochs

    # resume checkpoint
    resume_from = None
    if cfg.resume:
        resume_from = _find_latest_checkpoint(cfg.output_dir)

    # dispatch
    if cfg.method in ("lora", "qlora", "full") and task in (TASK_SFT, TASK_CHAT):
        final_dir = _train_sft(cfg, hf_ds, task, total_steps, progress_callback, resume_from)
    elif cfg.method == "dpo":
        final_dir = _train_dpo(cfg, hf_ds, total_steps, progress_callback, resume_from)
    else:
        raise ValueError(f"Unsupported method+task combination: method={cfg.method}, task={task}")

    return final_dir


def _train_sft(
    cfg: TrainingConfig,
    hf_ds,
    task: str,
    total_steps: int,
    progress_callback,
    resume_from: Optional[str],
) -> str:
    from trl import SFTTrainer, SFTConfig

    model, tokenizer = _load_base_model_and_tokenizer(cfg)

    peft_config = None
    if cfg.method in ("lora", "qlora"):
        from peft import get_peft_model, prepare_model_for_kbit_training
        if cfg.method == "qlora":
            model = prepare_model_for_kbit_training(model)
        peft_config = _make_lora_config(cfg)
        model = get_peft_model(model, peft_config)

    training_args = _build_training_args(cfg, total_steps)
    callback = _make_hf_callback(progress_callback, total_steps)

    # build formatting function based on task
    if task == TASK_SFT:
        def formatting_func(example):
            return [f"{p}{c}" for p, c in zip(example["prompt"], example["completion"])]
    else:
        # TASK_CHAT — flatten messages into a single string
        def formatting_func(example):
            results = []
            for msgs in example["messages"]:
                parts = [f"<|{m['role']}|>\n{m['content']}" for m in msgs]
                results.append("\n".join(parts))
            return results

    trainer = SFTTrainer(
        model=model,
        train_dataset=hf_ds,
        args=SFTConfig(
            output_dir=str(Path(cfg.output_dir) / "checkpoints"),
            num_train_epochs=cfg.epochs,
            per_device_train_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            learning_rate=cfg.learning_rate,
            warmup_steps=cfg.warmup_steps,
            weight_decay=cfg.weight_decay,
            fp16=training_args.fp16,
            bf16=training_args.bf16,
            logging_steps=1,
            save_strategy="epoch",
            save_total_limit=cfg.epochs,
            seed=cfg.seed,
            report_to="none",
            disable_tqdm=True,
            max_seq_length=cfg.max_seq_length,
        ),
        formatting_func=formatting_func,
        callbacks=[callback] if callback else [],
    )

    trainer.train(resume_from_checkpoint=resume_from)

    final_dir = str(Path(cfg.output_dir) / "model")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    return final_dir


def _train_dpo(
    cfg: TrainingConfig,
    hf_ds,
    total_steps: int,
    progress_callback,
    resume_from: Optional[str],
) -> str:
    from trl import DPOTrainer, DPOConfig

    model, tokenizer = _load_base_model_and_tokenizer(cfg)

    peft_config = None
    if cfg.method in ("lora", "qlora"):
        from peft import get_peft_model, prepare_model_for_kbit_training
        if cfg.method == "qlora":
            model = prepare_model_for_kbit_training(model)
        peft_config = _make_lora_config(cfg)
        model = get_peft_model(model, peft_config)

    training_args = _build_training_args(cfg, total_steps)
    callback = _make_hf_callback(progress_callback, total_steps)

    trainer = DPOTrainer(
        model=model,
        ref_model=None,     # uses implicit reference with LoRA; set explicitly for full
        args=DPOConfig(
            output_dir=str(Path(cfg.output_dir) / "checkpoints"),
            num_train_epochs=cfg.epochs,
            per_device_train_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            learning_rate=cfg.learning_rate,
            warmup_steps=cfg.warmup_steps,
            weight_decay=cfg.weight_decay,
            fp16=training_args.fp16,
            bf16=training_args.bf16,
            logging_steps=1,
            save_strategy="epoch",
            save_total_limit=cfg.epochs,
            seed=cfg.seed,
            report_to="none",
            disable_tqdm=True,
            beta=cfg.dpo_beta,
        ),
        train_dataset=hf_ds,
        tokenizer=tokenizer,
        peft_config=peft_config,
        callbacks=[callback] if callback else [],
    )

    trainer.train(resume_from_checkpoint=resume_from)

    final_dir = str(Path(cfg.output_dir) / "model")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    return final_dir
