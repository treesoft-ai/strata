"""
Strata training configuration.

TrainingConfig holds every hyperparameter that can be set by the user via
CLI flags.  Defaults are chosen to be sensible for a single-GPU LoRA run on
a consumer machine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class TrainingConfig:
    # ------------------------------------------------------------------ #
    # identity
    # ------------------------------------------------------------------ #
    model_name: str = ""
    run_name: str = ""          # used as output directory name
    data_path: str = ""

    # ------------------------------------------------------------------ #
    # method
    # ------------------------------------------------------------------ #
    method: Literal["lora", "qlora", "full", "dpo"] = "lora"
    # task is auto-detected from the dataset; can be overridden here
    task: Optional[Literal["sft", "chat", "dpo"]] = None

    # ------------------------------------------------------------------ #
    # LoRA / QLoRA settings (ignored for method=full)
    # ------------------------------------------------------------------ #
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )

    # ------------------------------------------------------------------ #
    # quantisation (QLoRA only)
    # ------------------------------------------------------------------ #
    load_in_4bit: bool = True        # used when method=qlora
    bnb_4bit_compute_dtype: str = "float16"

    # ------------------------------------------------------------------ #
    # training loop
    # ------------------------------------------------------------------ #
    epochs: int = 3
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_steps: int = 10
    max_seq_length: int = 512
    weight_decay: float = 0.01

    # ------------------------------------------------------------------ #
    # DPO-specific
    # ------------------------------------------------------------------ #
    dpo_beta: float = 0.1          # KL penalty coefficient for DPO

    # ------------------------------------------------------------------ #
    # checkpointing & output
    # ------------------------------------------------------------------ #
    save_checkpoints: bool = True   # always True per design; kept for explicitness
    resume: bool = True             # auto-resume from latest checkpoint if present
    output_dir: str = ""            # set at runtime from TRAINED_DIR / run_name

    # ------------------------------------------------------------------ #
    # misc
    # ------------------------------------------------------------------ #
    seed: int = 42
    fp16: bool = False              # set True automatically when CUDA + non-qlora
    bf16: bool = False              # preferred over fp16 on Ampere+
