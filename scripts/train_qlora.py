#!/usr/bin/env python3
"""
QLoRA fine-tuning script for text-to-SQL generation.

This script loads an open-weight LLM (e.g., Phi-3-mini), quantizes it to 4-bit,
adds LoRA adapters, and trains it on the processed Spider instruction dataset
using TRL's SFTTrainer.

All metrics, loss curves, and hyperparameters are logged to Weights & Biases.

Features:
    1. Config-driven design: Reads a base YAML config, merges optional sweep YAML
       configs, and accepts CLI arguments as overrides.
    2. Real QLoRA: Uses bitsandbytes for 4-bit NF4 quantization, peft for adapter
       injection, and trl for supervised fine-tuning.
    3. Target Module Flexibility: Automatically selects attention modules to target
       based on the model type (Phi-3 vs. Llama).
    4. completion-only training (optional): Uses DataCollatorForCompletionOnlyLM
       to compute loss only on the generated SQL, not the input instruction/schema.
    5. Weights & Biases integration: Tracks epochs, steps, training loss, validation
       loss, learning rate, VRAM usage, and saves model checkpoints directly.

Usage:
    # Train using the default config:
    python scripts/train_qlora.py \
        --config configs/default.yaml \
        --wandb_project text2sql-qlora

    # Train a sweep run (overriding hyperparams):
    python scripts/train_qlora.py \
        --config configs/sweep/r8_lr1e-4.yaml \
        --epochs 2 \
        --lr 1e-4
"""

from __future__ import annotations

import argparse
import os
import sys
import yaml
from pathlib import Path
from typing import Any, Dict, List

# Must be set BEFORE torch is imported so CUDA allocator picks it up at init time.
# Reduces memory fragmentation caused by many small alloc/free cycles during training.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

# ---------------------------------------------------------------------------
# DataCollatorForCompletionOnlyLM compatibility shim
# ---------------------------------------------------------------------------
# This class was removed from trl >= 0.13.  We try the known import locations
# first; if every one fails we supply our own minimal but correct implementation
# so the script runs on ANY trl version without requiring an upgrade.
# ---------------------------------------------------------------------------
_collator_imported = False
try:
    from trl import DataCollatorForCompletionOnlyLM
    _collator_imported = True
except ImportError:
    pass

if not _collator_imported:
    try:
        from trl.trainer import DataCollatorForCompletionOnlyLM
        _collator_imported = True
    except ImportError:
        pass

if not _collator_imported:
    from dataclasses import dataclass, field
    from typing import Optional, Union

    @dataclass
    class DataCollatorForCompletionOnlyLM:
        """Minimal self-contained replacement for trl's removed collator.

        Masks every token *before* (and including) the response template with
        ``ignore_index`` so the cross-entropy loss is only computed on the
        completion (the SQL query).

        Args:
            response_template: The string that separates prompt from completion,
                e.g. ``"SQL:"``.
            tokenizer: The HuggingFace tokenizer.
            ignore_index: Label value to ignore in loss computation (default -100).
        """

        response_template: Union[str, list]
        tokenizer: object
        ignore_index: int = -100

        def __post_init__(self) -> None:
            if isinstance(self.response_template, str):
                # Encode without special tokens so we get the raw sub-word ids
                self.response_token_ids: list[int] = self.tokenizer.encode(
                    self.response_template, add_special_tokens=False
                )
            else:
                self.response_token_ids = list(self.response_template)

        def __call__(self, features: list[dict]) -> dict:
            import torch

            # ----------------------------------------------------------------
            # In newer trl the dataset is pre-tokenised internally, so each
            # feature already has integer lists of potentially *different*
            # lengths.  We must pad to the longest sequence in the batch before
            # we can stack into a 2-D tensor.
            # ----------------------------------------------------------------
            pad_id = (
                self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else 0
            )
            max_len = max(len(f["input_ids"]) for f in features)

            input_ids_padded: list[list[int]] = []
            attention_mask_padded: list[list[int]] = []
            labels_padded: list[list[int]] = []

            for f in features:
                ids = list(f["input_ids"])
                # attention_mask may be absent in newer trl (only input_ids + labels
                # are stored after pre-tokenisation).  Generate it from the ids.
                if "attention_mask" in f:
                    mask = list(f["attention_mask"])
                else:
                    mask = [1] * len(ids)  # all real tokens before padding
                pad_len = max_len - len(ids)

                input_ids_padded.append(ids + [pad_id] * pad_len)
                attention_mask_padded.append(mask + [0] * pad_len)

                # Use pre-built labels if the trainer already created them
                # (new trl does this during "Building labels" dataset step).
                # Otherwise fall back to input_ids as the starting point.
                if "labels" in f:
                    lbls = list(f["labels"]) + [self.ignore_index] * pad_len
                else:
                    lbls = ids + [self.ignore_index] * pad_len

                labels_padded.append(lbls)

            input_ids = torch.tensor(input_ids_padded, dtype=torch.long)
            attention_mask = torch.tensor(attention_mask_padded, dtype=torch.long)
            labels = torch.tensor(labels_padded, dtype=torch.long)

            # ----------------------------------------------------------------
            # Apply completion-only masking: find the last occurrence of the
            # response template (e.g. "SQL:") and mask everything before it
            # with ignore_index so loss is only computed on the SQL output.
            # ----------------------------------------------------------------
            tmpl = self.response_token_ids
            tmpl_len = len(tmpl)

            for i in range(labels.size(0)):
                seq = input_ids[i].tolist()
                # Search from right so we hit the last "SQL:" occurrence
                found = False
                for j in range(len(seq) - tmpl_len, -1, -1):
                    if seq[j : j + tmpl_len] == tmpl:
                        labels[i, : j + tmpl_len] = self.ignore_index
                        found = True
                        break
                if not found:
                    # Template absent — mask entire sequence; no spurious loss
                    labels[i, :] = self.ignore_index

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }

# Import shared utilities
# Add project root to path if running directly
sys.path.append(str(Path(__file__).parent.parent))
from scripts.utils import load_config_with_overrides, format_prompt


# =============================================================================
# Helper: Find Target Modules
# =============================================================================

def get_target_modules(model_name: str) -> list[str]:
    """Determine the appropriate LoRA target modules based on model name.

    Phi-3-mini uses fused projection layers ('qkv_proj').
    Llama models use separate attention projections ('q_proj', 'k_proj', 'v_proj').

    Args:
        model_name: HuggingFace model path or identifier.

    Returns:
        List of target modules to apply LoRA to.
    """
    model_name_lower = model_name.lower()
    if "phi-3" in model_name_lower:
        # Phi-3 mini uses fused qkv projection
        return ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]
    elif "llama" in model_name_lower:
        # Llama-3/3.1 uses standard projection names
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    else:
        # Conservative default: project attention weight layers only
        return ["q_proj", "k_proj", "v_proj", "qkv_proj"]


# =============================================================================
# Dataset Preprocessing
# =============================================================================

def format_sft_dataset(
    example: dict[str, str],
    prompt_template: str | None = None,
) -> dict[str, str]:
    """Format a single dataset row for causal language modeling.

    Combines instructions, schema, and question into the prompt input,
    and appends the target SQL query (plus the EOS token) so the model learns
    when to stop generating.

    Args:
        example: Dataset example with 'instruction', 'schema', 'question', 'output'.
        prompt_template: Optional prompt template override.

    Returns:
        Dict containing the full text string.
    """
    prompt = format_prompt(
        schema=example["schema"],
        question=example["question"],
        template=prompt_template,
    )
    # The output SQL query
    target = example["output"]

    # Combine into a single text block. During training, the loss will
    # ideally only be calculated on the target part (using completion collator).
    # We append a trailing space and the query, then SFTTrainer will tokenize it.
    text = f"{prompt} {target}"

    return {"text": text}


# =============================================================================
# Training Execution
# =============================================================================

def run_training(config: dict[str, Any], cli_args: argparse.Namespace) -> None:
    """Execute the QLoRA training run using the loaded config.

    Args:
        config: Merged configuration dictionary.
        cli_args: Parsed command-line arguments (for logging run properties).
    """
    # Override config values with CLI args if specified
    if cli_args.epochs is not None:
        config["training"]["num_epochs"] = cli_args.epochs
    if cli_args.lr is not None:
        config["training"]["learning_rate"] = cli_args.lr
    if cli_args.lora_r is not None:
        config["lora"]["r"] = cli_args.lora_r
        config["lora"]["lora_alpha"] = 2 * cli_args.lora_r
    if cli_args.wandb_project is not None:
        config["wandb"]["project"] = cli_args.wandb_project
    if cli_args.run_name is not None:
        config["wandb"]["run_name"] = cli_args.run_name

    # Set up Weights & Biases environment variables before initializing wandb
    os.environ["WANDB_PROJECT"] = config["wandb"]["project"]
    if config["wandb"].get("entity"):
        os.environ["WANDB_ENTITY"] = config["wandb"]["entity"]

    # Create run name if not provided
    if not config["wandb"].get("run_name"):
        model_name = config["model"]["base_model"].split("/")[-1]
        r = config["lora"]["r"]
        lr = config["training"]["learning_rate"]
        config["wandb"]["run_name"] = f"{model_name}-r{r}-lr{lr:.0e}"

    os.environ["WANDB_RUN_GROUP"] = "qlora-sweeps"
    os.environ["WANDB_JOB_TYPE"] = "train"

    # Make output directory
    output_dir = Path(config["output"]["output_dir"]) / config["wandb"]["run_name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    config["training"]["output_dir"] = str(output_dir)

    print("=" * 70)
    print(f"Starting QLoRA training run: {config['wandb']['run_name']}")
    print(f"Base model:                  {config['model']['base_model']}")
    print(f"LoRA rank:                   {config['lora']['r']}")
    print(f"Learning rate:               {config['training']['learning_rate']}")
    print(f"Epochs:                      {config['training']['num_epochs']}")
    print(f"Checkpoints saving to:      {config['training']['output_dir']}")
    print("=" * 70)

    # --- Step 1: Load Tokenizer ---
    model_id = config["model"]["base_model"]
    print(f"\n[1/6] Loading tokenizer for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=False,
        padding_side="right",  # Standard for training (prevents attention mask issues)
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # --- Step 2: Load and Format Dataset ---
    print(f"\n[2/6] Loading datasets...")
    train_path = config["data"]["train_path"]
    val_path = config["data"]["val_path"]

    dataset = load_dataset(
        "json",
        data_files={"train": train_path, "validation": val_path},
    )

    prompt_template = config["data"].get("prompt_template")

    print("Formatting datasets for SFT...")
    # SFTDataset is mapped to have a single "text" field
    formatted_dataset = dataset.map(
        lambda ex: format_sft_dataset(ex, prompt_template),
        remove_columns=dataset["train"].column_names,
        desc="Formatting dataset",
    )

    print(f"  Train dataset size: {len(formatted_dataset['train'])} examples")
    print(f"  Val dataset size:   {len(formatted_dataset['validation'])} examples")
    print("\nSample formatted training input:")
    print("-" * 60)
    print(formatted_dataset["train"][0]["text"])
    print("-" * 60)

    # --- Step 3: Load Model in 4-Bit ---
    print(f"\n[3/6] Loading model {model_id} in 4-bit NF4...")
    # 4-bit compute dtype: honour the fp16/bf16 flag from config.
    # T4 GPUs have native fp16 tensor cores; bf16 is not hardware-accelerated.
    _use_bf16 = config["training"].get("bf16", False) and not config["training"].get("fp16", False)
    _compute_dtype = torch.bfloat16 if _use_bf16 else torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config["quantization"]["load_in_4bit"],
        bnb_4bit_quant_type=config["quantization"]["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=_compute_dtype,
        bnb_4bit_use_double_quant=config["quantization"]["bnb_4bit_use_double_quant"],
    )

    # Note: trust_remote_code=False leverages the native implementation.
    # device_map={"": 0} forces the entire model onto GPU 0.
    # Using "auto" on Kaggle's dual-T4 environment splits layers across
    # cuda:0 and cuda:1, which triggers a device-mismatch bug inside
    # trl's _chunked_cross_entropy_loss (valid mask on CPU, hidden on cuda:1).
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=False,
        attn_implementation="eager",  # Avoids flash-attention issues on T4 GPUs
    )

    # Prepare model for PEFT training (enables gradient checkpointing & inputs cast to active dtype)
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=config["training"]["gradient_checkpointing"],
    )

    # --- Step 4: Configure LoRA Adapters ---
    print(f"\n[4/6] Configuring LoRA...")
    target_modules = config["lora"].get("target_modules")
    if not target_modules:
        target_modules = get_target_modules(model_id)
        print(f"  Auto-selected target modules: {target_modules}")
    else:
        print(f"  Target modules from config: {target_modules}")

    lora_config = LoraConfig(
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["lora_alpha"],
        lora_dropout=config["lora"]["lora_dropout"],
        target_modules=target_modules,
        bias=config["lora"]["bias"],
        task_type=config["lora"]["task_type"],
    )

    model = get_peft_model(model, lora_config)
    print("\nTrainable parameters summary:")
    model.print_trainable_parameters()

    # --- Step 5: Setup Training Arguments ---
    print(f"\n[5/6] Setting up training arguments...")
    t_cfg = config["training"]

    # Compute total steps for warmup/logging calculation
    effective_batch_size = t_cfg["per_device_train_batch_size"] * t_cfg["gradient_accumulation_steps"]
    total_steps = (len(formatted_dataset["train"]) // effective_batch_size) * t_cfg["num_epochs"]

    # ---------------------------------------------------------------------------
    # trl API compatibility:
    #   trl < 0.10  → SFTTrainer accepts TrainingArguments + max_seq_length etc.
    #   trl >= 0.10 → SFTTrainer requires SFTConfig (subclass of TrainingArguments)
    #                 and no longer accepts max_seq_length / dataset_text_field /
    #                 tokenizer as direct constructor kwargs.
    # We detect which API is available and build the right objects accordingly.
    # ---------------------------------------------------------------------------
    _common_args = dict(
        output_dir=t_cfg["output_dir"],
        num_train_epochs=t_cfg["num_epochs"],
        per_device_train_batch_size=t_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=t_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=t_cfg["gradient_accumulation_steps"],
        learning_rate=t_cfg["learning_rate"],
        weight_decay=t_cfg["weight_decay"],
        warmup_steps=max(1, int(total_steps * t_cfg["warmup_ratio"])),
        lr_scheduler_type=t_cfg["lr_scheduler_type"],
        logging_steps=t_cfg["logging_steps"],
        eval_strategy="steps",
        eval_steps=t_cfg["eval_steps"],
        save_strategy="steps",
        save_steps=t_cfg["save_steps"],
        save_total_limit=t_cfg["save_total_limit"],
        fp16=t_cfg["fp16"] if torch.cuda.is_available() else False,
        bf16=t_cfg["bf16"] if torch.cuda.is_available() else False,
        gradient_checkpointing=t_cfg["gradient_checkpointing"],
        # use_reentrant=False uses the non-reentrant checkpoint implementation
        # which has lower peak memory during backward recomputation and is
        # the recommended default in PyTorch >= 2.1.
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=t_cfg["optim"],
        seed=t_cfg["seed"],
        report_to=["wandb"],
        run_name=config["wandb"]["run_name"],
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
    )

    # Try new-style SFTConfig first (trl >= 0.10).
    # Parameter names inside SFTConfig changed across versions, so we inspect
    # the signature at runtime rather than hardcoding version assumptions:
    #   max_seq_length  → trl 0.10 – 0.12
    #   max_length      → trl 0.13+
    #   dataset_text_field exists in some versions, absent in others
    try:
        import inspect
        from trl import SFTConfig

        _sft_sig = inspect.signature(SFTConfig.__init__).parameters

        _sft_extra: dict = {}
        if "max_seq_length" in _sft_sig:
            _sft_extra["max_seq_length"] = t_cfg["max_seq_length"]
        elif "max_length" in _sft_sig:
            _sft_extra["max_length"] = t_cfg["max_seq_length"]
        # dataset_text_field tells SFTConfig which column holds the text
        if "dataset_text_field" in _sft_sig:
            _sft_extra["dataset_text_field"] = "text"

        training_args = SFTConfig(**_common_args, **_sft_extra)
        _use_sft_config = True
        print(f"  Using SFTConfig (trl >= 0.10 API), extra kwargs: {list(_sft_extra)}")
    except ImportError:
        training_args = TrainingArguments(**_common_args)
        _use_sft_config = False
        print("  Using TrainingArguments (trl < 0.10 API)")

    # --- Step 6: Setup SFTTrainer ---
    print(f"\n[6/6] Initializing trainer...")

    # Explicitly initialise W&B before building the trainer so that
    # wandb.config.update() has an active run. (In new trl/transformers,
    # wandb.init() is only called lazily during trainer.train(), which is
    # too late for our config logging.)
    import wandb
    if not wandb.run:
        wandb.init(
            project=config["wandb"]["project"],
            name=config["wandb"]["run_name"],
            tags=config["wandb"].get("tags", []),
            group="qlora-sweeps",
            job_type="train",
        )
    wandb.config.update(config, allow_val_change=True)

    # Design decision: Compute loss only on the completion target (SQL)
    # The template starts the assistant response after "SQL:"
    response_template = "SQL:"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )

    # Build SFTTrainer kwargs — some args moved into SFTConfig in newer trl
    _trainer_kwargs: dict = dict(
        model=model,
        train_dataset=formatted_dataset["train"],
        eval_dataset=formatted_dataset["validation"],
        data_collator=collator,
        args=training_args,
    )
    if _use_sft_config:
        # New API: tokenizer → processing_class; max_seq_length/dataset_text_field in config
        _trainer_kwargs["processing_class"] = tokenizer
    else:
        # Old API: tokenizer and SFT-specific args passed directly to trainer
        _trainer_kwargs["tokenizer"] = tokenizer
        _trainer_kwargs["max_seq_length"] = t_cfg["max_seq_length"]
        _trainer_kwargs["dataset_text_field"] = "text"

    trainer = SFTTrainer(**_trainer_kwargs)

    print("\nStarting SFT Trainer...")
    trainer.train()

    # Save the final best model
    print(f"\nTraining completed. Saving best model checkpoint to: {t_cfg['output_dir']}")
    trainer.save_model(t_cfg["output_dir"])
    tokenizer.save_pretrained(t_cfg["output_dir"])
    wandb.finish()
    print("✓ Model and tokenizer saved successfully.")


# =============================================================================
# CLI Setup & Main
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="QLoRA training script with config merging and W&B integration.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to base/sweep configuration file (default: configs/default.yaml).",
    )
    parser.add_argument(
        "--epochs",
        type=float,
        default=None,
        help="Override training epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate.",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=None,
        help="Override LoRA rank.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="Override W&B project name.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Override W&B run name.",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for QLoRA training."""
    args = parse_args()

    # Resolve config paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    default_config_path = project_root / "configs" / "default.yaml"
    override_config_path = project_root / args.config if args.config != "configs/default.yaml" else None

    # Load and merge configurations
    try:
        config = load_config_with_overrides(
            default_path=default_config_path,
            override_path=override_config_path,
        )
    except FileNotFoundError as e:
        print(f"Error loading configs: {e}")
        sys.exit(1)

    # Execute training
    run_training(config, args)


if __name__ == "__main__":
    main()
