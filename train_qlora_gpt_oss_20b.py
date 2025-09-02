import os
import json
from dataclasses import dataclass
from typing import Dict, Iterable, List

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig


# ------------------------------
# Config
# ------------------------------
MODEL_ID = os.environ.get("MODEL_ID", "openai/gpt-oss-20b")
DATA_DIR = os.environ.get("DATA_DIR", os.path.join("jsonl_text"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "gpt-oss-20b-qlora-finetune")

# JSONL files must exist inside DATA_DIR
TRAIN_FILE = os.path.join(DATA_DIR, "train.jsonl")
VAL_FILE = os.path.join(DATA_DIR, "val.jsonl")


def assert_files():
    for p in [TRAIN_FILE, VAL_FILE]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing file: {p}")


def load_jsonl_as_hf_dataset(path: str):
    # Use datasets' json loader which supports JSON Lines
    return load_dataset("json", data_files=path, split="train")


@dataclass
class FormattingConfig:
    prompt_key: str = "prompt"
    completion_key: str = "completion"
    add_eos: bool = True


def build_text(example: Dict, tok, cfg: FormattingConfig) -> str:
    # Compact formatting for classification/QA: prompt + completion
    text = f"{example[cfg.prompt_key]}{example[cfg.completion_key]}"
    if cfg.add_eos and tok.eos_token:  # keep compact format for classification
        text += tok.eos_token
    return text


def prepare_dataset(ds, tok, cfg: FormattingConfig):
    def _map_fn(batch):
        prompts = batch[cfg.prompt_key]
        completions = batch[cfg.completion_key]
        texts = []
        for p, c in zip(prompts, completions):
            p = p or ""
            c = c or ""
            # Ensure a single space between prompt and label if missing
            if len(c) and not p.endswith(" ") and not c.startswith(" "):
                c = " " + c
            texts.append(p + c)
        if cfg.add_eos and tok.eos_token:
            texts = [t + tok.eos_token for t in texts]
        return {"text": texts}

    cols = ds.column_names
    if FormattingConfig.prompt_key not in cols or FormattingConfig.completion_key not in cols:
        # fall back to common variants
        if "instruction" in cols and "output" in cols:
            cfg.prompt_key, cfg.completion_key = "instruction", "output"
        else:
            raise ValueError(
                f"Dataset must contain 'prompt' and 'completion' keys. Found: {cols}"
            )
    ds = ds.map(_map_fn, batched=True, remove_columns=cols)
    return ds


def get_tokenizer(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def get_4bit_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def get_model(model_id: str):
    quant_config = get_4bit_config()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_config,
        device_map="auto",
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        use_cache=False,
    )
    return model


def attach_lora(base_model):
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules="all-linear",
    )
    lora_model = get_peft_model(base_model, peft_config)
    lora_model.print_trainable_parameters()
    return lora_model


def main():
    assert_files()

    tokenizer = get_tokenizer(MODEL_ID)

    # Load datasets
    train_ds = load_jsonl_as_hf_dataset(TRAIN_FILE)
    val_ds = load_jsonl_as_hf_dataset(VAL_FILE)

    # Prepare into single text field for SFT
    fmt_cfg = FormattingConfig()
    train_ds = prepare_dataset(train_ds, tokenizer, fmt_cfg)
    val_ds = prepare_dataset(val_ds, tokenizer, fmt_cfg)

    model = get_model(MODEL_ID)
    model = attach_lora(model)

    # Defaults tuned for ~48GB VRAM; adjust via env vars if needed
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=int(os.environ.get("BATCH_SIZE", 1)),
        per_device_eval_batch_size=int(os.environ.get("EVAL_BATCH_SIZE", 1)),
        gradient_accumulation_steps=int(os.environ.get("GRAD_ACCUM", 32)),
        learning_rate=float(os.environ.get("LR", 2e-4)),
        num_train_epochs=float(os.environ.get("EPOCHS", 1)),
        logging_steps=int(os.environ.get("LOG_STEPS", 10)),
        eval_strategy="steps",
        eval_steps=int(os.environ.get("EVAL_STEPS", 200)),
        save_steps=int(os.environ.get("SAVE_STEPS", 200)),
        save_total_limit=3,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        max_seq_length=int(os.environ.get("MAX_SEQ_LEN", 2048)),
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to=os.environ.get("REPORT_TO", "none"),
        optim=os.environ.get("OPTIM", "paged_adamw_8bit"),
        packing=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        dataset_text_field="text",
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)


if __name__ == "__main__":
    main()


