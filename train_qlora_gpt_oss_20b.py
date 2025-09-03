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
from transformers import AutoConfig


# ------------------------------
# Config
# ------------------------------
MODEL_ID = os.environ.get("MODEL_ID", "openai/gpt-oss-20b")
DATA_DIR = os.environ.get("DATA_DIR", os.path.join("jsonl_text"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "gpt-oss-20b-qlora-finetune")

# JSONL files must exist inside DATA_DIR - sử dụng format instruction mới
TRAIN_FILE = os.path.join(DATA_DIR, "train_instruction.jsonl")
VAL_FILE = os.path.join(DATA_DIR, "val_instruction.jsonl")


def assert_files():
    for p in [TRAIN_FILE, VAL_FILE]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing file: {p}")


def load_jsonl_as_hf_dataset(path: str):
    # Use datasets' json loader which supports JSON Lines
    return load_dataset("json", data_files=path, split="train")


@dataclass
class FormattingConfig:
    instruction_key: str = "instruction"
    input_key: str = "input"
    output_key: str = "output"
    add_eos: bool = False  # Không cần thêm EOS vì output đã có </s>


def build_text(example: Dict, tok, cfg: FormattingConfig) -> str:
    # Format mới: instruction + input + output (đã có </s)
    instruction = example.get(cfg.instruction_key, "")
    input_text = example.get(cfg.input_key, "")
    output = example.get(cfg.output_key, "")
    
    # Tạo text theo format: instruction + input + output
    if input_text:
        text = f"{instruction}\n\n{input_text}\n\n{output}"
    else:
        text = f"{instruction}\n\n{output}"
    
    return text


def prepare_dataset(ds, tok, cfg: FormattingConfig):
    def _map_fn(batch):
        instructions = batch[cfg.instruction_key]
        inputs = batch[cfg.input_key]
        outputs = batch[cfg.output_key]
        texts = []
        
        for instruction, input_text, output in zip(instructions, inputs, outputs):
            instruction = instruction or ""
            input_text = input_text or ""
            output = output or ""
            
            # Tạo text theo format mới
            if input_text:
                text = f"{instruction}\n\n{input_text}\n\n{output}"
            else:
                text = f"{instruction}\n\n{output}"
            
            texts.append(text)
        
        return {"text": texts}

    cols = ds.column_names
    # Kiểm tra xem có đủ các trường cần thiết không
    required_cols = [cfg.instruction_key, cfg.output_key]
    missing_cols = [col for col in required_cols if col not in cols]
    
    if missing_cols:
        raise ValueError(
            f"Dataset missing required columns: {missing_cols}. Found: {cols}"
        )
    
    ds = ds.map(_map_fn, batched=True, remove_columns=cols)
    return ds


def get_tokenizer(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def format_text_field(example: Dict) -> str:
    # SFTTrainer (this TRL version) expects a formatting_func to return a string
    # We already mapped dataset to have a single 'text' field
    return example["text"]


def get_4bit_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def get_model(model_id: str):
    cfg = AutoConfig.from_pretrained(model_id)

    has_prequant = hasattr(cfg, "quantization_config") and cfg.quantization_config is not None

    if has_prequant:
        # Model already carries a quantization config (e.g., MXFP4). Load as-is.
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            attn_implementation="eager",
            use_cache=False,
            low_cpu_mem_usage=True,
        )
    else:
        # Fallback to 4-bit bnb quantization
        quant_config = get_4bit_config()
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quant_config,
            device_map="auto",
            attn_implementation="eager",
            use_cache=False,
            low_cpu_mem_usage=True,
        )

    return model


def attach_lora(base_model):
    peft_config = LoraConfig(
        r=32,  # Tăng từ 16 lên 32 để có capacity cao hơn
        lora_alpha=64,  # Tăng từ 32 lên 64
        lora_dropout=0.1,  # Tăng dropout để tránh overfitting
        bias="none",
        target_modules="all-linear",
        # Thêm các tham số mới
        inference_mode=False,
        task_type="CAUSAL_LM",
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

    # Hyperparameters tối ưu cho format instruction mới
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=int(os.environ.get("BATCH_SIZE", 1)),
        per_device_eval_batch_size=int(os.environ.get("EVAL_BATCH_SIZE", 1)),
        gradient_accumulation_steps=int(os.environ.get("GRAD_ACCUM", 16)),  # Giảm từ 32 xuống 16
        learning_rate=float(os.environ.get("LR", 5e-4)),  # Tăng từ 2e-4 lên 5e-4
        num_train_epochs=float(os.environ.get("EPOCHS", 5)),  # Tăng từ 1 lên 5 epochs
        logging_steps=int(os.environ.get("LOG_STEPS", 10)),
        save_steps=int(os.environ.get("SAVE_STEPS", 100)),  # Giảm để save thường xuyên hơn
        save_total_limit=3,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        warmup_ratio=0.1,  # Tăng warmup ratio
        lr_scheduler_type="cosine",
        report_to=os.environ.get("REPORT_TO", "none"),
        optim=os.environ.get("OPTIM", "paged_adamw_8bit"),
        packing=True,
        # Thêm các tham số mới để tối ưu
        dataloader_pin_memory=False,  # Tiết kiệm memory
        remove_unused_columns=False,  # Giữ columns để debug
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        formatting_func=format_text_field,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)


if __name__ == "__main__":
    main()


