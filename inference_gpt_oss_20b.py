import os
import json
from typing import List

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


MODEL_ID = os.environ.get("MODEL_ID", "openai/gpt-oss-20b")
ADAPTER_DIR = os.environ.get("ADAPTER_DIR", "gpt-oss-20b-qlora-finetune")
DATA_DIR = os.environ.get("DATA_DIR", os.path.join("jsonl_text"))
TEST_FILE = os.environ.get("TEST_FILE", os.path.join(DATA_DIR, "test.jsonl"))


def load_jsonl_as_list(path: str) -> List[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model in 4-bit and attach LoRA without merging (lower VRAM)
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        quantization_config=quant_config,
        use_cache=True,
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.eval()

    test_records = load_jsonl_as_list(TEST_FILE)
    to_show = int(os.environ.get("NUM_SAMPLES", 5))
    for ex in test_records[:to_show]:
        prompt = ex.get("prompt", "")
        inputs = tokenizer([prompt + tokenizer.eos_token], return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=int(os.environ.get("MAX_NEW_TOKENS", 32)),
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        text = tokenizer.batch_decode(out, skip_special_tokens=True)[0]
        print("==== Prompt ====")
        print(prompt)
        print("==== Model output ====")
        print(text[len(prompt):].strip())
        print()


if __name__ == "__main__":
    main()


