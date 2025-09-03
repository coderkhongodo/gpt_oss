## Fine-tune GPT-OSS 20B on local JSONL with QLoRA

This project provides a minimal QLoRA pipeline to fine-tune `openai/gpt-oss-20b` on your local JSONL dataset located at `jsonl_text/`.

### Dataset format
Each line in `train.jsonl`, `val.jsonl`, `test.jsonl` should be a JSON object with keys:
- `prompt`: input text
- `completion`: expected target text (for classification could be " 0" or " 1")

The trainer concatenates them as `prompt + completion + eos`.

### Requirements
Install Python packages:

```bash
pip install -r requirements.txt
```

Hardware: QLoRA allows training on a single high-memory GPU (e.g., 80GB). Smaller GPUs might work with smaller batch sizes and gradient accumulation.

### Train (QLoRA)

Recommended env for ~48GB VRAM (QLoRA 4-bit, paged AdamW 8-bit, packing):

```bash
# Windows PowerShell examples
$env:DATA_DIR = "/root/gpt_oss/jsonl_text"
$env:OUTPUT_DIR = "gpt-oss-20b-qlora-finetune"
$env:BATCH_SIZE = "1"
$env:EVAL_BATCH_SIZE = "1"
$env:GRAD_ACCUM = "32"
$env:LR = "2e-4"
$env:EPOCHS = "1"
$env:MAX_SEQ_LEN = "2048"
$env:EVAL_STEPS = "200"
$env:SAVE_STEPS = "200"
$env:OPTIM = "paged_adamw_8bit"
python train_qlora_gpt_oss_20b.py
```

Environment variables (optional, defaults shown):
- `MODEL_ID` (default: `openai/gpt-oss-20b`)
- `DATA_DIR` (default: `jsonl_text`)
- `OUTPUT_DIR` (default: `gpt-oss-20b-qlora-finetune`)
- `BATCH_SIZE` (default: 1)
- `EVAL_BATCH_SIZE` (default: 1)
- `GRAD_ACCUM` (default: 32)
- `LR` (default: 2e-4)
- `EPOCHS` (default: 1)
- `MAX_SEQ_LEN` (default: 2048)
- `EVAL_STEPS`/`SAVE_STEPS` (default: 200)
- `OPTIM` (default: `paged_adamw_8bit`)

### Inference (4-bit + adapters, no merge)

```bash
python inference_gpt_oss_20b.py
```

Environment variables (optional):
- `MODEL_ID`
- `ADAPTER_DIR` (directory produced by training)
- `TEST_FILE` (default: `jsonl_text/test.jsonl`)
- `NUM_SAMPLES` (how many examples to print)
- `MAX_NEW_TOKENS`

### Notes
- This repo uses a compact formatting suitable for short classification/QA tasks.
- For conversational Harmony-style prompts, extend the formatting in `train_qlora_gpt_oss_20b.py` to build structured text.

