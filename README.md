# GPT-OSS 20B Vietnamese Social Media Fake News Detection

A fine-tuned GPT-OSS 20B model for Vietnamese social media fake news detection using QLoRA (Quantized Low-Rank Adaptation) technique.

## 🎯 Project Overview

This project fine-tunes the `openai/gpt-oss-20b` model on Vietnamese social media posts to classify them as **THẬT (Real - 0)** or **GIẢ (Fake - 1)**. The model achieves **87.2% accuracy** on the test dataset with excellent precision and recall scores.

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | 87.2% |
| **Precision** | 89.5% |
| **Recall** | 87.2% |
| **F1-Score** | 88.0% |
| **Success Rate** | 87.0% |

### Test Dataset Statistics
- **Total Examples**: 486
- **Correct Predictions**: 423
- **Error Predictions**: 1

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/coderkhongodo/gpt_oss.git
cd gpt_oss

# Install dependencies
pip install -r requirements.txt
```

### 2. Model Usage

#### Using Hugging Face Model Hub

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# Load base model and tokenizer
model_name = "openai/gpt-oss-20b"
adapter_name = "PhaaNe/gpt_oss"

tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load fine-tuned adapter
model = PeftModel.from_pretrained(base_model, adapter_name)

# Example inference
def classify_news(text):
    prompt = f"Hãy phân loại bài đăng mạng xã hội sau đây là THẬT (0) hay GIẢ (1): {text}"
    
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            temperature=0.1,
            do_sample=True
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split()[-1]  # Extract prediction

# Test example
news_text = "Dự báo thời tiết hôm nay: Nắng nóng gia tăng ở Bắc Bộ và Trung Bộ"
result = classify_news(news_text)
print(f"Prediction: {result}")  # Should output "0" for real news
```

#### Using Local Model

```bash
# Run inference script
python inference_gpt_oss_20b.py
```

## 🛠️ Training from Scratch

### Prerequisites

- **Hardware**: GPU with at least 24GB VRAM (recommended: 48GB+)
- **Software**: Python 3.8+, CUDA 11.8+

### Dataset Format

Your dataset should be in JSONL format with the following structure:

```json
{"instruction": "Hãy phân loại bài đăng mạng xã hội sau đây là THẬT (0) hay GIẢ (1):", "input": "Your news text here", "output": "0"}
```

Place your files in the `jsonl_text/` directory:
- `train_instruction.jsonl` - Training data
- `val_instruction.jsonl` - Validation data
- `test_instruction.jsonl` - Test data (optional)

### Training Configuration

#### Environment Variables

```bash
# Model and Data
export MODEL_ID="openai/gpt-oss-20b"
export DATA_DIR="jsonl_text"
export OUTPUT_DIR="gpt-oss-20b-qlora-finetune"

# Training Parameters
export BATCH_SIZE="1"
export EVAL_BATCH_SIZE="1"
export GRAD_ACCUM="32"
export LR="2e-4"
export EPOCHS="1"
export MAX_SEQ_LEN="2048"
export EVAL_STEPS="200"
export SAVE_STEPS="200"
export OPTIM="paged_adamw_8bit"
```

#### Start Training

```bash
# For 48GB VRAM setup
python train_qlora_gpt_oss_20b.py
```

### Training Features

- **QLoRA 4-bit Quantization**: Reduces memory usage by ~75%
- **Paged AdamW 8-bit**: Optimized memory management
- **Gradient Accumulation**: Simulates larger batch sizes
- **LoRA Adapters**: Efficient fine-tuning with minimal parameters
- **Automatic Mixed Precision**: Faster training with reduced memory

## 📁 Project Structure

```
gpt_oss/
├── train_qlora_gpt_oss_20b.py      # Training script
├── inference_gpt_oss_20b.py        # Inference script
├── evaluate_model.py               # Model evaluation
├── convert_data_format.py          # Data preprocessing
├── requirements.txt                # Dependencies
├── README.md                       # This file
├── .gitignore                      # Git ignore rules
├── jsonl_text/                     # Dataset directory
│   ├── train_instruction.jsonl
│   ├── val_instruction.jsonl
│   └── test_instruction.jsonl
└── gpt-oss-20b-qlora-finetune-v2/  # Trained model (excluded from git)
    ├── adapter_config.json
    ├── adapter_model.safetensors
    ├── tokenizer_config.json
    └── ...
```

## 🔧 Configuration Details

### QLoRA Configuration
- **Rank**: 64
- **Alpha**: 16
- **Dropout**: 0.1
- **Target Modules**: All linear layers
- **Quantization**: 4-bit (NF4)

### Training Parameters
- **Learning Rate**: 2e-4
- **Batch Size**: 1 (with gradient accumulation 32)
- **Max Sequence Length**: 2048
- **Epochs**: 1
- **Optimizer**: Paged AdamW 8-bit
- **Scheduler**: Cosine with warmup

## 📈 Evaluation

Run the evaluation script to test model performance:

```bash
python evaluate_model.py
```

This will generate:
- `my_results.csv`: Detailed predictions for each test sample
- `evaluation_summary.json`: Overall performance metrics

## 🌐 Model Availability

- **Hugging Face Hub**: [PhaaNe/gpt_oss](https://huggingface.co/PhaaNe/gpt_oss)
- **GitHub Repository**: [coderkhongodo/gpt_oss](https://github.com/coderkhongodo/gpt_oss)

## 📋 Requirements

```
torch>=2.0.0
transformers>=4.35.0
peft>=0.6.0
trl>=0.7.0
datasets>=2.14.0
accelerate>=0.24.0
bitsandbytes>=0.41.0
scikit-learn>=1.3.0
pandas>=2.0.0
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for the GPT-OSS 20B base model
- Hugging Face for the Transformers library
- QLoRA paper authors for the efficient fine-tuning technique
- Vietnamese NLP community for dataset contributions

## 📞 Contact

For questions or support, please open an issue on GitHub or contact the maintainers.

---

**Note**: This model is specifically trained for Vietnamese social media content and may not perform well on other languages or domains. Always validate predictions with human judgment for critical applications.