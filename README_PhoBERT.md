# PhoBERT Fake News Detection

Dự án fine-tune PhoBERT base cho tác vụ phát hiện tin giả trên mạng xã hội tiếng Việt.

## 🎯 Tổng quan

Dự án này sử dụng PhoBERT (vinai/phobert-base) để phân loại các bài đăng mạng xã hội tiếng Việt thành hai loại:
- **THẬT (0)**: Tin tức thật
- **GIẢ (1)**: Tin giả

## 📊 Dữ liệu

Dataset được chuyển đổi từ format instruction sang format đơn giản:
- **Train**: 14,519 mẫu (49.9% THẬT, 50.1% GIẢ)
- **Validation**: 483 mẫu (83.0% THẬT, 17.0% GIẢ)  
- **Test**: 486 mẫu (83.1% THẬT, 16.9% GIẢ)

## 🚀 Cài đặt và Sử dụng

### 1. Cài đặt dependencies

```bash
pip install -r requirements_phobert.txt
```

### 2. Chạy toàn bộ pipeline

```bash
# Chạy tất cả các bước
python run_phobert_pipeline.py

# Hoặc chạy từng bước riêng biệt
python run_phobert_pipeline.py --steps data train evaluate
```

### 3. Chạy từng bước riêng lẻ

#### Chuẩn bị dữ liệu
```bash
python prepare_data_for_phobert.py
```

#### Training model
```bash
python train_phobert_fake_news.py
```

#### Đánh giá model
```bash
python inference_phobert.py --mode evaluate
```

#### Demo tương tác
```bash
python inference_phobert.py --mode demo
```

## 📁 Cấu trúc Project

```
├── prepare_data_for_phobert.py     # Script chuyển đổi dữ liệu
├── train_phobert_fake_news.py      # Script training PhoBERT
├── inference_phobert.py            # Script inference và evaluation
├── run_phobert_pipeline.py         # Script chạy toàn bộ pipeline
├── requirements_phobert.txt        # Dependencies cho PhoBERT
├── README_PhoBERT.md              # Hướng dẫn này
├── phobert_data/                  # Dữ liệu đã chuyển đổi
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── phobert-fake-news-detector/    # Model đã training
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer_config.json
│   └── ...
└── evaluation_results/            # Kết quả đánh giá
    ├── detailed_predictions.csv
    └── evaluation_summary.json
```

## ⚙️ Cấu hình Training

- **Model**: vinai/phobert-base
- **Max Length**: 512 tokens
- **Batch Size**: 16
- **Learning Rate**: 2e-5
- **Epochs**: 3
- **Warmup Steps**: 500
- **Weight Decay**: 0.01

## 📈 Sử dụng Model

### Inference đơn lẻ

```python
from inference_phobert import PhoBERTFakeNewsDetector

# Khởi tạo detector
detector = PhoBERTFakeNewsDetector('phobert-fake-news-detector')

# Dự đoán
text = "Tin tức cần kiểm tra..."
result = detector.predict_single(text)

print(f"Dự đoán: {result['predicted_class']}")
print(f"Độ tin cậy: {result['confidence']:.4f}")
print(f"Xác suất THẬT: {result['probabilities']['THẬT']:.4f}")
print(f"Xác suất GIẢ: {result['probabilities']['GIẢ']:.4f}")
```

### Inference batch

```python
texts = ["Tin 1", "Tin 2", "Tin 3"]
results = detector.predict_batch(texts)

for i, result in enumerate(results):
    print(f"Text {i+1}: {result['predicted_class']} ({result['confidence']:.4f})")
```

## 🔧 Tùy chỉnh

### Thay đổi cấu hình training

Chỉnh sửa các tham số trong `train_phobert_fake_news.py`:

```python
# Cấu hình
MODEL_NAME = "vinai/phobert-base"  # Có thể thay bằng phobert-large
MAX_LENGTH = 512                   # Tăng nếu cần xử lý text dài hơn
BATCH_SIZE = 16                    # Giảm nếu thiếu GPU memory
LEARNING_RATE = 2e-5               # Điều chỉnh learning rate
NUM_EPOCHS = 3                     # Tăng số epochs nếu cần
```

### Sử dụng model khác

```python
# Thay đổi MODEL_NAME trong train_phobert_fake_news.py
MODEL_NAME = "vinai/phobert-large"  # Sử dụng PhoBERT large
```

## 📊 Kết quả Đánh giá

Sau khi training, kết quả đánh giá sẽ được lưu trong:
- `evaluation_results/evaluation_summary.json`: Tổng hợp metrics
- `evaluation_results/detailed_predictions.csv`: Chi tiết từng prediction

Các metrics được tính toán:
- **Accuracy**: Độ chính xác tổng thể
- **Precision**: Độ chính xác theo từng class
- **Recall**: Độ phủ theo từng class  
- **F1-score**: Điểm F1 weighted average

## 🚨 Lưu ý

1. **GPU Memory**: Training cần ít nhất 8GB GPU memory. Giảm batch_size nếu gặp lỗi OOM.

2. **Thời gian Training**: Với dataset ~15K mẫu, training mất khoảng 1-2 giờ trên GPU V100.

3. **Dữ liệu không cân bằng**: Val/Test set có nhiều mẫu THẬT hơn GIẢ, cần lưu ý khi đánh giá.

4. **Text dài**: Các text dài hơn 512 tokens sẽ bị cắt. Có thể tăng MAX_LENGTH nếu cần.

## 🤝 Đóng góp

1. Fork repository
2. Tạo feature branch
3. Commit changes
4. Push và tạo Pull Request

## 📄 License

MIT License - xem file LICENSE để biết chi tiết.

## 🙏 Tham khảo

- [PhoBERT](https://github.com/VinAIResearch/PhoBERT)
- [Transformers](https://huggingface.co/transformers/)
- [Dataset gốc](https://github.com/coderkhongodo/gpt_oss)
