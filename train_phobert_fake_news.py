import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import json
from datetime import datetime
import logging

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FakeNewsDataset(Dataset):
    """
    Dataset class cho tác vụ phát hiện tin giả
    """
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_data(data_dir):
    """
    Load dữ liệu từ các file CSV
    """
    train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    val_df = pd.read_csv(os.path.join(data_dir, 'val.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    
    logger.info(f"Loaded data:")
    logger.info(f"  Train: {len(train_df)} samples")
    logger.info(f"  Val: {len(val_df)} samples") 
    logger.info(f"  Test: {len(test_df)} samples")
    
    return train_df, val_df, test_df

def compute_metrics(eval_pred):
    """
    Tính toán các metrics cho evaluation với thông tin chi tiết cho từng lớp
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    # Metrics tổng thể
    accuracy = accuracy_score(labels, predictions)
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted'
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, predictions, average='macro'
    )

    # Metrics cho từng lớp
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        labels, predictions, average=None
    )

    # Confusion matrix
    cm = confusion_matrix(labels, predictions)

    # Tạo dictionary kết quả chi tiết
    results = {
        'accuracy': accuracy,
        'f1_weighted': f1_weighted,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_macro': f1_macro,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        # Metrics cho lớp THẬT (0)
        'f1_that': f1_per_class[0] if len(f1_per_class) > 0 else 0.0,
        'precision_that': precision_per_class[0] if len(precision_per_class) > 0 else 0.0,
        'recall_that': recall_per_class[0] if len(recall_per_class) > 0 else 0.0,
        'support_that': int(support_per_class[0]) if len(support_per_class) > 0 else 0,
        # Metrics cho lớp GIẢ (1)
        'f1_gia': f1_per_class[1] if len(f1_per_class) > 1 else 0.0,
        'precision_gia': precision_per_class[1] if len(precision_per_class) > 1 else 0.0,
        'recall_gia': recall_per_class[1] if len(recall_per_class) > 1 else 0.0,
        'support_gia': int(support_per_class[1]) if len(support_per_class) > 1 else 0,
        # Confusion matrix elements
        'true_negatives': int(cm[0, 0]) if cm.shape == (2, 2) else 0,
        'false_positives': int(cm[0, 1]) if cm.shape == (2, 2) else 0,
        'false_negatives': int(cm[1, 0]) if cm.shape == (2, 2) else 0,
        'true_positives': int(cm[1, 1]) if cm.shape == (2, 2) else 0,
    }

    # Để tương thích với code cũ
    results['f1'] = f1_weighted
    results['precision'] = precision_weighted
    results['recall'] = recall_weighted

    return results

def main():
    # Cấu hình
    MODEL_NAME = "vinai/phobert-base"
    DATA_DIR = "phobert_data"
    OUTPUT_DIR = "phobert-fake-news-detector"
    MAX_LENGTH = 256  # PhoBERT maximum sequence length is 256, not 512
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 3
    WARMUP_STEPS = 500
    WEIGHT_DECAY = 0.01
    
    # Tạo thư mục output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load dữ liệu
    logger.info("Loading data...")
    train_df, val_df, test_df = load_data(DATA_DIR)
    
    # Load tokenizer và model
    logger.info(f"Loading tokenizer and model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,  # 0: THẬT, 1: GIẢ
        id2label={0: "THẬT", 1: "GIẢ"},
        label2id={"THẬT": 0, "GIẢ": 1}
    )
    
    # Tạo datasets
    logger.info("Creating datasets...")
    train_dataset = FakeNewsDataset(
        train_df['text'].tolist(),
        train_df['label'].tolist(),
        tokenizer,
        MAX_LENGTH
    )
    
    val_dataset = FakeNewsDataset(
        val_df['text'].tolist(),
        val_df['label'].tolist(),
        tokenizer,
        MAX_LENGTH
    )
    
    test_dataset = FakeNewsDataset(
        test_df['text'].tolist(),
        test_df['label'].tolist(),
        tokenizer,
        MAX_LENGTH
    )
    
    # Cấu hình training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        learning_rate=LEARNING_RATE,
        logging_dir=f'{OUTPUT_DIR}/logs',
        logging_steps=100,
        eval_strategy="steps",  # Đã thay đổi từ evaluation_strategy
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to=None,  # Tắt wandb/tensorboard
        dataloader_num_workers=0,  # Tránh lỗi multiprocessing trên Windows
    )
    
    # Tạo trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Training
    logger.info("Starting training...")
    train_result = trainer.train()
    
    # Lưu model
    logger.info("Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Evaluation trên test set
    logger.info("Evaluating on test set...")
    test_results = trainer.evaluate(test_dataset)
    
    # Lưu kết quả chi tiết
    results = {
        "model_name": MODEL_NAME,
        "training_args": {
            "max_length": MAX_LENGTH,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "num_epochs": NUM_EPOCHS,
            "warmup_steps": WARMUP_STEPS,
            "weight_decay": WEIGHT_DECAY
        },
        "train_results": {
            "train_loss": train_result.training_loss,
            "train_runtime": train_result.metrics['train_runtime'],
            "train_samples_per_second": train_result.metrics['train_samples_per_second']
        },
        "test_results": {
            "overall_metrics": {
                "accuracy": test_results['eval_accuracy'],
                "f1_weighted": test_results['eval_f1_weighted'],
                "f1_macro": test_results['eval_f1_macro'],
                "precision_weighted": test_results['eval_precision_weighted'],
                "precision_macro": test_results['eval_precision_macro'],
                "recall_weighted": test_results['eval_recall_weighted'],
                "recall_macro": test_results['eval_recall_macro']
            },
            "per_class_metrics": {
                "that_class": {
                    "f1": test_results['eval_f1_that'],
                    "precision": test_results['eval_precision_that'],
                    "recall": test_results['eval_recall_that'],
                    "support": test_results['eval_support_that']
                },
                "gia_class": {
                    "f1": test_results['eval_f1_gia'],
                    "precision": test_results['eval_precision_gia'],
                    "recall": test_results['eval_recall_gia'],
                    "support": test_results['eval_support_gia']
                }
            },
            "confusion_matrix": {
                "true_negatives": test_results['eval_true_negatives'],
                "false_positives": test_results['eval_false_positives'],
                "false_negatives": test_results['eval_false_negatives'],
                "true_positives": test_results['eval_true_positives']
            },
            "raw_results": test_results  # Giữ lại kết quả gốc để tham khảo
        },
        "timestamp": datetime.now().isoformat()
    }
    
    with open(f'{OUTPUT_DIR}/training_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # In kết quả chi tiết
    logger.info("=== TRAINING COMPLETED ===")
    logger.info(f"Training loss: {train_result.training_loss:.4f}")
    logger.info("")
    logger.info("=== OVERALL METRICS ===")
    logger.info(f"Test accuracy: {test_results['eval_accuracy']:.4f}")
    logger.info(f"Test F1 (weighted): {test_results['eval_f1_weighted']:.4f}")
    logger.info(f"Test F1 (macro): {test_results['eval_f1_macro']:.4f}")
    logger.info(f"Test precision (weighted): {test_results['eval_precision_weighted']:.4f}")
    logger.info(f"Test precision (macro): {test_results['eval_precision_macro']:.4f}")
    logger.info(f"Test recall (weighted): {test_results['eval_recall_weighted']:.4f}")
    logger.info(f"Test recall (macro): {test_results['eval_recall_macro']:.4f}")
    logger.info("")
    logger.info("=== PER-CLASS METRICS ===")
    logger.info("Class THẬT (0):")
    logger.info(f"  F1: {test_results['eval_f1_that']:.4f}")
    logger.info(f"  Precision: {test_results['eval_precision_that']:.4f}")
    logger.info(f"  Recall: {test_results['eval_recall_that']:.4f}")
    logger.info(f"  Support: {test_results['eval_support_that']}")
    logger.info("")
    logger.info("Class GIẢ (1):")
    logger.info(f"  F1: {test_results['eval_f1_gia']:.4f}")
    logger.info(f"  Precision: {test_results['eval_precision_gia']:.4f}")
    logger.info(f"  Recall: {test_results['eval_recall_gia']:.4f}")
    logger.info(f"  Support: {test_results['eval_support_gia']}")
    logger.info("")
    logger.info("=== CONFUSION MATRIX ===")
    logger.info("                 Predicted")
    logger.info("                THẬT   GIẢ")
    logger.info(f"Actual THẬT    {test_results['eval_true_negatives']:4d}  {test_results['eval_false_positives']:4d}")
    logger.info(f"       GIẢ     {test_results['eval_false_negatives']:4d}  {test_results['eval_true_positives']:4d}")
    logger.info("")
    logger.info(f"Model saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
