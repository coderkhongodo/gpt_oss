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
    def __init__(self, texts, labels, tokenizer, max_length=512):
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
    Tính toán các metrics cho evaluation
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    accuracy = accuracy_score(labels, predictions)
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def main():
    # Cấu hình
    MODEL_NAME = "vinai/phobert-base"
    DATA_DIR = "phobert_data"
    OUTPUT_DIR = "phobert-fake-news-detector"
    MAX_LENGTH = 512
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
        evaluation_strategy="steps",
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
    
    # Lưu kết quả
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
        "test_results": test_results,
        "timestamp": datetime.now().isoformat()
    }
    
    with open(f'{OUTPUT_DIR}/training_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # In kết quả
    logger.info("=== TRAINING COMPLETED ===")
    logger.info(f"Training loss: {train_result.training_loss:.4f}")
    logger.info(f"Test accuracy: {test_results['eval_accuracy']:.4f}")
    logger.info(f"Test F1: {test_results['eval_f1']:.4f}")
    logger.info(f"Test precision: {test_results['eval_precision']:.4f}")
    logger.info(f"Test recall: {test_results['eval_recall']:.4f}")
    logger.info(f"Model saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
