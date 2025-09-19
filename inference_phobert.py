import os
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import json
import argparse
from datetime import datetime
import logging

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PhoBERTFakeNewsDetector:
    """
    Class để thực hiện inference với PhoBERT đã fine-tune
    """
    def __init__(self, model_path, max_length=256):
        self.model_path = model_path
        self.max_length = max_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Loading model from: {model_path}")
        logger.info(f"Using device: {self.device}")
        
        # Load tokenizer và model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Label mapping
        self.id2label = {0: "THẬT", 1: "GIẢ"}
        self.label2id = {"THẬT": 0, "GIẢ": 1}
    
    def predict_single(self, text):
        """
        Dự đoán cho một văn bản đơn lẻ
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1).item()
            confidence = predictions[0][predicted_class].item()
        
        return {
            'predicted_label': predicted_class,
            'predicted_class': self.id2label[predicted_class],
            'confidence': confidence,
            'probabilities': {
                'THẬT': predictions[0][0].item(),
                'GIẢ': predictions[0][1].item()
            }
        }
    
    def predict_batch(self, texts, batch_size=32):
        """
        Dự đoán cho một batch văn bản
        """
        predictions = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_classes = torch.argmax(batch_predictions, dim=-1)
            
            # Lưu kết quả
            for j, pred_class in enumerate(predicted_classes):
                pred_class = pred_class.item()
                confidence = batch_predictions[j][pred_class].item()
                
                predictions.append({
                    'predicted_label': pred_class,
                    'predicted_class': self.id2label[pred_class],
                    'confidence': confidence,
                    'probabilities': {
                        'THẬT': batch_predictions[j][0].item(),
                        'GIẢ': batch_predictions[j][1].item()
                    }
                })
        
        return predictions

def evaluate_model(model_path, test_data_path, output_dir="evaluation_results"):
    """
    Đánh giá model trên test set
    """
    # Tạo thư mục output
    os.makedirs(output_dir, exist_ok=True)
    
    # Load test data
    logger.info(f"Loading test data from: {test_data_path}")
    test_df = pd.read_csv(test_data_path)
    
    # Khởi tạo detector
    detector = PhoBERTFakeNewsDetector(model_path)
    
    # Thực hiện prediction
    logger.info("Making predictions...")
    predictions = detector.predict_batch(test_df['text'].tolist())
    
    # Lấy predicted labels
    predicted_labels = [pred['predicted_label'] for pred in predictions]
    true_labels = test_df['label'].tolist()
    
    # Tính toán metrics tổng thể
    accuracy = accuracy_score(true_labels, predicted_labels)

    # Metrics weighted và macro
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        true_labels, predicted_labels, average='weighted'
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        true_labels, predicted_labels, average='macro'
    )

    # Metrics cho từng lớp
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        true_labels, predicted_labels, average=None
    )

    # Confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)

    # Classification report chi tiết
    class_report = classification_report(
        true_labels, predicted_labels,
        target_names=['THẬT', 'GIẢ'],
        output_dict=True
    )
    
    # Tạo detailed results
    detailed_results = []
    for i, (_, row) in enumerate(test_df.iterrows()):
        result = {
            'text': row['text'],
            'true_label': row['label'],
            'true_class': 'THẬT' if row['label'] == 0 else 'GIẢ',
            'predicted_label': predictions[i]['predicted_label'],
            'predicted_class': predictions[i]['predicted_class'],
            'confidence': predictions[i]['confidence'],
            'correct': row['label'] == predictions[i]['predicted_label'],
            'prob_that': predictions[i]['probabilities']['THẬT'],
            'prob_gia': predictions[i]['probabilities']['GIẢ']
        }
        detailed_results.append(result)
    
    # Lưu detailed results
    detailed_df = pd.DataFrame(detailed_results)
    detailed_df.to_csv(f'{output_dir}/detailed_predictions.csv', index=False, encoding='utf-8')
    
    # Tổng hợp kết quả chi tiết
    evaluation_results = {
        'model_path': model_path,
        'test_data_path': test_data_path,
        'total_samples': len(test_df),
        'correct_predictions': sum(1 for r in detailed_results if r['correct']),
        'overall_metrics': {
            'accuracy': accuracy,
            'f1_weighted': f1_weighted,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'precision_macro': precision_macro,
            'recall_weighted': recall_weighted,
            'recall_macro': recall_macro
        },
        'per_class_metrics': {
            'that_class': {
                'f1': f1_per_class[0] if len(f1_per_class) > 0 else 0.0,
                'precision': precision_per_class[0] if len(precision_per_class) > 0 else 0.0,
                'recall': recall_per_class[0] if len(recall_per_class) > 0 else 0.0,
                'support': int(support_per_class[0]) if len(support_per_class) > 0 else 0
            },
            'gia_class': {
                'f1': f1_per_class[1] if len(f1_per_class) > 1 else 0.0,
                'precision': precision_per_class[1] if len(precision_per_class) > 1 else 0.0,
                'recall': recall_per_class[1] if len(recall_per_class) > 1 else 0.0,
                'support': int(support_per_class[1]) if len(support_per_class) > 1 else 0
            }
        },
        'confusion_matrix': {
            'matrix': cm.tolist(),
            'true_negatives': int(cm[0, 0]) if cm.shape == (2, 2) else 0,
            'false_positives': int(cm[0, 1]) if cm.shape == (2, 2) else 0,
            'false_negatives': int(cm[1, 0]) if cm.shape == (2, 2) else 0,
            'true_positives': int(cm[1, 1]) if cm.shape == (2, 2) else 0
        },
        'classification_report': class_report,
        'timestamp': datetime.now().isoformat()
    }
    
    # Lưu kết quả tổng hợp
    with open(f'{output_dir}/evaluation_summary.json', 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
    
    # In kết quả chi tiết
    logger.info("=== EVALUATION RESULTS ===")
    logger.info(f"Total samples: {len(test_df)}")
    logger.info(f"Correct predictions: {evaluation_results['correct_predictions']}")
    logger.info("")
    logger.info("=== OVERALL METRICS ===")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"F1 (weighted): {f1_weighted:.4f}")
    logger.info(f"F1 (macro): {f1_macro:.4f}")
    logger.info(f"Precision (weighted): {precision_weighted:.4f}")
    logger.info(f"Precision (macro): {precision_macro:.4f}")
    logger.info(f"Recall (weighted): {recall_weighted:.4f}")
    logger.info(f"Recall (macro): {recall_macro:.4f}")
    logger.info("")
    logger.info("=== PER-CLASS METRICS ===")
    logger.info("Class THẬT (0):")
    logger.info(f"  F1: {evaluation_results['per_class_metrics']['that_class']['f1']:.4f}")
    logger.info(f"  Precision: {evaluation_results['per_class_metrics']['that_class']['precision']:.4f}")
    logger.info(f"  Recall: {evaluation_results['per_class_metrics']['that_class']['recall']:.4f}")
    logger.info(f"  Support: {evaluation_results['per_class_metrics']['that_class']['support']}")
    logger.info("")
    logger.info("Class GIẢ (1):")
    logger.info(f"  F1: {evaluation_results['per_class_metrics']['gia_class']['f1']:.4f}")
    logger.info(f"  Precision: {evaluation_results['per_class_metrics']['gia_class']['precision']:.4f}")
    logger.info(f"  Recall: {evaluation_results['per_class_metrics']['gia_class']['recall']:.4f}")
    logger.info(f"  Support: {evaluation_results['per_class_metrics']['gia_class']['support']}")
    logger.info("")
    logger.info("=== CONFUSION MATRIX ===")
    logger.info("                 Predicted")
    logger.info("                THẬT   GIẢ")
    logger.info(f"Actual THẬT    {evaluation_results['confusion_matrix']['true_negatives']:4d}  {evaluation_results['confusion_matrix']['false_positives']:4d}")
    logger.info(f"       GIẢ     {evaluation_results['confusion_matrix']['false_negatives']:4d}  {evaluation_results['confusion_matrix']['true_positives']:4d}")
    logger.info("")
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"  - Detailed predictions: {output_dir}/detailed_predictions.csv")
    logger.info(f"  - Evaluation summary: {output_dir}/evaluation_summary.json")
    
    return evaluation_results

def interactive_demo(model_path):
    """
    Demo tương tác để test model
    """
    detector = PhoBERTFakeNewsDetector(model_path)
    
    print("=== PhoBERT Fake News Detector Demo ===")
    print("Nhập văn bản để phân loại (gõ 'quit' để thoát):")
    
    while True:
        text = input("\nNhập văn bản: ").strip()
        
        if text.lower() == 'quit':
            break
        
        if not text:
            continue
        
        result = detector.predict_single(text)
        
        print(f"\nKết quả:")
        print(f"  Dự đoán: {result['predicted_class']}")
        print(f"  Độ tin cậy: {result['confidence']:.4f}")
        print(f"  Xác suất THẬT: {result['probabilities']['THẬT']:.4f}")
        print(f"  Xác suất GIẢ: {result['probabilities']['GIẢ']:.4f}")

def main():
    parser = argparse.ArgumentParser(description='PhoBERT Fake News Detection Inference')
    parser.add_argument('--model_path', type=str, default='phobert-fake-news-detector',
                       help='Path to trained model')
    parser.add_argument('--mode', type=str, choices=['evaluate', 'demo'], default='evaluate',
                       help='Mode: evaluate or demo')
    parser.add_argument('--test_data', type=str, default='phobert_data/test.csv',
                       help='Path to test data (for evaluate mode)')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Output directory for evaluation results')
    
    args = parser.parse_args()
    
    if args.mode == 'evaluate':
        evaluate_model(args.model_path, args.test_data, args.output_dir)
    elif args.mode == 'demo':
        interactive_demo(args.model_path)

if __name__ == "__main__":
    main()
