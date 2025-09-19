#!/usr/bin/env python3
"""
Script để chạy toàn bộ pipeline PhoBERT fine-tuning cho phát hiện tin giả
"""

import os
import sys
import subprocess
import argparse
import logging

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(command, description):
    """
    Chạy một command và log kết quả
    """
    logger.info(f"Starting: {description}")
    logger.info(f"Command: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        logger.info(f"✓ Completed: {description}")
        if result.stdout:
            logger.info(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ Failed: {description}")
        logger.error(f"Error: {e.stderr}")
        return False

def check_requirements():
    """
    Kiểm tra và cài đặt requirements
    """
    logger.info("Checking requirements...")
    
    if not os.path.exists("requirements_phobert.txt"):
        logger.error("requirements_phobert.txt not found!")
        return False
    
    # Cài đặt requirements
    if not run_command("pip install -r requirements_phobert.txt", "Installing requirements"):
        return False
    
    return True

def prepare_data():
    """
    Chuẩn bị dữ liệu
    """
    logger.info("Preparing data...")
    
    if not os.path.exists("prepare_data_for_phobert.py"):
        logger.error("prepare_data_for_phobert.py not found!")
        return False
    
    if not run_command("python prepare_data_for_phobert.py", "Converting data format"):
        return False
    
    # Kiểm tra dữ liệu đã được tạo
    data_files = ["phobert_data/train.csv", "phobert_data/val.csv", "phobert_data/test.csv"]
    for file_path in data_files:
        if not os.path.exists(file_path):
            logger.error(f"Data file {file_path} not found!")
            return False
    
    logger.info("✓ Data preparation completed")
    return True

def train_model():
    """
    Training model
    """
    logger.info("Starting model training...")
    
    if not os.path.exists("train_phobert_fake_news.py"):
        logger.error("train_phobert_fake_news.py not found!")
        return False
    
    if not run_command("python train_phobert_fake_news.py", "Training PhoBERT model"):
        return False
    
    # Kiểm tra model đã được lưu
    if not os.path.exists("phobert-fake-news-detector"):
        logger.error("Trained model directory not found!")
        return False
    
    logger.info("✓ Model training completed")
    return True

def evaluate_model():
    """
    Đánh giá model
    """
    logger.info("Evaluating model...")

    if not os.path.exists("inference_phobert.py"):
        logger.error("inference_phobert.py not found!")
        return False

    if not run_command("python inference_phobert.py --mode evaluate", "Evaluating model"):
        return False

    logger.info("✓ Model evaluation completed")
    return True

def create_evaluation_report():
    """
    Tạo báo cáo đánh giá chi tiết với biểu đồ
    """
    logger.info("Creating detailed evaluation report...")

    if not os.path.exists("create_evaluation_report.py"):
        logger.error("create_evaluation_report.py not found!")
        return False

    if not os.path.exists("evaluation_results/evaluation_summary.json"):
        logger.error("Evaluation results not found! Please run evaluation first.")
        return False

    if not run_command("python create_evaluation_report.py", "Creating evaluation report"):
        return False

    logger.info("✓ Detailed evaluation report created")
    return True

def main():
    parser = argparse.ArgumentParser(description='PhoBERT Fake News Detection Pipeline')
    parser.add_argument('--steps', type=str, nargs='+',
                       choices=['requirements', 'data', 'train', 'evaluate', 'report', 'all'],
                       default=['all'],
                       help='Steps to run (default: all)')
    parser.add_argument('--skip-requirements', action='store_true',
                       help='Skip requirements installation')
    
    args = parser.parse_args()
    
    steps = args.steps
    if 'all' in steps:
        steps = ['requirements', 'data', 'train', 'evaluate', 'report']
    
    if args.skip_requirements and 'requirements' in steps:
        steps.remove('requirements')
    
    logger.info("=== PhoBERT Fake News Detection Pipeline ===")
    logger.info(f"Steps to run: {steps}")
    
    success = True
    
    # Chạy từng bước
    if 'requirements' in steps:
        if not check_requirements():
            success = False
    
    if success and 'data' in steps:
        if not prepare_data():
            success = False
    
    if success and 'train' in steps:
        if not train_model():
            success = False
    
    if success and 'evaluate' in steps:
        if not evaluate_model():
            success = False

    if success and 'report' in steps:
        if not create_evaluation_report():
            success = False

    if success:
        logger.info("🎉 Pipeline completed successfully!")
        logger.info("\nNext steps:")
        logger.info("1. Check evaluation results in 'evaluation_results/' directory")
        logger.info("2. Open 'evaluation_results/evaluation_report.html' for detailed analysis")
        logger.info("3. Run demo: python inference_phobert.py --mode demo")
        logger.info("4. Use the trained model in 'phobert-fake-news-detector/' directory")
    else:
        logger.error("❌ Pipeline failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
