import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import argparse
from datetime import datetime

# Thiết lập style cho plots
plt.style.use('default')
sns.set_palette("husl")

def load_evaluation_results(results_path):
    """
    Load kết quả đánh giá từ file JSON
    """
    with open(results_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_confusion_matrix_plot(cm_data, output_dir):
    """
    Tạo biểu đồ confusion matrix
    """
    # Tạo confusion matrix từ dữ liệu
    cm = np.array([
        [cm_data['true_negatives'], cm_data['false_positives']],
        [cm_data['false_negatives'], cm_data['true_positives']]
    ])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['THẬT', 'GIẢ'], 
                yticklabels=['THẬT', 'GIẢ'])
    plt.title('Confusion Matrix - PhoBERT Fake News Detection')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_metrics_comparison_plot(results, output_dir):
    """
    Tạo biểu đồ so sánh metrics giữa các lớp
    """
    # Chuẩn bị dữ liệu
    classes = ['THẬT', 'GIẢ']
    metrics = ['f1', 'precision', 'recall']
    
    that_metrics = [
        results['per_class_metrics']['that_class']['f1'],
        results['per_class_metrics']['that_class']['precision'],
        results['per_class_metrics']['that_class']['recall']
    ]
    
    gia_metrics = [
        results['per_class_metrics']['gia_class']['f1'],
        results['per_class_metrics']['gia_class']['precision'],
        results['per_class_metrics']['gia_class']['recall']
    ]
    
    # Tạo biểu đồ
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, that_metrics, width, label='THẬT', alpha=0.8)
    bars2 = ax.bar(x + width/2, gia_metrics, width, label='GIẢ', alpha=0.8)
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Performance Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylim(0, 1.0)
    
    # Thêm giá trị trên các cột
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom')
    
    autolabel(bars1)
    autolabel(bars2)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/per_class_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_overall_metrics_plot(results, output_dir):
    """
    Tạo biểu đồ tổng quan các metrics
    """
    metrics = ['Accuracy', 'F1 (Weighted)', 'F1 (Macro)', 
               'Precision (Weighted)', 'Precision (Macro)',
               'Recall (Weighted)', 'Recall (Macro)']
    
    values = [
        results['overall_metrics']['accuracy'],
        results['overall_metrics']['f1_weighted'],
        results['overall_metrics']['f1_macro'],
        results['overall_metrics']['precision_weighted'],
        results['overall_metrics']['precision_macro'],
        results['overall_metrics']['recall_weighted'],
        results['overall_metrics']['recall_macro']
    ]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(metrics, values, alpha=0.8, color='skyblue')
    plt.title('Overall Performance Metrics')
    plt.ylabel('Score')
    plt.ylim(0, 1.0)
    plt.xticks(rotation=45, ha='right')
    
    # Thêm giá trị trên các cột
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/overall_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_support_distribution_plot(results, output_dir):
    """
    Tạo biểu đồ phân bố số lượng mẫu theo lớp
    """
    classes = ['THẬT', 'GIẢ']
    supports = [
        results['per_class_metrics']['that_class']['support'],
        results['per_class_metrics']['gia_class']['support']
    ]
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(classes, supports, alpha=0.8, color=['lightgreen', 'lightcoral'])
    plt.title('Test Set Distribution by Class')
    plt.ylabel('Number of Samples')
    
    # Thêm giá trị và phần trăm trên các cột
    total = sum(supports)
    for bar, support in zip(bars, supports):
        percentage = (support / total) * 100
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{support}\n({percentage:.1f}%)', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/class_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_html_report(results, output_dir):
    """
    Tạo báo cáo HTML tổng hợp
    """
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>PhoBERT Fake News Detection - Evaluation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ text-align: center; color: #333; }}
            .section {{ margin: 30px 0; }}
            .metrics-table {{ border-collapse: collapse; width: 100%; }}
            .metrics-table th, .metrics-table td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
            .metrics-table th {{ background-color: #f2f2f2; }}
            .plot {{ text-align: center; margin: 20px 0; }}
            .plot img {{ max-width: 100%; height: auto; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>PhoBERT Fake News Detection</h1>
            <h2>Evaluation Report</h2>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="section">
            <h3>Model Information</h3>
            <p><strong>Model Path:</strong> {results['model_path']}</p>
            <p><strong>Test Data:</strong> {results['test_data_path']}</p>
            <p><strong>Total Samples:</strong> {results['total_samples']}</p>
            <p><strong>Correct Predictions:</strong> {results['correct_predictions']}</p>
        </div>
        
        <div class="section">
            <h3>Overall Performance</h3>
            <div class="plot">
                <img src="overall_metrics.png" alt="Overall Metrics">
            </div>
            <table class="metrics-table">
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr><td>Accuracy</td><td>{results['overall_metrics']['accuracy']:.4f}</td></tr>
                <tr><td>F1 (Weighted)</td><td>{results['overall_metrics']['f1_weighted']:.4f}</td></tr>
                <tr><td>F1 (Macro)</td><td>{results['overall_metrics']['f1_macro']:.4f}</td></tr>
                <tr><td>Precision (Weighted)</td><td>{results['overall_metrics']['precision_weighted']:.4f}</td></tr>
                <tr><td>Precision (Macro)</td><td>{results['overall_metrics']['precision_macro']:.4f}</td></tr>
                <tr><td>Recall (Weighted)</td><td>{results['overall_metrics']['recall_weighted']:.4f}</td></tr>
                <tr><td>Recall (Macro)</td><td>{results['overall_metrics']['recall_macro']:.4f}</td></tr>
            </table>
        </div>
        
        <div class="section">
            <h3>Per-Class Performance</h3>
            <div class="plot">
                <img src="per_class_metrics.png" alt="Per-Class Metrics">
            </div>
            <table class="metrics-table">
                <tr>
                    <th>Class</th>
                    <th>F1</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>Support</th>
                </tr>
                <tr>
                    <td>THẬT</td>
                    <td>{results['per_class_metrics']['that_class']['f1']:.4f}</td>
                    <td>{results['per_class_metrics']['that_class']['precision']:.4f}</td>
                    <td>{results['per_class_metrics']['that_class']['recall']:.4f}</td>
                    <td>{results['per_class_metrics']['that_class']['support']}</td>
                </tr>
                <tr>
                    <td>GIẢ</td>
                    <td>{results['per_class_metrics']['gia_class']['f1']:.4f}</td>
                    <td>{results['per_class_metrics']['gia_class']['precision']:.4f}</td>
                    <td>{results['per_class_metrics']['gia_class']['recall']:.4f}</td>
                    <td>{results['per_class_metrics']['gia_class']['support']}</td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <h3>Confusion Matrix</h3>
            <div class="plot">
                <img src="confusion_matrix.png" alt="Confusion Matrix">
            </div>
        </div>
        
        <div class="section">
            <h3>Class Distribution</h3>
            <div class="plot">
                <img src="class_distribution.png" alt="Class Distribution">
            </div>
        </div>
    </body>
    </html>
    """
    
    with open(f'{output_dir}/evaluation_report.html', 'w', encoding='utf-8') as f:
        f.write(html_content)

def main():
    parser = argparse.ArgumentParser(description='Create detailed evaluation report for PhoBERT')
    parser.add_argument('--results_path', type=str, default='evaluation_results/evaluation_summary.json',
                       help='Path to evaluation results JSON file')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Output directory for report files')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_path):
        print(f"Error: Results file not found: {args.results_path}")
        return
    
    # Load kết quả
    print("Loading evaluation results...")
    results = load_evaluation_results(args.results_path)
    
    # Tạo thư mục output
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Tạo các biểu đồ
    print("Creating visualizations...")
    create_confusion_matrix_plot(results['confusion_matrix'], args.output_dir)
    create_metrics_comparison_plot(results, args.output_dir)
    create_overall_metrics_plot(results, args.output_dir)
    create_support_distribution_plot(results, args.output_dir)
    
    # Tạo báo cáo HTML
    print("Generating HTML report...")
    generate_html_report(results, args.output_dir)
    
    print(f"✓ Evaluation report created successfully!")
    print(f"  - HTML Report: {args.output_dir}/evaluation_report.html")
    print(f"  - Plots saved in: {args.output_dir}/")

if __name__ == "__main__":
    main()
