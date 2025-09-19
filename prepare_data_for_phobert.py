import json
import pandas as pd
import os
from sklearn.model_selection import train_test_split

def convert_instruction_to_simple_format(input_file, output_file):
    """
    Chuyển đổi dữ liệu từ format instruction/input/output sang format đơn giản text/label cho PhoBERT
    """
    texts = []
    labels = []
    
    print(f"Đang xử lý file: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                
                # Lấy text từ trường input
                text = data.get('input', '').strip()
                
                # Lấy label từ trường output, loại bỏ token </s>
                output = data.get('output', '').strip()
                label_str = output.replace('</s>', '').strip()
                
                # Chuyển đổi label thành số nguyên
                if label_str == '0':
                    label = 0  # THẬT
                elif label_str == '1':
                    label = 1  # GIẢ
                else:
                    print(f"Cảnh báo: Label không hợp lệ '{label_str}' tại dòng {line_num}")
                    continue
                
                if text:  # Chỉ thêm nếu có text
                    texts.append(text)
                    labels.append(label)
                    
            except json.JSONDecodeError as e:
                print(f"Lỗi JSON tại dòng {line_num}: {e}")
                continue
            except Exception as e:
                print(f"Lỗi khác tại dòng {line_num}: {e}")
                continue
    
    # Tạo DataFrame
    df = pd.DataFrame({
        'text': texts,
        'label': labels
    })
    
    # Lưu thành CSV
    df.to_csv(output_file, index=False, encoding='utf-8')
    
    print(f"Đã chuyển đổi {len(df)} mẫu từ {input_file} sang {output_file}")
    print(f"Phân bố nhãn:")
    print(f"  - THẬT (0): {sum(1 for l in labels if l == 0)} mẫu")
    print(f"  - GIẢ (1): {sum(1 for l in labels if l == 1)} mẫu")
    
    return df

def analyze_data_distribution(df, dataset_name):
    """
    Phân tích phân bố dữ liệu
    """
    print(f"\n=== Phân tích dữ liệu {dataset_name} ===")
    print(f"Tổng số mẫu: {len(df)}")
    print(f"Phân bố nhãn:")
    label_counts = df['label'].value_counts().sort_index()
    for label, count in label_counts.items():
        label_name = "THẬT" if label == 0 else "GIẢ"
        percentage = (count / len(df)) * 100
        print(f"  - {label_name} ({label}): {count} mẫu ({percentage:.1f}%)")
    
    print(f"\nĐộ dài text trung bình: {df['text'].str.len().mean():.1f} ký tự")
    print(f"Độ dài text tối đa: {df['text'].str.len().max()} ký tự")
    print(f"Độ dài text tối thiểu: {df['text'].str.len().min()} ký tự")
    
    # Hiển thị một vài mẫu
    print(f"\nMột vài mẫu từ {dataset_name}:")
    for i, (_, row) in enumerate(df.head(3).iterrows()):
        label_name = "THẬT" if row['label'] == 0 else "GIẢ"
        text_preview = row['text'][:100] + "..." if len(row['text']) > 100 else row['text']
        print(f"  {i+1}. [{label_name}] {text_preview}")

def main():
    """
    Chuyển đổi tất cả các file dữ liệu
    """
    data_dir = "jsonl_text"
    output_dir = "phobert_data"
    
    # Tạo thư mục output nếu chưa có
    os.makedirs(output_dir, exist_ok=True)
    
    # Danh sách các file cần chuyển đổi
    files_to_convert = [
        ("train_instruction.jsonl", "train.csv"),
        ("val_instruction.jsonl", "val.csv"),
        ("test_instruction.jsonl", "test.csv")
    ]
    
    all_dataframes = {}
    
    for input_file, output_file in files_to_convert:
        input_path = os.path.join(data_dir, input_file)
        output_path = os.path.join(output_dir, output_file)
        
        if os.path.exists(input_path):
            print(f"\n{'='*50}")
            df = convert_instruction_to_simple_format(input_path, output_path)
            dataset_name = output_file.replace('.csv', '').upper()
            analyze_data_distribution(df, dataset_name)
            all_dataframes[dataset_name.lower()] = df
        else:
            print(f"File {input_path} không tồn tại")
    
    # Tổng hợp thống kê
    if all_dataframes:
        print(f"\n{'='*50}")
        print("=== TỔNG HỢP THỐNG KÊ ===")
        total_samples = sum(len(df) for df in all_dataframes.values())
        print(f"Tổng số mẫu trong tất cả datasets: {total_samples}")
        
        for name, df in all_dataframes.items():
            print(f"{name.upper()}: {len(df)} mẫu")
    
    print(f"\nDữ liệu đã được lưu trong thư mục: {output_dir}")
    print("Các file CSV đã sẵn sàng để training PhoBERT!")

if __name__ == "__main__":
    main()
