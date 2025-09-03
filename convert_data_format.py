import json
import os

def convert_data_format(input_file, output_file):
    """
    Chuyển đổi dữ liệu từ format prompt/completion sang instruction/input/output
    """
    converted_data = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            data = json.loads(line)
            prompt = data.get('prompt', '')
            completion = data.get('completion', '')
            
            # Tách instruction và input từ prompt
            if 'BÀI ĐĂNG:' in prompt:
                parts = prompt.split('BÀI ĐĂNG:')
                instruction = parts[0].strip()
                input_text = parts[1].strip()
            else:
                instruction = prompt
                input_text = ""
            
            # Tạo format mới
            new_format = {
                "instruction": instruction,
                "input": input_text,
                "output": completion + "</s>"
            }
            
            converted_data.append(new_format)
    
    # Ghi ra file mới
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in converted_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Đã chuyển đổi {len(converted_data)} mẫu từ {input_file} sang {output_file}")
    
    # Hiển thị mẫu đầu tiên để kiểm tra
    if converted_data:
        print("\nMẫu đầu tiên:")
        print(json.dumps(converted_data[0], ensure_ascii=False, indent=2))

def main():
    # Chuyển đổi các file dữ liệu
    data_dir = "jsonl_text"
    
    files_to_convert = [
        ("train.jsonl", "train_instruction.jsonl"),
        ("val.jsonl", "val_instruction.jsonl"),
        ("test.jsonl", "test_instruction.jsonl")
    ]
    
    for input_file, output_file in files_to_convert:
        input_path = os.path.join(data_dir, input_file)
        output_path = os.path.join(data_dir, output_file)
        
        if os.path.exists(input_path):
            convert_data_format(input_path, output_path)
        else:
            print(f"File {input_path} không tồn tại")

if __name__ == "__main__":
    main()
