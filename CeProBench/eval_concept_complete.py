import os
import json
import pandas as pd

def evaluate_accuracy_numeric(csv_path, json_dir, k_max=6):
    try:
        try:
            df_gt = pd.read_csv(csv_path, encoding='utf-8')
        except UnicodeDecodeError:
            df_gt = pd.read_csv(csv_path, encoding='gbk')
    except Exception as e:
        print(f"读取 CSV 失败: {e}")
        return

    df_clean = df_gt[df_gt['是否存在问题（是/否）'] == '否'].copy()
    gt_mapping = {str(row['编号']): str(row['类型']).strip() for _, row in df_clean.iterrows()}

    results_list = []
    total_samples = 0
    k_counts = [0] * k_max 

    if not os.path.exists(json_dir):
        print(f"错误：路径不存在 -> {json_dir}")
        return

    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    
    json_files.sort(key=lambda x: int(x.split('_')[0]) if x.split('_')[0].isdigit() else 0)

    for filename in json_files:
        file_id = filename.split('_')[0]
        if file_id not in gt_mapping:
            continue

        file_path = os.path.join(json_dir, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            preds = data.get('completion', [])
            if not isinstance(preds, list): 
                preds = [preds]
            preds = [str(p).strip().lower() for p in preds]
            
            gt_val = gt_mapping[file_id].lower()
            total_samples += 1

            row_data = [file_id, gt_mapping[file_id]]
            for k in range(1, k_max + 1):
                is_hit = 1 if gt_val in preds[:k] else 0
                row_data.append(is_hit)
                if is_hit:
                    k_counts[k-1] += 1
            
            results_list.append(row_data)

        except Exception as e:
            print(f"处理 {filename} 出错: {e}")

    if not results_list:
        print("未发现匹配的 ID，请检查 CSV 编号与 JSON 文件名开头是否一致。")
        return

    headers = ["ID", "正确类型"] + [f"Top{i}" for i in range(1, k_max + 1)]
    report_df = pd.DataFrame(results_list, columns=headers)
    
    print(f"\n--- 每个文件的 Top-1 到 Top-{k_max} 命中详情 (1=命中, 0=未命中) ---")
    pd.set_option('display.max_rows', None)  
    pd.set_option('display.width', 1000)    
    print(report_df.to_string(index=False))
    
    print("\n" + "="*50)
    print(f"{f'总体准确率统计 (Top 1-{k_max})':^50}")
    print("="*50)
    print(f"测试样本总数: {total_samples}")
    print("-" * 50)
    for k in range(1, k_max + 1):
        acc = (k_counts[k-1] / total_samples) * 100
        print(f"Top-{k}: {acc:6.2f}% ({k_counts[k-1]}/{total_samples})")
    print("="*50)

if __name__ == "__main__":
    CSV_PATH = r'.csv'
    JSON_DIR = r'\\complete'

    evaluate_accuracy_numeric(CSV_PATH, JSON_DIR, k_max=10)