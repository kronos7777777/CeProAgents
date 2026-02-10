import os
import json
import argparse
import csv
import numpy as np
from tqdm import tqdm
from .utils.qa_eval import QAEvaluator

def parse_arguments():
    parser = argparse.ArgumentParser(description="Batch LLM Evaluation for Chemical Process QA with CSV export.")
    
    parser.add_argument("--input_file", type=str, 
                        default="./workdir/knowledge_augment/deepseek/answers.jsonl", 
                        help="Path to the input JSONL file containing predictions.")
    parser.add_argument("--output_dir", type=str, default="./workdir/knowledge_augment", 
                        help="Directory to save output files.")
    parser.add_argument("--model", type=str, default="deepseek", help="LLM model used.")
    
    parser.add_argument("--start_id", type=int, default=1, help="Start ID for evaluation.")
    parser.add_argument("--end_id", type=int, default=243, help="End ID for evaluation.")
    
    return parser.parse_args()

def load_data_map(file_path):
    data_map = {}
    if not os.path.exists(file_path):
        print(f"[ERROR] Input file not found: {file_path}")
        return data_map

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line.strip())
                    # Ensure ID is treated as integer for range mapping
                    item_id = int(item.get("id", -1))
                    if item_id != -1:
                        data_map[item_id] = item
    except Exception as e:
        print(f"[ERROR] Failed to load {file_path}: {e}")
    
    return data_map

def save_to_csv(results, csv_path):
    """Saves detailed per-case metrics to a CSV file."""
    
    headers = [
        "ID", "Class", 
        "Avg_Score", 
        "Correctness", "Rationality", "Clarity", "Completeness", "Format",
        "Critique", "Question_Snippet"
    ]

    try:
        with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

            for res in results:
                m = res['metrics']
                scores = [m.get(k, 0) for k in ["Correctness", "Rationality", "Clarity", "Completeness", "Format"]]
                avg_val = np.mean(scores)

                row = [
                    res['id'],
                    res['class'],
                    f"{avg_val:.2f}",
                    m.get("Correctness", 0),
                    m.get("Rationality", 0),
                    m.get("Clarity", 0),
                    m.get("Completeness", 0),
                    m.get("Format", 0),
                    m.get("Critique", "").replace("\n", " "), # Flatten text for CSV
                    res['question'][:50] + "..."
                ]
                writer.writerow(row)
        print(f"[Saved] CSV details: {csv_path}")
    except Exception as e:
        print(f"[ERROR] Failed to write CSV: {e}")

def process_single_item(evaluator, item_id, data_map):
    item = data_map.get(item_id)
    
    if not item:
        return None

    question = item.get("question", "")
    gt = item.get("ground_truth", "")
    pred = item.get("prediction", "")
    q_class = item.get("class", "Unknown")

    if not pred:
        print(f"[{item_id}] WARN: Prediction field is empty.")
        return None

    metrics = evaluator.evaluate(
        question=question,
        ground_truth=gt,
        prediction=pred
    )

    return {
        "id": item_id,
        "class": q_class,
        "question": question,
        "metrics": metrics 
    }

def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print("=== Initializing Evaluator ===")
    evaluator = QAEvaluator()

    data_map = load_data_map(args.input_file)
    if not data_map:
        return

    all_results = []
    print(f"=== Starting Batch Eval (IDs {args.start_id}-{args.end_id}) ===")
    print(f"Input File: {args.input_file}")

    # Loop strictly based on start_id and end_id
    for i in tqdm(range(args.start_id, args.end_id + 1), desc="Processing IDs"):
        # try:
        res = process_single_item(evaluator, i, data_map)
        if res:
            all_results.append(res)
            # Optional: print specific metric like F1 in original, here we print Avg
            avg = np.mean([res['metrics'][k] for k in ["Correctness", "Rationality", "Clarity", "Completeness", "Format"]])
                # print(f"[{i}] Processed. Avg Score: {avg:.1f}")
        # except Exception as e:
        #     print(f"[{i}] FATAL ERROR: An unexpected exception occurred: {str(e)}")

    if not all_results:
        print("No valid results were computed. Check ID range or data file.")
        return

    csv_path = os.path.join(args.output_dir, f"knowledge_augment_eval_summary_{args.model}.csv")
    save_to_csv(all_results, csv_path)

    metric_keys = ["Correctness", "Rationality", "Clarity", "Completeness", "Format"]
    global_stats = {}
    
    for k in metric_keys:
        vals = [r['metrics'].get(k, 0) for r in all_results]
        global_stats[k] = float(np.mean(vals)) if vals else 0.0
    
    overall_avg = np.mean(list(global_stats.values()))
    
    print("\n" + "="*20 + " Final Summary " + "="*20)
    print(f"Total Items Processed: {len(all_results)}")
    print(f"Overall Average Score: {overall_avg:.4f}")
    for k, v in global_stats.items():
        print(f"  - {k}: {v:.2f}")
    print("="*55)

    out_json = os.path.join(args.output_dir, f"knowledge_augement_eval_summary_{args.model}.json")
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump({
            "summary": {
                "overall_average": overall_avg,
                "metric_breakdown": global_stats
            }, 
            "details": all_results
        }, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    args = parse_arguments()
    main(args)