import os
import json
import glob
import argparse
import csv
import numpy as np
from .utils.kg_entity_eval import KGExtractionEvaluator
from .utils.kg_graph_metrics import (
    build_adjacency_from_triplets,
    compute_mec_med,
)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Batch KG evaluation with CSV export.")
    
    parser.add_argument("--gt_dir", type=str, default="./CeProBench/knowledge/knowledge_extract", 
                        help="Directory containing ground truth files.")
    parser.add_argument("--pred_work_dir", type=str, default="./workdir/knowledge_extract/deepseek", 
                        help="Root directory for prediction outputs.")
    parser.add_argument("--output_dir", type=str, default="./workdir/knowledge_extract", 
                        help="Directory to save output files.")
    parser.add_argument("--model", type=str, default="deepseek", help="LLM model used.")
    
    parser.add_argument("--start_id", type=int, default=1)
    parser.add_argument("--end_id", type=int, default=10)
    parser.add_argument("--config_path", type=str, default=None, help="Entity eval config path.")
    
    return parser.parse_args()

def load_and_parse_json(file_path):
    """Parses KG JSON into entity list and triplet list."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load {file_path}: {e}")
        return [], []

    entities = []
    triplets = []

    raw_ents = data.get("entities") or data.get("nodes") or []
    for e in raw_ents:
        if isinstance(e, str):
            entities.append(e)
        elif isinstance(e, dict):
            val = e.get("name") or e.get("label") or e.get("id") or str(e)
            entities.append(val)
    entities = list(dict.fromkeys(entities))

    raw_rels = data.get("relations") or data.get("triplets") or data.get("edges") or []
    for r in raw_rels:
        if isinstance(r, (list, tuple)) and len(r) >= 3:
            triplets.append([str(r[0]), str(r[1]), str(r[2])])
        elif isinstance(r, dict):
            h = r.get("head") or r.get("source") or r.get("subject")
            t = r.get("tail") or r.get("target") or r.get("object")
            rel = r.get("relation") or r.get("type") or r.get("predicate") or "related_to"
            if h and t:
                triplets.append([str(h), str(rel), str(t)])

    return entities, triplets

def get_gt_edges_indices(gt_entities, gt_triplets):
    ent2idx = {name: i for i, name in enumerate(gt_entities)}
    edges = []
    for h, r, t in gt_triplets:
        if h in ent2idx and t in ent2idx:
            edges.append((ent2idx[h], ent2idx[t]))
    return edges

def save_to_csv(results, csv_path):
    headers = [
        "ID", 
        "F1", "ER", "Precision", "Num_GT_Ent", "Num_Pred_Ent",
        "MEC", "MED", "MED_Coverage", "Avg_Path_Len", "Num_GT_Edges"
    ]

    try:
        with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

            for res in results:
                em = res['entity_metrics']
                gm = res['graph_metrics']
                
                med_val = f"{gm['med']:.4f}" if gm['med'] is not None else "N/A"
                dbar_val = f"{gm['d_bar']:.4f}" if gm['d_bar'] is not None else "N/A"

                row = [
                    res['id'],
                    f"{em['f1']:.4f}", f"{em['er']:.4f}", f"{em['pc']:.4f}",
                    em['num_gt'], em['num_pred'],
                    f"{gm['mec']:.4f}", med_val, f"{gm['med_coverage']:.4f}",
                    dbar_val, gm['num_gt_edges']
                ]
                writer.writerow(row)
        print(f"[Saved] CSV details: {csv_path}")
    except Exception as e:
        print(f"[ERROR] Failed to write CSV: {e}")

def find_path_by_prefix(root_dir, prefix):
    """Finds a file or directory in root_dir that starts with the given prefix."""
    if not os.path.exists(root_dir):
        return None
    for name in os.listdir(root_dir):
        if name.startswith(prefix) and not name.endswith(".pdf"):
            return os.path.join(root_dir, name)
    return None

def process_single_item(evaluator, file_id, gt_dir, pred_work_dir):
    """
    Processes a single item by dynamically finding GT and prediction files based on ID prefix.
    """
    prefix = f"{file_id}_"

    gt_path = find_path_by_prefix(gt_dir, prefix)
    if not gt_path or not gt_path.endswith("_tgt.json"):
        # Fallback for simple numeric names if prefixed name fails
        gt_path_fallback = os.path.join(gt_dir, f"{file_id}_tgt.json")
        if os.path.exists(gt_path_fallback):
             gt_path = gt_path_fallback
        else:
             print(f"[{file_id}] WARN: No ground truth file found with prefix '{prefix}' or name '{file_id}_tgt.json'.")
             return None

    pred_subdir = find_path_by_prefix(pred_work_dir, prefix)
    if not pred_subdir:
        # Fallback for simple numeric names
        pred_subdir_fallback = os.path.join(pred_work_dir, str(file_id))
        if os.path.exists(pred_subdir_fallback):
            pred_subdir = pred_subdir_fallback
        else:
            print(f"[{file_id}] WARN: No prediction directory found with prefix '{prefix}' or name '{file_id}'.")
            return None

    base_name = os.path.basename(pred_subdir)
    pred_path = os.path.join(pred_subdir, f"{base_name}_pred.json")
    
    if not os.path.exists(pred_path):
        print(f"[{file_id}] WARN: Prediction file not found at '{pred_path}'.")
        return None
    
    gt_entities, gt_triplets = load_and_parse_json(gt_path)
    pred_entities, pred_triplets = load_and_parse_json(pred_path)

    if not gt_entities:
        print(f"[{file_id}] WARN: Ground truth entities are empty in '{gt_path}'. Skipping.")
        return None

    entity_stats = evaluator.evaluate(
        gt_entities=gt_entities,
        extracted_entities=pred_entities,
        max_eval=None
    )
    
    gt_edges = get_gt_edges_indices(gt_entities, gt_triplets)
    pred_adj = build_adjacency_from_triplets(pred_entities, pred_triplets, directed=False)
    
    graph_res = compute_mec_med(
        gt_edges=gt_edges,
        mapping_gt2pred=entity_stats["mapping_gt2pred"],
        pred_adj=pred_adj,
        med_sample_size=200
    )

    return {
        "id": file_id,
        "entity_metrics": {
            "num_gt": entity_stats['num_gt'],
            "num_pred": entity_stats['num_pred'],
            "er": entity_stats['er'],
            "pc": entity_stats['pc'],
            "f1": entity_stats['f1']
        },
        "graph_metrics": {
            "num_gt_edges": graph_res.num_gt_edges,
            "mec": graph_res.mec,
            "med": graph_res.med,
            "med_coverage": graph_res.med_coverage,
            "d_bar": graph_res.d_bar
        }
    }

def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print("=== Initializing Evaluator ===")
    evaluator = KGExtractionEvaluator(
        config_path=args.config_path,
        top_k=5,
        max_eval_default=None,
    )

    all_results = []
    print(f"=== Starting Batch Eval (IDs {args.start_id}-{args.end_id}) ===")
    print(f"GT Directory: {args.gt_dir}")
    print(f"Prediction Work Directory: {args.pred_work_dir}")

    for i in range(args.start_id, args.end_id + 1):
        try:
            res = process_single_item(evaluator, i, args.gt_dir, args.pred_work_dir)
            if res:
                all_results.append(res)
                print(f"[{i}] Processed Successfully. F1: {res['entity_metrics']['f1']:.3f} | MEC: {res['graph_metrics']['mec']:.3f}")
        except Exception as e:
            print(f"[{i}] FATAL ERROR: An unexpected exception occurred: {str(e)}")

    if not all_results:
        print("No valid results were computed. Please check paths and file formats.")
        return

    csv_path = os.path.join(args.output_dir, f"knowledge_extract_eval_summary_{args.model}.csv")
    save_to_csv(all_results, csv_path)

    avg_f1 = np.mean([r['entity_metrics']['f1'] for r in all_results])
    avg_mec = np.mean([r['graph_metrics']['mec'] for r in all_results])
    
    print("\n" + "="*20 + " Final Summary " + "="*20)
    print(f"Total Items Processed: {len(all_results)}")
    print(f"Average Entity F1-Score: {avg_f1:.4f}")
    print(f"Average Graph MEC Score: {avg_mec:.4f}")
    print("="*55)

    out_json = os.path.join(args.output_dir, f"knowledge_extract_eval_summary_{args.model}.json")
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump({"summary": {"f1": avg_f1, "mec": avg_mec}, "details": all_results}, f, indent=2)

if __name__ == "__main__":
    args = parse_arguments()
    main(args)