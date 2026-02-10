import json
import os
from difflib import SequenceMatcher


def string_similarity(a, b):
    if not a or not b: return 0.0
    a = str(a).upper().strip()
    b = str(b).upper().strip()
    ratio = SequenceMatcher(None, a, b).ratio()
    if a in b or b in a:
        ratio = max(ratio, 0.9)
    return ratio

def evaluate_pid_independent(gt_data, pred_data, threshold=0.4):
    gt_eqs = gt_data.get('equipments', [])
    pred_eqs = pred_data.get('equipments', [])
    gt_conns = gt_data.get('connections', [])
    pred_conns = pred_data.get('connections', [])

    tp_nodes = 0
    matched_gt_node_indices = set()
    
    gt_node_strings = [f"{e.get('identifier', '')} {e.get('type', '')}" for e in gt_eqs]
    pred_node_strings = [f"{e.get('identifier', '')} {e.get('type', '')}" for e in pred_eqs]

    for p_str in pred_node_strings:
        best_score = 0
        best_gt_idx = -1
        for g_idx, g_str in enumerate(gt_node_strings):
            if g_idx in matched_gt_node_indices: continue
            
            score = string_similarity(p_str, g_str)
            if score >= threshold and score > best_score:
                best_score = score
                best_gt_idx = g_idx
        
        if best_gt_idx != -1:
            tp_nodes += 1
            matched_gt_node_indices.add(best_gt_idx)

    node_precision = tp_nodes / len(pred_eqs) if pred_eqs else 0.0
    node_recall = tp_nodes / len(gt_eqs) if gt_eqs else 0.0

    tp_conns = 0
    matched_gt_conn_indices = set()

    for p_conn in pred_conns:
        p_src = p_conn.get('source', '')
        p_tgt = p_conn.get('target', '')
        
        best_conn_gt_idx = -1
        for g_idx, g_conn in enumerate(gt_conns):
            if g_idx in matched_gt_conn_indices: continue
            
            g_src = g_conn.get('source', '')
            g_tgt = g_conn.get('target', '')
            
            if string_similarity(p_src, g_src) >= threshold and \
               string_similarity(p_tgt, g_tgt) >= threshold:
                best_conn_gt_idx = g_idx
                break 
        
        if best_conn_gt_idx != -1:
            tp_conns += 1
            matched_gt_conn_indices.add(best_conn_gt_idx)

    conn_precision = tp_conns / len(pred_conns) if pred_conns else 0.0
    conn_recall = tp_conns / len(gt_conns) if gt_conns else 0.0

    return tp_nodes, node_precision, node_recall, tp_conns, conn_precision, conn_recall

base_path_gt = r"\PID_parse"
base_path_pred = r"\\parse"

all_results = []

print(f"{'FileID':<7} | {'Match_N':<7} | {'Node_P':<9} | {'Node_R':<9} | {'Match_C':<7} | {'Conn_P':<9} | {'Conn_R':<9}")
print("-" * 85)

for i in range(1, 116):
    gt_file = os.path.join(base_path_gt, f"{i}_parse.json")
    pred_file = os.path.join(base_path_pred, f"{i}_parse_claude-opus-4-5-20251101.json")
    
    if not os.path.exists(gt_file) or not os.path.exists(pred_file):
        continue

    try:
        with open(gt_file, 'r', encoding='utf-8') as f:
            gt_json = json.load(f)
        with open(pred_file, 'r', encoding='utf-8') as f:
            pred_json = json.load(f)

        tp_n, n_p, n_r, tp_c, c_p, c_r = evaluate_pid_independent(gt_json, pred_json, 0.4)
        
        all_results.append((n_p, n_r, c_p, c_r))
        # print(f"{i:<7} | {tp_n:<7} | {n_p:>8.1%} | {n_r:>8.1%} | {tp_c:<7} | {c_p:>8.1%} | {c_r:>8.1%}")
        print(f"{i:<7} {n_p:>8.1%} {n_r:>8.1%} {c_p:>8.1%} {c_r:>8.1%}")

    except Exception as e:
        print(f"{i:<7} | Error: {e}")

if all_results:
    count = len(all_results)
    avg_node_p = sum(r[0] for r in all_results) / count
    avg_node_r = sum(r[1] for r in all_results) / count
    avg_conn_p = sum(r[2] for r in all_results) / count
    avg_conn_r = sum(r[3] for r in all_results) / count

    print("\n" + "="*85)
    print("FINAL SUMMARY REPORT (Independent Fuzzy Matching)")
    print("="*85)
    print(f"Total Files Processed: {count}")
    print("-" * 85)
    print(f"Average Equipment Precision: {avg_node_p:.2%}")
    print(f"Average Equipment Recall:    {avg_node_r:.2%}")
    print(f"Average Connection Precision: {avg_conn_p:.2%}")
    print(f"Average Connection Recall:    {avg_conn_r:.2%}")
    print("="*85)