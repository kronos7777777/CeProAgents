# evaluate_optimization_results_final.py
# -*- coding: utf-8 -*-

import os
import json
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = "optimize_results"  
OUTPUT_JSON = os.path.join(ROOT_DIR, "evaluation_summary.json")
OUTPUT_CSV = os.path.join(ROOT_DIR, "evaluation_summary.csv")

PURITY_REQUIREMENT = None

DEFAULT_PURITY_FOR_YIELD_TASK = 1.0    
DEFAULT_YIELD_FOR_PURITY_TASK = 100.0   

def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def pct_change(base: float, new: float) -> Optional[float]:
    if base == 0:
        return None
    return (new - base) / base * 100.0


def list_all_json_files(root_dir: str) -> List[str]:
    files: List[str] = []
    if os.path.isfile(root_dir) and root_dir.lower().endswith(".json"):
        return [root_dir]
    for r, _, names in os.walk(root_dir):
        for n in names:
            if n.lower().endswith(".json"):
                files.append(os.path.join(r, n))
    files.sort()
    return files


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_full_history(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    hist = data.get("full_history")
    return hist if isinstance(hist, list) else []


def get_equipment_ids(iteration: Dict[str, Any]) -> List[str]:
    input_params = iteration.get("simulation_results", {}) or {}
    if isinstance(input_params, dict):
        return [str(k) for k in input_params.keys()]
    return []


def find_metric_dicts(sim_results: Dict[str, Any], metric_name: str) -> List[Tuple[str, Dict[str, Any]]]:

    metric_upper = metric_name.upper()
    out: List[Tuple[str, Dict[str, Any]]] = []
    for k, v in sim_results.items():
        if metric_upper in str(k).upper() and isinstance(v, dict) and v:
            out.append((str(k), v))
    return out


def extract_metric_average(
    iteration: Dict[str, Any],
    metric_name: str
) -> Tuple[Optional[float], List[str], List[str], int]:

    sim = iteration.get("simulation_results", {}) or {}
    if not isinstance(sim, dict) or not sim:
        return None, [], [], 0

    candidates = find_metric_dicts(sim, metric_name)
    if not candidates:
        return None, [], [], 0

    values: List[float] = []
    units: List[str] = []
    src_keys: List[str] = []

    for src_key, md in candidates:
        src_keys.append(src_key)
        for _, val_unit in md.items():
            if not (isinstance(val_unit, list) and len(val_unit) >= 1):
                continue
            v = safe_float(val_unit[0])
            if v is None:
                continue
            values.append(v)
            if len(val_unit) >= 2 and isinstance(val_unit[1], str):
                units.append(val_unit[1])

    if not values:
        return None, units, src_keys, 0

    avg = sum(values) / len(values)
    return avg, units, src_keys, len(values)


def infer_task_type_by_metrics(iter1: Dict[str, Any]) -> Tuple[str, bool, bool]:
    sim = iter1.get("simulation_results", {}) or {}
    if not isinstance(sim, dict):
        return "unknown", False, False

    has_mole_flow = len(find_metric_dicts(sim, "MOLEFLOW")) > 0
    has_mass_flow = len(find_metric_dicts(sim, "MASSFLOW")) > 0
    has_flow = has_mole_flow or has_mass_flow

    has_mole_frac = len(find_metric_dicts(sim, "MOLEFRAC")) > 0
    has_mass_frac = len(find_metric_dicts(sim, "MASSFRAC")) > 0
    has_frac = has_mole_frac or has_mass_frac

    if has_flow and has_frac:
        return "combined", True, True
    if has_flow:
        return "yield", True, False
    if has_frac:
        return "purity", False, True
    return "unknown", False, False


def filter_equipment_ids(eq_ids: List[str], task_type: str) -> List[str]:
    if task_type == "yield":
        return [e for e in eq_ids if e.startswith("R") or e.startswith("E")]
    if task_type == "purity":
        return [e for e in eq_ids if e.startswith("T") or e.startswith("E")]
    if task_type == "combined":
        return [e for e in eq_ids if e.startswith("R") or e.startswith("T") or e.startswith("V") or e.startswith("E")]
    return eq_ids


def sum_equipment_cost(
    iteration: Dict[str, Any],
    eq_ids: List[str],
    task_type: str
) -> Tuple[Optional[float], Optional[str], Dict[str, float]]:

    sim = iteration.get("simulation_results", {}) or {}
    if not isinstance(sim, dict):
        return None, "需要额外计算", {}

    total = 0.0
    breakdown: Dict[str, float] = {}

    for eq in eq_ids:
        block = sim.get(eq)
        if not isinstance(block, dict):
            return None, "需要额外计算", breakdown

        if eq.startswith("R"):
            util = block.get("UTIL_COST")
            if not (isinstance(util, list) and len(util) >= 1):
                return None, "需要额外计算", breakdown
            v = safe_float(util[0])
            if (v is None or v == 0) and task_type == "yield":
                qcalc = block.get("QCALC")
                v = safe_float(qcalc[0])
            elif (v is None or v == 0) and task_type == "combined":
                qcalc = block.get("QCALC")
                v = safe_float(qcalc[0] * 2.12e-7)
            elif v is None or v == 0:
                return None, "需要额外计算", breakdown
            breakdown[eq] = v
            total += v

        elif eq.startswith("V") or eq.startswith("E"):
            util = block.get("UTIL_COST")
            if not (isinstance(util, list) and len(util) >= 1):
                return None, "需要额外计算", breakdown
            v = safe_float(util[0])
            breakdown[eq] = v
            total += v

        elif eq.startswith("T"):
            cond = block.get("COND_COST")
            reb = block.get("REB_COST")
            if not (isinstance(cond, list) and len(cond) >= 1):
                return None, "需要额外计算", breakdown
            if not (isinstance(reb, list) and len(reb) >= 1):
                return None, "需要额外计算", breakdown

            vc = safe_float(cond[0])
            vr = safe_float(reb[0])
            if vc is None or vr is None:
                return None, "需要额外计算", breakdown

            v = vc + vr
            breakdown[eq] = v
            total += v
        else:
            return None, "需要额外计算", breakdown

    return total, None, breakdown


def _extract_flow_frac_with_fallback(iteration: Dict[str, Any]) -> Tuple[
    Optional[float], List[str], List[str], int,
    Optional[float], List[str], List[str], int,
    str, str
]:
    flow, flow_units, flow_src, flow_n = extract_metric_average(iteration, "MOLEFLOW")
    flow_metric = "MOLEFLOW"
    if flow is None:
        flow, flow_units, flow_src, flow_n = extract_metric_average(iteration, "MASSFLOW")
        flow_metric = "MASSFLOW"

    frac, frac_units, frac_src, frac_n = extract_metric_average(iteration, "MOLEFRAC")
    frac_metric = "MOLEFRAC"
    if frac is None:
        frac, frac_units, frac_src, frac_n = extract_metric_average(iteration, "MASSFRAC")
        frac_metric = "MASSFRAC"

    return (
        flow, flow_units, flow_src, flow_n,
        frac, frac_units, frac_src, frac_n,
        flow_metric, frac_metric
    )


def _compute_score_for_iteration(
    iteration: Dict[str, Any],
    task_type: str,
    eq_ids: List[str]
) -> Dict[str, Any]:
    # flow/frac
    (flow, flow_units, flow_src, flow_n,
     frac, frac_units, frac_src, frac_n,
     flow_metric, frac_metric) = _extract_flow_frac_with_fallback(iteration)

    if task_type == "yield":
        purity = DEFAULT_PURITY_FOR_YIELD_TASK
        yld = flow
    elif task_type == "purity":
        purity = frac
        yld = DEFAULT_YIELD_FOR_PURITY_TASK
    elif task_type == "combined":
        purity = frac
        yld = flow
    else:
        purity, yld = frac, flow

    # cost
    cost, cost_note, cost_bd = sum_equipment_cost(iteration, eq_ids, task_type) if eq_ids else (None, "需要额外计算", {})

    # score (float or None)
    score_val: Optional[float] = None
    score_note = None
    if cost is None:
        score_note = "需要额外计算"
    else:
        if purity is None or yld is None:
            score_note = "缺少纯度/产量数据，无法计算"
        elif cost == 0:
            score_note = "需要额外计算"
        else:
            score_val = purity * yld / cost

    return {
        "iteration_number": iteration.get("iteration"),
        "flow_metric_used": flow_metric,
        "frac_metric_used": frac_metric,

        "flow": flow,
        "flow_units": flow_units,
        "flow_src": flow_src,
        "flow_n": flow_n,

        "frac": frac,
        "frac_units": frac_units,
        "frac_src": frac_src,
        "frac_n": frac_n,

        "purity": purity,
        "yield": yld,

        "cost": cost,
        "cost_note": cost_note,
        "cost_breakdown": cost_bd,

        "score": score_val,
        "score_note": score_note,
    }


def evaluate_one_file(path: str) -> Dict[str, Any]:
    data = load_json(path)
    hist = get_full_history(data)
    if not hist:
        return {"_file": path, "error": "full_history 为空或不存在"}

    iter1 = hist[0]
    task_type, _, _ = infer_task_type_by_metrics(iter1)

    eq_ids_raw = get_equipment_ids(iter1)
    eq_ids = filter_equipment_ids(eq_ids_raw, task_type)

    base = _compute_score_for_iteration(iter1, task_type, eq_ids)
    flow1, flow_units1, flow_src1, flow_n1 = base["flow"], base["flow_units"], base["flow_src"], base["flow_n"]
    frac1, frac_units1, frac_src1, frac_n1 = base["frac"], base["frac_units"], base["frac_src"], base["frac_n"]
    purity1, yield1 = base["purity"], base["yield"]
    cost1, cost_breakdown_1 = base["cost"], base["cost_breakdown"]
    score1 = base["score"]
    best = None
    best_idx = None
    for idx, it in enumerate(hist):
        if idx == 0:
            continue
        cur = _compute_score_for_iteration(it, task_type, eq_ids)
        if cur["score"] is None:
            continue
        if best is None or cur["score"] > best["score"]:
            best = cur
            best_idx = idx

    if best is None:
        best = _compute_score_for_iteration(hist[-1], task_type, eq_ids)
        best_idx = len(hist) - 1

    purityn, yieldn = best["purity"], best["yield"]
    flown, flow_unitsn, flow_srcn, flow_nn = best["flow"], best["flow_units"], best["flow_src"], best["flow_n"]
    fracn, frac_unitsn, frac_srcn, frac_nn = best["frac"], best["frac_units"], best["frac_src"], best["frac_n"]
    costn, cost_breakdown_n = best["cost"], best["cost_breakdown"]
    scoren = best["score"]

    purity_impr_pct = None if (purity1 is None or purityn is None) else pct_change(purity1, purityn)
    yield_impr_pct = None if (yield1 is None or yieldn is None) else pct_change(yield1, yieldn)

    meets_purity = None
    if PURITY_REQUIREMENT is not None and purityn is not None and task_type in ("purity", "combined"):
        meets_purity = bool(purityn >= PURITY_REQUIREMENT)

    cost_change_pct = None
    cost_reduction_pct = None
    cost_note = None
    if isinstance(cost1, (int, float)) and isinstance(costn, (int, float)):
        cost_change_pct = pct_change(cost1, costn) 
        cost_reduction_pct = ((cost1 - costn) / cost1 * 100.0) if cost1 != 0 else None
    else:
        cost_note = "需要额外计算"

    score_impr_pct = None
    score_note = None
    if score1 is None or scoren is None:
        score_note = "需要额外计算"
    else:
        score_impr_pct = pct_change(score1, scoren) if score1 != 0 else None

    return {
        "_file": path,
        "task_type": task_type,
        "iterations": len(hist),

        "best_iteration_index_0based": best_idx,
        "best_iteration_number": best.get("iteration_number"),

        "yield_baseline_avg": yield1,
        "yield_final_avg": yieldn,
        "yield_improvement_pct_vs_first": yield_impr_pct,
        "yield_value_count_baseline": flow_n1,
        "yield_value_count_final": flow_nn,
        "yield_source_keys_baseline": flow_src1,
        "yield_source_keys_final": flow_srcn,
        "yield_units_seen_baseline": flow_units1,
        "yield_units_seen_final": flow_unitsn,

        "purity_baseline_avg": purity1,
        "purity_final_avg": purityn,
        "purity_improvement_pct_vs_first": purity_impr_pct,
        "purity_value_count_baseline": frac_n1,
        "purity_value_count_final": frac_nn,
        "purity_source_keys_baseline": frac_src1,
        "purity_source_keys_final": frac_srcn,
        "purity_units_seen_baseline": frac_units1,
        "purity_units_seen_final": frac_unitsn,

        "purity_requirement": PURITY_REQUIREMENT,
        "purity_meets_requirement": meets_purity,

        "equipment_ids_for_cost": eq_ids,
        "equipment_cost_sum_baseline": cost1 if cost1 is not None else "需要额外计算",
        "equipment_cost_sum_final": costn if costn is not None else "需要额外计算",
        "equipment_cost_change_pct_vs_first": cost_change_pct if cost_change_pct is not None else (cost_note or "需要额外计算"),
        "equipment_cost_reduction_pct_vs_first": cost_reduction_pct if cost_reduction_pct is not None else (cost_note or "需要额外计算"),
        "equipment_cost_breakdown_baseline": cost_breakdown_1,
        "equipment_cost_breakdown_final": cost_breakdown_n,
        "score_baseline_purity_times_yield_over_cost": score1 if score1 is not None else (score_note or "需要额外计算"),
        "score_final_purity_times_yield_over_cost": scoren if scoren is not None else (score_note or "需要额外计算"),
        "score_improvement_pct_vs_first": score_impr_pct if score_impr_pct is not None else (score_note or "需要额外计算"),
    }


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def to_csv_rows(reports: List[Dict[str, Any]]) -> List[List[str]]:
    header = [
        "file", "task_type", "iterations",
        "best_iteration_number",
        "yield_baseline_avg", "yield_final_avg", "yield_improvement_pct_vs_first",
        "purity_baseline_avg", "purity_final_avg", "purity_improvement_pct_vs_first",
        "purity_requirement", "purity_meets_requirement",
        "equipment_cost_sum_baseline", "equipment_cost_sum_final",
        "equipment_cost_change_pct_vs_first", "equipment_cost_reduction_pct_vs_first",
        "score_baseline", "score_final", "score_improvement_pct_vs_first",
        "equipment_ids_for_cost"
    ]
    rows = [header]

    def s(x: Any) -> str:
        if x is None:
            return ""
        if isinstance(x, float):
            return f"{x:.6g}"
        if isinstance(x, list):
            return ";".join(map(str, x))
        if isinstance(x, dict):
            return json.dumps(x, ensure_ascii=False)
        return str(x)

    for rep in reports:
        rows.append([
            s(rep.get("_file")),
            s(rep.get("task_type")),
            s(rep.get("iterations")),
            s(rep.get("best_iteration_number")),

            s(rep.get("yield_baseline_avg")),
            s(rep.get("yield_final_avg")),
            s(rep.get("yield_improvement_pct_vs_first")),

            s(rep.get("purity_baseline_avg")),
            s(rep.get("purity_final_avg")),
            s(rep.get("purity_improvement_pct_vs_first")),

            s(rep.get("purity_requirement")),
            s(rep.get("purity_meets_requirement")),

            s(rep.get("equipment_cost_sum_baseline")),
            s(rep.get("equipment_cost_sum_final")),
            s(rep.get("equipment_cost_change_pct_vs_first")),
            s(rep.get("equipment_cost_reduction_pct_vs_first")),

            s(rep.get("score_baseline_purity_times_yield_over_cost")),
            s(rep.get("score_final_purity_times_yield_over_cost")),
            s(rep.get("score_improvement_pct_vs_first")),

            s(rep.get("equipment_ids_for_cost")),
        ])
    return rows


def write_csv(path: str, rows: List[List[str]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            escaped = []
            for cell in row:
                cell = "" if cell is None else str(cell)
                if any(ch in cell for ch in [",", "\"", "\n"]):
                    cell = "\"" + cell.replace("\"", "\"\"") + "\""
                escaped.append(cell)
            f.write(",".join(escaped) + "\n")


def main():
    json_files = list_all_json_files(ROOT_DIR)
    reports: List[Dict[str, Any]] = []

    for p in json_files:
        try:
            rep = evaluate_one_file(p)
            reports.append(rep)
        except Exception as e:
            reports.append({"_file": p, "error": str(e)})

    write_json(OUTPUT_JSON, reports)
    write_csv(OUTPUT_CSV, to_csv_rows(reports))

    counts: Dict[str, int] = {}
    for r in reports:
        t = r.get("task_type", "error") if "error" not in r else "error"
        counts[t] = counts.get(t, 0) + 1

    print(f"[DONE] scanned json files: {len(json_files)}")
    print(f"[DONE] saved json: {OUTPUT_JSON}")
    print(f"[DONE] saved csv : {OUTPUT_CSV}")
    print("[COUNT BY TASK_TYPE]", counts)


if __name__ == "__main__":
    main()
