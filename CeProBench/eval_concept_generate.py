import os
import json
import glob
from typing import Any, Dict, Optional, Tuple
from eval_generate_prompts import system_prompt
from openai import OpenAI

def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    if start == -1: return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{": depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start:i+1]
                try: return json.loads(candidate)
                except Exception: return None
    return None

def _build_user_message(process_spec_txt: str, generated_flow_json: str) -> str:
    return (
        "process_spec_txt:\n<<<\n" f"{process_spec_txt}\n" ">>>\n\n"
        "generated_flow_json:\n<<<\n" f"{generated_flow_json}\n" ">>>\n"
    )

def evaluate_one(
    client: OpenAI,
    model: str,
    txt_path: str,
    json_path: str,
    temperature: float = 0,
    use_response_format_json: bool = True,
) -> Tuple[Dict[str, Any], str]:
    with open(txt_path, "r", encoding="utf-8") as f:
        process_spec_txt = f.read()
    with open(json_path, "r", encoding="utf-8") as f:
        generated_flow_json = f.read()

    user_msg = _build_user_message(process_spec_txt, generated_flow_json)
    print("*******************************")
    print(user_msg)
    kwargs = dict(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        temperature=temperature,
    )

    # if use_response_format_json:
    #     kwargs["response_format"] = {"type": "json_object"}

    try:
        resp = client.chat.completions.create(**kwargs)
        raw = resp.choices[0].message.content or ""
    except Exception as e:
        print(f"API 请求失败: {e}")
        if use_response_format_json:
            kwargs.pop("response_format", None)
            try:
                resp = client.chat.completions.create(**kwargs)
                raw = resp.choices[0].message.content or ""
            except Exception:
                raw = "{}"
        else:
            raw = "{}"
    print("*******************************")
    print(raw)
    parsed = _extract_json_object(raw)
    if parsed is None:
        parsed = {"category": "不正确", "score": 0, "analysis": {"error": "解析失败或AI未按格式输出"}}
    
    return parsed, raw

def evaluate_matching_batch(
    txt_dir: str,
    json_dir: str,
    summary_filename: str = "evaluation_summary_deepseek_final.json",
    model: str = "gemini-3-pro-preview",
):
    client = OpenAI(
        api_key="sk-",
        base_url="",
    )

    txt_files = glob.glob(os.path.join(txt_dir, "*.txt"))
    if not txt_files:
        print(f"Error: 在 {txt_dir} 未找到 txt 文件")
        return

    all_details = []
    stats = {
        "正确": 0,
        "合理但不正确": 0,
        "不正确": 0,
        "其他": 0 
    }

    print(f"开始评估，任务总数: {len(txt_files)}")

    for t_path in txt_files:
        base_name = os.path.splitext(os.path.basename(t_path))[0]
        j_path = os.path.join(json_dir, f"{base_name}_final.json")

        if not os.path.exists(j_path):
            print(f"[跳过] 未找到对应的 JSON: {base_name}_final.json")
            continue
        parsed, raw = evaluate_one(client, model, t_path, j_path)
        
        individual_eval_path = os.path.join(r"\geresults", f"{base_name}_eval_deepseek_final.json")
        with open(individual_eval_path, "w", encoding="utf-8") as f:
            json.dump({
                "case_name": base_name,
                "result": parsed,
                "raw_output": raw
            }, f, ensure_ascii=False, indent=2)

        cat = parsed.get("category", "其他")
        if cat in stats:
            stats[cat] += 1
        else:
            stats["其他"] += 1

        all_details.append({
            "case_name": base_name,
            "category": cat,
            "score": parsed.get("score", 0),
            "eval_file": os.path.basename(individual_eval_path)
        })
        
        print(f"[OK] {base_name} -> 分类: {cat} | 详情已保存至: {os.path.basename(individual_eval_path)}")

    total_eval = len(all_details)
    stats_with_ratio = {}
    if total_eval > 0:
        for key in ["正确", "合理但不正确", "不正确", "其他"]:
            count = stats[key]
            ratio = (count / total_eval) * 100
            stats_with_ratio[key] = {"count": count, "ratio": f"{ratio:.2f}%"}
    
    final_report = {
        "statistics": stats_with_ratio,
        "total_count": total_eval,
        "details": all_details  
    }

    out_path = os.path.join(json_dir, summary_filename)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, ensure_ascii=False, indent=2)

    print("\n" + "="*40)
    print(f"评估任务全部完成！")
    print(f"汇总报告路径: {out_path}")
    print("-"*40)
    print("统计结果摘要:")
    for k, v in stats_with_ratio.items():
        if v['count'] >= 0:
            print(f" - {k}: {v['count']} 个 (占比 {v['ratio']})")
    print("="*40)

if __name__ == "__main__":
    TXT_FOLDER = r"\huizong2"
    JSON_FOLDER = r"\new_generate\new_generate\generate_new_deepseekchat"
    
    evaluate_matching_batch(
        txt_dir=TXT_FOLDER,
        json_dir=JSON_FOLDER,
        summary_filename="eval_results_summary_deepseek_final.json",
        model="gemini-3-pro-preview"
    )