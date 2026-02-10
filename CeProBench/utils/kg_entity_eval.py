# CeProAgents/eval/kg_entity_eval.py
"""
KG Entity Extraction Evaluation Tool (Entity-Level)

This module implements the evaluation logic for Knowledge Graph entity extraction.
It utilizes a hybrid approach combining semantic embedding retrieval and LLM-based verification
to calculate Tree-KG metrics: Entity Recall (ER), Precision (PC), and F1-score.

Key Optimizations:
1. Semantic Retrieval: Uses embeddings to find top-k candidates.
2. Waterfall Logic: Exact Match -> High Confidence -> Low Confidence -> LLM Fallback.
3. Concurrency: Uses ThreadPoolExecutor for parallel LLM requests.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import yaml
from openai import OpenAI
from tqdm import tqdm

from CeProAgents.groups.knowledge_group.knowledge_extraction.sem_embed import (
    embed_texts,
)


class KGExtractionEvaluator:
    """
    Evaluator for entity extraction performance using embedding similarity and LLM verification.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        client: Optional[OpenAI] = None,
        model: Optional[str] = None,
        top_k: Optional[int] = None,
        max_eval_default: Optional[int] = None,
        high_conf_thresh: Optional[float] = None,
        low_conf_thresh: Optional[float] = None,
        max_workers: int = 10,
    ) -> None:
        if config_path is None:
            this_dir = Path(__file__).resolve().parent
            config_path = this_dir / "config" / "entity_eval.yaml"

        config_path = str(config_path)
        
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        self._cfg = cfg
        self.model = model or cfg.get("model", "gpt-4o-mini")
        self.top_k = int(top_k if top_k is not None else cfg.get("top_k", 5))
        
        self.max_eval_default = (
            max_eval_default if max_eval_default is not None else cfg.get("max_eval", None)
        )

        self.temperature = float(cfg.get("temperature", 0.0))
        self.max_try_times = int(cfg.get("max_try_times", 3))
        self.max_workers = int(cfg.get("max_workers", max_workers))
        self.high_conf_thresh = (
            high_conf_thresh 
            if high_conf_thresh is not None 
            else float(cfg.get("high_conf_thresh", 0.98))
        )
        self.low_conf_thresh = (
            low_conf_thresh 
            if low_conf_thresh is not None 
            else float(cfg.get("low_conf_thresh", 0.65))
        )
        api_key = (
            cfg.get("api_key")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("ONEAPI_API_KEY")
        )
        base_url = cfg.get("base_url") or os.getenv("OPENAI_BASE_URL")

        if client is not None:
            self.client = client
        else:
            if not api_key:
                raise ValueError("KGExtractionEvaluator: API Key not found in config or environment variables.")
            
            if base_url:
                self.client = OpenAI(api_key=api_key, base_url=base_url)
            else:
                self.client = OpenAI(api_key=api_key)

        self.system_prompt: str = cfg.get("system_prompt", "").strip()
        self.eval_prompt_template: str = cfg.get("eval_prompt", "").strip()
        
        if not self.system_prompt or not self.eval_prompt_template:
            raise ValueError("Configuration error: 'system_prompt' or 'eval_prompt' missing in YAML.")

    def _embed(self, texts: List[str]) -> np.ndarray:
        """Wrapper for text embedding."""
        if not texts:
            return np.zeros((0, 1), dtype="float32")
        vec = embed_texts(texts)
        return np.asarray(vec, dtype=np.float32)

    def _topk_candidates(
        self,
        query_entities: List[str],
        candidate_entities: List[str],
    ) -> List[Tuple[str, List[Tuple[int, str, float]]]]:
        if not query_entities or not candidate_entities:
            return []

        print(
            f"[EVAL] Vector Search: #queries={len(query_entities)}, "
            f"#candidates={len(candidate_entities)} (Top-{self.top_k})"
        )

        q_emb = self._embed(query_entities)        # Shape: (Q, D)
        c_emb = self._embed(candidate_entities)    # Shape: (C, D)
        
        sim = np.matmul(q_emb, c_emb.T)            # Shape: (Q, C)

        results = []
        for i, q in enumerate(query_entities):
            row = sim[i]
            k = min(self.top_k, len(candidate_entities))
            
            if k == 0:
                results.append((q, []))
                continue

            top_idx = np.argpartition(-row, k - 1)[:k]
            top_idx = top_idx[np.argsort(-row[top_idx])]

            candidates = [
                (int(j), candidate_entities[int(j)], float(row[int(j)]))
                for j in top_idx
            ]
            results.append((q, candidates))

        return results

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Execute LLM completion with retry logic."""
        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_try_times + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.temperature,
                )
                content = resp.choices[0].message.content
                return content.strip() if content else ""
            except Exception as e:
                last_err = e
        
        # print(f"[EVAL] LLM call failed: {last_err}")
        return ""

    def _llm_judge_match(
        self,
        target_entity: str,
        candidates: List[Tuple[int, str, float]],
    ) -> Dict[str, Any]:
        """Core LLM logic to determine semantic equivalence."""
        if not candidates:
            return {"match": False, "matched_indices": [], "raw": ""}

        numbered_lines = []
        for idx, (_, cand, score) in enumerate(candidates, start=1):
            numbered_lines.append(f"{idx}. {cand}  (sim={score:.3f})")
        numbered_str = "\n".join(numbered_lines)

        user_prompt = self.eval_prompt_template.replace("{target}", target_entity)
        user_prompt = user_prompt.replace("{candidates}", numbered_str)

        content = self._call_llm(self.system_prompt, user_prompt)

        match = False
        matched_indices: List[int] = []

        try:
            data = json.loads(content)
            match = bool(data.get("match", False))
            idx_list = data.get("indices", []) or []

            matched_indices = [
                int(i) - 1
                for i in idx_list
                if isinstance(i, int) and 1 <= int(i) <= len(candidates)
            ]
            
            if match and not matched_indices:
                matched_indices = [0]
        except Exception:
            low_content = content.lower()
            if "true" in low_content or "yes" in low_content:
                match = True
                matched_indices = [0]

        return {
            "match": match,
            "matched_indices": matched_indices,
            "raw": content,
        }

    def _smart_judge(
        self,
        target_entity: str,
        candidates: List[Tuple[int, str, float]],
    ) -> Dict[str, Any]:
        if not candidates:
            return {"match": False, "matched_indices": [], "raw": "", "source": "empty"}
        target_norm = target_entity.strip().lower()
        for local_idx, (_, cand_str, _) in enumerate(candidates):
            if cand_str.strip().lower() == target_norm:
                return {
                    "match": True,
                    "matched_indices": [local_idx],
                    "raw": "Exact string match",
                    "source": "exact_match",
                }

        top_score = candidates[0][2]

        if top_score >= self.high_conf_thresh:
            return {
                "match": True,
                "matched_indices": [0],
                "raw": f"Score {top_score:.3f} >= threshold {self.high_conf_thresh}",
                "source": "high_conf_shortcut",
            }

        if top_score < self.low_conf_thresh:
            return {
                "match": False,
                "matched_indices": [],
                "raw": f"Score {top_score:.3f} < threshold {self.low_conf_thresh}",
                "source": "low_conf_shortcut",
            }
        res = self._llm_judge_match(target_entity, candidates)
        res["source"] = "llm"
        return res

    def _process_single_match(
        self,
        target: str,
        candidates: List[Tuple[int, str, float]]
    ) -> Tuple[Dict[str, Any], Optional[int]]:
        judge = self._smart_judge(target, candidates)
        
        best_global_idx = None
        if judge["match"] and judge["matched_indices"]:
            local_idx = judge["matched_indices"][0]
            if 0 <= local_idx < len(candidates):
                best_global_idx = candidates[local_idx][0]

        matched_strs = [
            candidates[idx][1] 
            for idx in judge["matched_indices"] 
            if 0 <= idx < len(candidates)
        ]

        detail = {
            "query_text": target,
            "candidates": [(c[1], round(c[2], 3)) for c in candidates],
            "match": bool(judge["match"]),
            "source": judge.get("source", "unknown"),
            "best_idx": best_global_idx,
            "matched_strs": matched_strs,
            "raw_judge": judge.get("raw", "")
        }
        return detail, best_global_idx

    def evaluate(
        self,
        gt_entities: List[str],
        extracted_entities: List[str],
        max_eval: Optional[int] = None,
    ) -> Dict[str, Any]:
        all_gt = list(gt_entities)
        all_pred = list(extracted_entities)
        num_gt_total = len(all_gt)
        num_pred_total = len(all_pred)

        # Apply evaluation limit
        if max_eval is None:
            max_eval = self.max_eval_default
        
        gt_eval = all_gt[:max_eval] if max_eval is not None else all_gt

        if not gt_eval or not all_pred:
            return {
                "num_gt": num_gt_total, "num_pred": num_pred_total,
                "num_eval_gt": len(gt_eval), "num_eval_pred": num_pred_total,
                "er": 0.0, "pc": 0.0, "f1": 0.0,
                "mapping_gt2pred": [None] * len(gt_eval),
                "mapping_pred2gt": [None] * num_pred_total,
                "details_gt2pred": [], "details_pred2gt": [],
            }

        print(f"[EVAL] Starting Parallel Evaluation (Threads={self.max_workers})...")
        topk_gt2pred = self._topk_candidates(gt_eval, all_pred)
        num_hit_gt = 0
        mapping_gt2pred = [None] * len(gt_eval)
        details_gt2pred = [None] * len(gt_eval)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_idx = {
                executor.submit(self._process_single_match, gt, cands): i 
                for i, (gt, cands) in enumerate(topk_gt2pred)
            }
            
            for future in tqdm(as_completed(future_to_idx), total=len(gt_eval), desc="ER (GT->Pred)"):
                idx = future_to_idx[future]
                try:
                    detail, best_idx = future.result()
                    # Fix key name for specific direction
                    detail["gt"] = detail.pop("query_text")
                    
                    details_gt2pred[idx] = detail
                    mapping_gt2pred[idx] = best_idx
                    if detail["match"]:
                        num_hit_gt += 1
                except Exception as exc:
                    print(f"[EVAL] Error processing GT item {idx}: {exc}")

        er = num_hit_gt / len(gt_eval) if len(gt_eval) > 0 else 0.0
        topk_pred2gt = self._topk_candidates(all_pred, all_gt)
        num_hit_pred = 0
        mapping_pred2gt = [None] * num_pred_total
        details_pred2gt = [None] * num_pred_total

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_idx = {
                executor.submit(self._process_single_match, pred, cands): i 
                for i, (pred, cands) in enumerate(topk_pred2gt)
            }
            
            for future in tqdm(as_completed(future_to_idx), total=num_pred_total, desc="PC (Pred->GT)"):
                idx = future_to_idx[future]
                try:
                    detail, best_idx = future.result()
                    # Fix key name for specific direction
                    detail["pred"] = detail.pop("query_text")
                    
                    details_pred2gt[idx] = detail
                    mapping_pred2gt[idx] = best_idx
                    if detail["match"]:
                        num_hit_pred += 1
                except Exception as exc:
                    print(f"[EVAL] Error processing Pred item {idx}: {exc}")

        pc = num_hit_pred / num_pred_total if num_pred_total > 0 else 0.0
        f1 = 0.0 if (er + pc) == 0 else 2 * er * pc / (er + pc)

        print(f"[EVAL] Complete. ER={er:.2%}, PC={pc:.2%}, F1={f1:.2%}")

        return {
            "num_gt": num_gt_total,
            "num_pred": num_pred_total,
            "num_eval_gt": len(gt_eval),
            "num_eval_pred": num_pred_total,
            "top_k": self.top_k,
            "er": er,
            "pc": pc,
            "f1": f1,
            "mapping_gt2pred": mapping_gt2pred,
            "mapping_pred2gt": mapping_pred2gt,
            "details_gt2pred": details_gt2pred,
            "details_pred2gt": details_pred2gt,
        }