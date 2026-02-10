from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import networkx as nx


def _safe_stem(name: str) -> str:
    """
    把 doc_id / 文件名变成 Windows 也安全的 stem。
    """
    s = (name or "doc").strip()
    s = s.replace("\\", "_").replace("/", "_")
    s = re.sub(r"[<>:\"|?*\x00-\x1F]", "_", s)  # Windows 非法字符
    s = re.sub(r"\s+", "_", s)
    s = s.strip("._")
    return s or "doc"


def export_pred_kg(
    doc_id: str,
    entities: List[str],
    triplets: List[Dict[str, str]],
    out_dir: Optional[str] = None,
    fmt: Optional[str] = None,
) -> Dict[str, str]:
    """
    将“预测KG（本次抽取结果）”导出到本地文件。

    环境变量（都可选）：
      - EXPORT_PRED_KG=1            是否导出（默认 1）
      - PRED_KG_DIR=./outputs/pred_kgs
      - PRED_KG_FORMAT=json|graphml|both  （默认 both）

    返回：{"json": "...", "graphml": "..."}（按实际写出的内容返回）
    """
    enabled = os.getenv("EXPORT_PRED_KG", "1") == "1"
    if not enabled:
        return {}

    out_dir = out_dir or os.getenv("PRED_KG_DIR", "./outputs/pred_kgs")
    fmt = (fmt or os.getenv("PRED_KG_FORMAT", "both")).lower().strip()

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    stem_name = Path(doc_id).stem
    stem = _safe_stem(stem_name)
    wrote: Dict[str, str] = {}

    # ---- 1) JSON 导出（推荐一定留一个，调试最方便）----
    if fmt in ("json", "both"):
        json_fp = out_path / f"{stem}_pred.json"
        payload = {
            "doc_id": doc_id,
            "entities": list(entities or []),
            "triplets": list(triplets or []),  # [{"subject","relation","object"}, ...]
        }
        json_fp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        wrote["json"] = str(json_fp.resolve())

    # ---- 2) GraphML 导出（networkx 可直接读写，用于图指标最方便）----
    if fmt in ("graphml", "both"):
        g = nx.MultiDiGraph()
        # 加节点：优先用 entities；同时兜底把 triplets 里出现的节点也加上
        for e in (entities or []):
            if e and str(e).strip():
                g.add_node(str(e).strip())

        for t in (triplets or []):
            h = (t.get("subject") or "").strip()
            r = (t.get("relation") or "").strip()
            o = (t.get("object") or "").strip()
            if not h or not o:
                continue
            # MultiDiGraph 允许同一对节点多条边，不会互相覆盖
            g.add_edge(h, o, relation=r)

        graphml_fp = out_path / f"{stem}_pred.graphml"
        nx.write_graphml(g, graphml_fp)
        wrote["graphml"] = str(graphml_fp.resolve())

    return wrote
