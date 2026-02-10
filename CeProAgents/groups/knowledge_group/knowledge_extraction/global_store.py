# src/kg_extraction/global_store.py
# -*- coding: utf-8 -*-
"""
Global assets for multi-document entity disambiguation:
- FAISS vector index (global_entities.faiss)
- Sidecar SQLite (global_alias.sqlite): alias_map & vectors metadata
- JSON meta (global_meta.json): embedding/model/config info

Defaults:
  base_dir = <this_file_dir>/indices
  files   = global_entities.faiss / global_meta.json / global_alias.sqlite
Override via env:
  MERGE_INDEX_LOAD, MERGE_INDEX_SAVE, INDEX_META_PATH, ALIAS_DB_PATH
  FAISS_TYPE=flat|hnsw|ivf, FAISS_THREADS, FAISS_SEARCH_BATCH
"""

from __future__ import annotations

import os
import re
import json
import time
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterable

import numpy as np

# ---- Project-local imports (embedding helpers) ----
# embed_texts: returns L2-normalized float vectors (shape: [N, D])
# cosine_matrix: used only for diagnostics elsewhere (not required here)
from .sem_embed import embed_texts  # type: ignore


# =========================
# Path & Env Helpers
# =========================

_THIS_DIR = Path(__file__).resolve().parent
_DEFAULT_INDICES_DIR = _THIS_DIR / "indices"
_DEFAULT_INDEX_PATH = _DEFAULT_INDICES_DIR / "global_entities.faiss"
_DEFAULT_META_PATH = _DEFAULT_INDICES_DIR / "global_meta.json"
_DEFAULT_DB_PATH = _DEFAULT_INDICES_DIR / "global_alias.sqlite"


def _get_paths_from_env() -> Dict[str, str]:
    """Read paths from env or fall back to defaults inside src/kg_extraction/indices/"""
    base_dir = _DEFAULT_INDICES_DIR
    base_dir.mkdir(parents=True, exist_ok=True)

    idx_load = os.getenv("MERGE_INDEX_LOAD", str(_DEFAULT_INDEX_PATH))
    # 保存路径默认与读取路径一致（可单独覆盖 MERGE_INDEX_SAVE）
    idx_save = os.getenv("MERGE_INDEX_SAVE", idx_load)
    meta = os.getenv("INDEX_META_PATH", str(_DEFAULT_META_PATH))
    db = os.getenv("ALIAS_DB_PATH", str(_DEFAULT_DB_PATH))

    # Ensure parent dirs
    Path(idx_load).parent.mkdir(parents=True, exist_ok=True)
    Path(idx_save).parent.mkdir(parents=True, exist_ok=True)
    Path(meta).parent.mkdir(parents=True, exist_ok=True)
    Path(db).parent.mkdir(parents=True, exist_ok=True)

    return {
        "index_load": idx_load,
        "index_save": idx_save,
        "meta": meta,
        "db": db,
    }


# =========================
# Text Normalization
# =========================

def norm_text(s: str) -> str:
    """Normalize surface string for alias_map keying (lowercase, trim, simple punctuation unification)."""
    if not s:
        return ""
    t = s.strip().lower()
    t = t.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    t = t.replace("–", "-").replace("—", "-").replace("‒", "-")
    t = re.sub(r"\s+", " ", t)             # collapse spaces
    t = re.sub(r"\s*-\s*", "-", t)         # normalize hyphen spacing
    # unify parentheses spacing but keep content
    t = re.sub(r"\(\s*(.*?)\s*\)", r"(\1)", t)
    return t


# =========================
# SQLite Sidecar (alias/vectors)
# =========================

_DDL = """
CREATE TABLE IF NOT EXISTS alias_map (
  surface_norm TEXT PRIMARY KEY,
  canonical_id TEXT NOT NULL,
  decision_source TEXT NOT NULL,  -- exact_name | prior_llm_approved | internal_llm_approved | global_llm_approved | manual_override | heuristic
  ts INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_alias_canonical ON alias_map(canonical_id);

CREATE TABLE IF NOT EXISTS vectors (
  vec_id INTEGER PRIMARY KEY,
  canonical_id TEXT NOT NULL,
  surface TEXT NOT NULL,
  ts INTEGER NOT NULL,
  alive INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_vectors_canonical ON vectors(canonical_id);
"""

# ---- Alias source priority (higher wins) ----
ALIAS_SOURCE_PRIORITY = {
    "exact_name": 0,
    "heuristic": 1,
    "prior_llm_approved": 2,
    "internal_llm_approved": 3,
    "global_llm_approved": 4,
    "manual_override": 5,
}



class AliasDB:
    """Sidecar DB for alias memory and vector metadata."""
    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self.conn.executescript(_DDL)

    # --- alias_map ---

    def get_alias(self, surface_norm: str) -> Optional[Tuple[str, str]]:
        """Return (canonical_id, decision_source) or None."""
        cur = self.conn.execute(
            "SELECT canonical_id, decision_source FROM alias_map WHERE surface_norm=?",
            (surface_norm,)
        )
        row = cur.fetchone()
        return (row[0], row[1]) if row else None

    def upsert_alias(self, surface_norm: str, canonical_id: str, source: str) -> bool:
        """
        带优先级的 UPSERT：
        - 低优先级不会覆盖高优先级
        - 同级且 canonical 不同不覆盖（避免抖动）
        返回 True 表示写入/更新生效；False 表示被跳过（保持旧值）
        """
        ts = int(time.time())
        cur = self.conn.execute(
            "SELECT canonical_id, decision_source FROM alias_map WHERE surface_norm=?",
            (surface_norm,)
        )
        row = cur.fetchone()
        new_p = ALIAS_SOURCE_PRIORITY.get(source, 0)
        if row:
            old_cid, old_src = row
            old_p = ALIAS_SOURCE_PRIORITY.get(old_src, 0)

            # 高优先级保留（禁止降级覆盖）
            if old_p > new_p:
                return False

            # 同级不同 canonical，保守不改，避免左右摇摆
            if old_p == new_p and old_cid != canonical_id:
                return False

        # 插入或升级覆盖
        self.conn.execute(
            """
            INSERT INTO alias_map(surface_norm, canonical_id, decision_source, ts)
            VALUES(?,?,?,?)
            ON CONFLICT(surface_norm) DO UPDATE SET
                canonical_id=excluded.canonical_id,
                decision_source=excluded.decision_source,
                ts=excluded.ts
            """,
            (surface_norm, canonical_id, source, ts)
        )
        self.conn.commit()
        return True
    
    def has_vector_for_surface(self, surface: str) -> bool:
        """
        判断某个 surface 是否已经有一条处于 alive=1 的向量元数据，
        有则返回 True（无需再追加向量）。
        """
        cur = self.conn.execute(
            "SELECT 1 FROM vectors WHERE surface=? AND alive=1 LIMIT 1",
            (surface,)
        )
        return cur.fetchone() is not None


    def upsert_many_aliases(self, rows: Iterable[Tuple[str, str, str]]) -> int:
        """
        批量写入（逐条走 upsert_alias 的优先级规则）
        rows: iterable of (surface_norm, canonical_id, source)
        返回成功写入/更新的条数
        """
        n = 0
        for sn, cid, src in rows:
            if sn:
                n += 1 if self.upsert_alias(sn, cid, src) else 0
        return n
    
    def upsert_many_aliases_tx(self, rows: Iterable[Tuple[str, str, str]]) -> int:
        """
        批量 UPSERT（带优先级），单事务提交。
        rows: (surface_norm, canonical_id, source)
        返回最终实际写入/更新条数
        """
        rows = [(sn, cid, src) for (sn, cid, src) in rows if sn]
        if not rows:
            return 0
        ts = int(time.time())
        # 先把现有的拉一遍用于优先级判定
        sns = [sn for sn, _, _ in rows]
        placeholders = ",".join("?" * len(sns))
        cur = self.conn.execute(
            f"SELECT surface_norm, canonical_id, decision_source FROM alias_map WHERE surface_norm IN ({placeholders})",
            sns
        )
        exist = {sn: (cid, src) for (sn, cid, src) in cur.fetchall()}

        to_write = []
        for sn, cid, src in rows:
            new_p = ALIAS_SOURCE_PRIORITY.get(src, 0)
            old = exist.get(sn)
            if old:
                old_cid, old_src = old
                old_p = ALIAS_SOURCE_PRIORITY.get(old_src, 0)
                if old_p > new_p:
                    continue
                if old_p == new_p and old_cid != cid:
                    continue
            to_write.append((sn, cid, src, ts))

        if not to_write:
            return 0

        with self.conn:  # 单事务
            self.conn.executemany(
                """
                INSERT INTO alias_map(surface_norm, canonical_id, decision_source, ts)
                VALUES(?,?,?,?)
                ON CONFLICT(surface_norm) DO UPDATE SET
                    canonical_id=excluded.canonical_id,
                    decision_source=excluded.decision_source,
                    ts=excluded.ts
                """,
                to_write
            )
        return len(to_write)


    def upsert_alias_group(self, canonical_label: str, surfaces: Iterable[str], decision_source: str) -> int:
        """
        一次写入“规范名 + 全部同义词”
        - 会先确保 canonical_label 的自名映射（exact_name）
        - 对于 surfaces 中等于规范名的条目自动跳过
        返回成功写入/更新的条数
        """
        if not canonical_label:
            return 0
        cano = canonical_label
        cano_norm = norm_text(cano)

        written = 0
        # 先确保规范名自身的自名映射（不会覆盖更高优先级）
        written += 1 if self.upsert_alias(cano_norm, cano, "exact_name") else 0

        for s in (surfaces or []):
            sn = norm_text(s)
            if not sn or sn == cano_norm:
                continue
            written += 1 if self.upsert_alias(sn, cano, decision_source) else 0

        return written

    def get_aliases_for(self, canonical_label: str, include_canonical: bool = True) -> List[str]:
        """
        返回某规范名下的全部 alias（surface_norm 列表）。
        include_canonical=True 时包含规范名自身（若存在自名映射）。
        """
        if not canonical_label:
            return []
        cur = self.conn.execute(
            "SELECT surface_norm FROM alias_map WHERE canonical_id=?",
            (canonical_label,)
        )
        al = [r[0] for r in cur.fetchall()]
        if not include_canonical:
            cn = norm_text(canonical_label)
            al = [x for x in al if x != cn]
        return al

    # --- vectors metadata ---

    def add_vectors_meta(self, rows: Iterable[Tuple[int, str, str]]) -> None:
        """
        rows: list of (vec_id, canonical_id, surface)
        """
        ts = int(time.time())
        self.conn.executemany(
            "INSERT OR REPLACE INTO vectors(vec_id, canonical_id, surface, ts, alive) "
            "VALUES(?,?,?,?,1)",
            [(vid, cid, s, ts) for (vid, cid, s) in rows]
        )
        self.conn.commit()

    def tombstone_vecs_by_canonical(self, canonical_id: str) -> None:
        """Mark vectors of a canonical as not alive (soft delete)."""
        self.conn.execute("UPDATE vectors SET alive=0 WHERE canonical_id=?", (canonical_id,))
        self.conn.commit()

    def get_alive_vecs_for_canonical(self, canonical_id: str) -> List[int]:
        cur = self.conn.execute(
            "SELECT vec_id FROM vectors WHERE canonical_id=? AND alive=1 ORDER BY vec_id ASC",
            (canonical_id,)
        )
        return [r[0] for r in cur.fetchall()]

    def close(self) -> None:
        self.conn.close()



# =========================
# FAISS Utilities
# =========================

def ensure_float32_contig(X: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(X, dtype="float32")


def set_faiss_threads(threads: Optional[int] = None) -> int:
    """Set FAISS OMP threads. If None/<=0, use os.cpu_count(). Return threads used."""
    try:
        import faiss  # type: ignore
        th = threads if threads and threads > 0 else (os.cpu_count() or 1)
        faiss.omp_set_num_threads(th)
        return th
    except Exception:
        return 1


def load_or_create_index(
    index_path: str,
    dim: int,
    faiss_type: str = "flat",
    hnsw_cfg: Optional[Dict] = None,
    ivf_cfg: Optional[Dict] = None
):
    """
    Load if exists, else create a new index per faiss_type.
    NOTE: For IVF, training is required before add(); use train_index_ivf() with samples.
    """
    import faiss  # type: ignore
    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    if os.path.exists(index_path):
        return faiss.read_index(index_path)

    ft = (faiss_type or "flat").lower()
    if ft == "hnsw":
        M = int((hnsw_cfg or {}).get("M", 32))
        idx = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
        # efConstruction/efSearch set on idx.hnsw.* by caller if needed
        return idx
    elif ft == "ivf":
        nlist = int((ivf_cfg or {}).get("nlist", 1024))
        quant = faiss.IndexFlatIP(dim)
        idx = faiss.IndexIVFFlat(quant, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        return idx
    else:
        return faiss.IndexFlatIP(dim)


def train_index_ivf(index, train_X: np.ndarray, niter: int = 25) -> None:
    """Train IVF index; safe no-op if already trained."""
    import faiss  # type: ignore
    if hasattr(index, "is_trained") and not index.is_trained:
        index.train(ensure_float32_contig(train_X).astype("float32"))
        # (optional) niter tuning via faiss parameters if needed


def save_index(index, index_path: str) -> None:
    import faiss  # type: ignore
    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, index_path)


# =========================
# Meta JSON
# =========================

def read_meta(meta_path: str) -> Dict:
    if os.path.exists(meta_path):
        try:
            return json.loads(Path(meta_path).read_text("utf-8"))
        except Exception:
            return {}
    return {}


def write_meta(meta_path: str, meta: Dict) -> None:
    Path(meta_path).parent.mkdir(parents=True, exist_ok=True)
    Path(meta_path).write_text(json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")


def get_or_probe_dim() -> int:
    """Probe embedding dim by embedding a dummy token."""
    vec = embed_texts(["__probe__"])
    return int(vec.shape[1])


# =========================
# High-level Open/Close
# =========================

def open_global_assets() -> Tuple[object, AliasDB, Dict, Dict[str, str]]:
    """
    Open index & sidecar DB & meta with defaults/ENV overrides.
    Returns: (index, db, meta, paths)
    """
    paths = _get_paths_from_env()
    meta = read_meta(paths["meta"])
    dim = int(meta.get("dimension", 0)) or get_or_probe_dim()

    faiss_type = os.getenv("FAISS_TYPE", meta.get("faiss_type", "flat"))
    index = load_or_create_index(paths["index_load"], dim, faiss_type=faiss_type)

    # Optional: set HNSW/IVF runtime params from env
    try:
        import faiss  # type: ignore
        if faiss_type.lower() == "hnsw":
            efC = int(os.getenv("FAISS_EF_CONSTRUCTION", "200"))
            efS = int(os.getenv("FAISS_EF_SEARCH", "64"))
            if hasattr(index, "hnsw"):
                index.hnsw.efConstruction = efC
                index.hnsw.efSearch = efS
    except Exception:
        print("[WARN] Failed to set FAISS HNSW parameters from env.")

    # Ensure meta basics
    if "embedding_model" not in meta:
        meta["embedding_model"] = os.getenv(
            "EMBEDDING_MODEL",
            "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
        )
    meta["dimension"] = dim
    meta["l2_normalized"] = True
    meta["faiss_type"] = faiss_type

    # Persist meta (idempotent)
    write_meta(paths["meta"], meta)

    # Open DB
    db = AliasDB(paths["db"])

    # Threads
    set_faiss_threads(int(os.getenv("FAISS_THREADS", "0")))

    return index, db, meta, paths


def save_and_close(index, db: AliasDB, paths: Optional[Dict[str, str]] = None) -> None:
    """Save index to MERGE_INDEX_SAVE (or load path), and close DB."""
    if paths is None:
        paths = _get_paths_from_env()
    save_to = paths.get("index_save") or paths.get("index_load")
    save_index(index, save_to)
    db.close()


# =========================
# Batch Add & Alias Helpers
# =========================

def add_aliases_to_index(
    index,
    db: AliasDB,
    pairs: List[Tuple[str, str]],
    decision_source: str = "exact_name",
    write_alias=True
) -> int:
    """
    Append alias surfaces to index and write alias_map & vectors metadata.

    pairs: list of (canonical_id, surface_str)

    Returns: number of added vectors.
    """
    if not pairs:
        return 0

    surfaces = [s for (_, s) in pairs]
    X = embed_texts(surfaces)          # already L2-normalized
    X = ensure_float32_contig(X)       # (N, D) float32 contiguous

    start = int(getattr(index, "ntotal", 0))
    index.add(X)
    added = X.shape[0]

    # vectors metadata
    vec_rows = []
    for k, (cid, s) in enumerate(pairs):
        vec_rows.append((start + k, cid, s))
    db.add_vectors_meta(vec_rows)

    if write_alias:
        alias_rows = [(norm_text(s), cid, decision_source) for cid, s in pairs]
        db.upsert_many_aliases_tx(alias_rows)   # 受开关控制
    return added


# src/kg_extraction/global_store.py

def quick_alias_or_exact(db: AliasDB, name: str) -> Optional[Tuple[str, str]]:
    """
    Fast path: try alias_map by normalized exact name.
    Returns: (canonical_id, decision_source) or None
    """
    nm = norm_text(name)
    row = db.get_alias(nm)

    # === 调试打印（可选）===
    if os.getenv("MERGE_DEBUG_ALIAS", "0") == "1":
        if row is None:
            print(f"[ALIAS-LOOKUP] miss  query={repr(name)}  norm={repr(nm)}")
        else:
            cano, src = row
            print(f"[ALIAS-LOOKUP] HIT   query={repr(name)}  norm={repr(nm)}  -> canonical={repr(cano)}  source={src}")
    # ======================

    return row



# =========================
# Batch Search Utilities
# =========================

def batch_search_matrix(
    index,
    Xq: np.ndarray,
    topk: int,
    batch: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Search in batches for a query matrix Xq (already float32 contiguous, L2-normalized).
    Returns: (D, I) of shape (nq, topk)
    """
    Xq = ensure_float32_contig(Xq)
    nq = Xq.shape[0]
    K = int(topk)

    if batch is None or batch <= 0:
        batch = int(os.getenv("FAISS_SEARCH_BATCH", "512"))

    D_all, I_all = [], []
    for i0 in range(0, nq, batch):
        i1 = min(nq, i0 + batch)
        D, I = index.search(Xq[i0:i1], K)
        D_all.append(D)
        I_all.append(I)
    return np.vstack(D_all), np.vstack(I_all)


def batch_search_texts(
    index,
    texts: List[str],
    topk: int,
    batch: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience wrapper: embed texts then batch_search.
    """
    Xq = embed_texts(texts)  # L2-normalized
    return batch_search_matrix(index, Xq, topk=topk, batch=batch)


# =========================
# LLM-confirmed alias write-back
# =========================

def record_llm_equivalences(
    index,
    db: AliasDB,
    canonical_id: str,
    alias_names: List[str],
    also_add_vectors: bool = True,
    write_alias=True
) -> int:
    """
    After LLM confirms these alias names belong to canonical_id,
    write alias_map(decision_source=prior_llm_approved), and optionally add vectors to index.

    Returns: number of vectors added (if also_add_vectors=True).
    """
    # Update alias_map
    if write_alias:
        rows = [(norm_text(s), canonical_id, "prior_llm_approved") for s in alias_names]
        db.upsert_many_aliases_tx(rows)
    if also_add_vectors:
        pairs = [(canonical_id, s) for s in alias_names]
        return add_aliases_to_index(index, db, pairs, decision_source="prior_llm_approved", write_alias=write_alias)
    return 0
