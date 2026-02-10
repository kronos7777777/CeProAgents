# CeProAgents/eval/kg_graph_metrics.py
"""
Tree-KG style graph-structure evaluation metrics:
- MEC: Mapping-based Edge Connectivity
- MED: Mapping-based Edge Distance (normalized by avg shortest path distance in predicted graph)

Inputs:
- gt_entities: List[str]
- gt_edges: List[Tuple[int,int]]  (edge endpoints are indices in gt_entities, 0-based)
- extracted_entities: List[str]
- pred_triplets: List[[h,r,t]] or List[{"subject","relation","object"}]
- mapping_gt2pred: List[Optional[int]] length == len(gt_entities), mapping GT idx -> predicted entity idx (0-based) or None

Notes:
- For MEC, a GT edge counts as preserved if both endpoints map to predicted nodes and are connected by some path in predicted graph.
- For MED, Tree-KG defines shortest-path distance between mapped pairs and normalizes by avg shortest path distance in predicted graph.
  For disconnected graphs, MED can be ill-defined; we compute MED over *mapped+connected* GT edges and also return coverage.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


Triplet = Union[
    Sequence[str],                 # [h, r, t]
    Dict[str, str],                # {"subject","relation","object"}
]


def _triplet_to_ht(t: Triplet) -> Tuple[Optional[str], Optional[str]]:
    try:
        if isinstance(t, dict):
            h = t.get("subject")
            tail = t.get("object")
            return h, tail
        # assume list/tuple
        if len(t) >= 3:
            return t[0], t[2]
    except Exception:
        pass
    return None, None


def build_adjacency_from_triplets(
    entities: List[str],
    triplets: List[Triplet],
    directed: bool = False,
) -> List[List[int]]:
    idx = {e: i for i, e in enumerate(entities)}
    adj: List[List[int]] = [[] for _ in range(len(entities))]

    def add_edge(u: int, v: int):
        adj[u].append(v)

    for t in triplets or []:
        h, tail = _triplet_to_ht(t)
        if h is None or tail is None:
            continue
        if h not in idx or tail not in idx:
            continue
        u = idx[h]
        v = idx[tail]
        add_edge(u, v)
        if not directed:
            add_edge(v, u)

    for i in range(len(adj)):
        if adj[i]:
            adj[i] = list(dict.fromkeys(adj[i]))
    return adj


def connected_components(adj: List[List[int]]) -> List[int]:
    n = len(adj)
    comp = [-1] * n
    cid = 0
    for s in range(n):
        if comp[s] != -1:
            continue
        # isolated ok
        q = deque([s])
        comp[s] = cid
        while q:
            u = q.popleft()
            for v in adj[u]:
                if comp[v] == -1:
                    comp[v] = cid
                    q.append(v)
        cid += 1
    return comp


def shortest_path_length(adj: List[List[int]], src: int, dst: int) -> Optional[int]:
    if src == dst:
        return 0
    n = len(adj)
    dist = [-1] * n
    q = deque([src])
    dist[src] = 0
    while q:
        u = q.popleft()
        du = dist[u]
        for v in adj[u]:
            if dist[v] != -1:
                continue
            dist[v] = du + 1
            if v == dst:
                return dist[v]
            q.append(v)
    return None


def avg_shortest_path_distance(
    adj: List[List[int]],
    sample_size: int = 200,
) -> Optional[float]:
    n = len(adj)
    if n <= 1:
        return None

    sources = [i for i in range(n) if adj[i]]
    if not sources:
        return None
    sources = sources[: min(sample_size, len(sources))]

    total = 0
    cnt = 0

    for s in sources:
        dist = [-1] * n
        q = deque([s])
        dist[s] = 0
        while q:
            u = q.popleft()
            du = dist[u]
            for v in adj[u]:
                if dist[v] != -1:
                    continue
                dist[v] = du + 1
                q.append(v)
        for d in dist:
            if d > 0:
                total += d
                cnt += 1

    if cnt == 0:
        return None
    return total / cnt


@dataclass
class GraphMetricResult:
    mec: float
    med: Optional[float]
    med_coverage: float
    d_bar: Optional[float]
    num_gt_edges: int
    num_mapped_connected_edges: int
    num_mapped_edges: int


def compute_mec_med(
    gt_edges: List[Tuple[int, int]],
    mapping_gt2pred: List[Optional[int]],
    pred_adj: List[List[int]],
    med_sample_size: int = 200,
) -> GraphMetricResult:
    num_gt_edges = len(gt_edges)
    if num_gt_edges == 0:
        return GraphMetricResult(
            mec=0.0,
            med=None,
            med_coverage=0.0,
            d_bar=None,
            num_gt_edges=0,
            num_mapped_connected_edges=0,
            num_mapped_edges=0,
        )

    comp = connected_components(pred_adj)

    mapped_edges = 0
    mapped_connected_edges = 0

    d_bar = avg_shortest_path_distance(pred_adj, sample_size=med_sample_size)
    dist_sum_norm = 0.0
    dist_cnt = 0

    for (gu, gv) in gt_edges:
        pu = mapping_gt2pred[gu] if 0 <= gu < len(mapping_gt2pred) else None
        pv = mapping_gt2pred[gv] if 0 <= gv < len(mapping_gt2pred) else None
        if pu is None or pv is None:
            continue

        mapped_edges += 1

        if 0 <= pu < len(comp) and 0 <= pv < len(comp) and comp[pu] == comp[pv]:
            mapped_connected_edges += 1
            if d_bar is not None and d_bar > 0:
                d = shortest_path_length(pred_adj, pu, pv)
                if d is not None:
                    dist_sum_norm += (d / d_bar)
                    dist_cnt += 1

    mec = mapped_connected_edges / num_gt_edges

    med = (dist_sum_norm / dist_cnt) if dist_cnt > 0 else None
    med_coverage = mapped_connected_edges / num_gt_edges

    return GraphMetricResult(
        mec=mec,
        med=med,
        med_coverage=med_coverage,
        d_bar=d_bar,
        num_gt_edges=num_gt_edges,
        num_mapped_connected_edges=mapped_connected_edges,
        num_mapped_edges=mapped_edges,
    )
