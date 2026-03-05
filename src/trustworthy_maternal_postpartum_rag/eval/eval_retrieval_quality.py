# src/trustworthy_maternal_postpartum_rag/eval/eval_retrieval_quality.py

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# -------------------------
# PATH-DRIVEN CONFIG
# -------------------------
EVAL_RUNS_DIR = Path("eval_runs")
INPUT_LOG_PATH = Path("eval_runs/eval_run.jsonl")

OUTPUT_REPORT_PATH = Path("eval_runs/retrieval_quality_report.json")
OUTPUT_CASES_PATH = Path("eval_runs/retrieval_quality_cases.jsonl")

# Tokenization and stopwords (minimal, deterministic)
STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "is", "are", "was", "were",
    "be", "been", "being", "it", "this", "that", "these", "those", "i", "me", "my", "we", "you", "your",
    "he", "she", "they", "them", "as", "at", "by", "from", "after", "before", "during", "when", "what",
    "how", "can", "could", "should", "would", "do", "does", "did",
}


def _latest_timestamped_log(eval_dir: Path) -> Optional[Path]:
    if not eval_dir.exists():
        return None
    candidates = sorted(eval_dir.glob("eval_run_*.jsonl"))
    return candidates[-1] if candidates else None


def _resolve_input_log_path() -> Path:
    latest = _latest_timestamped_log(EVAL_RUNS_DIR)
    if latest is not None:
        return latest
    return INPUT_LOG_PATH


def _read_jsonl_all(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing log file: {path}")
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            out.append(json.loads(s))
    return out


def _scope_to_last_run_meta(objs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    last_meta_idx = -1
    for i, o in enumerate(objs):
        if o.get("type") == "run_meta":
            last_meta_idx = i
    if last_meta_idx == -1:
        return [o for o in objs if o.get("type") == "result"]
    return [o for o in objs[last_meta_idx + 1 :] if o.get("type") == "result"]


def _tokens(text: str) -> List[str]:
    toks = re.findall(r"[A-Za-z0-9]+", (text or "").lower())
    return [t for t in toks if t and t not in STOPWORDS and len(t) >= 2]


def _jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


def _pairwise_max_similarity(texts: List[str]) -> float:
    # For k=4 this is trivial; still safe for small lists
    toks = [_tokens(t) for t in texts]
    best = 0.0
    for i in range(len(toks)):
        for j in range(i + 1, len(toks)):
            best = max(best, _jaccard(toks[i], toks[j]))
    return float(best)


def _publisher_stats(evidence: List[Dict[str, Any]]) -> Tuple[int, Optional[float]]:
    pubs = [(e.get("publisher") or "").strip() for e in evidence]
    pubs = [p for p in pubs if p and p != "UNKNOWN"]
    if not pubs:
        return 0, None
    counts: Dict[str, int] = {}
    for p in pubs:
        counts[p] = counts.get(p, 0) + 1
    distinct = len(counts)
    dom_share = max(counts.values()) / max(1, len(pubs))
    return distinct, float(dom_share)


def _meta_missing_counts(evidence: List[Dict[str, Any]]) -> Dict[str, int]:
    missing = {"publisher": 0, "source_file": 0, "page_number": 0, "text": 0}
    for e in evidence:
        if not (e.get("publisher") or "").strip():
            missing["publisher"] += 1
        if not (e.get("source_file") or "").strip():
            missing["source_file"] += 1
        if e.get("page_number", None) in (None, -1, ""):
            missing["page_number"] += 1
        if not (e.get("text") or "").strip():
            missing["text"] += 1
    return missing


def _query_coverage(question: str, evidence_texts: List[str]) -> float:
    qt = set(_tokens(question))
    if not qt:
        return 0.0
    blob = " ".join(evidence_texts).lower()
    hit = 0
    for t in qt:
        if t in blob:
            hit += 1
    return hit / max(1, len(qt))


def main() -> None:
    log_path = _resolve_input_log_path()
    objs = _read_jsonl_all(log_path)
    rows = _scope_to_last_run_meta(objs)

    case_out: List[Dict[str, Any]] = []

    distinct_pubs_list: List[float] = []
    dom_share_list: List[float] = []
    redundancy_list: List[float] = []
    qcov_list: List[float] = []
    meta_missing_total: List[float] = []

    for r in rows:
        idx = r.get("idx")
        question = r.get("question", "")
        audit = r.get("audit", {}) or {}
        lifecycle = str(audit.get("lifecycle", ""))
        topic = str(audit.get("topic", ""))

        evidence = r.get("evidence", []) or []
        if not isinstance(evidence, list):
            evidence = []

        ev_texts = [(e.get("text") or "").strip() for e in evidence if isinstance(e, dict)]
        ev_texts = [t for t in ev_texts if t]

        distinct_pubs, dom_share = _publisher_stats([e for e in evidence if isinstance(e, dict)])
        redundancy = _pairwise_max_similarity(ev_texts) if ev_texts else 0.0
        qcov = _query_coverage(question, ev_texts) if ev_texts else 0.0
        miss = _meta_missing_counts([e for e in evidence if isinstance(e, dict)])
        miss_total = float(sum(miss.values()))

        distinct_pubs_list.append(float(distinct_pubs))
        if dom_share is not None:
            dom_share_list.append(float(dom_share))
        redundancy_list.append(float(redundancy))
        qcov_list.append(float(qcov))
        meta_missing_total.append(miss_total)

        case_out.append(
            {
                "idx": idx,
                "lifecycle": lifecycle,
                "topic": topic,
                "n_evidence": len(evidence),
                "distinct_publishers": distinct_pubs,
                "publisher_dominance_share": dom_share,
                "max_pairwise_redundancy": redundancy,
                "query_token_coverage": qcov,
                "missing_meta_counts": miss,
            }
        )

    report = {
        "input_log_path": str(log_path),
        "n_results": len(rows),
        "mean_distinct_publishers": float(np.mean(distinct_pubs_list)) if distinct_pubs_list else 0.0,
        "mean_publisher_dominance_share": float(np.mean(dom_share_list)) if dom_share_list else None,
        "mean_max_pairwise_redundancy": float(np.mean(redundancy_list)) if redundancy_list else 0.0,
        "mean_query_token_coverage": float(np.mean(qcov_list)) if qcov_list else 0.0,
        "mean_missing_meta_total": float(np.mean(meta_missing_total)) if meta_missing_total else 0.0,
        "notes": [
            "Log-only retrieval-quality evaluation (no retrieval/LLM calls).",
            "Redundancy is max pairwise token-Jaccard across evidence chunk texts.",
            "Query coverage is fraction of query tokens found in concatenated evidence text.",
        ],
    }

    OUTPUT_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with OUTPUT_CASES_PATH.open("w", encoding="utf-8") as f:
        for c in case_out:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
