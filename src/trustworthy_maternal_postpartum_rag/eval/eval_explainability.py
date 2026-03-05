from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# -------------------------
# PATH-DRIVEN CONFIG
# -------------------------
EVAL_RUNS_DIR = Path("eval_runs")

# Backward compatible default; script will auto-select latest timestamped run if present
INPUT_LOG_PATH = Path("eval_runs/eval_run.jsonl")

OUTPUT_REPORT_PATH = Path("eval_runs/explainability_report.json")
OUTPUT_CASES_PATH = Path("eval_runs/explainability_cases.jsonl")

SENT_SUPPORT_THRESHOLD = 0.10  # lexical score in [0,1]

# If a citation.supports string does not overlap the cited chunk text at least this much,
# treat it as unreliable and fall back to chunk text.
SUPPORTS_VALIDATION_THRESHOLD = 0.08

# Generic/label-like supports strings that often indicate hallucinated placeholders.
_GENERIC_SUPPORT_PATTERNS = [
    r"^\s*results?\b",
    r"^\s*evidence\b",
    r"^\s*study\b",
    r"^\s*information\b",
    r"^\s*guidance\b",
    r"^\s*recommendations?\b",
]


def _split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return []
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def support_score(sentence: str, evidence_text: str) -> float:
    s = re.findall(r"[A-Za-z0-9]+", (sentence or "").lower())
    e = re.findall(r"[A-Za-z0-9]+", (evidence_text or "").lower())
    if not s or not e:
        return 0.0

    s_set, e_set = set(s), set(e)
    j = len(s_set & e_set) / max(1, len(s_set | e_set))  # token Jaccard

    s_bi = set(zip(s, s[1:])) if len(s) >= 2 else set()
    e_bi = set(zip(e, e[1:])) if len(e) >= 2 else set()
    b = (len(s_bi & e_bi) / max(1, len(s_bi))) if s_bi else 0.0  # bigram recall

    return float(0.4 * j + 0.6 * b)


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
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _scope_to_last_run_meta(objs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    last_meta_idx = -1
    for i, o in enumerate(objs):
        if o.get("type") == "run_meta":
            last_meta_idx = i
    if last_meta_idx == -1:
        return [o for o in objs if o.get("type") == "result"]
    return [o for o in objs[last_meta_idx + 1 :] if o.get("type") == "result"]


def _build_chunk_map_from_evidence(result_row: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    evidence = result_row.get("evidence", []) or []
    return {f"E{i+1}": ev for i, ev in enumerate(evidence)}


def _supports_is_valid(supports: str, ev_text: str) -> bool:
    s = (supports or "").strip()
    e = (ev_text or "").strip()
    if not s or not e:
        return False

    # Very short supports strings are usually labels, not faithful quote/paraphrase.
    if len(s) < 25:
        return False

    # Generic placeholders: only allow them if they still overlap well with evidence text.
    s_l = s.lower()
    if any(re.search(p, s_l) for p in _GENERIC_SUPPORT_PATTERNS):
        return support_score(s, e) >= 0.12

    # General validation: overlap with cited evidence text.
    return support_score(s, e) >= SUPPORTS_VALIDATION_THRESHOLD


def main() -> None:
    log_path = _resolve_input_log_path()
    objs = _read_jsonl_all(log_path)
    rows = _scope_to_last_run_meta(objs)

    case_out = []
    trace_ok = []
    coverages = []
    specificities = []
    ok_citation_hygiene = []

    # OK-only metrics
    coverages_ok: List[float] = []
    specificities_ok: List[float] = []

    for r in rows:
        idx = r.get("idx")
        question = r.get("question", "")
        status = r.get("status", "")
        answer = r.get("answer", "") or ""

        llm = (((r.get("audit") or {}).get("llm")) or {})
        evidence_used = llm.get("evidence_used", []) or []
        citations = llm.get("citations", []) or []

        chunk_map = _build_chunk_map_from_evidence(r)
        available_chunk_ids = set(chunk_map.keys())

        cited_ids = {c.get("chunk_id") for c in citations if c.get("chunk_id")}
        used_ids = set(evidence_used)

        trace_complete = cited_ids.issubset(available_chunk_ids) and used_ids.issubset(available_chunk_ids)
        trace_ok.append(trace_complete)

        hygiene = True
        if status == "ok":
            hygiene = bool(evidence_used) and bool(citations)
        ok_citation_hygiene.append(hygiene)

        # -------------------------
        # Evidence basis selection (hardened):
        # Prefer citation.supports only if it is validated against the cited chunk text.
        # Otherwise fall back to chunk text.
        # -------------------------
        valid_supports_texts: List[str] = []
        invalid_supports_n = 0

        for c in citations:
            if not isinstance(c, dict):
                continue
            cid = c.get("chunk_id")
            if not cid or cid not in chunk_map:
                continue
            supports = (c.get("supports") or "").strip()
            ev_text = (chunk_map[cid].get("text") or "").strip()

            if _supports_is_valid(supports, ev_text):
                valid_supports_texts.append(supports)
            else:
                if supports:
                    invalid_supports_n += 1

        used_texts = [chunk_map[cid].get("text", "") for cid in evidence_used if cid in chunk_map]
        used_texts = [t for t in used_texts if (t or "").strip()]

        all_texts = [chunk_map[cid].get("text", "") for cid in sorted(chunk_map.keys(), key=lambda x: int(x[1:]))]
        all_texts = [t for t in all_texts if (t or "").strip()]

        ev_texts = valid_supports_texts or used_texts or all_texts
        sents = _split_sentences(answer)

        if not sents or not ev_texts:
            coverage = 0.0
            specificity = 0.0
        else:
            best_scores = []
            supported = 0
            for s in sents:
                best = max((support_score(s, t) for t in ev_texts), default=0.0)
                best_scores.append(best)
                if best >= SENT_SUPPORT_THRESHOLD:
                    supported += 1
            coverage = supported / max(1, len(sents))
            specificity = float(np.mean(best_scores)) if best_scores else 0.0

        coverages.append(coverage)
        specificities.append(specificity)

        if status == "ok":
            coverages_ok.append(coverage)
            specificities_ok.append(specificity)

        case_out.append(
            {
                "idx": idx,
                "question": question,
                "status": status,
                "trace_complete": trace_complete,
                "ok_requires_citations_passed": hygiene,
                "sentence_coverage": coverage,
                "mean_specificity": specificity,
                "used_chunk_ids": list(evidence_used),
                "cited_chunk_ids": sorted([x for x in cited_ids if x]),
                "n_evidence_total": len(chunk_map),
                "n_supports_strings_total": sum(1 for c in citations if (c.get("supports") or "").strip()),
                "n_supports_strings_valid": len(valid_supports_texts),
                "n_supports_strings_invalid": invalid_supports_n,
                "supports_validation_threshold": SUPPORTS_VALIDATION_THRESHOLD,
            }
        )

    n_ok = sum(1 for r in rows if r.get("status") == "ok")

    report = {
        "input_log_path": str(log_path),
        "n_results": len(rows),

        "trace_complete_rate": float(np.mean(trace_ok)) if trace_ok else 0.0,
        "mean_sentence_coverage": float(np.mean(coverages)) if coverages else 0.0,
        "mean_specificity": float(np.mean(specificities)) if specificities else 0.0,
        "ok_citation_hygiene_rate": float(np.mean(ok_citation_hygiene)) if ok_citation_hygiene else 0.0,

        # OK-only summary (more interpretable)
        "n_ok": n_ok,
        "mean_sentence_coverage_ok": float(np.mean(coverages_ok)) if coverages_ok else None,
        "mean_specificity_ok": float(np.mean(specificities_ok)) if specificities_ok else None,

        "thresholds": {
            "sentence_support": SENT_SUPPORT_THRESHOLD,
            "supports_validation": SUPPORTS_VALIDATION_THRESHOLD,
        },
        "notes": [
            "Support computed via deterministic lexical+bigram overlap (no sentence-transformers).",
            "Evidence basis prefers audit.llm.citations[].supports only when validated against the cited chunk text; otherwise falls back to chunk text.",
            "If the input log contains multiple appended runs, only the most recent run is evaluated.",
            "OK-only metrics exclude insufficient_evidence and safety_escalation (which are often intentionally non-evidence-quoting templates).",
        ],
    }

    OUTPUT_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with OUTPUT_CASES_PATH.open("w", encoding="utf-8") as f:
        for c in case_out:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
