# src/trustworthy_maternal_postpartum_rag/app/local_qa.py

import logging
import os
import re
import uuid
from dataclasses import dataclass
from typing import List, Dict, Any, Callable, Optional
from collections import Counter

from trustworthy_maternal_postpartum_rag.retrieval.chroma_retriever import retrieve as chroma_retrieve

logger = logging.getLogger(__name__)

MAX_CONTEXT_CHUNKS = 4
MAX_CHUNK_CHARS = 700

# Evidence sufficiency thresholds
MIN_EVIDENCE_CHUNKS = 2
MIN_DISTINCT_PUBLISHERS = 2  # try to avoid single-publisher evidence when possible

# Prefer stronger medical publishers where applicable
PUBLISHER_PRIORITY = {
    "WHO": 3,
    "NHS": 2,
    "ACOG": 2,
    "Cleveland Clinic": 3,
}

# Governance knobs (env-driven)
FORCE_PUBLISHER_INJECTION = os.getenv("TMPRAG_FORCE_PUBLISHER_INJECTION", "true").strip().lower() in {
    "1", "true", "yes", "y"
}
INJECT_PUBLISHER_NAME = os.getenv("TMPRAG_INJECT_PUBLISHER", "Cleveland Clinic")

# Optional: reduce noisy warnings in demos/CLI
# - Chroma telemetry banner: export ANONYMIZED_TELEMETRY=false
# - HF tokenizers fork warning: export TOKENIZERS_PARALLELISM=false
_DEFAULT_ENV_HINTS = {
    "TOKENIZERS_PARALLELISM": "false",
    "ANONYMIZED_TELEMETRY": "false",
}


@dataclass
class RetrievedChunk:
    text: str
    source_file: str
    page_number: int
    lifecycle: str
    stage: str
    country: str
    publisher: str
    source_type: str
    topic_hint: str


# ============================================================
# Lifecycle inference (query-side)
# ============================================================

def infer_lifecycle_from_query(query: str) -> str:
    q = (query or "").lower()

    postpartum_markers = [
        "postpartum", "after delivery", "after birth", "after childbirth", "childbirth",
        "child birth", "after giving birth", "after i gave birth",
        "lochia", "perineal", "episiotomy", "c-section", "cesarean", "stitches",
        "period", "periods", "menstruation", "menses",
    ]
    if any(w in q for w in postpartum_markers):
        return "postpartum"

    breastfeeding_markers = [
        "breastfeed", "breastfeeding", "lactation", "milk supply", "letdown",
        "pump", "pumping", "expressing milk",
        "nipple", "crack", "cracked", "fissure", "sore nipples", "mastitis", "engorgement",
        "latch", "attachment",
    ]
    if any(w in q for w in breastfeeding_markers):
        return "breastfeeding"

    pregnancy_markers = [
        "gestational diabetes", "preeclampsia", "eclampsia",
        "placenta", "amniotic", "antenatal", "trimester",
        "pregnant", "pregnancy", "fetus", "fetal", "foetal",
        "kick count", "contractions", "labor", "labour",
    ]
    if any(k in q for k in pregnancy_markers):
        return "pregnancy"

    newborn_markers = [
        "newborn", "neonate", "first week", "few days old",
        "umbilical cord", "meconium", "jaundice",
    ]
    if any(w in q for w in newborn_markers):
        return "newborn"

    infant_markers = ["infant", "6 months", "weaning", "solid foods"]
    if any(w in q for w in infant_markers):
        return "infant"

    toddler_markers = ["toddler", "2 year", "3 year", "4 year"]
    if any(w in q for w in toddler_markers):
        return "toddler"

    return "general"


# ============================================================
# Topic inference (simple heuristics)
# ============================================================

def infer_topic_from_query(query: str) -> str:
    q = (query or "").lower()

    if any(k in q for k in ["vaccine", "vaccination", "immunization", "immunisation", "hep b", "bcg", "polio"]):
        return "immunization"

    if any(k in q for k in [
        "gestational diabetes", "diabetes", "blood sugar", "glucose", "insulin",
    ]):
        return "medical-condition"

    # IMPORTANT: Breastfeeding care must be checked BEFORE diet/recovery to avoid misclassification
    breastfeeding_care = [
        "breastfeed", "breastfeeding", "lactation", "milk supply", "letdown", "latch",
        "nipple", "crack", "cracked", "fissure", "sore nipple", "sore nipples",
        "mastitis", "engorgement",
        "pump", "pumping", "expressing milk",
    ]
    if any(w in q for w in breastfeeding_care):
        return "breastfeeding"

    if any(w in q for w in ["back pain", "pelvic pain", "sciatica", "hip pain", "round ligament"]):
        return "pregnancy-discomfort"

    if any(w in q for w in [
        "period", "periods", "menstruation", "menses", "lochia",
        "stitch", "bleeding", "recovery", "healing", "infection", "fever",
    ]):
        return "recovery"

    if any(w in q for w in ["eat", "food", "diet", "nutrition", "avoid", "safe to eat", "raw", "undercooked"]):
        return "diet"

    if any(w in q for w in ["sleep", "nap", "wake", "insomnia"]):
        return "sleep"

    return "general"


# ============================================================
# Stage alignment guardrail
# ============================================================

def stage_aligns(meta: Dict[str, Any], lifecycle: str) -> bool:
    """
    Conservative guardrail to prevent cross-stage leakage.
    """
    stage = (meta.get("stage") or "").lower()

    if lifecycle == "pregnancy":
        return "pregnancy" in stage
    if lifecycle == "postpartum":
        return "postpartum" in stage
    if lifecycle == "breastfeeding":
        # breastfeeding can be discussed in pregnancy+postpartum guides
        return ("breast" in stage) or ("postpartum" in stage) or ("pregnancy" in stage)
    if lifecycle in {"newborn", "infant", "toddler"}:
        return ("baby" in stage) or ("child" in stage) or ("infant" in stage) or ("newborn" in stage)

    return True


# ============================================================
# Metadata hygiene helpers
# ============================================================

def _normalize_meta(meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Ensure required fields exist so downstream logic does not degrade into UNKNOWN.
    """
    if meta is None:
        meta = {}
    meta.setdefault("publisher", "UNKNOWN")
    meta.setdefault("source_file", "unknown")
    meta.setdefault("page_number", -1)
    meta.setdefault("lifecycle", "general")
    meta.setdefault("stage", "")
    meta.setdefault("country", "")
    meta.setdefault("source_type", "")
    meta.setdefault("topic_hint", "general")
    return meta


def _count_missing_critical_meta(hits: List[Dict[str, Any]]) -> Dict[str, int]:
    required = ["publisher", "source_file", "page_number", "lifecycle", "stage"]
    missing = Counter()
    for h in hits:
        meta = h.get("metadata") or {}
        for k in required:
            v = meta.get(k)
            if v is None or (isinstance(v, str) and not v.strip()):
                missing[k] += 1
    return dict(missing)


def publisher_counts(hits: List[Dict[str, Any]]) -> Dict[str, int]:
    c = Counter()
    for h in hits:
        meta = (h.get("metadata") or {})
        c[meta.get("publisher", "UNKNOWN")] += 1
    return dict(c)


# ============================================================
# Ranking helpers
# ============================================================

def meta_match_score(meta: Dict[str, Any], topic: str, lifecycle: str) -> int:
    score = 0
    if meta.get("topic_hint") == topic:
        score += 2
    if meta.get("lifecycle") == lifecycle:
        score += 2
    score += PUBLISHER_PRIORITY.get(meta.get("publisher"), 0)
    return score


def toc_or_nav_penalty(text: str) -> float:
    """
    Returns a penalty in [0, 0.25]. Higher penalty = less desirable chunk.
    Detects table-of-contents / navigation / section-list pages.
    """
    t = (text or "").lower()
    if not t.strip():
        return 0.25

    penalty = 0.0

    if "table of contents" in t:
        penalty += 0.20
    if "return to table of contents" in t:
        penalty += 0.20
    if t.count("›") >= 4:
        penalty += 0.10

    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    short_caps = sum(1 for ln in lines if len(ln) <= 60 and ln.isupper())
    if short_caps >= 3:
        penalty += 0.05

    return min(0.25, penalty)


def _tokenize_basic(s: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (s or "").lower())


def looks_relevant(query: str, text: str, *, min_overlap: float = 0.10) -> bool:
    """
    Lightweight sanity gate to prevent obvious cross-topic mismatches.
    Tokenization is punctuation-robust.
    """
    q_tokens = set(_tokenize_basic(query))
    t_tokens = set(_tokenize_basic(text))
    if not q_tokens:
        return False
    overlap = len(q_tokens & t_tokens) / max(1, len(q_tokens))
    return overlap >= min_overlap


# ============================================================
# Source-diverse selection (publisher-aware) + fallback
# ============================================================

def select_diverse_chunks(hits: List[Dict[str, Any]], k: int, min_pubs: int = 2) -> List[Dict[str, Any]]:
    """
    Select k chunks while encouraging publisher diversity.
    Scans deeper than k to avoid missing a publisher that appears slightly later.
    """
    selected: List[Dict[str, Any]] = []
    seen_publishers = set()

    def pub_of(h):
        meta = h.get("metadata") or {}
        pub = (meta.get("publisher") or "UNKNOWN").strip()
        return pub if pub else "UNKNOWN"

    scan_limit = min(len(hits), max(k * 8, 64))

    for h in hits[:scan_limit]:
        pub = pub_of(h)
        if pub == "UNKNOWN":
            continue
        if pub not in seen_publishers:
            selected.append(h)
            seen_publishers.add(pub)
        if len(selected) >= k:
            break

    if len(selected) < k:
        for h in hits[:scan_limit]:
            if h in selected:
                continue
            selected.append(h)
            if len(selected) >= k:
                break

    logger.info("Publishers selected: %s", sorted(seen_publishers))
    if len(seen_publishers) < min_pubs:
        logger.warning(
            "Low publisher diversity (%d). Consider increasing retrieval candidate pool or improving metadata.",
            len(seen_publishers),
        )
    return selected


def ensure_min_publisher_diversity(
    ranked_hits: List[Dict[str, Any]],
    chosen_hits: List[Dict[str, Any]],
    *,
    k: int,
    min_distinct_publishers: int,
) -> List[Dict[str, Any]]:
    pubs = {(h.get("metadata") or {}).get("publisher", "UNKNOWN") for h in chosen_hits}
    pubs.discard(None)

    if len(pubs) >= min_distinct_publishers:
        return chosen_hits

    logger.warning(
        "Low publisher diversity (%d). Attempting fallback selection (k=%d, min_pubs=%d).",
        len(pubs), k, min_distinct_publishers,
    )

    expanded_k = min(len(ranked_hits), k * 2)
    reselected = select_diverse_chunks(ranked_hits[:expanded_k], k)
    return reselected[:k]


def _dedupe_by_id(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Prevent accidental duplicate IDs after injection/replacement logic.
    """
    out: List[Dict[str, Any]] = []
    seen = set()
    for h in hits:
        hid = h.get("id")
        key = hid if hid is not None else id(h)
        if key in seen:
            continue
        seen.add(key)
        out.append(h)
    return out


def inject_best_publisher_hit(
    ranked_hits: List[Dict[str, Any]],
    final_hits: List[Dict[str, Any]],
    *,
    publisher: str,
    k: int,
) -> List[Dict[str, Any]]:
    """
    If a trusted publisher exists somewhere in ranked hits, ensure at least one
    appears in the final evidence list (without exceeding k).
    Injects the best-ranked candidate (ranked_hits is already sorted best-first).
    """
    pubs = {(h.get("metadata") or {}).get("publisher", "UNKNOWN") for h in final_hits}
    if publisher in pubs:
        return final_hits

    candidates = [h for h in ranked_hits if (h.get("metadata") or {}).get("publisher") == publisher]
    if not candidates:
        return final_hits

    best = candidates[0]
    if len(final_hits) >= k:
        out = final_hits[:-1] + [best]
    else:
        out = final_hits + [best]
    return _dedupe_by_id(out)[:k]


# ============================================================
# Prompt builder
# ============================================================

def build_rag_prompt(query: str, chunks: List[RetrievedChunk], lifecycle: str, topic: str) -> str:
    context = "\n\n".join(c.text[:MAX_CHUNK_CHARS] for c in chunks)

    sources = "\n".join(
        f"- {c.publisher} | {c.source_file} (page {c.page_number})"
        f" | stage={c.stage or 'NA'} | lifecycle={c.lifecycle or 'NA'}"
        for c in chunks
    )

    return f"""
You are an evidence-based maternal/postpartum/newborn RAG assistant.

Lifecycle: {lifecycle}
Topic: {topic}

RULES:
- Use ONLY the provided context.
- Do not invent facts, recommendations, doses, thresholds, or contraindications.
- If evidence is limited or conflicting, say so explicitly.
- Attribute key statements to their publisher when possible (WHO, NHS, ACOG, Cleveland Clinic, etc.).
- Do NOT add any links/URLs that are not present in the context.

---------------- CONTEXT ----------------
{context}

--------------- QUESTION ----------------
{query}

---------------- ANSWER -----------------
Write a direct, helpful answer grounded in the context.

Evidence & Sources:
{sources}

Uncertainty & Notes:
State evidence gaps explicitly.
""".strip()


# ============================================================
# High-level QA (retrieval-only and LLM modes)
# ============================================================

def answer_question(
    query: str,
    *,
    k: int = MAX_CONTEXT_CHUNKS,
    use_llm: bool = False,
    llm_fn: Optional[Callable[[str], str]] = None,
    retrieve_fn: Callable[..., List[Dict[str, Any]]] = chroma_retrieve,
    debug: bool = False,
) -> dict:
    request_id = str(uuid.uuid4())

    pipeline_run_id = os.getenv("TMPRAG_RUN_ID")  # optional experiment-grouping id
    audit_run_id = pipeline_run_id or request_id

    lifecycle = infer_lifecycle_from_query(query)
    topic = infer_topic_from_query(query)

    logger.info("[%s] Inferred lifecycle: %s", audit_run_id, lifecycle)
    logger.info("[%s] Inferred topic: %s", audit_run_id, topic)

    # Retrieval-time filter (important): reduces leakage and improves diversity selection
    where = {"lifecycle": lifecycle} if lifecycle != "general" else None

    candidate_k = max(k * 12, 48)
    raw_hits = retrieve_fn(query, k=candidate_k, where=where)

    logger.info("[%s] Retrieved %d raw chunks", audit_run_id, len(raw_hits))
    logger.info("[%s] Raw publisher distribution: %s", audit_run_id, publisher_counts(raw_hits))

    missing_meta = _count_missing_critical_meta(raw_hits)
    if any(v > 0 for v in missing_meta.values()):
        logger.warning("[%s] Missing critical metadata counts: %s", audit_run_id, missing_meta)

    aligned_hits = [h for h in raw_hits if stage_aligns(h.get("metadata") or {}, lifecycle)]
    logger.info("[%s] Stage-aligned hits: %d", audit_run_id, len(aligned_hits))

    if len(aligned_hits) >= MIN_EVIDENCE_CHUNKS:
        candidate_hits = aligned_hits
    else:
        logger.warning("[%s] Low stage-aligned evidence; keeping unfiltered hits", audit_run_id)
        candidate_hits = raw_hits

    lifecycle_hits = [h for h in candidate_hits if (h.get("metadata") or {}).get("lifecycle") == lifecycle]
    logger.info("[%s] Lifecycle-matched hits: %d", audit_run_id, len(lifecycle_hits))

    if lifecycle != "general" and len(lifecycle_hits) >= MIN_EVIDENCE_CHUNKS:
        filtered_for_ranking = lifecycle_hits
    else:
        if lifecycle != "general":
            logger.warning("[%s] Low lifecycle evidence; using candidate hits", audit_run_id)
        filtered_for_ranking = candidate_hits

    def _rank_key(h):
        meta = h.get("metadata") or {}
        score = meta_match_score(meta, topic, lifecycle)

        dist = float(h.get("distance", 1.0))
        penalty = toc_or_nav_penalty(h.get("text", ""))

        # lower "effective distance" is better; penalty makes TOC chunks worse
        effective_dist = dist + penalty
        return (score, -effective_dist)

    ranked = sorted(filtered_for_ranking, key=_rank_key, reverse=True)

    final_hits = select_diverse_chunks(ranked, k)
    final_hits = ensure_min_publisher_diversity(
        ranked_hits=ranked,
        chosen_hits=final_hits,
        k=k,
        min_distinct_publishers=MIN_DISTINCT_PUBLISHERS,
    )

    # Governance: optionally inject a trusted publisher if present
    injected = False
    if FORCE_PUBLISHER_INJECTION and INJECT_PUBLISHER_NAME:
        before_pubs = {(h.get("metadata") or {}).get("publisher", "UNKNOWN") for h in final_hits}
        final_hits = inject_best_publisher_hit(ranked, final_hits, publisher=INJECT_PUBLISHER_NAME, k=k)
        after_pubs = {(h.get("metadata") or {}).get("publisher", "UNKNOWN") for h in final_hits}
        injected = (INJECT_PUBLISHER_NAME not in before_pubs) and (INJECT_PUBLISHER_NAME in after_pubs)

    # Sanity gate: drop egregiously irrelevant hits, but never drop all evidence
    filtered = [h for h in final_hits if looks_relevant(query, h.get("text", ""))]
    if filtered:
        final_hits = filtered
    final_hits = _dedupe_by_id(final_hits)[:k]

    logger.info("[%s] Final selected hits: %d", audit_run_id, len(final_hits))
    logger.info("[%s] Final publisher distribution: %s", audit_run_id, publisher_counts(final_hits))

    if debug:
        for i, h in enumerate(final_hits, start=1):
            meta = _normalize_meta(h.get("metadata") or {})
            snip = (h.get("text") or "").replace("\n", " ")[:160]
            logger.info(
                "[%s] HIT %d | pub=%s | stage=%s | lc=%s | file=%s | page=%s | type=%s | topic=%s | dist=%.4f | toc_pen=%.3f | text=%s",
                audit_run_id,
                i,
                meta.get("publisher"),
                meta.get("stage"),
                meta.get("lifecycle"),
                meta.get("source_file"),
                meta.get("page_number"),
                meta.get("source_type"),
                meta.get("topic_hint"),
                float(h.get("distance", 1.0)),
                toc_or_nav_penalty(h.get("text", "")),
                snip,
            )

    chunks: List[RetrievedChunk] = []
    for h in final_hits:
        meta = _normalize_meta(h.get("metadata") or {})
        chunks.append(
            RetrievedChunk(
                text=h.get("text", "") or "",
                source_file=meta.get("source_file", "unknown"),
                page_number=int(meta.get("page_number", -1)),
                lifecycle=meta.get("lifecycle", "general"),
                stage=meta.get("stage", ""),
                country=meta.get("country", ""),
                publisher=meta.get("publisher", "UNKNOWN"),
                source_type=meta.get("source_type", ""),
                topic_hint=meta.get("topic_hint", "general"),
            )
        )

    prompt = build_rag_prompt(query, chunks, lifecycle, topic)

    audit = {
        "request_id": request_id,
        "pipeline_run_id": pipeline_run_id,
        "run_id": audit_run_id,  # backward-compatible field
        "lifecycle": lifecycle,
        "topic": topic,
        "retrieved_chunks": len(raw_hits),
        "used_chunks": len(chunks),
        "publisher_counts": publisher_counts(final_hits),
        "missing_meta_counts": missing_meta,
        "env_hints": dict(_DEFAULT_ENV_HINTS),
        "retrieval_where": where,
        "candidate_k": candidate_k,
        "publisher_injection": {
            "enabled": FORCE_PUBLISHER_INJECTION,
            "publisher": INJECT_PUBLISHER_NAME,
            "injected": injected,
        },
    }

    if not use_llm:
        return {
            "status": "retrieval_only",
            "prompt": prompt,
            "audit": audit,
            "evidence": [c.__dict__ for c in chunks],
        }

    if llm_fn is None:
        raise ValueError("use_llm=True requires llm_fn(prompt)->str")

    answer = llm_fn(prompt)

    return {
        "status": "ok",
        "answer": answer,
        "prompt": prompt,
        "audit": audit,
        "evidence": [c.__dict__ for c in chunks],
    }
