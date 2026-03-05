# src/trustworthy_maternal_postpartum_rag/retrieval/chroma_retriever.py

import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Defaults (read from env, but re-read lazily when needed)
# -----------------------------------------------------------------------------

def _get_defaults() -> Dict[str, str]:
    """
    Lazy env reads so unit tests / notebooks can override env vars without
    needing a fresh Python process.
    """
    return {
        "chroma_path": os.getenv("TMPRAG_CHROMA_PATH", "data/chroma_db"),
        "collection": os.getenv("TMPRAG_COLLECTION_NAME", "maternal_postpartum_chunks"),
        "embed_model": os.getenv("TMPRAG_EMBED_MODEL", "all-MiniLM-L6-v2"),
        "device": os.getenv("TMPRAG_EMBED_DEVICE", "cpu"),
        # If true, always rebuild collection objects (useful for debugging)
        "disable_cache": os.getenv("TMPRAG_DISABLE_CHROMA_CACHE", "false").lower() in {"1", "true", "yes"},
        # Extra logs for retrieval
        "debug": os.getenv("TMPRAG_RETRIEVER_DEBUG", "false").lower() in {"1", "true", "yes"},
    }


# -----------------------------------------------------------------------------
# Cache: keyed by (path, collection, model, device)
# -----------------------------------------------------------------------------

_client_cache: Dict[str, chromadb.PersistentClient] = {}
_collection_cache: Dict[Tuple[str, str, str, str], Any] = {}


def reset_cache() -> None:
    """
    Useful for tests (pytest) or experiments where you want a clean slate.
    """
    _client_cache.clear()
    _collection_cache.clear()
    logger.info("Chroma retriever cache cleared.")


# -----------------------------------------------------------------------------
# Metadata normalization
# -----------------------------------------------------------------------------

_REQUIRED_META_DEFAULTS: Dict[str, Any] = {
    "publisher": "UNKNOWN",
    "source_file": "unknown",
    "page_number": -1,
    "lifecycle": "general",
    "stage": "",
    "country": "",
    "source_type": "",
    "topic_hint": "general",
    "language": "en",
}

def _norm_str(v: Any, default: str) -> str:
    if v is None:
        return default
    s = str(v).strip()
    return s if s else default


def _normalize_metadata(meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Ensure critical keys exist so downstream logic (publisher balancing,
    stage filtering, audits) behaves deterministically.

    Notes:
    - We do not aggressively canonicalize case for publisher because you already
      have meaningful casing ("Cleveland Clinic"). We only strip/blank-check.
    """
    meta_in = dict(meta or {})
    out = dict(_REQUIRED_META_DEFAULTS)

    # overlay stored keys
    for k, v in meta_in.items():
        out[k] = v

    out["publisher"] = _norm_str(out.get("publisher"), "UNKNOWN")
    out["source_file"] = _norm_str(out.get("source_file"), "unknown")
    out["source_type"] = _norm_str(out.get("source_type"), "")
    out["stage"] = _norm_str(out.get("stage"), "")
    out["lifecycle"] = _norm_str(out.get("lifecycle"), "general")
    out["country"] = _norm_str(out.get("country"), "")
    out["topic_hint"] = _norm_str(out.get("topic_hint"), "general")
    out["language"] = _norm_str(out.get("language"), "en")

    # Page number should be int if possible
    pn = out.get("page_number", -1)
    try:
        out["page_number"] = int(pn)
    except Exception:
        out["page_number"] = -1

    # Keep doc_id if present (handy for debugging)
    if "doc_id" in out and out["doc_id"] is None:
        out["doc_id"] = ""

    return out


# -----------------------------------------------------------------------------
# Where validation (lightweight)
# -----------------------------------------------------------------------------

def _validate_where(where: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Chroma expects a dict with simple comparable values or operator dicts.
    We keep it permissive but avoid the most common footguns.
    """
    if where is None:
        return None
    if not isinstance(where, dict):
        raise TypeError(f"where must be a dict or None, got {type(where)}")
    # Avoid empty dict: behaves like no filter
    if not where:
        return None
    return where


# -----------------------------------------------------------------------------
# Collection getter (cached)
# -----------------------------------------------------------------------------

def _get_collection(
    chroma_path: Path,
    collection_name: str,
    embed_model: str,
    device: str,
    *,
    disable_cache: bool = False,
):
    key = (str(chroma_path), collection_name, embed_model, device)

    if not disable_cache and key in _collection_cache:
        return _collection_cache[key]

    chroma_path = Path(chroma_path)
    chroma_path.mkdir(parents=True, exist_ok=True)

    # Cache client by path
    client_key = str(chroma_path)
    client = None if disable_cache else _client_cache.get(client_key)
    if client is None:
        client = chromadb.PersistentClient(path=str(chroma_path))
        if not disable_cache:
            _client_cache[client_key] = client

    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=embed_model,
        device=device,
    )

    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    if not disable_cache:
        _collection_cache[key] = collection

    logger.info(
        "Chroma ready | path=%s collection=%s model=%s device=%s",
        chroma_path, collection_name, embed_model, device
    )
    return collection


# -----------------------------------------------------------------------------
# Public retrieve()
# -----------------------------------------------------------------------------

def retrieve(
    query: str,
    k: int = 8,
    *,
    chroma_path: Optional[Path] = None,
    collection_name: Optional[str] = None,
    embed_model: Optional[str] = None,
    device: Optional[str] = None,
    where: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Returns a list of hits:
      [{"id": str, "text": str, "metadata": dict, "distance": float}, ...]

    Notes:
    - metadata is normalized to include keys your local_qa expects.
    - where is passed directly to Chroma (after light validation).
    """
    if not query or not query.strip():
        return []

    defaults = _get_defaults()

    chroma_path = Path(chroma_path or defaults["chroma_path"])
    collection_name = collection_name or defaults["collection"]
    embed_model = embed_model or defaults["embed_model"]
    device = device or defaults["device"]
    disable_cache = bool(defaults["disable_cache"])
    debug = bool(defaults["debug"])

    where = _validate_where(where)

    col = _get_collection(
        chroma_path=chroma_path,
        collection_name=collection_name,
        embed_model=embed_model,
        device=device,
        disable_cache=disable_cache,
    )

    n_results = max(1, int(k))

    # Be explicit about include fields so output is stable across Chroma versions
    res = col.query(
        query_texts=[query],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    ids = (res.get("ids") or [[]])[0]
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    hits: List[Dict[str, Any]] = []
    limit = min(len(ids), len(docs), len(metas), len(dists))

    for i in range(limit):
        text = (docs[i] or "").strip()
        if not text:
            continue

        meta = _normalize_metadata(metas[i] or {})
        dist_val = float(dists[i]) if dists[i] is not None else 0.0

        hits.append(
            {
                "id": ids[i],
                "text": text,
                "metadata": meta,
                "distance": dist_val,
            }
        )

    # Safety: ensure non-decreasing distance ordering if anything upstream reorders
    hits.sort(key=lambda h: float(h.get("distance", 1e9)))

    if debug:
        pub_counts: Dict[str, int] = {}
        for h in hits:
            p = (h.get("metadata") or {}).get("publisher", "UNKNOWN")
            pub_counts[p] = pub_counts.get(p, 0) + 1

        logger.info(
            "Retriever debug | query=%r k=%d where=%s hits=%d pubs=%s",
            query, k, where, len(hits), pub_counts
        )
        if hits:
            h0 = hits[0]
            m0 = h0["metadata"]
            logger.info(
                "Retriever top hit | dist=%.4f pub=%s file=%s page=%s stage=%s lc=%s snip=%s",
                float(h0.get("distance", 0.0)),
                m0.get("publisher"),
                m0.get("source_file"),
                m0.get("page_number"),
                m0.get("stage"),
                m0.get("lifecycle"),
                (h0.get("text", "")[:160].replace("\n", " ")),
            )

    return hits
