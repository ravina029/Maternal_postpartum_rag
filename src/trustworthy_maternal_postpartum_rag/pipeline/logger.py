# src/trustworthy_maternal_postpartum_rag/pipeline/logger.py

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import uuid

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / "reasoner_audit.log"
LOG_VERSION = "1.2"


def _safe_json_dumps(obj: Any) -> str:
    """
    Ensure logging never crashes the pipeline due to non-serializable objects.
    """
    return json.dumps(obj, ensure_ascii=False, default=str)


def log_reasoning(
    query: str,
    decision: Dict[str, Any],
    *,
    status: Optional[str] = None,
    lifecycle: Optional[str] = None,
    topic: Optional[str] = None,
    publisher_counts: Optional[Dict[str, int]] = None,
    retrieved_chunks: Optional[int] = None,
    used_chunks: Optional[int] = None,
    error: Optional[str] = None,
    run_id: Optional[str] = None,
):
    """
    Central audit logger for RAG decision-making.

    Append-only JSONL format.
    All extra fields are optional to preserve backward compatibility.
    """

    entry: Dict[str, Any] = {
        "log_version": LOG_VERSION,
        "timestamp_utc": datetime.utcnow().isoformat(),
        "run_id": run_id or str(uuid.uuid4()),
        "query": query,
        "decision": decision,
    }

    # Optional context (added only if provided)
    if status:
        entry["status"] = status
    if lifecycle:
        entry["lifecycle"] = lifecycle
    if topic:
        entry["topic"] = topic
    if publisher_counts:
        entry["publisher_counts"] = publisher_counts
    if retrieved_chunks is not None:
        entry["retrieved_chunks"] = retrieved_chunks
    if used_chunks is not None:
        entry["used_chunks"] = used_chunks
    if error:
        entry["error"] = error

    # Write safely; avoid partial records where possible
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(_safe_json_dumps(entry) + "\n")
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            # fsync may not be available on some filesystems; ignore safely
            pass
