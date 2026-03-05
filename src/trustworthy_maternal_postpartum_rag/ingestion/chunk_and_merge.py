# src/trustworthy_maternal_postpartum_rag/ingestion/chunking.py

import json
import re
import uuid
import hashlib
import logging
from pathlib import Path
import os
from typing import Dict, Any, List

# ============================================================
# Pipeline RUN_ID
# ============================================================

RUN_ID_ENV = "TMPRAG_RUN_ID"
print("chunking started.")


def get_run_id() -> str:
    rid = os.environ.get(RUN_ID_ENV)
    if rid:
        return rid
    rid = str(uuid.uuid4())
    os.environ[RUN_ID_ENV] = rid
    return rid


RUN_ID = get_run_id()


class RunIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = RUN_ID
        return True


# ============================================================
# Logging
# ============================================================

LOG_FILE = Path("logs/data_logs/chunking.log")
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("tmprag.ingestion.chunking")
logger.setLevel(logging.INFO)

fmt = "%(asctime)s - %(levelname)s - run_id=%(run_id)s - %(message)s"

if not any(
    isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == str(LOG_FILE)
    for h in logger.handlers
):
    fh = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(fmt))
    fh.addFilter(RunIdFilter())
    logger.addHandler(fh)

if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter(fmt))
    sh.addFilter(RunIdFilter())
    logger.addHandler(sh)

# ============================================================
# CONFIG (path-driven)
# ============================================================

PROCESSED_DIR = Path("data/processed")
CHUNKS_DIR = Path("data/chunks")
PATTERN = "*_preprocessed.jsonl"

MAX_TOKENS = int(os.getenv("TMPRAG_CHUNK_MAX_TOKENS", "450"))
OVERLAP_TOKENS = int(os.getenv("TMPRAG_CHUNK_OVERLAP_TOKENS", "60"))

# Dedup: default OFF because it can silently destroy PDFs with repetitive headers.
# Turn on explicitly when you trust your text cleaning.
ENABLE_CHUNK_DEDUP = os.getenv("TMPRAG_ENABLE_CHUNK_DEDUP", "false").lower() == "true"
DEDUP_MODE = os.getenv("TMPRAG_DEDUP_MODE", "page").lower()
# DEDUP_MODE: "page" (safe), "global" (aggressive), "off"
DEDUP_FINGERPRINT_CHARS = int(os.getenv("TMPRAG_DEDUP_CHARS", "8000"))
STRIP_BOILERPLATE_FOR_DEDUP = os.getenv("TMPRAG_STRIP_BOILERPLATE", "true").lower() == "true"

# Emergency/table heuristics can be tuned by env if needed
TABLE_BULLET_THRESHOLD = int(os.getenv("TMPRAG_TABLE_BULLET_THRESHOLD", "12"))
TABLE_COLONLINE_THRESHOLD = int(os.getenv("TMPRAG_TABLE_COLONLINE_THRESHOLD", "12"))

# ============================================================
# Token estimation
# ============================================================

def est_tokens(text: str) -> int:
    # Simple approximation (sufficient for chunk sizing)
    return len(text.split())


# ============================================================
# Lifecycle inference (text-side)
# ============================================================

def infer_lifecycle(text: str) -> str:
    t = (text or "").lower()

    if any(w in t for w in [
        "postpartum", "after delivery", "after childbirth",
        "lochia", "perineal", "c-section", "cesarean",
        "six weeks after", "maternal recovery"
    ]):
        return "postpartum"

    if any(w in t for w in [
        "breastfeeding", "lactation", "milk supply",
        "nipple", "expressing milk"
    ]):
        return "breastfeeding"

    if any(w in t for w in [
        "newborn", "first 28 days", "neonate",
        "umbilical cord", "meconium"
    ]):
        return "newborn"

    if any(w in t for w in [
        "infant", "months old", "6 months", "weaning",
        "complementary feeding"
    ]):
        return "infant"

    if any(w in t for w in [
        "toddler", "1 year", "2 years"
    ]):
        return "toddler"

    if any(w in t for w in [
        "pregnant", "pregnancy", "antenatal",
        "trimester", "gestation"
    ]):
        return "pregnancy"

    return "general"


# ============================================================
# Splitting helpers
# ============================================================

def split_on_headings(text: str) -> List[str]:
    """
    Split on plausible headings while avoiding over-fragmentation.
    Falls back to [full_text] if nothing is detected.
    """
    lines = [l for l in (text or "").split("\n") if l.strip()]
    blocks: List[str] = []
    current_lines: List[str] = []

    heading_pattern = re.compile(
        r"""^(
            (\d+(\.\d+)*\s+.+)                 # numbered headings
            |([A-Z][A-Z0-9 ,\-]{8,})           # ALLCAPS headings (>=8 chars)
            |([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){1,5})  # Title Case headings
        )$""",
        re.X,
    )

    # Remove some extremely common boilerplate headings that can appear on every page
    boilerplate_exact = {
        "RETURN TO TABLE OF CONTENTS",
        "TABLE OF CONTENTS",
    }

    for line in lines:
        s = line.strip()
        up = s.upper()

        if up in boilerplate_exact:
            continue

        # Skip a few known long institutional banner lines (customize as you learn)
        if re.match(r"^(CLINICAL PRACTICE GUIDELINES|NATIONAL INTEGRATED MATERNAL)", up):
            continue

        if heading_pattern.match(s):
            if current_lines:
                blocks.append("\n".join(current_lines).strip())
                current_lines = []
            # Do not store heading itself as a standalone chunk
        else:
            current_lines.append(s)

    if current_lines:
        blocks.append("\n".join(current_lines).strip())

    return blocks if blocks else [text.strip()]


def split_block_by_length(block_text: str, max_tokens: int = MAX_TOKENS, overlap_tokens: int = OVERLAP_TOKENS) -> List[str]:
    if not block_text:
        return []

    # Split by sentence boundary and paragraph breaks.
    raw_parts = re.split(r"(?<=[\.\!\?])\s+|\n\n+", block_text)
    parts = [p.strip() for p in raw_parts if p and p.strip()]

    if not parts:
        return []

    chunks: List[str] = []
    cur_parts: List[str] = []
    cur_tokens = 0

    for part in parts:
        t = est_tokens(part)

        # If a single part is extremely long, force-split it by words.
        if t > max_tokens * 2:
            words = part.split()
            start = 0
            while start < len(words):
                end = min(len(words), start + max_tokens)
                chunks.append(" ".join(words[start:end]).strip())
                # Overlap by overlap_tokens words (approx)
                start = end - min(overlap_tokens, (end - start))
            continue

        if cur_parts and cur_tokens + t > max_tokens:
            chunks.append(" ".join(cur_parts).strip())
            if overlap_tokens > 0 and cur_parts:
                cur_parts = [cur_parts[-1]]
                cur_tokens = est_tokens(cur_parts[0])
            else:
                cur_parts, cur_tokens = [], 0

        cur_parts.append(part)
        cur_tokens += t

    if cur_parts:
        chunks.append(" ".join(cur_parts).strip())

    return chunks


# ============================================================
# Table/emergency heuristics
# ============================================================

def is_table_page(text: str) -> bool:
    low = (text or "").lower()
    if "table " in low:
        return True
    bullet_like = (text or "").count("•") + (text or "").count("- ")
    if bullet_like >= TABLE_BULLET_THRESHOLD:
        return True
    colon_lines = sum(1 for l in (text or "").splitlines() if ":" in l and len(l.strip()) < 80)
    if colon_lines >= TABLE_COLONLINE_THRESHOLD:
        return True
    return False


def split_table_rows(text: str) -> List[str]:
    lines = [l.strip() for l in (text or "").split("\n") if l.strip()]
    if not lines:
        return []

    # Use first non-empty line as title; remove repetitive banners if present
    title = lines[0]
    rest = " ".join(lines[1:])

    # Split on 2+ spaces or " - " patterns
    rows = re.split(r"\s{2,}| - ", rest)
    out = []
    for r in rows:
        r = r.strip()
        if r:
            out.append(f"{title}: {r}")
    return out


def is_emergency_card(text: str) -> bool:
    t = text or ""
    return (
        "EMERGENCY TREATMENTS FOR THE WOMAN" in t
        or (t.count("If the woman") >= 2 and t.count("Give ") >= 2)
    )


def split_emergency_card(text: str) -> List[str]:
    markers = [
        "Give oxytocin",
        "Give magnesium sulphate",
        "Refer the woman urgently",
    ]
    t = text or ""
    for m in markers:
        t = t.replace(m, f"\n\n### {m}\n")

    parts = re.split(r"\n{2,}###\s+", t)
    return [p.strip() for p in parts if p and p.strip()]


# ============================================================
# Dedup helpers (safe defaults)
# ============================================================

def _strip_boilerplate_lines(text: str) -> str:
    """
    Conservative removal of common repeated header/footer patterns that dominate PDF extraction.
    Only removes lines that are likely boilerplate (short or numeric).
    """
    lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
    kept: List[str] = []

    for l in lines:
        low = l.lower()

        # Page numbers (standalone)
        if re.fullmatch(r"\d{1,4}", l):
            continue

        # Very common guide headers; remove only if short-ish
        if any(p in low for p in [
            "return to table of contents",
            "table of contents",
            "cleveland clinic",
            "women's health - pregnancy",
            "your guide to a healthy pregnancy",
            "healthy pregnancy",
            "edition",
        ]):
            if len(l) <= 80:
                continue

        kept.append(l)

    return "\n".join(kept).strip() if kept else (text or "").strip()


def _normalize_for_dedup(text: str) -> str:
    t = (text or "").lower()
    t = re.sub(r"\s+", " ", t).strip()
    if STRIP_BOILERPLATE_FOR_DEDUP:
        t2 = _strip_boilerplate_lines(t)
        t2 = re.sub(r"\s+", " ", t2).strip()
        # Keep the “stripped” variant only if it still has substance
        if len(t2) >= 50:
            t = t2
    return t


def _chunk_fingerprint(text: str) -> str:
    """
    Head+tail hashing prevents a shared intro boilerplate from collapsing everything.
    """
    norm = _normalize_for_dedup(text)

    n = DEDUP_FINGERPRINT_CHARS
    if len(norm) <= n:
        material = norm
    else:
        half = n // 2
        material = norm[:half] + " || " + norm[-half:]

    return hashlib.sha256(material.encode("utf-8", errors="ignore")).hexdigest()


# ============================================================
# Chunk builder
# ============================================================

def make_chunk(doc_id: str, source_file: str, page_number: int, text: str, doc_meta: Dict[str, Any], topic_hint: str = None) -> Dict[str, Any]:
    lifecycle = infer_lifecycle(text)

    # Heuristic fallback for "general" pages
    if lifecycle == "general":
        low = (text or "").lower()
        if any(word in low for word in [
            "mother", "baby", "feeding", "breast",
            "diet", "nutrition", "sleep", "care", "recovery",
        ]):
            lifecycle = "postpartum"

    if lifecycle == "general":
        low = (text or "").lower()
        if any(word in low for word in [
            "woman", "labour", "labor", "delivery", "birth",
            "episiotomy", "uterus", "uterine",
            "eclampsia", "pre-eclampsia",
            "postpartum haemorrhage", "hemorrhage",
            "magnesium sulphate", "refer the woman",
        ]):
            lifecycle = "pregnancy"

    return {
        "chunk_id": str(uuid.uuid4()),
        "doc_id": doc_id,
        "source_file": source_file,
        "page_number": page_number,
        "text": text,
        "language": "en",
        "version": "1.5",
        "country": doc_meta.get("country"),
        "stage": doc_meta.get("stage"),
        "target": doc_meta.get("target"),
        "source_type": doc_meta.get("source_type"),
        "publisher": doc_meta.get("publisher"),
        "topic_hint": topic_hint,
        "lifecycle": lifecycle,
    }


# ============================================================
# Page chunking (CRITICAL FIX: always produce chunks if text exists)
# ============================================================

def chunk_page(record: Dict[str, Any], max_tokens: int = MAX_TOKENS, overlap_tokens: int = OVERLAP_TOKENS) -> List[Dict[str, Any]]:
    text = record.get("text", "") or ""
    if record.get("skipped") or not text.strip():
        return []

    doc_meta = record.get("doc_metadata", {}) or {}
    source_file = record.get("source_file")
    page_number = record.get("page_number")
    doc_id = record.get("doc_id")

    chunks: List[Dict[str, Any]] = []

    # Emergency cards
    if is_emergency_card(text):
        for body in split_emergency_card(text):
            bodies = split_block_by_length(body, max_tokens, overlap_tokens) or [body.strip()]
            for b in bodies:
                if b:
                    chunks.append(make_chunk(doc_id, source_file, page_number, b, doc_meta, "emergency"))
        return chunks

    # Table-heavy pages
    if is_table_page(text):
        rows = split_table_rows(text)
        if rows:
            for row in rows:
                for body in split_block_by_length(row, max_tokens, overlap_tokens):
                    if body:
                        chunks.append(make_chunk(doc_id, source_file, page_number, body, doc_meta, "table"))
            if chunks:
                return chunks
        # If table heuristic triggers but row split fails, fall through to normal splitting.

    # Normal splitting
    blocks = split_on_headings(text)
    for block in blocks:
        bodies = split_block_by_length(block, max_tokens, overlap_tokens)
        for body in bodies:
            if body:
                chunks.append(make_chunk(doc_id, source_file, page_number, body, doc_meta))

    # FINAL SAFETY NET:
    # If for any reason we produced zero chunks (this is what you are seeing for Cleveland Clinic),
    # force a simple chunking over the entire page.
    if not chunks:
        forced = split_block_by_length(text.strip(), max_tokens, overlap_tokens) or [text.strip()]
        for body in forced:
            if body:
                chunks.append(make_chunk(doc_id, source_file, page_number, body, doc_meta, topic_hint="forced_fallback"))

    return chunks


# ============================================================
# Batch chunking
# ============================================================

def chunk_preprocessed_files(
    processed_dir: Path = PROCESSED_DIR,
    pattern: str = PATTERN,
    chunks_dir: Path = CHUNKS_DIR,
) -> None:
    chunks_dir.mkdir(parents=True, exist_ok=True)

    pre_files = sorted(processed_dir.glob(pattern), key=lambda p: p.name.lower())
    logger.info(
        "[Chunking] Batch start | processed_dir=%s files=%d dedup_enabled=%s dedup_mode=%s",
        processed_dir, len(pre_files), ENABLE_CHUNK_DEDUP, DEDUP_MODE
    )

    total_pages_in = 0
    total_pages_skipped = 0
    total_chunks_out = 0
    total_chunks_deduped = 0

    for pre_file in pre_files:
        out_file = chunks_dir / pre_file.name.replace("_preprocessed", "_chunks")
        logger.info("[Chunking] File start | in=%s out=%s", pre_file.name, out_file.name)

        # Dedup state
        seen_global = set()
        seen_by_page: Dict[Any, set] = {}

        pages_in = 0
        pages_skipped = 0
        chunks_out = 0
        chunks_deduped = 0

        with open(pre_file, "r", encoding="utf-8") as fin, open(out_file, "w", encoding="utf-8") as fout:
            for line in fin:
                rec = json.loads(line)
                pages_in += 1

                page_text = (rec.get("text") or "").strip()
                if rec.get("skipped") or not page_text:
                    pages_skipped += 1
                    continue

                page_chunks = chunk_page(rec)
                if not page_chunks:
                    # This should not happen with the fallback, but keep it explicit for audits.
                    logger.warning(
                        "[Chunking] Empty chunk_page output | file=%s page_number=%s",
                        pre_file.name, rec.get("page_number")
                    )
                    continue

                for ch in page_chunks:
                    if ENABLE_CHUNK_DEDUP and DEDUP_MODE != "off":
                        fp = _chunk_fingerprint(ch.get("text", ""))

                        if DEDUP_MODE == "page":
                            pn = ch.get("page_number", -1)
                            bucket = seen_by_page.setdefault(pn, set())
                            if fp in bucket:
                                chunks_deduped += 1
                                continue
                            bucket.add(fp)

                        elif DEDUP_MODE == "global":
                            if fp in seen_global:
                                chunks_deduped += 1
                                continue
                            seen_global.add(fp)

                        else:
                            # Unknown mode -> do not dedup
                            pass

                    fout.write(json.dumps(ch, ensure_ascii=False) + "\n")
                    chunks_out += 1

        logger.info(
            "[Chunking] File done | file=%s pages_in=%d pages_skipped=%d chunks_out=%d chunks_deduped=%d dedup_enabled=%s dedup_mode=%s",
            pre_file.name, pages_in, pages_skipped, chunks_out, chunks_deduped, ENABLE_CHUNK_DEDUP, DEDUP_MODE
        )

        total_pages_in += pages_in
        total_pages_skipped += pages_skipped
        total_chunks_out += chunks_out
        total_chunks_deduped += chunks_deduped

    logger.info(
        "[Chunking] Batch done | pages_in=%d pages_skipped=%d chunks_out=%d chunks_deduped=%d",
        total_pages_in, total_pages_skipped, total_chunks_out, total_chunks_deduped
    )


if __name__ == "__main__":
    chunk_preprocessed_files()

print("chunking completed.")
