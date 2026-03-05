# src/trustworthy_maternal_postpartum_rag/ingestion/preprocessing.py

import logging
import time
import fitz  # PyMuPDF
import re
import uuid
from pathlib import Path
import unicodedata
import json
from collections import Counter
import hashlib
import os

# ============================================================
# Pipeline RUN_ID
# ============================================================
print("Preprocessing started.")
RUN_ID_ENV = "TMPRAG_RUN_ID"

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

LOG_FILE = Path("logs/data_logs/preprocessing.log")
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("tmprag.ingestion.preprocessing")
logger.setLevel(logging.INFO)

fmt = "%(asctime)s - %(levelname)s - run_id=%(run_id)s - %(message)s"

if not any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == str(LOG_FILE) for h in logger.handlers):
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
# CONFIG – Manual Per-PDF Skip Logic
# ============================================================

MANUAL_SKIP_MAP = {
    "who_antenatal care.pdf": {
        *range(0, 0),
        *range(9, 11),
    },
    "who_pcpnc_third_edition.pdf": {
        *range(1, 31),
        *range(206, 244),
    },
    "who_postnatal_positive_experience.pdf": {
        *range(1, 21),
        *range(170, 186),
    },
    "baby 411 clear answers and smart advice for your babys first year (brown, ari, fields, denise).pdf": {
        *range(1, 39),
        *range(946, 1102),
    },
}
MANUAL_SKIP_MAP = {k.lower(): v for k, v in MANUAL_SKIP_MAP.items()}


def normalize_unicode(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    replacements = {
        "“": '"', "”": '"', "‘": "'", "’": "'",
        "\xa0": " ", "ﬁ": "fi", "ﬂ": "fl",
        "•": "-", "–": "-", "—": "-",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return "".join(c for c in text if unicodedata.category(c)[0] != "C")


def dehyphenate(text: str) -> str:
    return re.sub(r"(\w+)-\n([a-z])", r"\1\2", text)


def clean_headers_footers(text: str) -> str:
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        s = line.strip()
        if re.fullmatch(r"[A-Z ,\-]{15,}", s):
            continue
        if re.fullmatch(r"[A-Z]\d{1,3}", s):
            continue
        if re.fullmatch(r"\d{1,4}", s):
            continue
        if re.match(r"^Page \d+ of \d+$", s, re.I):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def detect_repeated_headers_footers(pages_raw_text):
    line_counter = Counter()
    for text in pages_raw_text:
        for line in text.splitlines():
            c = line.strip()
            if c:
                line_counter[c] += 1
    threshold = len(pages_raw_text) * 0.2
    return {line for line, count in line_counter.items() if count > threshold}


def remove_noise_lines(text: str, repeated_noise=None) -> str:
    lines = text.splitlines()
    out = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if re.match(r"^[-=*_.]{3,}$", s):
            continue
        if re.fullmatch(r"\d{1,3}", s):
            continue
        if repeated_noise and s in repeated_noise:
            continue
        out.append(s)
    return "\n".join(out)


def merge_paragraph_lines(text: str) -> str:
    paragraphs = text.split("\n\n")
    merged = [" ".join(p.splitlines()) for p in paragraphs]
    return "\n\n".join(merged)


def should_remove_page(text: str) -> bool:
    lower = text.lower()

    if not text.strip():
        return True

    if any(term in lower for term in [
        "circumcision", "foreskin", "penis", "hiv acquisition",
        "sexually transmitted infections in men", "balanitis",
        "paraphimosis", "hpv infection in men",
    ]):
        return True

    if "table of contents" in lower or lower.startswith("contents"):
        if len(re.findall(r"\s\d{1,4}$", text, re.M)) >= 5:
            return True

    if lower.startswith("references"):
        return True

    if re.search(r"\b(appendix|annex|annexure)\b", lower):
        return True

    num_ratio = sum(c.isdigit() for c in text) / max(len(text), 1)
    if num_ratio > 0.60:
        return True

    boilerplate_hit = re.search(
        r"(isbn|©|copyright|all rights reserved|produced by|available from:|tel:|fax:|email:|contact us|privacy statement|terms and conditions)",
        lower
    )
    if boilerplate_hit:
        alpha_chars = sum(ch.isalpha() for ch in text)
        total_chars = max(len(text), 1)
        alpha_ratio = alpha_chars / total_chars
        if len(text) < 700 or alpha_ratio < 0.35:
            return True

    return False


def manual_page_skip(pdf_name, page_num) -> bool:
    pdf_name = pdf_name.lower()
    if pdf_name in MANUAL_SKIP_MAP:
        return page_num in MANUAL_SKIP_MAP[pdf_name]
    return False


def infer_doc_metadata(pdf_name: str):
    name = pdf_name.lower()
    meta = {
        "country": None,
        "stage": None,
        "target": None,
        "source_type": "guideline",
        "publisher": None,
        "year": None,
    }

    if "who_antenatal care" in name:
        meta.update({"country": "Global/WHO", "stage": "pregnancy", "target": "mother", "publisher": "WHO"})
    elif "who_pcpnc_third_edition" in name:
        meta.update({"country": "Global/WHO", "stage": "pregnancy+postpartum+newborn", "target": "mother+baby", "publisher": "WHO"})
    elif "who_postnatal_positive_experience" in name:
        meta.update({"country": "Global/WHO", "stage": "postpartum+newborn", "target": "mother+baby", "publisher": "WHO"})
    elif "india_pmsma_high-risk-conditions-in-preg-modified-final" in name or "high-risk-conditions-in-preg-modified-final" in name:
        meta.update({"country": "India", "stage": "pregnancy", "target": "mother", "source_type": "national_guideline", "publisher": "Government of India"})
    elif "nhs_pregnancy_postpartum_guide" in name:
        meta.update({"country": "United Kingdom", "stage": "pregnancy+postpartum", "target": "mother+baby", "source_type": "patient_guide", "publisher": "NHS"})
    elif "newborn_and_children_care" in name and "nhs" in name:
        meta.update({"country": "United Kingdom", "stage": "newborn+child", "target": "baby+child", "source_type": "patient_guide", "publisher": "NHS"})
    elif "baby 411 clear answers and smart advice" in name:
        meta.update({"country": "United States", "stage": "newborn+infant", "target": "baby", "source_type": "book", "publisher": "Baby 411"})
    elif "acog" in name and "pregnancy" in name:
        meta.update({"country": "United States", "stage": "pregnancy+postpartum", "target": "mother+baby", "source_type": "patient_guide", "publisher": "ACOG"})
    elif "cleveland clinic" in name and "pregnancy" in name:
        meta.update({"country": "United States", "stage": "pregnancy+postpartum", "target": "mother+baby", "source_type": "patient_guide", "publisher": "Cleveland Clinic"})

    return meta


def _normalize_for_dedup(text: str) -> str:
    if not text:
        return ""
    t = text.lower()
    t = re.sub(r"\s+", " ", t).strip()
    return t[:2000]


def _page_fingerprint(text: str) -> str:
    norm = _normalize_for_dedup(text)
    return hashlib.sha256(norm.encode("utf-8", errors="ignore")).hexdigest()


def preprocess_pdf_to_pages(pdf_path: Path, doc_id=None):
    doc_id = doc_id or str(uuid.uuid4())
    pdf_name = pdf_path.name.lower()
    doc_meta = infer_doc_metadata(pdf_path.name)
    doc_metadata = {
    "country": doc_meta["country"],
    "stage": doc_meta["stage"],
    "target": doc_meta["target"],
    "source_type": doc_meta["source_type"],
    "publisher": doc_meta.get("publisher"),
}

    t0 = time.time()
    logger.info(f"[Preprocess] Start | file={pdf_path.name} | doc_id={doc_id} | publisher={doc_meta.get('publisher')}")

    doc = fitz.open(pdf_path)
    raw_pages = [page.get_text("text") for page in doc]
    doc.close()

    repeated_noise = detect_repeated_headers_footers(raw_pages)

    preprocessed_pages = []
    seen_page_fps = set()

    manual_skipped = 0
    auto_filtered = 0
    deduped = 0
    kept = 0

    for i, raw in enumerate(raw_pages, start=1):
        if manual_page_skip(pdf_name, i):
            manual_skipped += 1
            preprocessed_pages.append({
                "page_id": str(uuid.uuid4()),
                "doc_id": doc_id,
                "source_file": pdf_path.name,
                "page_number": i,
                "text": "",
                "skipped": True,
                "skip_reason": "manual_skip",
                "doc_metadata": doc_metadata,
                "language": "en",
                "version": "1.2",
            })
            continue

        text = normalize_unicode(raw)
        text = clean_headers_footers(text)
        text = dehyphenate(text)
        text = remove_noise_lines(text, repeated_noise)
        text = merge_paragraph_lines(text)

        fp = _page_fingerprint(text)
        if fp in seen_page_fps:
            deduped += 1
            preprocessed_pages.append({
                "page_id": str(uuid.uuid4()),
                "doc_id": doc_id,
                "source_file": pdf_path.name,
                "page_number": i,
                "text": "",
                "skipped": True,
                "skip_reason": "deduped_duplicate_page",
                "doc_metadata": doc_metadata,
                "language": "en",
                "version": "1.2",
            })
            continue
        seen_page_fps.add(fp)

        skipped = False
        skip_reason = None
        if should_remove_page(text):
            skipped = True
            skip_reason = "auto_filter"
            auto_filtered += 1
        else:
            kept += 1

        preprocessed_pages.append({
            "page_id": str(uuid.uuid4()),
            "doc_id": doc_id,
            "source_file": pdf_path.name,
            "page_number": i,
            "text": text if not skipped else "",
            "skipped": skipped,
            "skip_reason": skip_reason,
            "doc_metadata": doc_metadata,
            "language": "en",
            "version": "1.2",
        })

    dt = time.time() - t0
    logger.info(
        f"[Preprocess] Done | file={pdf_path.name} pages={len(raw_pages)} kept={kept} manual={manual_skipped} auto={auto_filtered} deduped={deduped} seconds={dt:.2f}"
    )

    return preprocessed_pages


if __name__ == "__main__":
    raw_root = Path("data/raw")
    pdf_paths = list(raw_root.rglob("*.pdf"))

    logger.info(f"[Preprocess] Batch start | root={raw_root} pdfs={len(pdf_paths)}")

    out_dir = Path("data/processed/")
    out_dir.mkdir(parents=True, exist_ok=True)

    for old in out_dir.glob("*_preprocessed.jsonl"):
        old.unlink()

    for pdf in pdf_paths:
        logger.info(f"[Preprocess] Running | rel={pdf.relative_to(raw_root)}")
        pages = preprocess_pdf_to_pages(pdf)

        rel_stem = "_".join(pdf.relative_to(raw_root).with_suffix("").parts)
        out_file = out_dir / f"{rel_stem}_preprocessed.jsonl"

        with open(out_file, "w", encoding="utf-8") as f:
            for page in pages:
                f.write(json.dumps(page, ensure_ascii=False) + "\n")

        kept = sum(1 for p in pages if not p.get("skipped"))
        skipped = sum(1 for p in pages if p.get("skipped"))
        deduped = sum(1 for p in pages if p.get("skip_reason") == "deduped_duplicate_page")
        logger.info(f"[Preprocess] Saved | out={out_file.name} pages={len(pages)} kept={kept} skipped={skipped} deduped={deduped}")

    logger.info("[Preprocess] Batch completed")
    print("Preprocessing completed.")
