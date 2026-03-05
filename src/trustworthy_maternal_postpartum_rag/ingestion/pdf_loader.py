# src/trustworthy_maternal_postpartum_rag/ingestion/pdf_loader.py

import logging
from pathlib import Path
from typing import Dict, List, Tuple
import hashlib
import re
import fitz  # PyMuPDF
import os
import uuid

# ============================================================
# Pipeline RUN_ID (propagated via env)
# ============================================================

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
# Logging (module logger + file handler)
# ============================================================

LOG_FILE = Path("logs/data_logs/ingestion.log")
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("tmprag.ingestion.pdf_loader")
logger.setLevel(logging.INFO)

fmt = "%(asctime)s - %(levelname)s - run_id=%(run_id)s - %(message)s"

if not any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == str(LOG_FILE) for h in logger.handlers):
    fh = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(fmt))
    fh.addFilter(RunIdFilter())
    logger.addHandler(fh)

# optional console
if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter(fmt))
    sh.addFilter(RunIdFilter())
    logger.addHandler(sh)


class PDFLoader:
    """
    Load PDFs from a directory and extract page-level plain text.
    Public API:
      - load_pdfs() -> Dict[pdf_stem, List[ (page_number:int, text:str) ] ]
      - load_pdfs_as_single_string() -> Dict[pdf_stem, full_text_str] (kept for compatibility)
    """

    def __init__(self, pdf_dir: str):
        self.pdf_dir = Path(pdf_dir)
        if not self.pdf_dir.exists():
            raise ValueError(f"PDF directory does not exist: {self.pdf_dir}")

        # Optional behavior flags (safe defaults)
        self.enable_page_dedup = True

    def load_pdfs(self) -> Dict[str, List[Tuple[int, str]]]:
        """
        Returns a dict mapping file stem -> list of (page_number starting at 1, page_text).
        Page order preserved.
        """
        pdf_texts: Dict[str, List[Tuple[int, str]]] = {}

        pdf_files = sorted(self.pdf_dir.rglob("*.pdf"), key=lambda p: str(p).lower())
        logger.info(f"[PDFLoader] Scan start | root={self.pdf_dir} | pdfs={len(pdf_files)}")

        total_pages_raw = 0
        total_pages_kept = 0
        total_pages_deduped = 0
        failed = 0

        for pdf_file in pdf_files:
            try:
                pages, stats = self._extract_pages(pdf_file)
                total_pages_raw += stats["raw_pages"]
                total_pages_kept += stats["kept_pages"]
                total_pages_deduped += stats["deduped_pages"]

                if pages:
                    key = pdf_file.stem
                    if key in pdf_texts:
                        key = "_".join(pdf_file.relative_to(self.pdf_dir).with_suffix("").parts)

                    pdf_texts[key] = pages
                    logger.info(
                        f"[PDFLoader] Loaded | file={pdf_file.name} | raw={stats['raw_pages']} kept={stats['kept_pages']} deduped={stats['deduped_pages']}"
                    )
                else:
                    logger.warning(f"[PDFLoader] No text extracted | file={pdf_file.name}")
            except Exception as e:
                failed += 1
                logger.exception(f"[PDFLoader] Failed | file={pdf_file.name} | error={e}")

        logger.info(
            f"[PDFLoader] Scan done | pdfs={len(pdf_files)} failed={failed} pages_raw={total_pages_raw} pages_kept={total_pages_kept} pages_deduped={total_pages_deduped}"
        )
        return pdf_texts

    def load_pdfs_as_single_string(self) -> Dict[str, str]:
        """
        Backwards-compatible: returns a mapping file_stem -> full single string of text.
        """
        out: Dict[str, str] = {}
        for name, pages in self.load_pdfs().items():
            joined = "\n".join(p for (_, p) in pages)
            out[name] = joined
        return out

    def _normalize_for_fingerprint(self, text: str) -> str:
        if not text:
            return ""
        t = text.lower()
        t = re.sub(r"\s+", " ", t).strip()
        return t[:2000]

    def _fingerprint_page(self, text: str) -> str:
        norm = self._normalize_for_fingerprint(text)
        return hashlib.sha256(norm.encode("utf-8", errors="ignore")).hexdigest()

    def _extract_pages(self, pdf_path: Path) -> Tuple[List[Tuple[int, str]], dict]:
        """
        Extract text per page. Returns (list of (1-based page_number, text), stats dict).
        """
        doc = fitz.open(pdf_path)
        pages: List[Tuple[int, str]] = []

        seen_fps = set()
        deduped = 0

        for i, page in enumerate(doc, start=1):
            try:
                text = page.get_text()
            except Exception:
                text = ""

            text = text if text else ""

            if self.enable_page_dedup:
                fp = self._fingerprint_page(text)
                if fp in seen_fps:
                    deduped += 1
                    logger.info(f"[PDFLoader] Deduped page | file={pdf_path.name} | page={i}")
                    continue
                seen_fps.add(fp)

            pages.append((i, text))

        raw_pages = len(doc)
        doc.close()

        stats = {
            "raw_pages": raw_pages,
            "kept_pages": len(pages),
            "deduped_pages": deduped,
        }
        return pages, stats
