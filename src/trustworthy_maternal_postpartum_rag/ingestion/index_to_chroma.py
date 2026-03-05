# src/trustworthy_maternal_postpartum_rag/ingestion/chroma_index.py

import json
from pathlib import Path
import logging
import uuid
import os

import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm

# ============================================================
# Pipeline RUN_ID
# ============================================================

RUN_ID_ENV = "TMPRAG_RUN_ID"
print("Indexing started.")

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

LOG_FILE = Path("logs/data_logs/indexing.log")
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("tmprag.ingestion.indexing")
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
# CONFIG (path-driven)
# ============================================================

CHUNKS_DIR = Path("data/chunks")
CHROMA_PATH = Path("data/chroma_db")
COLLECTION_NAME = "maternal_postpartum_chunks"
BATCH_SIZE = 128
RECURSIVE_CHUNK_SCAN = False


def iter_chunks():
    pattern = "**/*_chunks.jsonl" if RECURSIVE_CHUNK_SCAN else "*_chunks.jsonl"
    files = sorted(CHUNKS_DIR.glob(pattern), key=lambda p: str(p).lower())

    logger.info(f"[Index] Scan | dir={CHUNKS_DIR} files={len(files)} recursive={RECURSIVE_CHUNK_SCAN}")

    for f in files:
        logger.info(f"[Index] Read | file={f.name}")
        with open(f, "r", encoding="utf-8") as fin:
            for line in fin:
                if not line.strip():
                    continue
                rec = json.loads(line)
                text = (rec.get("text") or "").strip()
                chunk_id = rec.get("chunk_id")
                if not chunk_id or not text:
                    continue
                yield rec


def main():
    logger.info(f"[Index] Start | chroma_path={CHROMA_PATH} collection={COLLECTION_NAME}")

    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2",
        device="cpu",
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_func,
        metadata={"hnsw:space": "cosine"},
    )

    ids, documents, metadatas = [], [], []
    total_upserted = 0
    seen_ids_in_run = set()

    for rec in tqdm(iter_chunks(), desc="Indexing chunks"):
        chunk_id = rec.get("chunk_id")
        text = (rec.get("text") or "").strip()
        if not chunk_id or not text:
            continue

        if chunk_id in seen_ids_in_run:
            continue
        seen_ids_in_run.add(chunk_id)

        ids.append(chunk_id)
        documents.append(text)

        meta = {
            k: v
            for k, v in {
                "source_file": rec.get("source_file"),
                "page_number": rec.get("page_number"),
                "doc_id": rec.get("doc_id"),
                "country": rec.get("country"),
                "stage": rec.get("stage"),
                "target": rec.get("target"),
                "source_type": rec.get("source_type"),
                "publisher": rec.get("publisher"),
                "topic_hint": rec.get("topic_hint"),
                "lifecycle": rec.get("lifecycle"),
                "version": rec.get("version"),
            }.items()
            if v is not None
        }
        metadatas.append(meta)

        if len(ids) >= BATCH_SIZE:
            collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
            total_upserted += len(ids)
            logger.info(f"[Index] Upserted | total={total_upserted}")
            ids, documents, metadatas = [], [], []

    if ids:
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        total_upserted += len(ids)
        logger.info(f"[Index] Upserted | total={total_upserted}")

    logger.info("[Index] Done")


if __name__ == "__main__":
    main()
print("Indexing done.")