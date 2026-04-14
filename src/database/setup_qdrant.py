"""
setup_qdrant.py
Membaca dataset JSONL, melakukan chunking & embedding,
lalu mengupload ke Qdrant Cloud sebagai vector index untuk RAG.

Jalankan sekali sebelum menggunakan RAG agent:
    python src/database/setup_qdrant.py
"""

import os
import json
import uuid
import time
from pathlib import Path
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

load_dotenv()

# ── Konfigurasi ──────────────────────────────────────────────────────────────

QDRANT_URL     = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

COLLECTION_NAME = "indonesian_jobs"
EMBEDDING_MODEL = "text-embedding-3-small"   # 1536 dimensi
VECTOR_SIZE     = 1536

CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50

# Batch size kecil supaya tidak timeout saat upload ke Qdrant Cloud
BATCH_SIZE = 20

# Timeout (detik) untuk koneksi ke Qdrant
QDRANT_TIMEOUT = 60

# Path ke file JSONL — sesuaikan jika nama file berbeda
JSONL_PATH = Path("data/raw/jobs.jsonl")


# ── Helper: format satu job jadi teks ────────────────────────────────────────

def format_job_text(job: dict) -> str:
    """
    Gabungkan field-field penting jadi satu string teks
    yang siap di-embed oleh model.
    """
    parts = [
        f"Job Title: {job.get('job_title', '')}",
        f"Company: {job.get('company_name', '')}",
        f"Location: {job.get('location', '')}",
        f"Work Type: {job.get('work_type', '')}",
        f"Salary: {job.get('salary', '')}",
        f"Description: {job.get('job_description', '')}",
    ]
    return "\n".join(p for p in parts if p.split(": ", 1)[1])  # skip field kosong


# ── Load JSONL ────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    jobs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                jobs.append(json.loads(line))
    print(f"[✓] Loaded {len(jobs)} jobs dari {path}")
    return jobs


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_jobs(jobs: list[dict]) -> list[dict]:
    """
    Setiap job di-split jadi beberapa chunk.
    Tiap chunk membawa metadata job aslinya.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = []
    for job in jobs:
        text = format_job_text(job)
        splits = splitter.split_text(text)

        for i, split in enumerate(splits):
            job_id = str(job.get("id", ""))
            # ID deterministik: uuid5 dari job_id + chunk_index
            # Aman dijalankan ulang — tidak akan buat duplikat
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{job_id}_{i}"))
            chunks.append({
                "id":   point_id,
                "text": split,
                "metadata": {
                    "job_id":       job_id,
                    "job_title":    job.get("job_title", ""),
                    "company_name": job.get("company_name", ""),
                    "location":     job.get("location", ""),
                    "work_type":    job.get("work_type", ""),
                    "chunk_index":  i,
                },
            })

    print(f"[✓] Total chunks: {len(chunks)} dari {len(jobs)} jobs")
    return chunks


# ── Setup Qdrant collection ───────────────────────────────────────────────────

def setup_collection(client: QdrantClient, recreate: bool = False):
    existing = [c.name for c in client.get_collections().collections]

    if COLLECTION_NAME in existing:
        if recreate:
            client.delete_collection(COLLECTION_NAME)
            print(f"[!] Collection '{COLLECTION_NAME}' dihapus untuk dibuat ulang.")
        else:
            print(f"[!] Collection '{COLLECTION_NAME}' sudah ada — lanjut upload (upsert).")
            return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE,
        ),
    )
    print(f"[✓] Collection '{COLLECTION_NAME}' berhasil dibuat.")


# ── Embed & upload ke Qdrant ──────────────────────────────────────────────────

def upload_chunks(client: QdrantClient, chunks: list[dict], batch_size: int = BATCH_SIZE):
    embeddings_model = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=OPENAI_API_KEY,
    )

    total = len(chunks)
    for start in range(0, total, batch_size):
        batch = chunks[start : start + batch_size]
        texts = [c["text"] for c in batch]

        # Embed batch
        vectors = embeddings_model.embed_documents(texts)

        # Buat PointStruct dengan ID deterministik
        points = [
            PointStruct(
                id=chunk["id"],
                vector=vector,
                payload={
                    "text": chunk["text"],
                    **chunk["metadata"],
                },
            )
            for chunk, vector in zip(batch, vectors)
        ]

        # Retry 3x jika timeout
        for attempt in range(3):
            try:
                client.upsert(collection_name=COLLECTION_NAME, points=points)
                break
            except Exception as e:
                if attempt < 2:
                    wait = (attempt + 1) * 5
                    print(f"[!] Timeout batch {start}-{start+len(batch)}, retry {attempt+1}/3 dalam {wait}s... ({e})")
                    time.sleep(wait)
                else:
                    raise RuntimeError(
                        f"Gagal upload batch {start}-{start+len(batch)} setelah 3x retry: {e}"
                    ) from e

        end = min(start + batch_size, total)
        print(f"[✓] Upload {end}/{total} chunks...")

    print(f"\n[✓] Selesai! Semua {total} chunks berhasil diupload ke Qdrant.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Validasi env variables
    missing = [k for k, v in {
        "QDRANT_URL": QDRANT_URL,
        "QDRANT_API_KEY": QDRANT_API_KEY,
        "OPENAI_API_KEY": OPENAI_API_KEY,
    }.items() if not v]

    if missing:
        raise EnvironmentError(
            f"Environment variables berikut belum diisi di .env: {missing}"
        )

    if not JSONL_PATH.exists():
        raise FileNotFoundError(
            f"File JSONL tidak ditemukan di: {JSONL_PATH}\n"
            "Pastikan kamu sudah menjalankan setup_sqlite.py terlebih dahulu "
            "atau cek path file JSONL kamu."
        )

    # Jalankan pipeline
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=QDRANT_TIMEOUT,
    )
    print(f"[✓] Terhubung ke Qdrant: {QDRANT_URL}")

    jobs   = load_jsonl(JSONL_PATH)
    chunks = chunk_jobs(jobs)

    # recreate=True → hapus collection lama dan buat ulang (bersih)
    # Gunakan recreate=False jika ingin lanjut dari upload yang terputus
    setup_collection(client, recreate=True)
    upload_chunks(client, chunks)

    # Verifikasi
    info = client.get_collection(COLLECTION_NAME)
    points = client.count(collection_name=COLLECTION_NAME).count
    print(f"\n[✓] Verifikasi collection '{COLLECTION_NAME}':")
    print(f"    Points count  : {points}")


if __name__ == "__main__":
    main()
