"""
qdrant_tool.py
Wrapper untuk melakukan semantic search ke Qdrant Cloud.
Dipakai oleh rag_agent.py — jangan dipanggil langsung dari orchestrator.

Fungsi utama:
    qdrant_search(query, top_k=5) -> list[dict]
"""

import os
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient

load_dotenv()

# ── Konfigurasi ───────────────────────────────────────────────────────────────

QDRANT_URL      = os.getenv("QDRANT_URL")
QDRANT_API_KEY  = os.getenv("QDRANT_API_KEY")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")

COLLECTION_NAME = "indonesian_jobs"
EMBEDDING_MODEL = "text-embedding-3-small"

# ── Inisialisasi client (singleton) ───────────────────────────────────────────
# Client dibuat sekali saat module di-import, bukan setiap kali fungsi dipanggil

_qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

_embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    openai_api_key=OPENAI_API_KEY,
)


# ── Fungsi utama ──────────────────────────────────────────────────────────────

def qdrant_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Melakukan semantic search ke Qdrant berdasarkan query string.

    Args:
        query  : Pertanyaan atau kalimat dari user.
        top_k  : Jumlah hasil yang dikembalikan (default 5).

    Returns:
        List of dict, masing-masing berisi:
            - text         : Isi chunk yang relevan
            - score        : Similarity score (0.0 - 1.0, makin tinggi makin relevan)
            - job_id       : ID job asal
            - job_title    : Judul posisi
            - company_name : Nama perusahaan
            - location     : Lokasi kerja
            - work_type    : Tipe pekerjaan (full-time, remote, dll)
            - chunk_index  : Urutan chunk dalam dokumen asli
    """
    # 1. Embed query pakai model yang sama dengan saat indexing
    query_vector = _embeddings.embed_query(query)

    # 2. Cari di Qdrant
    # Gunakan query_points() — method search() sudah deprecated di qdrant-client >= 1.7
    response = _qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True,    # ikutkan metadata
        with_vectors=False,   # tidak perlu return vectornya
    )

    # 3. Format hasil jadi list of dict yang bersih
    hits = []
    for result in response.points:
        payload = result.payload or {}
        hits.append({
            "text":         payload.get("text", ""),
            "score":        round(result.score, 4),
            "job_id":       payload.get("job_id", ""),
            "job_title":    payload.get("job_title", ""),
            "company_name": payload.get("company_name", ""),
            "location":     payload.get("location", ""),
            "work_type":    payload.get("work_type", ""),
            "chunk_index":  payload.get("chunk_index", 0),
        })

    return hits


def format_search_results(hits: list[dict]) -> str:
    """
    Format hasil search jadi string konteks yang siap dikirim ke LLM.
    Dipakai oleh rag_agent untuk menyusun prompt.

    Contoh output:
        [1] Data Scientist — PT Tokopedia (Jakarta, Full-time) | score: 0.91
        ...teks chunk...

        [2] ...
    """
    if not hits:
        return "Tidak ada hasil yang ditemukan untuk query ini."

    parts = []
    for i, hit in enumerate(hits, start=1):
        header = (
            f"[{i}] {hit['job_title']} — {hit['company_name']}"
            f" ({hit['location']}, {hit['work_type']}) | score: {hit['score']}"
        )
        parts.append(f"{header}\n{hit['text']}")

    return "\n\n".join(parts)


# ── Quick test (jalankan langsung: python src/tools/qdrant_tool.py) ───────────

if __name__ == "__main__":
    test_query = "lowongan data scientist dengan skill Python dan machine learning"
    print(f"Query: {test_query}\n")

    hits = qdrant_search(test_query, top_k=3)

    if not hits:
        print("Tidak ada hasil. Pastikan setup_qdrant.py sudah dijalankan.")
    else:
        for i, h in enumerate(hits, 1):
            print(f"[{i}] {h['job_title']} — {h['company_name']} ({h['location']})")
            print(f"     Score: {h['score']}")
            print(f"     {h['text'][:150]}...")
            print()
