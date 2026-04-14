"""
rag_agent.py
RAG Agent — menjawab pertanyaan semantik tentang lowongan kerja
menggunakan Qdrant sebagai vector store dan OpenAI sebagai LLM.

Interface publik (dipanggil dari orchestrator):
    run(query: str) -> str
"""

import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from src.tools.qdrant_tool import qdrant_search, format_search_results

load_dotenv()

# ── LLM ──────────────────────────────────────────────────────────────────────

_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)

# ── Prompt ────────────────────────────────────────────────────────────────────

_RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Kamu adalah asisten pencari kerja yang membantu user menemukan lowongan kerja di Indonesia.
Jawab pertanyaan user berdasarkan konteks lowongan kerja yang diberikan di bawah.
Gunakan Bahasa Indonesia yang natural dan informatif.

Aturan:
- Hanya gunakan informasi dari konteks yang diberikan
- Jika konteks tidak cukup untuk menjawab, katakan dengan jujur
- Sebutkan nama perusahaan dan posisi jika relevan
- Jangan mengarang informasi yang tidak ada di konteks"""),
    ("human", """Pertanyaan: {query}

Konteks lowongan kerja yang relevan:
{context}

Berikan jawaban yang membantu berdasarkan konteks di atas.""")
])

# ── Core function ─────────────────────────────────────────────────────────────

def run(query: str, top_k: int = 5) -> str:
    """
    Menjawab query dengan cara:
    1. Semantic search ke Qdrant → ambil top-K chunk relevan
    2. Format hasil sebagai konteks
    3. Kirim ke LLM untuk di-synthesize jadi jawaban
    4. Return jawaban sebagai string

    Args:
        query : Pertanyaan dari user
        top_k : Jumlah chunk yang diambil dari Qdrant (default 5)

    Returns:
        Jawaban final sebagai string
    """
    # 1. Retrieve — cari chunk relevan di Qdrant
    hits = qdrant_search(query, top_k=top_k)

    if not hits:
        return (
            "Maaf, saya tidak menemukan lowongan yang relevan dengan pertanyaan tersebut. "
            "Coba gunakan kata kunci yang berbeda."
        )

    # 2. Format hasil jadi konteks untuk LLM
    context = format_search_results(hits)

    # 3. Generate jawaban dengan LLM
    chain = _RAG_PROMPT | _llm
    answer = chain.invoke({
        "query": query,
        "context": context,
    }).content

    return answer
