# 💼 Indonesian Job AI Agent

Sistem multi-agent berbasis AI untuk menjawab pertanyaan seputar lowongan kerja di Indonesia. Dibangun menggunakan **LangGraph**, **RAG (Retrieval-Augmented Generation)**, dan **SQL querying** — dikoordinasikan oleh orchestrator agent yang secara otomatis merouting setiap pertanyaan ke agent yang tepat.

> Final Project — AI Engineering | Purwadhika 2025

> Kelompok 3

> Tim: Fil Ardhi Kamza & Anintyo Herdadi

---

## 🏗️ Arsitektur Sistem

```
Input User (Streamlit UI)
        ↓
  [FastAPI /chat]
        ↓
[Orchestrator Agent]  ← LangGraph StateGraph
        ↓
  klasifikasi intent
        ↓
  ┌─────────────────────┐
  │                     │
[RAG Agent]       [SQL Agent]
  │                     │
Qdrant Cloud       SQLite DB
(pencarian semantik)  (query terstruktur)
  │                     │
  └──────────┬──────────┘
             ↓
    [Response Synthesizer]
             ↓
       Jawaban Final
```

**Logika routing:**
- `rag` → pertanyaan semantik: deskripsi pekerjaan, skill yang dibutuhkan, info perusahaan
- `sql` → pertanyaan terstruktur: rentang gaji, tipe pekerjaan, lokasi, jumlah lowongan
- `both` → pertanyaan kompleks yang membutuhkan kedua sumber

---

## 🛠️ Tech Stack

| Komponen | Tools |
|---|---|
| Agent Framework | LangChain + LangGraph |
| LLM | OpenAI GPT-4o-mini |
| Vector Database | Qdrant Cloud |
| SQL Database | SQLite |
| API Layer | FastAPI |
| UI | Streamlit |
| Deployment | Docker + GCP Cloud Run |

---

## 📁 Struktur Folder

```
indonesian-job-aiagent/
├── api/
│   └── main.py                 # FastAPI entrypoint (/chat, /health)
├── src/
│   ├── agents/
│   │   ├── orchestrator.py     # LangGraph orchestrator
│   │   ├── rag_agent.py        # RAG agent (Qdrant + OpenAI)
│   │   └── sql_agent.py        # SQL agent (SQLite + OpenAI)
│   ├── tools/
│   │   ├── qdrant_tool.py      # Wrapper semantic search ke Qdrant
│   │   └── sql_tool.py         # Wrapper query ke SQLite
│   ├── database/
│   │   ├── setup_sqlite.py     # Load JSONL → SQLite
│   │   └── setup_qdrant.py     # Embed & indexing JSONL → Qdrant Cloud
│   └── utils/
│       └── salary_parser.py    # Normalisasi string salary → integer
├── streamlit/
│   └── app.py                  # Streamlit chat UI
├── data/
│   ├── raw/                    # Dataset JSONL original
│   └── jobs.db                 # SQLite database
├── Dockerfile
├── requirements.txt
└── .env.example
```

---

## 🚀 Cara Menjalankan

### 1. Clone & Install

```bash
git clone https://github.com/anintyo/indonesian-job-aiagent.git
cd indonesian-job-aiagent
pip install -r requirements.txt
```

### 2. Konfigurasi Environment

```bash
cp .env.example .env
```

Isi file `.env`:
```
OPENAI_API_KEY=your_openai_api_key
QDRANT_URL=https://your-cluster.qdrant.io:6333
QDRANT_API_KEY=your_qdrant_api_key
```

### 3. Setup Database

```bash
# Load JSONL → SQLite
python src/database/setup_sqlite.py

# Embed & indexing → Qdrant Cloud
python src/database/setup_qdrant.py
```

### 4. Jalankan API

```bash
uvicorn api.main:app --reload
```

### 5. Jalankan Streamlit UI

```bash
streamlit run streamlit/app.py
```

---

## 🌐 API yang Sudah Di-deploy

**Base URL:** `https://indonesian-job-ai-421382217116.asia-southeast1.run.app`

| Endpoint | Method | Deskripsi |
|---|---|---|
| `/health` | GET | Cek status API |
| `/chat` | POST | Kirim pertanyaan ke agent |

**Contoh request:**
```bash
curl -X POST https://indonesian-job-ai-421382217116.asia-southeast1.run.app/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "skill apa yang dibutuhkan untuk posisi data scientist?"}'
```

**Contoh response:**
```json
{
  "response": "Untuk posisi data scientist, keterampilan yang dibutuhkan meliputi..."
}
```

---

## 🐳 Docker

```bash
# Build image
docker build -t indonesian-job-ai .

# Jalankan container
docker run -p 8000:8000 --env-file .env indonesian-job-ai
```

---

## 📋 Pembagian Tugas

| Tugas | Person |
|---|---|
| Data preprocessing & setup SQLite | Fil |
| SQL Agent | Fil |
| FastAPI | Fil |
| Orchestrator (LangGraph) | Fil |
| Docker & GCP Cloud Run | Fil |
| Qdrant indexing | Anin |
| RAG Agent | Anin |
