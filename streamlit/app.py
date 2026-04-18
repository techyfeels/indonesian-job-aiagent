"""
app.py — Streamlit UI untuk Indonesian Job AI Agent
Konek ke FastAPI yang sudah di-deploy di GCP Cloud Run.

Jalankan:
    streamlit run streamlit/app.py
"""

import streamlit as st
import requests

# ── Konfigurasi ───────────────────────────────────────────────────────────────

API_URL        = "https://indonesian-job-ai-421382217116.asia-southeast1.run.app"
CHAT_ENDPOINT  = f"{API_URL}/chat"
HEALTH_ENDPOINT = f"{API_URL}/health"

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Indonesian Job AI",
    page_icon="💼",
    layout="wide",
)

# ── Session state ─────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("💼 Indonesian Job AI")
    st.caption("Multi-agent sistem pencari lowongan kerja di Indonesia")
    st.divider()

    # Status API
    st.subheader("🔌 Status API")
    if st.button("Cek Koneksi", use_container_width=True):
        try:
            r = requests.get(HEALTH_ENDPOINT, timeout=10)
            if r.status_code == 200:
                st.success("● Online")
            else:
                st.error(f"● Error ({r.status_code})")
        except Exception:
            st.error("● Tidak terjangkau")

    st.divider()

    # Filter
    st.subheader("🔍 Filter Pencarian")
    st.caption("Filter akan otomatis ditambahkan ke pertanyaanmu.")

    filter_lokasi = st.selectbox(
        "Lokasi",
        ["Semua", "Jakarta", "Bandung", "Surabaya", "Yogyakarta",
         "Bali", "Medan", "Semarang", "Remote"],
    )

    filter_worktype = st.selectbox(
        "Tipe Pekerjaan",
        ["Semua", "Full Time", "Part Time", "Contract", "Internship", "Freelance"],
    )

    filter_salary = st.selectbox(
        "Kisaran Gaji",
        ["Semua", "Di bawah 5 juta", "5 - 10 juta", "10 - 20 juta", "Di atas 20 juta"],
    )

    st.divider()

    if st.button("🗑️ Hapus Riwayat Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Helper ────────────────────────────────────────────────────────────────────

def build_filter_suffix() -> str:
    parts = []
    if filter_lokasi != "Semua":
        parts.append(f"di {filter_lokasi}")
    if filter_worktype != "Semua":
        parts.append(f"tipe pekerjaan {filter_worktype}")
    if filter_salary != "Semua":
        parts.append(f"gaji {filter_salary}")
    return (", " + ", ".join(parts)) if parts else ""


def send_message(user_message: str) -> str:
    full_message = user_message + build_filter_suffix()
    try:
        response = requests.post(
            CHAT_ENDPOINT,
            json={"message": full_message},
            timeout=60,
        )
        if response.status_code == 200:
            return response.json().get("response", "Tidak ada jawaban.")
        else:
            return f"⚠️ Server error ({response.status_code}). Coba lagi."
    except requests.exceptions.Timeout:
        return "⏱️ Request timeout. Server sedang sibuk, coba lagi."
    except requests.exceptions.ConnectionError:
        return "❌ Tidak bisa konek ke server."
    except Exception as e:
        return f"❌ Error: {e}"

# ── Main ──────────────────────────────────────────────────────────────────────

st.title("🤖 Indonesian Job AI Agent")
st.caption("Tanyakan apa saja tentang lowongan kerja di Indonesia.")

# Tampilkan riwayat chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Input
user_input = st.chat_input("Contoh: 'lowongan data scientist di Jakarta'")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Mencari..."):
            reply = send_message(user_input)
        st.write(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
