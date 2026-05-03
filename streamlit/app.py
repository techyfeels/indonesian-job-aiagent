"""
app.py — Streamlit UI untuk Indonesian Job AI Agent
Konek ke FastAPI yang sudah di-deploy di GCP Cloud Run.

Jalankan:
    streamlit run streamlit/app.py
"""

import streamlit as st
import requests

# ── Konfigurasi ───────────────────────────────────────────────────────────────

API_URL         = "https://indonesian-job-ai-421382217116.asia-southeast1.run.app"
CHAT_ENDPOINT   = f"{API_URL}/chat"
HEALTH_ENDPOINT = f"{API_URL}/health"

PLACEHOLDER = (
    "Contoh RAG: 'Skill apa yang dibutuhkan untuk posisi data scientist?' | "
    "Contoh SQL: 'Ada berapa lowongan full-time di Jakarta?'"
)

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


def send_message(user_message: str) -> dict:
    full_message = user_message + build_filter_suffix()
    try:
        response = requests.post(
            CHAT_ENDPOINT,
            json={"message": full_message},
            timeout=60,
        )
        if response.status_code == 200:
            return {"ok": True, "data": response.json()}
        else:
            return {"ok": False, "error": f"⚠️ Server error ({response.status_code}). Coba lagi."}
    except requests.exceptions.Timeout:
        return {"ok": False, "error": "⏱️ Request timeout. Server sedang sibuk, coba lagi."}
    except requests.exceptions.ConnectionError:
        return {"ok": False, "error": "❌ Tidak bisa konek ke server."}
    except Exception as e:
        return {"ok": False, "error": f"❌ Error: {e}"}


def render_response(data: dict):
    """Tampilkan jawaban utama + metadata dalam expander.
    Kompatibel dengan API lama (response) dan baru (answer).
    """
    answer = data.get("answer") or data.get("response", "Tidak ada jawaban.")
    st.write(answer)

    # Hanya tampilkan Detail Response jika API baru sudah di-deploy (ada field 'answer')
    if "answer" in data:
        with st.expander("📊 Detail Response"):
            col1, col2, col3 = st.columns(3)
            col1.metric("🤖 Agent", data.get("agent", "-"))
            col2.metric("📥 Input Tokens", data.get("total_input_tokens", 0))
            col3.metric("📤 Output Tokens", data.get("total_output_tokens", 0))

            price = data.get("price_idr", 0)
            st.metric("💰 Estimated Price", f"Rp {price:,.4f}")

            tool_messages = data.get("tool_messages", [])
            if tool_messages:
                st.markdown("**🔧 Tool Messages**")
                for tm in tool_messages:
                    st.markdown(f"📌 **{tm.get('agent', 'agent')}**")
                    st.text(tm.get("result", "-"))

# ── Main ──────────────────────────────────────────────────────────────────────

st.title("🤖 Indonesian Job AI Agent")
st.caption("Tanyakan apa saja tentang lowongan kerja di Indonesia.")

# Tampilkan riwayat chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and isinstance(msg.get("data"), dict):
            render_response(msg["data"])
        else:
            st.write(msg["content"])

# Input
user_input = st.chat_input(PLACEHOLDER)

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Mencari..."):
            result = send_message(user_input)

        if result["ok"]:
            render_response(result["data"])
            st.session_state.messages.append({
                "role": "assistant",
                "data": result["data"],
            })
        else:
            st.error(result["error"])
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["error"],
            })
