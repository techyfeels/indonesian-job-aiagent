from typing import Literal, List
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.callbacks import get_openai_callback
from langgraph.graph import StateGraph, END
from src.agents import sql_agent, rag_agent

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    query: str
    intent: str
    sql_result: str
    rag_result: str
    final_answer: str
    suggested_prompts: List[str]

# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
_CLASSIFY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Kamu adalah intent classifier untuk sistem pencari lowongan kerja di Indonesia.
Tugasmu adalah menentukan jenis pertanyaan user dan menjawab HANYA dengan satu kata:
- "sql"  → pertanyaan terstruktur/filter: work type, salary range, lokasi, jumlah lowongan, perusahaan tertentu
- "rag"  → pertanyaan semantik/deskriptif: cocok untuk siapa, skill yang dibutuhkan, deskripsi pekerjaan
- "both" → pertanyaan yang butuh keduanya: rekomendasi lengkap, perbandingan, atau pertanyaan kompleks
Jawab HANYA dengan: sql / rag / both"""),
    ("human", "{query}")
])

def classify_node(state: AgentState) -> AgentState:
    chain = _CLASSIFY_PROMPT | llm
    result = chain.invoke({"query": state["query"]}).content.strip().lower()
    if result not in ("sql", "rag", "both"):
        result = "both"
    return {**state, "intent": result}

def sql_node(state: AgentState) -> AgentState:
    result = sql_agent.run(state["query"])
    return {**state, "sql_result": result}

def rag_node(state: AgentState) -> AgentState:
    result = rag_agent.run(state["query"])
    return {**state, "rag_result": result}

_SYNTHESIZE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Kamu adalah asisten pencari kerja yang membantu user menemukan lowongan di Indonesia.
Gabungkan informasi dari dua sumber di bawah menjadi satu jawaban yang natural dan informatif dalam Bahasa Indonesia.
Jangan sebutkan bahwa ada dua sumber — cukup berikan jawaban yang kohesif.
Jika salah satu sumber tidak relevan atau kosong, abaikan dan fokus ke sumber yang ada."""),
    ("human", """Pertanyaan user: {query}
Hasil pencarian semantik (RAG):
{rag_result}
Hasil dari database (SQL):
{sql_result}
Berikan jawaban gabungan yang membantu.""")
])

def synthesize_node(state: AgentState) -> AgentState:
    chain = _SYNTHESIZE_PROMPT | llm
    answer = chain.invoke({
        "query": state["query"],
        "rag_result": state.get("rag_result", "-"),
        "sql_result": state.get("sql_result", "-"),
    }).content
    return {**state, "final_answer": answer}

_SUGGEST_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Kamu adalah asisten pencari kerja di Indonesia.
Berdasarkan pertanyaan user dan jawaban yang sudah diberikan, buatlah tepat 3 pertanyaan lanjutan yang relevan dan mungkin ingin ditanyakan user.
Format output: hanya 3 pertanyaan, masing-masing di baris baru, tanpa penomoran atau bullet point."""),
    ("human", """Pertanyaan awal: {query}
Jawaban yang diberikan:
{final_answer}
Tulis 3 pertanyaan lanjutan:""")
])

def suggest_node(state: AgentState) -> AgentState:
    chain = _SUGGEST_PROMPT | llm
    raw = chain.invoke({
        "query": state["query"],
        "final_answer": state.get("final_answer", ""),
    }).content
    suggestions = [s.strip() for s in raw.strip().splitlines() if s.strip()][:3]
    return {**state, "suggested_prompts": suggestions}

# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------
def route_intent(state: AgentState) -> Literal["sql", "rag", "both_sql"]:
    intent = state.get("intent", "both")
    if intent == "sql":
        return "sql"
    elif intent == "rag":
        return "rag"
    else:
        return "both_sql"

# ---------------------------------------------------------------------------
# Build Graph
# ---------------------------------------------------------------------------
def _build_graph() -> StateGraph:
    graph = StateGraph(AgentState)
    graph.add_node("classify", classify_node)
    graph.add_node("sql", sql_node)
    graph.add_node("rag", rag_node)
    graph.add_node("both_sql", sql_node)
    graph.add_node("both_rag", rag_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("suggest", suggest_node)
    graph.set_entry_point("classify")
    graph.add_conditional_edges("classify", route_intent, {
        "sql": "sql",
        "rag": "rag",
        "both_sql": "both_sql",
    })
    graph.add_edge("sql", "synthesize")
    graph.add_edge("rag", "synthesize")
    graph.add_edge("both_sql", "both_rag")
    graph.add_edge("both_rag", "synthesize")
    graph.add_edge("synthesize", "suggest")
    graph.add_edge("suggest", END)
    return graph.compile()

_graph = _build_graph()

# ---------------------------------------------------------------------------
# Public interface — dipanggil dari api/main.py
# ---------------------------------------------------------------------------
def run(query: str) -> dict:
    with get_openai_callback() as cb:
        result = _graph.invoke({
            "query": query,
            "intent": "",
            "sql_result": "",
            "rag_result": "",
            "final_answer": "",
            "suggested_prompts": [],
        })

    intent = result.get("intent", "unknown")
    if intent == "sql":
        agent_used = "sql_agent"
    elif intent == "rag":
        agent_used = "rag_agent"
    else:
        agent_used = "rag_agent + sql_agent"

    # Kumpulkan tool messages (hasil mentah tiap agent sebelum synthesis)
    tool_messages = []
    if result.get("sql_result"):
        tool_messages.append({"agent": "sql_agent", "result": result["sql_result"]})
    if result.get("rag_result"):
        tool_messages.append({"agent": "rag_agent", "result": result["rag_result"]})

    input_tokens  = cb.prompt_tokens
    output_tokens = cb.completion_tokens
    price_idr     = round(17_000 * (input_tokens * 0.15 + output_tokens * 0.6) / 1_000_000, 4)

    return {
        "answer":              result["final_answer"],
        "agent":               agent_used,
        "total_input_tokens":  input_tokens,
        "total_output_tokens": output_tokens,
        "price_idr":           price_idr,
        "tool_messages":       tool_messages,
        "suggested_prompts":   result.get("suggested_prompts", []),
    }
