from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Any
from dotenv import load_dotenv

load_dotenv()

from src.agents.orchestrator import run as orchestrate

app = FastAPI(title="Indonesian Job AI")


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer:              str
    agent:               str
    total_input_tokens:  int
    total_output_tokens: int
    price_idr:           float
    tool_messages:       List[Any]
    suggested_prompts:   List[str]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message tidak boleh kosong")
    result = orchestrate(req.message)
    return ChatResponse(**result)
