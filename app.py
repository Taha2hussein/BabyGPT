"""FastAPI application and Server-Sent Events chat endpoint."""
import json
import uuid
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk, ToolMessage

from agent import get_agent
from config import OLLAMA_MODEL, OLLAMA_BASE_URL, LANGSMITH_TRACING
from database import init_db, save_chat_message, get_chat_history, create_or_update_conversation, list_conversations
from rag import add_document_to_rag, UPLOAD_DIR

app = FastAPI(title="BappyGPT Local", version="1.0.0", description="Local Ollama agent with LangGraph, RAG, web search and LangSmith tracing")
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR.mkdir(exist_ok=True)
(BASE_DIR / "data").mkdir(exist_ok=True)
init_db()

@app.get("/health")
def health():
    return {"status": "ok", "model": OLLAMA_MODEL, "ollama_base_url": OLLAMA_BASE_URL, "langsmith_tracing": LANGSMITH_TRACING}

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={})

@app.get("/conversations")
def conversations():
    items = list_conversations()
    return {"conversations": [{"thread_id": x.thread_id, "title": x.title, "created_at": x.created_at.isoformat(), "updated_at": x.updated_at.isoformat()} for x in items]}

@app.get("/history/{thread_id}")
def history(thread_id: str):
    messages = get_chat_history(thread_id)
    return {"messages": [{"role": m.role, "content": m.content} for m in messages]}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...), thread_id: str = Form(...)):
    allowed = {".pdf", ".docx", ".txt", ".md", ".py", ".csv"}
    filename = file.filename or "uploaded_file"
    suffix = Path(filename).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(status_code=400, detail="Unsupported file type. Upload PDF, DOCX, TXT, MD, PY, or CSV.")
    if not thread_id or len(thread_id) > 200:
        raise HTTPException(status_code=400, detail="Invalid thread_id.")
    content = await file.read()
    if len(content) > 15 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File is too large. Maximum size is 15 MB.")
    safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in Path(filename).name)[:120]
    path = UPLOAD_DIR / f"{uuid.uuid4().hex}_{safe_name}"
    path.write_bytes(content)
    try:
        create_or_update_conversation(thread_id, "Uploaded document")
        result = add_document_to_rag(str(path), thread_id)
        return {"success": True, "message": f"Uploaded {filename}; indexed {result['chunks']} chunks."}
    except Exception as exc:
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Document ingestion failed: {type(exc).__name__}: {exc}")


def sse_data(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

def should_stream_chunk(chunk, metadata) -> bool:
    metadata = metadata or {}
    if "tool" in str(metadata.get("langgraph_node", "")).lower(): return False
    if isinstance(chunk, ToolMessage): return False
    if not isinstance(chunk, (AIMessage, AIMessageChunk)): return False
    if getattr(chunk, "tool_calls", None) or getattr(chunk, "invalid_tool_calls", None): return False
    return True

def extract_text_from_chunk(chunk) -> str:
    content = getattr(chunk, "content", "")
    if isinstance(content, str): return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str): parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str): parts.append(item["text"])
        return "".join(parts)
    return ""

@app.post("/chat/stream")
async def chat_stream(request: Request):
    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body."}, status_code=400)
    user_message = str(data.get("message", ""))
    thread_id = str(data.get("thread_id", "default"))
    selected_model = str(data.get("model") or OLLAMA_MODEL)
    # Only permit the configured model to avoid arbitrary model selection/resource exhaustion.
    if selected_model != OLLAMA_MODEL:
        selected_model = OLLAMA_MODEL
    if not user_message.strip(): return JSONResponse({"error": "Message is required."}, status_code=400)
    if len(user_message) > 20000: return JSONResponse({"error": "Message is too long (maximum 20,000 characters)."}, status_code=413)
    if not thread_id or len(thread_id) > 200: return JSONResponse({"error": "Invalid thread_id."}, status_code=400)
    agent = get_agent(selected_model)
    create_or_update_conversation(thread_id, user_message)
    save_chat_message(thread_id, "user", user_message)
    config = {"configurable": {"thread_id": thread_id}, "metadata": {"app": "bappygpt-local"}, "tags": ["bappygpt-local"]}

    def event_generator():
        final_answer = ""
        try:
            inputs = {"messages": [HumanMessage(content=user_message)]}
            for chunk, metadata in agent.stream(inputs, config=config, stream_mode="messages"):
                if not should_stream_chunk(chunk, metadata): continue
                token = extract_text_from_chunk(chunk)
                if token:
                    final_answer += token
                    yield sse_data({"token": token})
            if final_answer.strip(): save_chat_message(thread_id, "assistant", final_answer)
            yield sse_data({"done": True})
        except Exception as exc:
            yield sse_data({"error": f"{type(exc).__name__}: {exc}"})
            yield sse_data({"done": True})
    return StreamingResponse(event_generator(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"})

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8080, reload=True)
