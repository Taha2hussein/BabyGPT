"""LangGraph agent powered by a local Ollama model."""
import sqlite3
from pathlib import Path
from functools import lru_cache

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver

from config import OLLAMA_BASE_URL, OLLAMA_MODEL
from tools import tools

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
_CHECKPOINT_CONNECTION = sqlite3.connect(str(DATA_DIR / "langgraph_checkpoints.sqlite"), check_same_thread=False)
_CHECKPOINT_CONNECTION.execute("PRAGMA journal_mode=WAL")
_CHECKPOINTER = SqliteSaver(_CHECKPOINT_CONNECTION)

SYSTEM_PROMPT = """You are BappyGPT Local, a helpful agentic AI assistant.
You can answer normal questions and use tools for calculations, web search, uploaded documents, and conversation memory.
Rules:
- For arithmetic, percentages, roots, or other calculations, use calculator; do not estimate mentally.
- For latest/current/time-sensitive information, use web_search and clearly summarize results. Include URLs from results when available; never invent sources.
- For questions about a file uploaded in this conversation, use search_uploaded_documents and ground your answer in retrieved text. Say when the answer is not found.
- Use remember_this only when the user explicitly asks you to remember/save something.
- Use recall_memory when the user asks about saved information or a previously remembered preference.
- Do not claim you browsed, calculated, searched a file, or saved a memory unless the corresponding tool was used successfully.
- Be clear and concise. The model runs locally; web search and LangSmith tracing (if enabled) are external services.
"""

def build_agent(model_name: str | None = None):
    model = (model_name or OLLAMA_MODEL).strip()
    llm = ChatOllama(
        model=model,
        base_url=OLLAMA_BASE_URL,
        temperature=0.2,
        num_ctx=8192,
    )
    llm_with_tools = llm.bind_tools(tools)

    def chatbot_node(state: MessagesState):
        response = llm_with_tools.invoke([SystemMessage(content=SYSTEM_PROMPT), *state["messages"]])
        return {"messages": [response]}

    workflow = StateGraph(MessagesState)
    workflow.add_node("chatbot", chatbot_node)
    workflow.add_node("tools", ToolNode(tools, handle_tool_errors=True))
    workflow.add_edge(START, "chatbot")
    workflow.add_conditional_edges("chatbot", tools_condition)
    workflow.add_edge("tools", "chatbot")
    return workflow.compile(checkpointer=_CHECKPOINTER)

@lru_cache(maxsize=4)
def get_agent(model_name: str | None = None):
    return build_agent(model_name)
