"""Tools available to the local LangGraph agent."""
import ast
import math
import operator
from typing import Any

from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from config import TAVILY_API_KEY
from database import save_memory, search_memory
from rag import retrieve_from_rag

# Safe, deliberately limited arithmetic evaluator. It never calls eval().
_BIN_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_MATH_FUNCS = {
    "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan,
    "log": math.log, "log10": math.log10, "exp": math.exp,
    "floor": math.floor, "ceil": math.ceil, "fabs": math.fabs,
    "factorial": math.factorial, "degrees": math.degrees, "radians": math.radians,
    "abs": abs, "round": round, "min": min, "max": max,
}
_MATH_CONSTS = {"pi": math.pi, "e": math.e, "tau": math.tau}

def _safe_eval(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _safe_eval(node.left), _safe_eval(node.right)
        if isinstance(node.op, ast.Pow) and (abs(right) > 100 or abs(left) > 1e100):
            raise ValueError("Exponent or base is too large.")
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_safe_eval(node.operand))
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_safe_eval(x) for x in node.elts]
    if isinstance(node, ast.Name) and node.id in _MATH_CONSTS:
        return _MATH_CONSTS[node.id]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _MATH_FUNCS:
        if node.keywords:
            raise ValueError("Keyword arguments are not supported.")
        args = [_safe_eval(a) for a in node.args]
        return _MATH_FUNCS[node.func.id](*args)
    raise ValueError("Unsupported expression. Use arithmetic, parentheses, pi/e, or supported functions such as sqrt(16).")

@tool
def calculator(expression: str) -> str:
    """Calculate a mathematical expression accurately. Supports + - * / // % **, parentheses, pi, e, and functions such as sqrt(16), sin(0), log(10), abs(-4), round(2.5), min(1,2), max(1,2)."""
    try:
        if len(expression) > 300:
            return "Calculation error: expression is too long."
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree)
        if isinstance(result, float) and not math.isfinite(result):
            return "Calculation error: result is not finite."
        return str(result)
    except Exception as exc:
        return f"Calculation error: {exc}"

@tool
def search_uploaded_documents(query: str, config: RunnableConfig) -> str:
    """Search documents uploaded in this conversation for relevant content. Use for questions about uploaded PDFs, DOCX, TXT, MD, PY, or CSV files."""
    thread_id = (config.get("configurable") or {}).get("thread_id", "default")
    return retrieve_from_rag(query=query, thread_id=thread_id)

@tool
def remember_this(memory: str, config: RunnableConfig) -> str:
    """Save an important fact or preference when the user explicitly asks you to remember it."""
    thread_id = (config.get("configurable") or {}).get("thread_id", "default")
    return save_memory(thread_id=thread_id, memory=memory)

@tool
def recall_memory(query: str, config: RunnableConfig) -> str:
    """Recall facts previously saved for this conversation thread. Use when the user asks what you remember about them."""
    thread_id = (config.get("configurable") or {}).get("thread_id", "default")
    return search_memory(thread_id=thread_id, query=query)

@tool
def web_search(query: str) -> str:
    """Search the public web for current events, recent releases, latest information, or facts that may have changed. Returns web results and URLs."""
    if not TAVILY_API_KEY:
        return "Web search is not configured. Add TAVILY_API_KEY to your .env file and restart the app. The rest of the assistant works without it."
    try:
        from langchain_tavily import TavilySearch
        search = TavilySearch(max_results=5, topic="general", search_depth="advanced")
        result = search.invoke({"query": query})
        return str(result)
    except Exception as exc:
        return f"Web search failed: {type(exc).__name__}: {exc}"

tools = [calculator, search_uploaded_documents, remember_this, recall_memory, web_search]
