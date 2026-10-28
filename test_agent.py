"""Manual end-to-end test; requires Ollama and qwen2.5:7b."""
from langchain_core.messages import HumanMessage
from agent import get_agent
from config import OLLAMA_MODEL

agent = get_agent(OLLAMA_MODEL)
config = {"configurable": {"thread_id": "manual-test-thread"}}
for chunk, metadata in agent.stream({"messages": [HumanMessage(content="Calculate 12500 * 37 using the calculator tool.")]}, config=config, stream_mode="messages"):
    content = getattr(chunk, "content", "")
    if isinstance(content, str) and content:
        print(content, end="", flush=True)
print()
