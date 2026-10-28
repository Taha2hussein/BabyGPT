# BappyGPT Local (Ollama + Qwen2.5:7B)

A local-first adaptation of BappyGPT. It keeps the original FastAPI/Jinja chat UI, streaming responses, conversation history, document uploads + RAG, web search, calculator, and memory tools. Gemini is removed.

## Requirements
- macOS with Python 3.11 recommended (Python 3.13 may work, but dependency compatibility varies).
- Ollama installed and running.
- `qwen2.5:7b` chat model and `nomic-embed-text` embeddings model pulled locally.
- Tavily API key is optional and only needed for web search.
- LangSmith is optional; enabling it sends traces to the configured LangSmith service. Do not enable tracing for sensitive conversations unless you understand the data being sent.

## Setup on macOS
```bash
cd BappyGPT-Local
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
ollama serve  # If Ollama is not already running; use a separate terminal
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
````
Edit `.env` if needed. For web search set `TAVILY_API_KEY`. For LangSmith set `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY`, and `LANGSMITH_PROJECT`.

LangSmith's environment-variable tracing is automatic through LangChain. Depending on the installed LangChain version, `LANGCHAIN_TRACING_V2=true` may also be recognized; this project uses `LANGSMITH_TRACING`.

## Run
```bash
python test.py
python app.py
```
Open http://127.0.0.1:8080. API docs are at http://127.0.0.1:8080/docs and health check at http://127.0.0.1:8080/health.

## Features
- Local chat: Ollama `qwen2.5:7b`
- LangGraph tool-calling workflow
- Safe calculator (AST allow-list; no `eval`)
- Optional Tavily web search
- Local RAG with Ollama `nomic-embed-text` + ChromaDB
- SQLite chat history, memory, and LangGraph checkpoints
- LangSmith tracing if configured
- GitHub Actions compile + smoke tests

## Notes
- Web search requires a Tavily key. The rest of the app works without it.
- LangSmith tracing is external, not fully local. Disable it for privacy (`LANGSMITH_TRACING=false`).
- RAG uses a new Chroma collection (`bappygpt_local_docs_v1`) so old Gemini embeddings are not mixed with local vectors.
- This is a learning project, not hardened for public deployment. Add authentication, CSRF/rate limiting, stronger upload isolation, and production secret management before exposing it publicly.
- Do not deploy the local Ollama setup with the provided Dockerfile unchanged: Docker would need network access to the host Ollama service and persistent data volumes.
