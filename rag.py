"""Local document ingestion and RAG using Ollama embeddings + ChromaDB."""
from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import docx2txt

from config import OLLAMA_BASE_URL, OLLAMA_EMBEDDING_MODEL

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
CHROMA_DIR = BASE_DIR / "chroma_db"
UPLOAD_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(exist_ok=True)

embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
# Use a new collection name so old Google-embedding vectors are never mixed with local vectors.
vectorstore = Chroma(
    collection_name="bappygpt_local_docs_v1",
    embedding_function=embeddings,
    persist_directory=str(CHROMA_DIR),
)

def read_file_text(file_path: str) -> str:
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if suffix == ".docx":
        return docx2txt.process(str(path))
    if suffix in {".txt", ".md", ".py", ".csv"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    raise ValueError("Unsupported file type. Upload PDF, DOCX, TXT, MD, PY, or CSV.")

def add_document_to_rag(file_path: str, thread_id: str) -> dict:
    text = read_file_text(file_path)
    if not text.strip():
        raise ValueError("No text could be extracted from this file.")
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
    chunks = splitter.split_text(text)
    filename = Path(file_path).name
    docs: List[Document] = [
        Document(page_content=chunk, metadata={"thread_id": thread_id, "source": filename})
        for chunk in chunks
    ]
    vectorstore.add_documents(docs)
    return {"filename": filename, "chunks": len(docs)}

def retrieve_from_rag(query: str, thread_id: str, k: int = 4) -> str:
    docs = vectorstore.similarity_search(query, k=k, filter={"thread_id": thread_id})
    if not docs:
        return "No relevant uploaded document content found for this conversation."
    return "\n\n".join(
        f"[Source {i}: {doc.metadata.get('source', 'uploaded document')}]\n{doc.page_content}"
        for i, doc in enumerate(docs, start=1)
    )
