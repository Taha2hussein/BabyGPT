"""SQLite persistence for conversation history and long-term memory."""
from datetime import datetime, timezone
from pathlib import Path
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
DATABASE_URL = f"sqlite:///{DATA_DIR / 'chatbot_memory.sqlite'}"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

def utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)

class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(Integer, primary_key=True, index=True)
    thread_id = Column(String, unique=True, index=True, nullable=False)
    title = Column(String, default="New Chat")
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow)

class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id = Column(Integer, primary_key=True, index=True)
    thread_id = Column(String, index=True, nullable=False)
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=utcnow)

class LongTermMemory(Base):
    __tablename__ = "long_term_memory"
    id = Column(Integer, primary_key=True, index=True)
    thread_id = Column(String, index=True, nullable=False)
    memory = Column(Text, nullable=False)
    created_at = Column(DateTime, default=utcnow)

def init_db():
    Base.metadata.create_all(bind=engine)

def create_or_update_conversation(thread_id: str, first_message: str | None = None):
    with SessionLocal() as db:
        conversation = db.query(Conversation).filter_by(thread_id=thread_id).first()
        if not conversation:
            title = (first_message or "New Chat").strip()[:40] or "New Chat"
            if first_message and len(first_message.strip()) > 40: title += "..."
            db.add(Conversation(thread_id=thread_id, title=title))
        else:
            conversation.updated_at = utcnow()
        db.commit()

def list_conversations():
    with SessionLocal() as db:
        return db.query(Conversation).order_by(Conversation.updated_at.desc()).all()

def save_chat_message(thread_id: str, role: str, content: str):
    with SessionLocal() as db:
        db.add(ChatMessage(thread_id=thread_id, role=role, content=content))
        conversation = db.query(Conversation).filter_by(thread_id=thread_id).first()
        if conversation: conversation.updated_at = utcnow()
        db.commit()

def get_chat_history(thread_id: str):
    with SessionLocal() as db:
        return db.query(ChatMessage).filter_by(thread_id=thread_id).order_by(ChatMessage.id.asc()).all()

def save_memory(thread_id: str, memory: str):
    with SessionLocal() as db:
        db.add(LongTermMemory(thread_id=thread_id, memory=memory))
        db.commit()
    return "Memory saved successfully."

def search_memory(thread_id: str, query: str):
    # Basic local recall: return recent memories for this thread. Query is retained for tool compatibility.
    with SessionLocal() as db:
        memories = db.query(LongTermMemory).filter_by(thread_id=thread_id).order_by(LongTermMemory.created_at.desc()).limit(20).all()
        if not memories: return "No saved memory found for this conversation."
        return "\n".join(f"- {item.memory}" for item in memories)
