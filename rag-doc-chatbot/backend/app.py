import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings

load_dotenv()

app = FastAPI(title="Single-Doc RAG Chatbot (Local, No LLM)", version="1.0")

# ---------------- Config ---------------- #
STORAGE_DIR = Path("storage")  # FAISS index location
TOP_K = int(os.getenv("TOP_K", 4))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


# ---------------- Models ---------------- #
class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    sources: List[str] = []


# ---------------- Utility Functions ---------------- #
def load_vectorstore():
    if not STORAGE_DIR.exists():
        raise RuntimeError("Vector store not found. Run `python ingest.py` first.")
    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
    vs = FAISS.load_local(
        str(STORAGE_DIR),
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vs


# ---------------- Routes ---------------- #
@app.get("/")
def root():
    return {"message": "RAG chatbot API is running locally (no LLM required)"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    # Load FAISS vectorstore
    vs = load_vectorstore()
    retriever = vs.as_retriever(search_kwargs={"k": TOP_K})

    # Retrieve relevant documents
    docs = retriever.get_relevant_documents(req.question)

    if not docs:
        return ChatResponse(answer="I don't know.", sources=[])

    # Combine retrieved documents as the answer
    answer_text = "\n\n".join([f"[{i+1}] {d.page_content}" for i, d in enumerate(docs)])
    sources = [d.metadata.get("source", "unknown") for d in docs][:TOP_K]

    return ChatResponse(answer=answer_text, sources=sources)


@app.post("/rebuild")
def rebuild_index():
    """Rebuild FAISS index from source documents."""
    import subprocess, sys
    try:
        subprocess.check_call([sys.executable, "ingest.py"])
        return {"status": "ok", "message": "Index rebuilt."}
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=str(e))
