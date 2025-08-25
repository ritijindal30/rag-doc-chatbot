import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings

def load_all_documents(data_dir: Path):
    docs = []
    for p in data_dir.glob("**/*"):
        if p.is_dir():
            continue
        if p.suffix.lower() == ".pdf":
            docs.extend(PyPDFLoader(str(p)).load())
        elif p.suffix.lower() in {".txt", ".md"}:
            docs.extend(TextLoader(str(p), encoding="utf-8").load())
        elif p.suffix.lower() in {".docx"}:
            docs.extend(Docx2txtLoader(str(p)).load())
    return docs

def main():
    load_dotenv()
    data_dir = Path("data")
    storage_dir = Path("storage")
    storage_dir.mkdir(exist_ok=True, parents=True)

    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 120))

    print("🔄 Loading documents from:", data_dir.resolve())
    raw_docs = load_all_documents(data_dir)
    if not raw_docs:
        raise SystemExit("No documents found in ./data. Add your company PDF/DOCX/TXT and rerun.")

    print(f"📄 Loaded {len(raw_docs)} raw documents. Chunking...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    docs = splitter.split_documents(raw_docs)
    print(f"✂️  Created {len(docs)} chunks. Building embeddings...")

    model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    embeddings = SentenceTransformerEmbeddings(model_name=model_name)

    print("🧠 Building FAISS index...")
    vs = FAISS.from_documents(docs, embeddings)
    vs.save_local(str(storage_dir))

    print("✅ Ingestion complete. Index saved to ./storage")

if __name__ == "__main__":
    main()
