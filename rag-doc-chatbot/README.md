# 📄 Single-Document RAG Chatbot (FastAPI + FAISS + LangChain + Streamlit)

A turnkey template to build a **live chatbot** that answers questions **only** from one (or a few) documents — perfect for things like a healthcare company's patient info sheet where queries like _"What are your business hours?"_ must be answered from the doc.

---

## ✨ Features
- **RAG pipeline** with FAISS vector store and Sentence-Transformer embeddings (local, free).
- **FastAPI** backend with `/chat` and `/rebuild` endpoints.
- **Streamlit** chat UI.
- **Doc ingestion** supports **PDF**, **DOCX**, **TXT**.
- **Grounded answers**: if the info isn’t in your doc, the bot says it doesn’t know.
- Minimal vendor lock-in; LLM is OpenAI by default but you can swap it.

---

## 🧱 Project Structure
```
rag-doc-chatbot/
├─ backend/
│  └─ app.py            # FastAPI server (RAG pipeline + endpoints)
├─ data/
│  └─ sample_clinic_info.txt  # Example doc with business hours
├─ storage/             # FAISS index gets stored here after ingestion
├─ ingest.py            # Load & chunk docs, build FAISS index
├─ streamlit_app.py     # Simple chat UI
├─ requirements.txt
└─ .env.example
```

---

## 🚀 Quickstart

> Prereqs: Python 3.10+ recommended.

```bash
# 1) Create and activate a venv (recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Add your docs
# Put a PDF/DOCX/TXT into ./data (sample already provided)

# 4) (Optional) Configure environment
cp .env.example .env
# - Put OPENAI_API_KEY in .env (for better generations)
# - Tweak CHUNK_SIZE/OVERLAP/TOP_K if needed

# 5) Build the vector index
python ingest.py

# 6) Start the backend API
uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000

# 7) Start the UI (new terminal)
streamlit run streamlit_app.py
```

Open Streamlit URL shown in the terminal, ask: **"What are the business hours?"**

---

## 🧠 How the RAG Flow Works

```
            ┌────────────────────┐
            │    Your Document   │  (PDF/DOCX/TXT)
            └─────────┬──────────┘
                      │ Load & Split
                      ▼
            ┌────────────────────┐
            │  Text Chunks       │  (Recursive splitter)
            └─────────┬──────────┘
                      │ Embed (local MiniLM)
                      ▼
            ┌────────────────────┐
            │  FAISS Vector DB   │  (./storage)
            └─────────┬──────────┘
                      │ Retrieve top-k
                      ▼
            ┌────────────────────┐
            │  Prompt Template   │  (Context + Question)
            └─────────┬──────────┘
                      │ Generate (OpenAI or other)
                      ▼
            ┌────────────────────┐
            │  Grounded Answer   │
            └────────────────────┘
```

**Why these choices?**
- **RecursiveCharacterTextSplitter**: robust across formats; preserves local coherence.
- **Sentence-Transformers (MiniLM)**: fast, small, solid recall; no API cost.
- **FAISS**: battle-tested local vector index; easy `save_local/load_local`.
- **ChatOpenAI (OpenAI)**: strong instruction-following for concise, accurate answers; you can swap it.

**Alternatives you can plug in:**
- **Embeddings**: `text-embedding-3-small` (OpenAI), `bge-small-en`, `all-mpnet-base-v2`.
- **Vector DB**: Chroma, Qdrant, Weaviate, Pinecone.
- **Chunking**: token-based splitters, semantic split (e.g., `TokenTextSplitter` or `SemanticChunker`).

---

## 🔄 Updating the Knowledge
- Replace files in `./data/` with your latest document(s).
- Run `python ingest.py` (or click **Rebuild Index** in the Streamlit sidebar).
- That’s it — the bot starts answering from the new content.

---

## 🔒 Guardrails & Tips
- The system prompt forces **"answer only from context"**. The bot will say _"I don't know"_ if info is missing.
- Keep documents **clean & up-to-date** (e.g., a single source of truth per clinic).
- For hours/phones/emails, ensure they appear **verbatim** in the doc — the bot will quote them.

---

## 🔁 Switching the LLM (Optional)
By default we use OpenAI via `langchain-openai`. To switch:
- **Ollama (local)**: Replace `get_llm()` in `backend/app.py` with an Ollama LLM wrapper.
- **Groq**: Use `langchain-groq` and set `GROQ_API_KEY`.
- **Transformers pipeline (local)**: Use `langchain`'s `HuggingFacePipeline` with an instruct model.

---

## 🧪 Test Prompts
- "What are the business hours?"
- "Do you offer lab tests on Saturday? Until what time are samples accepted?"
- "Which insurers are accepted and how to use cashless?"
- "What is the emergency after-hours contact?"

---

## 🐞 Common Issues
- **`Vector store not found`**: Run `python ingest.py` after placing docs in `./data/`.
- **OpenAI key missing**: Add `OPENAI_API_KEY` to `.env` or remove OpenAI usage and switch to a local LLM.
- **Poor recall**: Increase `TOP_K`, reduce `CHUNK_SIZE`, increase `CHUNK_OVERLAP`.
- **Duplicates**: Keep `./data` tidy; re-run ingestion after changes.

---

## 📦 Deploy (quick hints)
- **Backend**: `uvicorn` behind Nginx or run as a service (systemd) or Docker.
- **UI**: Streamlit can be containerized or replaced with a React frontend calling the same API.
- **Secrets**: Mount `.env` with proper permissions.

---

Happy building! 💪
