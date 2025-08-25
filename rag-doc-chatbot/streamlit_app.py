import os
import json
import httpx
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Single-Doc RAG Chatbot", page_icon="💬")
st.title("💬 Single-Document RAG Chatbot")

with st.sidebar:
    st.header("⚙️ Settings")
    st.write("1) Place your PDF/DOCX/TXT in the `data/` folder.\n2) Run `python ingest.py`.\n3) Start the backend.\n4) Ask questions here.")
    backend = st.text_input("Backend URL", BACKEND_URL)
    if st.button("Rebuild Index"):
        with st.spinner("Rebuilding index..."):
            try:
                r = httpx.post(f"{backend}/rebuild", timeout=600.0)
                st.success("Index rebuilt ✅")
            except Exception as e:
                st.error(f"Failed: {e}")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

q = st.chat_input("Ask something from your document (e.g., What are the business hours?)")
if q:
    st.session_state.messages.append({"role":"user","content":q})
    with st.chat_message("user"):
        st.markdown(q)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                r = httpx.post(f"{backend}/chat", json={"question": q}, timeout=120.0)
                r.raise_for_status()
                data = r.json()
                answer = data.get("answer","")
                sources = data.get("sources",[])
                st.markdown(answer)
                if sources:
                    st.caption("Sources: " + ", ".join(sorted(set(sources))))
                st.session_state.messages.append({"role":"assistant","content":answer})
            except Exception as e:
                st.error(f"Request failed: {e}")
