"""
Document Q&A (RAG) - Streamlit app powered by Groq.

Pipeline: upload file -> chunk -> embed (local) -> FAISS -> retrieve top-k
          -> Groq LLM answers using ONLY the retrieved chunks (streamed).

Install:
    pip install streamlit groq sentence-transformers faiss-cpu pypdf numpy

API key (any one of these):
    export GROQ_API_KEY="gsk_..."
    or .streamlit/secrets.toml  ->  GROQ_API_KEY = "gsk_..."
    or paste it in the sidebar

Run:
    streamlit run rag_groq_app.py
"""

import hashlib
import io
import os
import re
from dataclasses import dataclass

import faiss
import numpy as np
import streamlit as st
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "all-MiniLM-L6-v2"
GROQ_MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]

SYSTEM_PROMPT = (
    "You are a careful assistant that answers questions about a document. "
    "Use ONLY the provided context excerpts. If the answer is not in the context, "
    "say \"I couldn't find that in the document.\" Do not guess or use outside knowledge. "
    "Keep answers concise and cite the chunk(s) you used like [Chunk 3]."
)

st.set_page_config(page_title="Document Q&A (Groq RAG)", page_icon="📄", layout="wide")


# ----------------------------------------------------------------------------
# Pipeline pieces
# ----------------------------------------------------------------------------
def load_document(file_bytes: bytes, filename: str) -> str:
    name = filename.lower()
    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(file_bytes))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    return file_bytes.decode("utf-8", errors="ignore")


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            space = text.rfind(" ", start + chunk_size // 2, end)
            if space != -1:
                end = space
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


@dataclass
class Hit:
    chunk_id: int
    score: float
    text: str


@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL)


class VectorStore:
    def __init__(self, chunks: list[str]):
        self.chunks = chunks
        self.embedder = get_embedder()
        vectors = self.embedder.encode(chunks, normalize_embeddings=True).astype("float32")
        self.index = faiss.IndexFlatIP(vectors.shape[1])  # cosine on normalized vectors
        self.index.add(vectors)

    def search(self, query: str, k: int) -> list[Hit]:
        q = self.embedder.encode([query], normalize_embeddings=True).astype("float32")
        scores, ids = self.index.search(q, min(k, len(self.chunks)))
        return [
            Hit(int(i), float(s), self.chunks[int(i)])
            for s, i in zip(scores[0], ids[0])
            if i != -1
        ]


def stream_answer(client: Groq, model: str, question: str, hits: list[Hit], temperature: float):
    context = "\n\n".join(f"[Chunk {h.chunk_id}]\n{h.text}" for h in hits)
    stream = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=800,
        stream=True,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )
    for event in stream:
        delta = event.choices[0].delta.content
        if delta:
            yield delta


def get_api_key(sidebar_value: str) -> str:
    if sidebar_value:
        return sidebar_value
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    return os.getenv("GROQ_API_KEY", "")


def render_sources(hits_data: list[dict]) -> None:
    with st.expander(f"Sources ({len(hits_data)} chunks)"):
        for h in hits_data:
            st.markdown(f"**Chunk {h['chunk_id']}**  ·  similarity `{h['score']:.3f}`")
            st.caption(h["text"])


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")

    key_input = st.text_input("Groq API key", type="password",
                              help="Leave empty to use GROQ_API_KEY from env or secrets.")
    api_key = get_api_key(key_input)

    model_choice = st.selectbox("Groq model", GROQ_MODELS + ["Custom..."])
    if model_choice == "Custom...":
        model_choice = st.text_input("Model ID", value=GROQ_MODELS[0])
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)

    st.subheader("Retrieval")
    chunk_size = st.slider("Chunk size (chars)", 300, 2000, 800, 100)
    overlap = st.slider("Chunk overlap (chars)", 0, 400, 150, 25)
    top_k = st.slider("Top-k chunks", 1, 8, 3)

    if st.button("Clear chat", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
st.title("📄 Document Q&A")
st.caption("Basic RAG: chunk → embed → retrieve → answer with Groq, grounded in your document.")

uploaded = st.file_uploader("Upload a PDF, TXT or MD file", type=["pdf", "txt", "md"])

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if uploaded is None:
    st.info("Upload a document to start asking questions.")
    st.stop()

# (Re)build the index only when the file or chunk settings change
file_bytes = uploaded.getvalue()
index_key = (hashlib.md5(file_bytes).hexdigest(), chunk_size, overlap)

if st.session_state.get("index_key") != index_key:
    with st.spinner("Reading, chunking and embedding..."):
        text = load_document(file_bytes, uploaded.name)
        if not text.strip():
            st.error("No text could be extracted. Scanned PDFs need OCR first.")
            st.stop()
        chunks = chunk_text(text, chunk_size, overlap)
        st.session_state["store"] = VectorStore(chunks)
        st.session_state["index_key"] = index_key
        st.session_state["doc_chars"] = len(text)
        if st.session_state.get("doc_hash") != index_key[0]:
            st.session_state["messages"] = []  # new document -> fresh chat
            st.session_state["doc_hash"] = index_key[0]

store: VectorStore = st.session_state["store"]
c1, c2, c3 = st.columns(3)
c1.metric("Document", uploaded.name)
c2.metric("Characters", f"{st.session_state['doc_chars']:,}")
c3.metric("Chunks", len(store.chunks))

# Replay history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            render_sources(msg["sources"])

# New question
question = st.chat_input("Ask something about the document...")
if question:
    if not api_key:
        st.error("Add your Groq API key in the sidebar (or set GROQ_API_KEY).")
        st.stop()

    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    hits = store.search(question, k=top_k)
    sources = [{"chunk_id": h.chunk_id, "score": h.score, "text": h.text} for h in hits]

    with st.chat_message("assistant"):
        try:
            client = Groq(api_key=api_key)
            answer = st.write_stream(
                stream_answer(client, model_choice, question, hits, temperature)
            )
            render_sources(sources)
            st.session_state["messages"].append(
                {"role": "assistant", "content": answer, "sources": sources}
            )
        except Exception as e:
            st.error(f"Groq request failed: {e}")