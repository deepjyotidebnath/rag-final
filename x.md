1. **Basic Document Q&A (Single File)** — Load one text/PDF file, chunk it, embed it, and answer questions using an LLM with the most relevant chunk as context; your first working RAG pipeline.

2. **Multi-Document RAG with Vector Store** — Ingest many documents into a proper vector database (e.g. Chroma/FAISS) and retrieve the top-k relevant chunks across the whole collection.

3. **Improved Retrieval (Chunking, Metadata & Hybrid Search)** — Tune chunk size/overlap, attach metadata (source, page, date), and combine keyword + semantic search for better retrieval accuracy.

4. **RAG with Source Citations & Answer Evaluation** — Return answers with cited sources/page numbers, and build a simple eval script to check if retrieved chunks actually support the answer (catches hallucination).

5. **Enterprise RAG QA System (Full App)** — Wrap it all in a real app: file upload, user auth/access control per document, chat history, and a UI (Streamlit) — a deployable enterprise-ready Q&A system.