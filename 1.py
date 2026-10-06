"""
Project: Basic Document Q&A (Single File)
-------------------------------------------
Loads one text or PDF file, splits it into chunks, embeds the chunks,
and answers a question using an LLM with the most relevant chunk as context.
This is the simplest possible RAG (Retrieval-Augmented Generation) pipeline.

Requires: pip install groq sentence-transformers pypdf numpy
Set GROQ_API_KEY in your environment.

Usage:    python doc_qa.py mydoc.pdf "What is the refund policy?"
"""

import sys
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq

client = Groq()  # reads GROQ_API_KEY from env
embedder = SentenceTransformer("all-MiniLM-L6-v2")  # small, fast, runs locally


def load_text(path: str) -> str:
    """Load a .txt file directly, or extract text from a .pdf file."""
    if path.lower().endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks of ~chunk_size characters."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c.strip() for c in chunks if c.strip()]


def embed_chunks(chunks: list[str]) -> np.ndarray:
    return embedder.encode(chunks, convert_to_numpy=True)


def find_best_chunk(question: str, chunks: list[str], chunk_embeddings: np.ndarray) -> str:
    """Embed the question and return the chunk with the highest cosine similarity."""
    q_embedding = embedder.encode([question], convert_to_numpy=True)[0]

    # cosine similarity between the question and every chunk
    norms = np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(q_embedding)
    scores = chunk_embeddings @ q_embedding / norms

    best_idx = int(np.argmax(scores))
    return chunks[best_idx]


def answer_question(question: str, context: str) -> str:
    prompt = f"""Answer the question using ONLY the context below.
If the answer isn't in the context, say "I don't know based on the document."

Context:
{context}

Question: {question}
Answer:"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def main():
    if len(sys.argv) != 3:
        print('Usage: python doc_qa.py <file.txt|file.pdf> "your question"')
        sys.exit(1)

    file_path, question = sys.argv[1], sys.argv[2]

    print("Loading document...")
    text = load_text(file_path)

    print("Chunking document...")
    chunks = chunk_text(text)
    print(f"Created {len(chunks)} chunks.")

    print("Embedding chunks...")
    chunk_embeddings = embed_chunks(chunks)

    print("Finding most relevant chunk...")
    best_chunk = find_best_chunk(question, chunks, chunk_embeddings)

    print("Asking the LLM...\n")
    answer = answer_question(question, best_chunk)

    print("=== Answer ===")
    print(answer)
    print("\n=== Source chunk used ===")
    print(best_chunk[:300] + ("..." if len(best_chunk) > 300 else ""))


if __name__ == "__main__":
    main()