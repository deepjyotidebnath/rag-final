import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load variables from .env
load_dotenv()

st.set_page_config(page_title="PDF RAG Chatbot", page_icon="🤖")
st.title("🤖 PDF RAG Chatbot")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found. Add it to your .env file and restart the app.")
    st.stop()


@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource
def get_llm():
    return ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL, temperature=0)


@st.cache_resource(show_spinner="Processing PDF...")
def build_vector_store(file_bytes: bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        pdf_path = tmp.name

    try:
        documents = PyPDFLoader(pdf_path).load()
    finally:
        os.remove(pdf_path)  # always clean up, even on error

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    return FAISS.from_documents(chunks, get_embeddings())


PROMPT_TEMPLATE = """You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I couldn't find that in the document."

Context:
{context}

Question: {question}

Answer:"""


uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    db = build_vector_store(uploaded_file.getvalue())
    st.success("PDF Loaded Successfully!")

    query = st.text_input("Ask a Question")

    if query:
        results = db.similarity_search(query, k=3)
        context = "\n\n".join(doc.page_content for doc in results)

        prompt = PROMPT_TEMPLATE.format(context=context, question=query)

        with st.spinner("Thinking..."):
            try:
                response = get_llm().invoke(prompt)
            except Exception as e:
                st.error(f"Groq request failed: {e}")
                st.stop()

        st.subheader("Answer")
        st.write(response.content)

        with st.expander("Retrieved Context"):
            for i, doc in enumerate(results, 1):
                page = doc.metadata.get("page", "?")
                st.markdown(f"**Chunk {i}** (page {page + 1 if isinstance(page, int) else page})")
                st.write(doc.page_content)