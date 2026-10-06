import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import tempfile
import os

st.set_page_config(page_title="PDF RAG Chatbot", page_icon="🤖")

st.title("🤖 PDF RAG Chatbot")

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)

if uploaded_file:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    with st.spinner("Processing PDF..."):

        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        chunks = splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        db = FAISS.from_documents(chunks, embeddings)

    st.success("PDF Loaded Successfully!")

    query = st.text_input("Ask a Question")

    if query:

        docs = db.similarity_search(query, k=3)

        st.subheader("Retrieved Context")

        for i, doc in enumerate(docs, 1):
            st.write(f"### Chunk {i}")
            st.write(doc.page_content)

    os.remove(pdf_path)