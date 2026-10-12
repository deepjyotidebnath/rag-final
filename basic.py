import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import tempfile
import os


st.set_page_config(page_title="PDF RAG Chatbot", page_icon="🤖")

st.title("🤖 PDF RAG CHATBOT")
uploaded_file = st.file_uploader(
    "upload a pdf",
    type=["pdf"]
)

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf
                                     ")