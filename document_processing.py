from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from config import TMP_DIR, EMBEDDING_MODEL, OPENAI_API_KEY
import os
from pathlib import Path

def load_documents(uploaded_files):
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    documents = []
    for uploaded_file in uploaded_files:
        file_path = TMP_DIR / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        pdf_loader = PyPDFLoader(str(file_path))
        loaded_docs = pdf_loader.load()
        for doc in loaded_docs:
            doc.metadata["source"] = str(file_path)
        documents.extend(loaded_docs)
    return documents

def split_documents(documents):
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  
        chunk_overlap=200,
        length_function=lambda x: len(x.split())  
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

def get_embeddings():
    return OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)