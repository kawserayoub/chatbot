import os
import pickle
from typing import List
from PyPDF2 import PdfReader

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

def load_documents(folder_path: str) -> List[Document]:
    docs = []
    for filename in os.listdir(folder_path):
        path = os.path.join(folder_path, filename)
        if filename.endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        elif filename.endswith(".pdf"):
            reader = PdfReader(path)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        else:
            continue
        docs.append(Document(page_content=text, metadata={"source": filename}))
    return docs

def split_documents(docs: List[Document], chunk_size: int = 500, chunk_overlap: int = 50) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)

def embed_documents(docs: List[Document], api_key: str) -> FAISS:
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    db = FAISS.from_documents(docs, embeddings)
    return db

def save_faiss_index(db: FAISS, path: str):
    os.makedirs(path, exist_ok=True)
    db.save_local(path)
    with open(os.path.join(path, "chunks.pkl"), "wb") as f:
        pickle.dump(db.docstore._dict, f)

def load_faiss_index(path: str, api_key: str) -> FAISS:
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)