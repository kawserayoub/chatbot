from typing import List, Tuple
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.vectorstores import FAISS
import numpy as np


class ChatMemory:
    def __init__(self):
        self.history: List[Tuple[str, str]] = []

    def add(self, user: str, bot: str):
        self.history.append((user, bot))

    def to_context(self) -> List[dict]:
        context = []
        for user_msg, bot_msg in self.history:
            context.append({"role": "user", "content": user_msg})
            context.append({"role": "assistant", "content": bot_msg})
        return context

def expand_query(llm: ChatOpenAI, query: str, db: FAISS) -> List[Document]:
    retriever = MultiQueryRetriever.from_llm(retriever=db.as_retriever(), llm=llm)
    return retriever.invoke(query)

def rerank(query: str, docs: List[Document], api_key: str, top_k: int = 5) -> List[Document]:
    embedder = OpenAIEmbeddings(openai_api_key=api_key)
    query_vec = embedder.embed_query(query)
    scored = []
    for doc in docs:
        doc_vec = embedder.embed_query(doc.page_content[:1000])
        score = cosine_similarity(query_vec, doc_vec)
        scored.append((score, doc))
    scored.sort(reverse=True, key=lambda x: x[0])
    return [doc for _, doc in scored[:top_k]]

def cosine_similarity(a: List[float], b: List[float]) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def generate_answer(llm: ChatOpenAI, query: str, context_docs: List[Document], memory: ChatMemory) -> str:
    context_text = "\n\n".join(doc.page_content for doc in context_docs)
    messages = memory.to_context()
    messages += [
        {"role": "user", "content": f"Answer the following using the provided context:\n\n{context_text}\n\nQuestion: {query}"}
    ]
    response = llm.invoke(messages)
    return response.content.strip()