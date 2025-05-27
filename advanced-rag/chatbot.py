import os
import sys
from dotenv import load_dotenv

from utils import load_documents, split_documents, embed_documents, save_faiss_index, load_faiss_index
from enhancers import expand_query, rerank, generate_answer, ChatMemory

from langchain_openai import ChatOpenAI

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(root_dir, "data")
store_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "store")
chunk_size = 500
chunk_overlap = 50

def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Missing OPENAI_API_KEY in .env")
        sys.exit(1)

    llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=api_key)
    memory = ChatMemory()

    print("Welcome. You can now chat with your files.")
    rebuild = input("Rebuild vector store from scratch? (y/n): ").strip().lower() == "y"

# Om ingent index finns eller användaren vill återskapa det
    if rebuild or not os.path.exists(os.path.join(store_path, "index.faiss")):
        docs = load_documents(data_path)
        chunks = split_documents(docs, chunk_size, chunk_overlap)
        db = embed_documents(chunks, api_key)
        save_faiss_index(db, store_path)
    else:
        db = load_faiss_index(store_path, api_key)

    while True:
        query = input("\nYou: ").strip()
        if query.lower() in ["exit", "quit"]:
            print("Goodbye.")
            break

        expanded_docs = expand_query(llm, query, db)
        reranked_docs = rerank(query, expanded_docs, api_key)
        answer = generate_answer(llm, query, reranked_docs, memory)

        memory.add(query, answer)
        print(f"\nBot: {answer}")

# Ser till att koden bara körs vid direkt exekvering - möjliggör testing/import
if __name__ == "__main__":
    main()