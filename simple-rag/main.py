import os
import sys
import pickle
import faiss
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import TokenTextSplitter
from langchain.docstore.document import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


class chatbot:
    chunk_size = 500
    chunk_overlap = 50

    current_dir = os.path.dirname(os.path.abspath(__file__))
    index_file = os.path.join(current_dir, "vectors.faiss")
    store_file = os.path.join(current_dir, "chunks.pkl")
    data_folder = os.path.join(os.path.dirname(current_dir), "data")

    def __init__(self):
        load_dotenv()
        self.openai_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_key:
            print("Missing API key. Please set OPENAI_API_KEY in your .env file.")
            sys.exit(1)

        self.embedding_model = OpenAIEmbeddings(openai_api_key=self.openai_key)
        self.llm = ChatOpenAI(temperature=0, openai_api_key=self.openai_key)
        self.db = None
        self.chat_history = []

# Ger kontroll över cachbeteende
    def prompt_for_reset(self) -> bool:
        while True:
            response = input("Would you like to rebuild the system from scratch? (y/n): ").strip().lower()
            if response in {"y", "n"}:
                return response == "y"
            print("Please respond with 'y' or 'n'")

    def load_documents(self):
        if not os.path.isdir(self.data_folder):
            print(f"Folder '{self.data_folder}' not found")
            sys.exit(1)

        txt_files = [f for f in os.listdir(self.data_folder) if f.endswith(".txt")]
        pdf_files = [f for f in os.listdir(self.data_folder) if f.endswith(".pdf")]

        if not txt_files and not pdf_files:
            print("No .txt or .pdf files found in data folder")
            sys.exit(1)

        docs = []

        for name in txt_files:
            path = os.path.join(self.data_folder, name)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                    docs.append(Document(page_content=content))
            except Exception as e:
                print(f"Failed to read {name}: {e}")

        for name in pdf_files:
            path = os.path.join(self.data_folder, name)
            try:
                reader = PdfReader(path)
                text = ""
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text
                if text.strip():
                    docs.append(Document(page_content=text))
            except Exception as e:
                print(f"Failed to read {name}: {e}")

        # Tokenbaserad splittning minksar risken för trasiga meningar jämfört med teckenbaserad delning
        splitter = TokenTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        chunks = splitter.split_documents(docs)
        print(f"{len(txt_files) + len(pdf_files)} files loaded and processed into smaller parts.")
        return chunks

    # Balanserar mellan snabbhet och kontroll
    def prepare_index(self, chunks):
        reset = self.prompt_for_reset()
        existing = os.path.exists(self.index_file) and os.path.exists(self.store_file)

        if reset or not existing:
            print("Building a new system. May take a second...")
            db = FAISS.from_documents(chunks, self.embedding_model)
            faiss.write_index(db.index, self.index_file)
            with open(self.store_file, "wb") as f:
                pickle.dump(db.docstore._dict, f)
            print("The system is ready to use.")
        else:
            print("Loading existing index")
            index = faiss.read_index(self.index_file)
            with open(self.store_file, "rb") as f:
                store = pickle.load(f)
            db = FAISS(
                embedding_function=self.embedding_model,
                index=index,
                docstore=store,
                index_to_docstore_id={}
            )
            print("The system has been loaded successfully.")

        return db

    def run(self):
        print("Welcome! Lets have a chat with your files.")
        chunks = self.load_documents()
        self.db = self.prepare_index(chunks)

        # Säkerställer att modellen fokuserar på rätt underlag
        prompt = ChatPromptTemplate.from_template(
            """
            answer the question based only on the following context:

            {context}

            question: {question}
            """.strip()
        )

        retriever = self.db.as_retriever()
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
        )

        print("\nAsk your questions below. Type 'exit' to end the session.\n")

        while True:
            try:
                query = input("You: ").strip()
                if query.lower() == "exit":
                    print("Goodbye!")
                    break
                if not query:
                    continue

                response = chain.invoke(query)
                print("Assistant:", response.content)
                self.chat_history.append((query, response.content))

            except KeyboardInterrupt:
                print("\nSession ended")
                break
            except Exception as e:
                print(f"error: {e}")


if __name__ == "__main__":
    chatbot().run()