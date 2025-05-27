import os
import tempfile
import time
import streamlit as st
from dotenv import load_dotenv

from utils import load_documents, split_documents, embed_documents
from enhancers import expand_query, rerank, generate_answer, ChatMemory
from langchain_openai import ChatOpenAI

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY is missing.")
    st.stop()

st.set_page_config(page_title="RAG Assistant", layout="centered")

st.markdown("""
    <style>
        html, body, .stApp {
            background-color: #f4f4f4;
            color: #212529;
        }

        .chat-container {
            max-height: 500px;
            overflow-y: auto;
            padding-bottom: 20px;
        }

        .chat-message {
            padding: 1rem;
            border-radius: 12px;
            margin-bottom: 10px;
            font-size: 16px;
            max-width: 80%;
            word-wrap: break-word;
        }

        .user-msg {
            background-color: #003566;
            color: #ffffff;
            margin-left: auto;
            text-align: right;
        }

        .bot-msg {
            background-color: #ffffff;
            color: #000000;
            margin-right: auto;
            text-align: left;
            border: 1px solid #dee2e6;
        }
    </style>
""", unsafe_allow_html=True)

# Initierar chatthistorik endast en gång per session
if "memory" not in st.session_state:
    st.session_state.memory = ChatMemory()
    
# Skapar plats för vektordatabas i session, undviker ominläsning vid varje prompt
if "db" not in st.session_state:
    st.session_state.db = None

st.title("🧠 Ask Your Documents")

with st.expander("📂 Upload TXT or PDF documents"):
    uploaded_files = st.file_uploader("Browse files", type=["txt", "pdf"], accept_multiple_files=True)
    if uploaded_files:
        
        # Temporär katalog används för att hantera filerna lokalt innan indexering
        with st.spinner("Indexing documents..."):
            with tempfile.TemporaryDirectory() as tmpdir:
                for file in uploaded_files:
                    path = os.path.join(tmpdir, file.name)
                    with open(path, "wb") as f:
                        f.write(file.getvalue())
                        
                # Dokumentet laddas, chunkas och embeddas direkt, ingen cache lösning används
                docs = load_documents(tmpdir)
                chunks = split_documents(docs)
                db = embed_documents(chunks, api_key)
                st.session_state.db = db
        st.success("✅ Documents are ready.")

if st.session_state.db:
    st.markdown("""<div class="chat-container">""", unsafe_allow_html=True)

    for user_msg, bot_msg in st.session_state.memory.history:
        st.markdown(f"<div class='chat-message user-msg'>{user_msg}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='chat-message bot-msg'>{bot_msg}</div>", unsafe_allow_html=True)

    st.markdown("""</div>""", unsafe_allow_html=True)

    query = st.chat_input("Type your question...")
    if query and query.strip():
        st.markdown(f"<div class='chat-message user-msg'>{query}</div>", unsafe_allow_html=True)

        with st.spinner("Thinking..."):
            try:
                
                # Tempratur noll för deterministiska svar - lämplig vid faktabaserad analys
                llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=api_key)
                
                # Query expansion ökas chansen att träffa relevant material vid vaga frågor
                docs = expand_query(llm, query, st.session_state.db)
                
                # Reranking förbättrar precision genom att väga relevans efter expansion
                reranked = rerank(query, docs, api_key)
                
                # Kontextuell svargenerering med chatthistorik som stöd
                answer = generate_answer(llm, query, reranked, st.session_state.memory)
                st.session_state.memory.add(query, answer)

                placeholder = st.empty()
                typing_text = ""
                for word in answer.split():
                    typing_text += word + " "
                    placeholder.markdown(f"<div class='chat-message bot-msg'>{typing_text}</div>", unsafe_allow_html=True)
                    time.sleep(0.03)
            except Exception as e:
                st.error(f"Error: {str(e)}")
else:
    st.info("Please upload documents to start.")
