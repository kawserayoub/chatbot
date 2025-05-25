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

st.set_page_config(page_title="RAG Assistant", layout="centered")
st.markdown("""
    <style>
        .stApp {
            background-color: white;
        }
        .chat-message {
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 10px;
        }
        .user-msg {
            background-color: #DCF8C6;
            text-align: right;
        }
        .bot-msg {
            background-color: #F1F0F0;
            text-align: left;
        }
        .chat-container {
            max-height: 500px;
            overflow-y: auto;
            padding-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# init
if "memory" not in st.session_state:
    st.session_state.memory = ChatMemory()
if "db" not in st.session_state:
    st.session_state.db = None

st.title("🧠 Ask Your Documents")

# file upload
with st.expander("📂 Upload TXT or PDF documents"):
    uploaded_files = st.file_uploader("Drag and drop or browse", type=["txt", "pdf"], accept_multiple_files=True)
    if uploaded_files:
        with st.spinner("Indexing documents..."):
            with tempfile.TemporaryDirectory() as tmpdir:
                for file in uploaded_files:
                    path = os.path.join(tmpdir, file.name)
                    with open(path, "wb") as f:
                        f.write(file.getvalue())

                docs = load_documents(tmpdir)
                chunks = split_documents(docs)
                db = embed_documents(chunks, api_key)
                st.session_state.db = db
        st.success("✅ Documents are ready to chat with!")

# chat UI
if st.session_state.db:
    st.markdown("""<div class="chat-container">""", unsafe_allow_html=True)

    for user_msg, bot_msg in st.session_state.memory.history:
        st.markdown(f"<div class='chat-message user-msg'>{user_msg}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='chat-message bot-msg'>{bot_msg}</div>", unsafe_allow_html=True)

    st.markdown("""</div>""", unsafe_allow_html=True)

    query = st.chat_input("Type your question...")
    if query:
        st.markdown(f"<div class='chat-message user-msg'>{query}</div>", unsafe_allow_html=True)

        with st.spinner("Thinking..."):
            llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=api_key)
            docs = expand_query(llm, query, st.session_state.db)
            reranked = rerank(query, docs, api_key)
            answer = generate_answer(llm, query, reranked, st.session_state.memory)
            st.session_state.memory.add(query, answer)

            placeholder = st.empty()
            typing_text = ""
            for word in answer.split():
                typing_text += word + " "
                placeholder.markdown(f"<div class='chat-message bot-msg'>{typing_text}</div>", unsafe_allow_html=True)
                time.sleep(0.03)

            st.rerun()
