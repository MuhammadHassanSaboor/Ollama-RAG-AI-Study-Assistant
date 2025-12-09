import streamlit as st
import os
import gc
import importlib
import base64
from automatic_quiz import show_page as show_automatic_quiz_page  
import shutil  # noqa: F401
from typing import List
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
    TextLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.docstore.document import Document
import chromadb

llm = Ollama(model="gemma2:2b", temperature=0.1)
# llm = Ollama(model="phi3:latest", temperature=0.1)
# llm = Ollama(model="llama3", temperature=0.1)
# llm = Ollama(model="llama3:8b-instruct", temperature=0.1)
# llm = Ollama(model="qwen3-vl:4b", temperature=0.1)

PERSIST_DIRECTORY = "embeddings"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 3

embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

def load_docs(file_paths: List[str]) -> List[Document]:
    docs = []
    for path in file_paths:
        print(f"📂 Loading: {path}")
        try:
            if path.endswith(".pdf"):
                loader = PyPDFLoader(path)
            elif path.endswith(".pptx"):
                loader = UnstructuredPowerPointLoader(path)
            elif path.endswith(".docx"):
                loader = UnstructuredWordDocumentLoader(path)
            elif path.endswith((".txt", ".text")):
                loader = TextLoader(path, encoding="utf-8")
            else:
                print(f"❗ Unsupported file format: {path}")
                continue
            docs.extend(loader.load())
        except Exception as e:
            print(f"❌ Error loading {path}: {e}")
    print(f"✅ Total documents loaded: {len(docs)}")
    return docs


def split_docs(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)
    print(f"🔪 Total chunks created: {len(chunks)}")
    return chunks


def reset_chroma_db_without_deleting():
    try:
        client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
        collections = client.list_collections()
        for collection in collections:
            client.delete_collection(name=collection.name)
        print("✅ All Chroma collections deleted.")
    except Exception as e:
        print(f"❌ Error clearing collections: {e}")

    os.makedirs(PERSIST_DIRECTORY, exist_ok=True)


def store_embeddings(splits: List[Document]) -> Chroma:
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embedding,
        persist_directory=PERSIST_DIRECTORY,
    )
    print("📦 Embeddings stored and vector DB persisted.")
    return vectordb


def get_relevant_chunks(query: str, k: int = TOP_K) -> List[Document]:
    vectordb = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedding)
    return vectordb.similarity_search(query, k=k)

def get_answer(query: str) -> str:
    chunks = get_relevant_chunks(query)
    if not chunks:
        return "❌ No relevant information found in the documents."

    context = "\n\n".join(doc.page_content for doc in chunks)
    prompt = f"""You are a helpful assistant. Use the context below to answer the question.

    Context:
    {context}

    Question: {query}

    Answer:"""

    # ✅ Use the same instance each time
    return llm.invoke(prompt)

def build_vector_db(file_paths: List[str]):
    reset_chroma_db_without_deleting()
    docs = load_docs(file_paths)
    splits = split_docs(docs)
    store_embeddings(splits)
# -------------------------------
# Cleanup
# -------------------------------
gc.collect()
# -------------------------------
# RAG Pipeline
# -------------------------------

# -------------------------------
# Streamlit Config
# -------------------------------
st.set_page_config(
    page_title="AI Study Assistant",
    page_icon="🎓",
    layout="wide",
)
# -------------------------------
# Utility: Encode logo
# -------------------------------
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

logo_base64 = get_base64_image("./assets/logo.png")
# -------------------------------
# Custom CSS
# -------------------------------
st.markdown(
    """
    <style>
    /* Sidebar styles */
    .sidebar .sidebar-content { padding-top: 0.5rem; }
    .sidebar-logo { text-align:center; margin-bottom:8px; }
    .sidebar-title { text-align:center; font-size:20px; font-weight:700; color:white; }
    .sidebar-sub { text-align:center; color:gray; margin-bottom:12px; }

    /* Buttons */
    div.stButton > button {
        width: 100% !important;
        background-color: black;
        color: white;
        border: 2px solid white;
        padding: 10px 12px;
        border-radius: 10px;
        font-size: 18px;
        font-weight: bold;
        cursor: pointer;
        transition: transform 0.12s ease, background-color 0.12s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        background-color: white;
        color:black;
    }

    /* Title */
    .main-title { font-weight:700; color:#0b5cff; margin-bottom: 5px; }

    /* Chat messages */
    .user-msg {
        background-color: green;
        color: white;
        padding: 10px 14px;
        border-radius: 10px;
        margin-bottom: 8px;
        max-width: 80%;
        margin-left: auto;
        text-align: right;
    }
    .assistant-msg {
        background-color: #E8E8E8;
        color: black;
        padding: 10px 14px;
        border-radius: 10px;
        margin-bottom: 8px;
        max-width: 80%;
        margin-right: auto;
        text-align: left;
    }

    /* Extra space for fixed input */
    .block-container { padding-bottom: 90px; }
    </style>
    """,
    unsafe_allow_html=True,
)
# -------------------------------
# Sidebar Navigation
# -------------------------------
with st.sidebar:
    st.markdown(
        f"""
        <div style="text-align:center;">
            <img src="data:image/png;base64,{logo_base64}" width="90">
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sidebar-title">AI Study Assistant</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="sidebar-sub">Your Smart Learning Partner 🎓</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    if "page" not in st.session_state:
        st.session_state.page = "Page 1"

    if st.button("📄 Document Q&A", key="nav_p1"):
        st.session_state.page = "Page 1"
    if st.button("🧠 Auto Quiz Generator", key="nav_p2"):
        st.session_state.page = "Page 2"
    if st.button("📊 Explore Lectures", key="nav_p3"):
        st.session_state.page = "Page 3"
    if st.button("🎯 Live Quiz Mode", key="nav_p4"):
        st.session_state.page = "Page 4"
    if st.button("📅 Study Planner", key="nav_p5"):
        st.session_state.page = "Page 5"
    if st.button("🧭 Concept Map Generator", key="nav_p6"):
        st.session_state.page = "Page 6"
# -------------------------------
# Page Routing
# -------------------------------
current_page = st.session_state.page

# ---- Page 1: Document Q&A Chatbot ----
if current_page == "Page 1":
    st.markdown(
        "<h1 class='main-title'>💬 RAG-Powered Chatbot</h1>", unsafe_allow_html=True
    )
    st.markdown(
        "Upload **PDF**, **PPTX**, **DOCX**, or **TXT** files. "
        "Ask multiple questions and get context-aware answers from your documents."
    )

    uploaded_files = st.file_uploader(
        "📤 Upload documents",
        type=["pdf", "pptx", "docx", "txt"],
        accept_multiple_files=True,
        key="doc_qa_uploader",
        help="Upload multiple documents for deeper context.",
    )

    if uploaded_files:
        os.makedirs("docs", exist_ok=True)
        doc_paths = []
        progress_bar = st.progress(0)
        total_files = len(uploaded_files)

        for i, file in enumerate(uploaded_files):
            filepath = os.path.join("docs", file.name)
            with open(filepath, "wb") as f:
                f.write(file.getbuffer())
            doc_paths.append(filepath)
            progress_bar.progress(int(((i + 1) / total_files) * 100))

        st.info("📚 Processing uploaded documents. Please wait...")
        build_vector_db(doc_paths)
        st.success("✅ Documents processed successfully!")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for sender, message in st.session_state.chat_history:
        if sender == "You":
            st.markdown(
                f"<div class='user-msg'><b>You:</b> {message}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div class='assistant-msg'><b>Assistant:</b> {message}</div>",
                unsafe_allow_html=True,
            )

    query = st.text_input(
        "", key="chat_input", placeholder="💬 Ask a question from your documents..."
    )
    send_clicked = st.button("Send", key="send_button")

    if send_clicked and query:
        with st.spinner("🤔 Thinking..."):
            try:
                response = get_answer(query)
                st.session_state.chat_history.append(("You", query))
                st.session_state.chat_history.append(("Assistant", response))
                st.rerun()
            except Exception as e:
                st.session_state.chat_history.append(
                    ("Assistant", f"❗ LLM Error: {e}")
                )
                st.rerun()

# ---- Page 2: Auto Quiz Generator ----
elif current_page == "Page 2":
    try:
        show_automatic_quiz_page()
    except Exception as e:
        st.error(f"⚠️ Error loading Auto Quiz Generator: {e}")

# ---- Page 3: Explore Lectures ----
elif current_page == "Page 3":
    try:
        page_module = importlib.import_module("explore_lectures")
        if hasattr(page_module, "show_page"):
            page_module.show_page()
        else:
            st.title("📊 Explore Lectures")
            st.info("`show_page()` not found — module imported directly.")
            importlib.reload(page_module)
    except ModuleNotFoundError:
        st.title("📊 Explore Lectures")
        st.warning("`explore_lectures.py` not found.")
    except Exception as e:
        st.error(f"An error occurred while loading Page 3: {e}")

# ---- Page 4: Live Quiz Mode ----
elif current_page == "Page 4":
    try:
        page_module = importlib.import_module("live_quiz")
        if hasattr(page_module, "show_page"):
            page_module.show_page()
        else:
            st.title("🎯 Live Quiz Mode")
            st.info("`show_page()` not found.")
    except ModuleNotFoundError:
        st.title("🎯 Live Quiz Mode")
        st.warning("`live_quiz.py` not found.")
    except Exception as e:
        st.error(f"An error occurred while loading Page 4: {e}")

# ---- Page 5: Study Planner ----
elif current_page == "Page 5":
    try:
        import study_planner
        study_planner.show_page()
    except Exception as e:
        st.error(f"⚠️ Error loading Study Planner: {e}")


# ---- Page 6: Concept Map Generator ----
elif current_page == "Page 6":
    try:
        import concept_map
        concept_map.show_page()
    except Exception as e:
        st.error(f"⚠️ Error loading Concept Map Generator: {e}")
# -------------------------------
# Footer
# -------------------------------
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")
st.caption("AI Study Assistant • Developed with ❤️ using Streamlit")