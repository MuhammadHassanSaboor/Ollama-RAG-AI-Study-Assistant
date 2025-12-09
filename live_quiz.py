import PyPDF2
import streamlit as st
from docx import Document
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import Ollama
from langchain.chains import RetrievalQA

def read_text_for_quiz(file):
    """Extracts text from PDF, DOCX, or TXT for quiz generation."""
    if file.name.endswith(".pdf"):
        pdf_reader = PyPDF2.PdfReader(file)
        text = "".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
    elif file.name.endswith(".docx"):
        doc = Document(file)
        text = "\n".join([para.text for para in doc.paragraphs])
    else:
        text = file.read().decode("utf-8", errors="ignore")
    return text.strip()

def create_quiz_vector_db(text):
    """Splits text and builds a FAISS vector store for quiz generation."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(text)
    embeddings = OllamaEmbeddings(model="gemma2:2b")
    # embeddings = OllamaEmbeddings(model="phi3:latest")
    # embeddings = OllamaEmbeddings(model="llama3")
    db = FAISS.from_texts(chunks, embedding=embeddings)
    return db

def generate_quiz_questions(db, num_questions):
    """Generates multiple-choice quiz questions using RAG pipeline."""
    retriever = db.as_retriever()
    llm = Ollama(model="llama3")

    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    
    prompt = f"""
    You are a quiz master. Based on the provided document context,
    generate {num_questions} multiple-choice questions (MCQs).
    Use this format exactly:

    Q: Question text
    A) option1
    B) option2
    C) option3
    D) option4
    Correct: <A/B/C/D>

    Ensure the options are relevant and the correct answer is clearly indicated.
    """
    
    
    response = qa_chain.run(prompt)
    return response



def show_page():
    st.set_page_config(page_title=" Evaluate yourself", layout="wide")

    st.title("🏆 Evaluate Yourself")

    uploaded_file = st.file_uploader("📂 Upload a document", 
                                     type=["pdf", "pptx", "docx", "txt"],
                                     accept_multiple_files=True,
                                      key="live_quiz_uploader", 
        help="You can upload multiple documents together for better context.",)
    num_qs = st.number_input("🧮 Number of Questions", min_value=1, max_value=20, value=5)

    if uploaded_file:
        all_text = ""
        for f in uploaded_file:
            text_part = read_text_for_quiz(f)
            all_text += text_part + "\n\n"
        text = all_text.strip()
        st.success(f"✅ {len(uploaded_file)} document(s) loaded successfully!")

        if st.button("🚀 Generate Quiz"):
            with st.spinner("Generating quiz... please wait ⏳"):
                db = create_quiz_vector_db(text)
                quiz_text = generate_quiz_questions(db, num_qs)

                # Store quiz in session
                st.session_state.quiz = quiz_text.split("\nQ:")
                st.session_state.answers = {}
                st.session_state.correct = {}

    if "quiz" in st.session_state:
        st.subheader("🧠 Take the Quiz")
        for i, q in enumerate(st.session_state.quiz[1:], start=1):
            question_lines = q.strip().split("\n")
            if len(question_lines) < 6:
                continue  # skip malformed question

            question = question_lines[0].strip()
            options = [l.strip() for l in question_lines[1:5]]  # noqa: E741
            correct_line = [l for l in question_lines if l.startswith("Correct")]  # noqa: E741

            if correct_line:
                correct = correct_line[0].split(":")[-1].strip()
                st.session_state.correct[i] = correct

            user_choice = st.radio(f"Q{i}. {question}", options, key=f"q{i}")
            st.session_state.answers[i] = user_choice

        if st.button("✅ Check Results"):
            score = 0
            total = len(st.session_state.answers)
            for i in st.session_state.answers:
                if st.session_state.answers[i].startswith(st.session_state.correct.get(i, "")):
                    score += 1
            st.success(f"🎯 Your Score: {score} / {total}")