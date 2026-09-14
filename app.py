import os
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from google import genai
from pypdf import PdfReader

load_dotenv()

try:
    api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)

def extract_text(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def chunk_text(text, chunk_size=600, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return chunks

def get_embedding(text):
    result = client.models.embed_content(
        model="models/gemini-embedding-001",
        contents=text
    )
    return result.embeddings[0].values

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def find_best_chunks(question, chunks, chunk_embeddings, top_k=2):
    q_emb = get_embedding(question)
    sims = [cosine_similarity(q_emb, emb) for emb in chunk_embeddings]
    top_indices = np.argsort(sims)[::-1][:top_k]
    return [chunks[i] for i in top_indices]

def answer_question(question, chunks, chunk_embeddings):
    context = "\n\n".join(find_best_chunks(question, chunks, chunk_embeddings))
    prompt = f"""Answer using ONLY the context below.
If the answer isn't there, say "I don't know based on the provided document."

Context:
{context}

Question: {question}

Answer:"""
    response = client.models.generate_content(
        model="models/gemini-3.6-flash",
        contents=prompt
    )
    return response.text

def summarize_document(full_text):
    prompt = f"""Summarize the following document in a clear, well-organized way.
Give a short overview paragraph, followed by 4-6 key bullet points covering the most important information.

Document:
{full_text}

Summary:"""
    response = client.models.generate_content(
        model="models/gemini-3.6-flash",
        contents=prompt
    )
    return response.text

# ---------- Page setup ----------

st.set_page_config(page_title="DocuChat", page_icon="📄", layout="centered")

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stAppDeployButton {display: none;}
    [data-testid="stStatusWidget"] {display: none;}
    [data-testid="stToolbarActions"] {display: none;}
    </style>
""", unsafe_allow_html=True)

if "theme" not in st.session_state:
    st.session_state.theme = "dark"
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []

# ---------- Sidebar ----------

with st.sidebar:
    st.markdown("## 📄 DocuChat")
    st.caption("RAG-powered document assistant")

    st.divider()

    st.markdown("**Built with**")
    st.caption("Python · Gemini API · Streamlit")

    with st.expander("How it works"):
        st.markdown("""
        1. Extract text from your PDF
        2. Split into overlapping chunks
        3. Embed each chunk
        4. Retrieve relevant chunks per question
        5. Generate a grounded answer
        """)

    st.divider()
    st.caption("Built by Rengabalaji")

# ---------- Theme styling ----------

if st.session_state.theme == "dark":
    bg_color, text_color = "#0e1117", "#fafafa"
else:
    bg_color, text_color = "#ffffff", "#0e1117"

st.markdown(f"""
    <style>
    .stApp {{
        background-color: {bg_color};
        color: {text_color};
    }}
    h1 {{
        background: linear-gradient(90deg, #4F8BF9, #A66CFF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }}
    .stButton>button {{
        border-radius: 8px;
        border: 1px solid #4F8BF9;
        padding: 0.5em 1.5em;
    }}
    .stTabs [data-baseweb="tab"] {{
        font-size: 16px;
        font-weight: 600;
    }}
    </style>
""", unsafe_allow_html=True)

# ---------- Main UI ----------

col1, col2 = st.columns([5, 1])
with col1:
    st.title("📄 Document Q&A & Summarizer")
with col2:
    if st.button("🌓 Theme"):
        st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
        st.rerun()

st.write("Upload any PDF, then ask questions about it or get a quick summary.")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is None:
    st.info("👆 Upload a PDF to get started.")
else:
    if "processed_filename" not in st.session_state or st.session_state.processed_filename != uploaded_file.name:
        with st.spinner("Reading and processing your document..."):
            full_text = extract_text(uploaded_file)

            if not full_text.strip():
                st.error("⚠️ This PDF appears to have no extractable text (it may be a scanned image). Please try a different PDF with selectable text.")
                st.stop()

            chunks = chunk_text(full_text)
            chunk_embeddings = [get_embedding(c) for c in chunks]

        st.session_state.full_text = full_text
        st.session_state.chunks = chunks
        st.session_state.chunk_embeddings = chunk_embeddings
        st.session_state.processed_filename = uploaded_file.name
        st.session_state.qa_history = []

    st.success(f"'{uploaded_file.name}' is ready.")

    tab1, tab2 = st.tabs(["💬 Ask a question", "📝 Summarize"])

    with tab1:
        question = st.text_input("Your question:")
        if question:
            with st.spinner("Thinking..."):
                answer = answer_question(
                    question,
                    st.session_state.chunks,
                    st.session_state.chunk_embeddings
                )
            st.session_state.qa_history.insert(0, (question, answer))

        if st.session_state.qa_history:
            st.subheader("Conversation")
            for q, a in st.session_state.qa_history:
                st.markdown(f"**Q: {q}**")
                st.write(a)
                st.divider()

    with tab2:
        if st.button("Generate summary"):
            with st.spinner("Summarizing..."):
                summary = summarize_document(st.session_state.full_text)
            st.session_state.last_summary = summary

        if "last_summary" in st.session_state:
            st.write(st.session_state.last_summary)
            st.download_button(
                "⬇️ Download summary as text",
                st.session_state.last_summary,
                file_name="summary.txt"
            )