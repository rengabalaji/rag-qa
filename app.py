import os
import logging
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from pypdf import PdfReader

load_dotenv()

# ---------- Logging setup ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

try:
    api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("⚠️ No API key found. Please set GEMINI_API_KEY in your .env file or Streamlit secrets.")
    st.stop()

client = genai.Client(api_key=api_key)

def extract_text(uploaded_file):
    try:
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        return text
    except Exception as e:
        logger.error(f"PDF extraction failed: {e}")
        st.error("⚠️ Couldn't read this file. Please make sure it's a valid PDF.")
        st.stop()

def chunk_text(text, chunk_size=600, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return chunks

def get_embedding(text, retries=2):
    for attempt in range(retries + 1):
        try:
            result = client.models.embed_content(
                model="models/gemini-embedding-001",
                contents=text
            )
            return result.embeddings[0].values
        except genai_errors.ClientError as e:
            logger.error(f"Embedding API error (attempt {attempt+1}): {e}")
            if attempt == retries:
                st.error("⚠️ Couldn't process the document right now (API error). Please try again in a moment.")
                st.stop()
        except Exception as e:
            logger.error(f"Unexpected embedding error: {e}")
            st.error("⚠️ Something went wrong while processing the document.")
            st.stop()

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return np.dot(a, b) / denom

def find_best_chunks(question, chunks, chunk_embeddings, top_k=2):
    q_emb = get_embedding(question)
    sims = [cosine_similarity(q_emb, emb) for emb in chunk_embeddings]
    top_indices = np.argsort(sims)[::-1][:top_k]
    return [chunks[i] for i in top_indices]

def answer_question(question, chunks, chunk_embeddings):
    try:
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
    except genai_errors.ClientError as e:
        logger.error(f"Generation API error: {e}")
        return "⚠️ Sorry, I couldn't get an answer right now due to an API error. Please try again."
    except Exception as e:
        logger.error(f"Unexpected error answering question: {e}")
        return "⚠️ Something unexpected went wrong. Please try again."

def summarize_document(full_text):
    try:
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
    except genai_errors.ClientError as e:
        logger.error(f"Summarization API error: {e}")
        return "⚠️ Sorry, I couldn't generate a summary right now due to an API error. Please try again."
    except Exception as e:
        logger.error(f"Unexpected error summarizing: {e}")
        return "⚠️ Something unexpected went wrong while summarizing."

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

    if st.button("🔄 Reset / Start Over"):
        for key in ["processed_filename", "full_text", "chunks", "chunk_embeddings", "qa_history", "last_summary"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

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

col1, col2, col3 = st.columns([4, 1, 1])
with col1:
    st.title("📄 Document Q&A & Summarizer")
with col2:
    if st.button("🌓 Theme"):
        st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
        st.rerun()
with col3:
    if st.button("❄️ Snow"):
        st.snow()

st.write("Upload any PDF, then ask questions about it or get a quick summary.")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is None:
    st.info("👆 Upload a PDF to get started.")
else:
    if uploaded_file.size > 20 * 1024 * 1024:  # 20MB safety limit
        st.error("⚠️ File too large. Please upload a PDF under 20MB.")
        st.stop()

    is_new_document = (
        "processed_filename" not in st.session_state
        or st.session_state.processed_filename != uploaded_file.name
    )

    if is_new_document:
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
        logger.info(f"Processed document: {uploaded_file.name}, {len(chunks)} chunks")

    # Word/character count
    word_count = len(st.session_state.full_text.split())
    char_count = len(st.session_state.full_text)

    st.success(f"'{uploaded_file.name}' is ready.")
    st.caption(f"📊 {word_count:,} words · {char_count:,} characters · {len(st.session_state.chunks)} chunks")

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
            logger.info(f"Question answered: {question[:50]}")

        if st.session_state.qa_history:
            st.subheader("Conversation")
            for q, a in st.session_state.qa_history:
                st.markdown(f"**Q: {q}**")
                st.write(a)
                # Copy-friendly display: code block gives a built-in copy icon
                st.code(a, language=None)
                st.divider()

            # Export full conversation
            transcript = "\n\n".join([f"Q: {q}\nA: {a}" for q, a in reversed(st.session_state.qa_history)])
            st.download_button(
                "⬇️ Export full conversation",
                transcript,
                file_name="conversation.txt"
            )

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