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
        return None

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

def find_best_chunks_multi(question, doc_names, documents, top_k=3):
    """Search across one or more selected documents, tagging each result with its source."""
    q_emb = get_embedding(question)
    all_candidates = []  # (similarity, chunk_text, source_filename)
    for name in doc_names:
        doc = documents[name]
        for chunk, emb in zip(doc["chunks"], doc["embeddings"]):
            sim = cosine_similarity(q_emb, emb)
            all_candidates.append((sim, chunk, name))
    all_candidates.sort(key=lambda x: x[0], reverse=True)
    return all_candidates[:top_k]

def answer_question_multi(question, doc_names, documents):
    try:
        top_matches = find_best_chunks_multi(question, doc_names, documents)
        context_parts = [f"[From: {name}]\n{chunk}" for _, chunk, name in top_matches]
        context = "\n\n".join(context_parts)

        prompt = f"""Answer using ONLY the context below. Each piece of context is labeled with its source document.
If the answer isn't there, say "I don't know based on the provided document(s)."
When relevant, mention which document the answer came from.

Context:
{context}

Question: {question}

Answer:"""
        response = client.models.generate_content(
            model="models/gemini-3.6-flash",
            contents=prompt
        )
        sources_used = sorted(set(name for _, _, name in top_matches))
        return response.text, sources_used
    except genai_errors.ClientError as e:
        logger.error(f"Generation API error: {e}")
        return "⚠️ Sorry, I couldn't get an answer right now due to an API error. Please try again.", []
    except Exception as e:
        logger.error(f"Unexpected error answering question: {e}")
        return "⚠️ Something unexpected went wrong. Please try again.", []

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
if "question_input" not in st.session_state:
    st.session_state.question_input = ""
if "documents" not in st.session_state:
    st.session_state.documents = {}  # filename -> {full_text, chunks, embeddings}

# ---------- Sidebar ----------

with st.sidebar:
    st.markdown("## 📄 DocuChat")
    st.caption("RAG-powered document assistant")

    st.divider()

    st.markdown("**Built with**")
    st.caption("Python · Gemini API · Streamlit")

    with st.expander("How it works"):
        st.markdown("""
        1. Extract text from your PDFs
        2. Split into overlapping chunks
        3. Embed each chunk
        4. Retrieve relevant chunks per question, across selected documents
        5. Generate a grounded answer with source labels
        """)

    st.divider()

    if st.session_state.documents:
        st.markdown("**Loaded documents**")
        for name in st.session_state.documents:
            st.caption(f"• {name}")

    if st.button("🔄 Reset / Clear All Documents"):
        for key in ["documents", "qa_history", "last_summary", "question_input"]:
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
    st.title("📄 Multi-Document Q&A & Summarizer")
with col2:
    if st.button("🌓 Theme"):
        st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
        st.rerun()
with col3:
    if st.button("❄️ Snow"):
        st.snow()

st.write("Upload one or more PDFs, then ask questions across all of them or summarize one at a time.")

uploaded_files = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        if uploaded_file.name in st.session_state.documents:
            continue  # already processed, skip re-embedding

        if uploaded_file.size > 20 * 1024 * 1024:
            st.error(f"⚠️ '{uploaded_file.name}' is too large (over 20MB). Skipped.")
            continue

        with st.spinner(f"Processing '{uploaded_file.name}'..."):
            full_text = extract_text(uploaded_file)

            if full_text is None:
                st.error(f"⚠️ Couldn't read '{uploaded_file.name}'. Please make sure it's a valid PDF.")
                continue

            if not full_text.strip():
                st.error(f"⚠️ '{uploaded_file.name}' has no extractable text (may be a scanned image). Skipped.")
                continue

            chunks = chunk_text(full_text)
            embeddings = [get_embedding(c) for c in chunks]

        st.session_state.documents[uploaded_file.name] = {
            "full_text": full_text,
            "chunks": chunks,
            "embeddings": embeddings
        }
        logger.info(f"Processed document: {uploaded_file.name}, {len(chunks)} chunks")

if not st.session_state.documents:
    st.info("👆 Upload at least one PDF to get started.")
else:
    doc_names = list(st.session_state.documents.keys())
    total_words = sum(len(d["full_text"].split()) for d in st.session_state.documents.values())
    st.success(f"{len(doc_names)} document(s) ready · {total_words:,} total words")

    tab1, tab2 = st.tabs(["💬 Ask a question", "📝 Summarize"])

    with tab1:
        selected_docs = st.multiselect(
            "Search in:",
            options=doc_names,
            default=doc_names
        )

        st.caption("Try asking:")
        sample_questions = [
            "What is this about?",
            "Summarize the key points",
            "What are the main rules mentioned?"
        ]
        chip_cols = st.columns(len(sample_questions))
        for i, sq in enumerate(sample_questions):
            with chip_cols[i]:
                if st.button(sq, key=f"chip_{i}"):
                    st.session_state.question_input = sq
                    st.rerun()

        question = st.text_input("Your question:", key="question_input")

        ask_col, regen_col = st.columns([1, 1])
        ask_clicked = ask_col.button("Ask")
        regen_clicked = regen_col.button("🔁 Regenerate last answer", disabled=len(st.session_state.qa_history) == 0)

        if (question and ask_clicked and selected_docs) or regen_clicked:
            target_question = st.session_state.qa_history[0][0] if regen_clicked else question
            with st.spinner("Thinking..."):
                answer, sources = answer_question_multi(target_question, selected_docs, st.session_state.documents)
            if regen_clicked:
                st.session_state.qa_history[0] = (target_question, answer, sources)
            else:
                st.session_state.qa_history.insert(0, (target_question, answer, sources))
            logger.info(f"Question answered: {target_question[:50]}")
            st.rerun()
        elif question and ask_clicked and not selected_docs:
            st.warning("Please select at least one document to search in.")

        if st.session_state.qa_history:
            st.subheader("Conversation")
            for entry in st.session_state.qa_history:
                q, a = entry[0], entry[1]
                sources = entry[2] if len(entry) > 2 else []
                st.markdown(f"**Q: {q}**")
                st.write(a)
                if sources:
                    st.caption(f"📎 Sources: {', '.join(sources)}")
                st.code(a, language=None)
                st.divider()

            transcript = "\n\n".join([f"Q: {e[0]}\nA: {e[1]}" for e in reversed(st.session_state.qa_history)])
            st.download_button(
                "⬇️ Export full conversation",
                transcript,
                file_name="conversation.txt"
            )

    with tab2:
        doc_to_summarize = st.selectbox("Choose a document to summarize:", doc_names)

        if st.button("Generate summary"):
            with st.spinner(f"Summarizing '{doc_to_summarize}'..."):
                summary = summarize_document(st.session_state.documents[doc_to_summarize]["full_text"])
            st.session_state.last_summary = summary
            st.session_state.last_summary_doc = doc_to_summarize

        if "last_summary" in st.session_state:
            st.caption(f"Summary of: {st.session_state.get('last_summary_doc', '')}")
            st.write(st.session_state.last_summary)
            st.code(st.session_state.last_summary, language=None)
            st.download_button(
                "⬇️ Download summary as text",
                st.session_state.last_summary,
                file_name="summary.txt"
            )