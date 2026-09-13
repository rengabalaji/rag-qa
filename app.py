import os
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from google import genai
from pypdf import PdfReader

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

@st.cache_resource
def setup():
    reader = PdfReader("sample_company_handbook.pdf")
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text()

    def chunk_text(text, chunk_size=600, overlap=100):
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end].strip())
            start += chunk_size - overlap
        return chunks

    chunks = chunk_text(full_text)

    def get_embedding(text):
        result = client.models.embed_content(
            model="models/gemini-embedding-001",
            contents=text
        )
        return result.embeddings[0].values

    chunk_embeddings = [get_embedding(c) for c in chunks]
    return chunks, chunk_embeddings

chunks, chunk_embeddings = setup()

def get_embedding(text):
    result = client.models.embed_content(
        model="models/gemini-embedding-001",
        contents=text
    )
    return result.embeddings[0].values

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def find_best_chunks(question, top_k=2):
    q_emb = get_embedding(question)
    sims = [cosine_similarity(q_emb, emb) for emb in chunk_embeddings]
    top_indices = np.argsort(sims)[::-1][:top_k]
    return [chunks[i] for i in top_indices]

def answer_question(question):
    context = "\n\n".join(find_best_chunks(question))
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

st.title("📄 Nimbora Handbook Q&A")
st.write("Ask a question about the sample employee handbook.")

question = st.text_input("Your question:")

if question:
    with st.spinner("Thinking..."):
        answer = answer_question(question)
    st.write(answer)