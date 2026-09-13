import os
import numpy as np
from dotenv import load_dotenv
from google import genai
from pypdf import PdfReader

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

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

print("Creating embeddings for each chunk...")
chunk_embeddings = [get_embedding(chunk) for chunk in chunks]

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def find_best_chunks(question, top_k=2):
    question_embedding = get_embedding(question)
    similarities = [cosine_similarity(question_embedding, emb) for emb in chunk_embeddings]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [(chunks[i], similarities[i]) for i in top_indices]

def answer_question(question):
    top_chunks = find_best_chunks(question)
    context = "\n\n".join([chunk for chunk, score in top_chunks])

    prompt = f"""Answer the question using ONLY the context below.
If the answer isn't in the context, say "I don't know based on the provided document."

Context:
{context}

Question: {question}

Answer:"""

    response = client.models.generate_content(
        model="models/gemini-3.6-flash",
        contents=prompt
    )
    return response.text

# Test it
question = "How many paid leave days do employees get?"
answer = answer_question(question)
print(f"\nQuestion: {question}")
print(f"Answer: {answer}")