\# RAG Document Q\&A



A Retrieval-Augmented Generation (RAG) pipeline built from scratch in Python.



\## What it does

\- Extracts text from a PDF

\- Splits it into overlapping chunks

\- Generates embeddings for each chunk using Gemini's embedding model

\- Finds the most relevant chunks for a question using cosine similarity

\- Sends the question and retrieved context to Gemini to generate an answer

\- Has a simple Streamlit web interface for asking questions



\## Tech used

\- Python

\- Google Gemini API

\- numpy

\- Streamlit



🔗 \*\*Live demo:\*\* \[Try it here](https://rag-app-aztqetcxxpxjhmddstrrrm.streamlit.app/)

