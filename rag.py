import os
import chromadb
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables (.env file) - works locally
load_dotenv()

# Try to get the key from .env first (local), otherwise from Streamlit Cloud secrets
api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", "")
genai.configure(api_key=api_key)

# Load embedding model once
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Gemini model
gemini_model = genai.GenerativeModel("gemini-3-flash-preview")

# ChromaDB client (persistent - saves to disk)
chroma_client = chromadb.PersistentClient(path="./chroma_db")


def extract_text_from_pdf(pdf_path):
    """Extract full text from a PDF file"""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def get_embeddings(chunks):
    """Convert chunks into embedding vectors"""
    return embedding_model.encode(chunks).tolist()


def store_chunks(chunks, embeddings, collection_name="pdf_docs"):
    """Store chunks and embeddings in ChromaDB"""
    # Delete old collection if it exists, then start fresh
    try:
        chroma_client.delete_collection(name=collection_name)
    except Exception:
        pass

    collection = chroma_client.create_collection(name=collection_name)
    ids = [str(i) for i in range(len(chunks))]
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids
    )
    return collection


def search_relevant_chunks(query, collection_name="pdf_docs", n_results=3):
    """Fetch chunks relevant to the query from ChromaDB"""
    collection = chroma_client.get_collection(name=collection_name)
    query_embedding = embedding_model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results
    )
    return results['documents'][0]


def generate_answer(query, context_chunks):
    """Generate an answer from Gemini using the retrieved context"""
    context = "\n\n".join(context_chunks)
    prompt = f"""You are a helpful assistant. Answer the question using ONLY the context given below.
If the answer is not present in the context, say "I don't know based on the given document."

Context:
{context}

Question: {query}

Answer:"""

    response = gemini_model.generate_content(prompt)
    return response.text