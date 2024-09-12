from flask import Flask, request, render_template, jsonify, session, redirect, url_for
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from flask_session import Session
import pickle  


app = Flask(__name__)


load_dotenv()

LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


chat = ChatGoogleGenerativeAI(
    model="gemini-pro",
    temperature=0.4,
    google_api_key=GOOGLE_API_KEY,
    convert_system_message_to_human=True
)


def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


def chunk_text(text, chunk_size=1000):
    sentences = text.split('. ')
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        if current_length + len(sentence.split()) > chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(sentence)
        current_length += len(sentence.split())

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def save_index(index, filename):
    faiss.write_index(index, filename)


def load_index(filename):
    return faiss.read_index(filename)


def save_embeddings(embeddings, filename):
    with open(filename, 'wb') as f:
        pickle.dump(embeddings, f)


def load_embeddings(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def create_faiss_index(chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Using a smaller, efficient model
    chunk_embeddings = model.encode(chunks)
    index = faiss.IndexFlatL2(chunk_embeddings.shape[1])  # L2 distance for similarity
    index.add(np.array(chunk_embeddings))
    return index, chunk_embeddings


def retrieve_relevant_chunks(query, index, chunks, chunk_embeddings, top_k=3):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode([query])
    _, I = index.search(query_embedding, top_k)
    return [chunks[i] for i in I[0] if i < len(chunks)]


def answer_question(chat, retrieved_chunks, user_query):
    context = " ".join(retrieved_chunks)
    message = [
        SystemMessage(content=f"Use the following context to answer: {context}"),
        HumanMessage(content=user_query)
    ]
    result = chat.invoke(message)
    return result


@app.route("/", methods=["GET", "POST"])
def input_pdf_path():
    if request.method == "POST":
        file_path = request.form['file_path']
        
     
        if not os.path.isfile(file_path):
            return "Invalid file path. Please provide a valid PDF file path."

        
        pdf_text = extract_text_from_pdf(file_path)
        chunks = chunk_text(pdf_text)
        index, chunk_embeddings = create_faiss_index(chunks)

        
        save_index(index, 'faiss_index.idx')
        save_embeddings(chunk_embeddings, 'embeddings.pkl')
        
        
        session['chunks'] = chunks
        session['pdf_processed'] = True

        return redirect(url_for('ask_question'))  
    return render_template("input_pdf_path.html")


@app.route("/ask", methods=["GET", "POST"])
def ask_question():
    if 'pdf_processed' not in session:
        return redirect(url_for('input_pdf_path')) 

    if request.method == "POST":
        user_query = request.form['query']

        
        index = load_index('faiss_index.idx')
        if not index:
            return "Failed to load FAISS index."

        chunk_embeddings = load_embeddings('embeddings.pkl')
        if chunk_embeddings is None or len(chunk_embeddings) == 0:
            return "No embeddings found for the given chunks."

       
        chunks = session.get('chunks', [])

        
        retrieved_chunks = retrieve_relevant_chunks(user_query, index, chunks, chunk_embeddings)

        
        response = answer_question(chat, retrieved_chunks, user_query)

        return jsonify({"answer": response.content})

    return render_template("ask.html")


if __name__ == "__main__":
    app.secret_key = os.urandom(24)  
    app.config['SESSION_TYPE'] = 'filesystem'  
    Session(app)  
    app.run(debug=True, port=5004)
