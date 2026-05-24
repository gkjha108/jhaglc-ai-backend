from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import faiss
import pickle
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# =========================
# OPENAI CLIENT
# =========================

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

# =========================
# LOAD FAISS DATABASE
# =========================

print("Loading FAISS index...")

index = faiss.read_index("faiss.index")

with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

print("FAISS Loaded Successfully")

# =========================
# CREATE EMBEDDING
# =========================

def create_embedding(text):

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )

    embedding = response.data[0].embedding

    return np.array([embedding], dtype="float32")

# =========================
# SEARCH RELEVANT CHUNKS
# =========================

def search_chunks(question, top_k=5):

    question_embedding = create_embedding(question)

    distances, indices = index.search(question_embedding, top_k)

    results = []

    for i in indices[0]:

        if i < len(chunks):

            results.append(chunks[i])

    return "\n\n".join(results)

# =========================
# CHAT API
# =========================

@app.route("/chat", methods=["POST"])
def chat():

    data = request.json

    question = data.get("message", "")

    if not question:

        return jsonify({
            "response": "Please ask a question."
        })

    try:

        relevant_text = search_chunks(question)

        prompt = f"""
You are an expert AI assistant for Indian Labour Laws.

Answer ONLY from the provided context.

If answer is not available in context, say:
"उत्तर उपलब्ध दस्तावेज़ों में नहीं मिला।"

Rules:
- Give pointwise answers
- Use simple Hindi
- Mention section/rule if available
- Do not make fake answers
- Keep formatting clean

Context:
{relevant_text}

Question:
{question}
"""

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful Indian Labour Law AI assistant."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2
        )

        answer = response.choices[0].message.content

        return jsonify({
            "response": answer
        })

    except Exception as e:

        return jsonify({
            "response": str(e)
        })

# =========================
# ROOT ROUTE
# =========================

@app.route("/")
def home():

    return "JhaGLC AI Backend Running"

# =========================
# RUN APP
# =========================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 10000))

    app.run(
        host="0.0.0.0",
        port=port
    )
```
