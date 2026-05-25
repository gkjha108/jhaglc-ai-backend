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
# LOAD FAISS INDEX
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

    return results

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

        retrieved_chunks = search_chunks(question)

        context = ""

        sources = set()

        for item in retrieved_chunks:

            context += item["text"] + "\n\n"

            sources.add(item["source"])

        source_text = ", ".join(sources)

        prompt = f"""
You are an expert Indian Labour Law AI assistant.

STRICT INSTRUCTIONS:

1. Reply in the SAME language as the user's question.
2. If user asks in English, answer in English.
3. If user asks in Hindi, answer in Hindi.
4. Give answers in structured pointwise format.
5. Use markdown headings, numbering, and bullets.
6. Never give very long paragraphs.
7. Mention exact legal provisions whenever available.
8. Mention:
   - Act name
   - Rule number
   - Section number
   - Chapter if available
9. Prefer exact wording from legal documents.
10. Do not invent legal information.
11. If exact answer is unavailable, clearly say so.
12. Keep formatting clean and professional.
13. Explain practical meaning in simple language.

User Question:
{question}

Relevant Legal Context:
{context}

Source Documents:
{source_text}

Now provide the best accurate legal answer.
"""

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a highly accurate Indian Labour Law AI assistant."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1
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
# HOME ROUTE
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
