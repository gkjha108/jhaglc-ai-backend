from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import faiss
import pickle
import numpy as np
import os

app = Flask(__name__)
CORS(app)

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

print("Loading FAISS index...")

index = faiss.read_index("faiss.index")

with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

print("FAISS Loaded Successfully")

def create_embedding(text):

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )

    embedding = response.data[0].embedding

    return np.array([embedding], dtype="float32")

def search_chunks(question, top_k=5):

    question_embedding = create_embedding(question)

    distances, indices = index.search(question_embedding, top_k)

    results = []

    for i in indices[0]:

        if i < len(chunks):

            results.append(chunks[i])

    return "\n\n".join(results)

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
You are an expert Labour Law AI assistant.

Instructions:
1. Reply in the SAME language as the user's question.
2. If user asks in English, answer in English.
3. If user asks in Hindi, answer in Hindi.
4. Give answers in clear pointwise format.
5. Use headings and numbering whenever possible.
6. Explain legal provisions simply and accurately.
7. Mention Section/Rule references if available.
8. If answer is not available in documents, say:
   "Answer not found in uploaded documents."
9. Before giving answer read code or rule carefully

User Question:
{question}

Relevant Legal Context:
{context}

Now provide a structured answer.
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

@app.route("/")
def home():

    return "JhaGLC AI Backend Running"

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 10000))

    app.run(
        host="0.0.0.0",
        port=port
    )
