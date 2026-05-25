from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import faiss
import pickle
import numpy as np
import os

# =====================================================
# FLASK SETUP
# =====================================================

app = Flask(__name__)
CORS(app)

# =====================================================
# OPENAI CLIENT
# =====================================================

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

# =====================================================
# INDEX FOLDER
# =====================================================

INDEX_FOLDER = "indexes"

# =====================================================
# DOCUMENT MAP
# =====================================================

DOCUMENT_MAP = {

    # =========================================
    # LABOUR CODES
    # =========================================

    "code_on_wages":
        "Code on Wages",

    "industrial_relations_code":
        "Industrial Relations Code",

    "social_security_code":
        "Social Security Code",

    "oshwc_code":
        "OSHWC Code",

    # =========================================
    # CENTRAL RULES
    # =========================================

    "central_rules_code_on_wages":
        "Central Rules - Code on Wages",

    "central_rules_industrial_relations":
        "Central Rules - Industrial Relations",

    "central_rules_social_security":
        "Central Rules - Social Security",

    "central_rules_oshwc":
        "Central Rules - OSHWC",

    # =========================================
    # BIHAR RULES
    # =========================================

    "bihar_rules_code_on_wages":
        "Bihar Rules - Code on Wages",

    "bihar_rules_industrial_relations":
        "Bihar Rules - Industrial Relations",

    "bihar_rules_social_security":
        "Bihar Rules - Social Security",

    "bihar_rules_oshwc":
        "Bihar Rules - OSHWC"
}

# =====================================================
# LOAD ALL INDEXES
# =====================================================

indexes = {}

print("===================================")
print("LOADING ALL INDEXES...")
print("===================================")

for key in DOCUMENT_MAP:

    try:

        index_path = f"{INDEX_FOLDER}/{key}.index"
        chunk_path = f"{INDEX_FOLDER}/{key}.pkl"

        index = faiss.read_index(index_path)

        with open(chunk_path, "rb") as f:
            chunks = pickle.load(f)

        indexes[key] = {
            "index": index,
            "chunks": chunks
        }

        print(f"Loaded: {key}")

    except Exception as e:

        print(f"ERROR loading {key}: {e}")

print("===================================")
print("ALL INDEXES READY")
print("===================================")

# =====================================================
# CREATE EMBEDDING
# =====================================================

def create_embedding(text):

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )

    embedding = response.data[0].embedding

    return np.array([embedding], dtype="float32")

# =====================================================
# SEARCH SINGLE DOCUMENT
# =====================================================

def search_document(question, document_key, top_k=15):

    if document_key not in indexes:

        return ""

    data = indexes[document_key]

    index = data["index"]
    chunks = data["chunks"]

    question_embedding = create_embedding(question)

    distances, indices = index.search(question_embedding, top_k)

    question_lower = question.lower()

    results = []

    for i in indices[0]:

        if i < len(chunks):

            chunk = chunks[i]

            # =================================
            # KEYWORD BOOST
            # =================================

            if any(word in chunk.lower() for word in question_lower.split()):

                results.insert(0, chunk)

            else:

                results.append(chunk)

    return "\n\n".join(results)

# =====================================================
# SEARCH ALL DOCUMENTS
# =====================================================

def search_all_documents(question, top_k=10):

    all_results = []

    for key in indexes:

        data = indexes[key]

        index = data["index"]
        chunks = data["chunks"]

        question_embedding = create_embedding(question)

        distances, indices = index.search(question_embedding, top_k)

        question_lower = question.lower()

        for i in indices[0]:

            if i < len(chunks):

                chunk = chunks[i]

                if any(word in chunk.lower() for word in question_lower.split()):

                    all_results.insert(0, chunk)

                else:

                    all_results.append(chunk)

    return "\n\n".join(all_results[:20])

# =====================================================
# CHAT ROUTE
# =====================================================

@app.route("/chat", methods=["POST"])
def chat():

    try:

        data = request.json

        question = data.get("message", "")
        document = data.get("document", "all")
        language = data.get("language", "english")

        if not question:

            return jsonify({
                "response": "Please ask a question."
            })

        # =========================================
        # SEARCH CONTEXT
        # =========================================

        if document == "all":

            context = search_all_documents(question)

        else:

            context = search_document(question, document)

        # =========================================
        # CONTEXT LIMIT
        # =========================================

        context = context[:6000]

        # =========================================
        # LANGUAGE
        # =========================================

        if language.lower() == "hindi":

            reply_language = "Hindi"

        else:

            reply_language = "English"

        # =========================================
        # STRICT RAG PROMPT
        # =========================================

       prompt = f"""
You are an intelligent Indian Labour Law AI Assistant.

IMPORTANT RULES:

1. First try to answer from the provided legal context.
2. Prefer the selected document context whenever possible.
3. If exact answer is not available, use the closest matching legal context.
4. If still not available, provide a general legal explanation related to Indian Labour Law.
5. Clearly mention when the answer is based on general legal understanding.
6. Reply ONLY in {reply_language}.
7. Give structured pointwise answers.
8. Use headings and numbering.
9. Mention relevant Sections/Rules whenever available.
10. Keep answers professional, simple, and legally accurate.
11. Avoid hallucination and unrelated information.
12. If answer is partially available in context, combine context + general explanation carefully.

USER QUESTION:
{question}

LEGAL CONTEXT:
{context}

NOW PROVIDE A STRUCTURED ANSWER.
"""

        # =========================================
        # GPT RESPONSE
        # =========================================

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional Indian Labour Law AI Assistant."
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

# =====================================================
# HOME ROUTE
# =====================================================

@app.route("/")
def home():

    return "JhaGLC AI Backend Running Successfully"

# =====================================================
# RUN APP
# =====================================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 10000))

    app.run(
        host="0.0.0.0",
        port=port
    )
