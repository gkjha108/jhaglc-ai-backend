from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import faiss
import pickle
import numpy as np
import os

# =========================================================
# FLASK APP
# =========================================================

app = Flask(__name__)
CORS(app)

# =========================================================
# OPENAI CLIENT
# =========================================================

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

# =========================================================
# LOAD ALL INDEXES
# =========================================================

print("===================================")
print("LOADING ALL INDEXES...")
print("===================================")

INDEX_FOLDER = "indexes"

indexes = {}

DOCUMENT_MAP = {

    # =====================================================
    # LABOUR CODES
    # =====================================================

    "code_on_wages":
        "Code on Wages",

    "industrial_relations_code":
        "Industrial Relations Code",

    "social_security_code":
        "Social Security Code",

    "oshwc_code":
        "OSHWC Code",

    # =====================================================
    # CENTRAL RULES
    # =====================================================

    "central_rules_code_on_wages":
        "Central Rules - Code on Wages",

    "central_rules_industrial_relations":
        "Central Rules - Industrial Relations",

    "central_rules_social_security":
        "Central Rules - Social Security",

    "central_rules_oshwc":
        "Central Rules - OSHWC",

    # =====================================================
    # BIHAR RULES
    # =====================================================

    "bihar_rules_code_on_wages":
        "Bihar Rules - Code on Wages",

    "bihar_rules_industrial_relations":
        "Bihar Rules - Industrial Relations",

    "bihar_rules_social_security":
        "Bihar Rules - Social Security",

    "bihar_rules_oshwc":
        "Bihar Rules - OSHWC"
}

# =========================================================
# LOAD EACH INDEX
# =========================================================

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

# =========================================================
# CREATE EMBEDDING
# =========================================================

def create_embedding(text):

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )

    embedding = response.data[0].embedding

    return np.array([embedding], dtype="float32")

# =========================================================
# SEARCH SINGLE DOCUMENT
# =========================================================

def search_single_document(question, document_key, top_k=8):

    if document_key not in indexes:
        return ""

    index = indexes[document_key]["index"]
    chunks = indexes[document_key]["chunks"]

    question_embedding = create_embedding(question)

    distances, indices = index.search(question_embedding, top_k)

    results = []

    for i in indices[0]:

        if i < len(chunks):

            results.append(chunks[i])

    return "\n\n".join(results)

# =========================================================
# SEARCH ALL DOCUMENTS
# =========================================================

def search_all_documents(question, top_k=5):

    all_results = []

    question_embedding = create_embedding(question)

    for key in indexes:

        try:

            index = indexes[key]["index"]
            chunks = indexes[key]["chunks"]

            distances, indices = index.search(
                question_embedding,
                top_k
            )

            for i in indices[0]:

                if i < len(chunks):

                    all_results.append(chunks[i])

        except:
            pass

    return "\n\n".join(all_results)

# =========================================================
# CHAT API
# =========================================================

@app.route("/chat", methods=["POST"])
def chat():

    try:

        data = request.json

        question = data.get("message", "").strip()

        language = data.get("language", "English")

        document = data.get("document", "all")

        # =================================================
        # EMPTY QUESTION
        # =================================================

        if not question:

            return jsonify({
                "response": "Please ask a question."
            })

        # =================================================
        # DOCUMENT SEARCH
        # =================================================

        if document == "all":

            context = search_all_documents(question)

        else:

            context = search_single_document(
                question,
                document
            )

        # =================================================
        # CONTEXT LIMIT
        # =================================================

        context = context[:6000]

        # =================================================
        # NO RESULT
        # =================================================

        if not context.strip():

            return jsonify({
                "response":
                "Answer not found in selected document."
            })

        # =================================================
        # LANGUAGE
        # =================================================

        if language.lower() == "hindi":

            reply_language = "Hindi"

        else:

            reply_language = "English"

        # =================================================
        # STRICT RAG PROMPT
        # =================================================

        prompt = f"""
You are a STRICT Labour Law AI Assistant.

VERY IMPORTANT RULES:

1. Answer ONLY from the provided legal context.
2. Do NOT use general legal knowledge.
3. Do NOT make up information.
4. If answer is unavailable in context, say:
   "Answer not found in selected document."
5. Reply ONLY in {reply_language}.
6. Give structured pointwise answers.
7. Use headings and numbering.
8. Mention exact Section / Rule references.
9. Keep answer professional and accurate.
10. Explain in simple language.

USER QUESTION:
{question}

LEGAL CONTEXT:
{context}

NOW GIVE THE FINAL ANSWER.
"""

        # =================================================
        # GPT RESPONSE
        # =================================================

        response = client.chat.completions.create(

            model="gpt-4.1-mini",

            messages=[

                {
                    "role": "system",
                    "content":
                    "You are an expert Indian Labour Law Assistant."
                },

                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0.1
        )

        answer = response.choices[0].message.content

        # =================================================
        # FINAL RESPONSE
        # =================================================

        return jsonify({
            "response": answer
        })

    except Exception as e:

        return jsonify({
            "response": str(e)
        })

# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():

    return "JhaGLC AI Backend Running Successfully"

# =========================================================
# RUN APP
# =========================================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 10000))

    app.run(
        host="0.0.0.0",
        port=port
    )
