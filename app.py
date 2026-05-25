from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import faiss
import pickle
import numpy as np
import os

# =========================================
# FLASK SETUP
# =========================================

app = Flask(__name__)
CORS(app)

# =========================================
# OPENAI CLIENT
# =========================================

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

# =========================================
# INDEX FOLDER
# =========================================

INDEX_FOLDER = "indexes"

# =========================================
# DOCUMENT MAP
# =========================================

DOCUMENT_MAP = {

    # LABOUR CODES

    "code_on_wages": "code_on_wages",

    "industrial_relations_code":
        "industrial_relations_code",

    "social_security_code":
        "social_security_code",

    "oshwc_code":
        "oshwc_code",

    # CENTRAL RULES

    "central_rules_code_on_wages":
        "central_rules_code_on_wages",

    "central_rules_industrial_relations":
        "central_rules_industrial_relations",

    "central_rules_social_security":
        "central_rules_social_security",

    "central_rules_oshwc":
        "central_rules_oshwc",

    # BIHAR RULES

    "bihar_rules_code_on_wages":
        "bihar_rules_code_on_wages",

    "bihar_rules_industrial_relations":
        "bihar_rules_industrial_relations",

    "bihar_rules_social_security":
        "bihar_rules_social_security",

    "bihar_rules_oshwc":
        "bihar_rules_oshwc"
}

# =========================================
# LOAD ALL INDEXES
# =========================================

indexes = {}

print("===================================")
print("LOADING ALL INDEXES")
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

# =========================================
# CREATE EMBEDDING
# =========================================

def create_embedding(text):

    response = client.embeddings.create(

        model="text-embedding-3-small",

        input=text
    )

    embedding = response.data[0].embedding

    return np.array([embedding], dtype="float32")

# =========================================
# SEARCH SINGLE DOCUMENT
# =========================================

def search_document(question, document_key, top_k=8):

    if document_key not in indexes:

        return ""

    try:

        data = indexes[document_key]

        index = data["index"]

        chunks = data["chunks"]

        question_embedding = create_embedding(question)

        distances, indices = index.search(
            question_embedding,
            top_k
        )

        results = []

        for i in indices[0]:

            if i < len(chunks):

                chunk = chunks[i]

                # OPTIONAL STRICT FILTER

                if any(
                    word.lower() in chunk.lower()
                    for word in question.split()
                ):

                    results.append(chunk)

        return "\n\n".join(results)

    except Exception as e:

        print("SEARCH ERROR:", e)

        return ""

# =========================================
# SEARCH ALL DOCUMENTS
# =========================================

def search_all_documents(question, top_k=5):

    all_results = []

    for key in indexes:

        try:

            data = indexes[key]

            index = data["index"]

            chunks = data["chunks"]

            question_embedding = create_embedding(question)

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

# =========================================
# HOME ROUTE
# =========================================

@app.route("/")
def home():

    return "JhaGLC AI Running Successfully"

# =========================================
# CHAT ROUTE
# =========================================

@app.route("/chat", methods=["POST"])
def chat():

    try:

        data = request.json

        question = data.get("message", "")

        document = data.get("document", "all")

        language = data.get("language", "english")

        print("===================================")
        print("QUESTION:", question)
        print("DOCUMENT:", document)
        print("LANGUAGE:", language)
        print("===================================")

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

            context = search_document(
                question,
                document
            )

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
        # PROMPT
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
13. NEVER use generic titles like "Indian Labour Law".
14. Always mention the selected Code/Rule name in heading.
15. Mention exact Section/Rule numbers whenever available.
16. Do NOT use markdown symbols like ### or **.
17. Return clean plain text formatting only.
18. Use professional legal formatting.

USER QUESTION:
{question}

LEGAL CONTEXT:
{context}

NOW PROVIDE A STRUCTURED ANSWER.
"""

        # =========================================
        # OPENAI RESPONSE
        # =========================================

        response = client.chat.completions.create(

            model="gpt-4.1-mini",

            messages=[

                {
                    "role": "system",

                    "content":
                        "You are a professional Labour Law AI Assistant."
                },

                {
                    "role": "user",

                    "content": prompt
                }
            ],

            temperature=0.2
        )

        answer = response.choices[0].message.content

        # =========================================
        # RESPONSE
        # =========================================

        return jsonify({

            "response": answer
        })

    except Exception as e:

        print("ERROR:", str(e))

        return jsonify({

            "response": str(e)
        })

# =========================================
# MAIN
# =========================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 10000))

    app.run(

        host="0.0.0.0",

        port=port
    )
