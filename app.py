from flask import Flask, request, jsonify
from flask_cors import CORS

from openai import OpenAI

import faiss
import pickle
import numpy as np
import os

# =========================================================
# FLASK
# =========================================================

app = Flask(__name__)

CORS(app)

# =========================================================
# OPENAI
# =========================================================

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

# =========================================================
# INDEX FOLDER
# =========================================================

INDEX_FOLDER = "indexes"

# =========================================================
# CATEGORY MAP
# =========================================================

CATEGORY_MAP = {

    # =====================================================
    # LABOUR CODES
    # =====================================================

    "code_on_wages": "code_on_wages",
    "industrial_relations_code": "industrial_relations_code",
    "social_security_code": "social_security_code",
    "oshwc_code": "oshwc_code",

    # =====================================================
    # CENTRAL RULES
    # =====================================================

    "central_rules_code_on_wages": "central_rules_code_on_wages",
    "central_rules_industrial_relations": "central_rules_industrial_relations",
    "central_rules_social_security": "central_rules_social_security",
    "central_rules_oshwc": "central_rules_oshwc",

    # =====================================================
    # BIHAR RULES
    # =====================================================

    "bihar_rules_code_on_wages": "bihar_rules_code_on_wages",
    "bihar_rules_industrial_relations": "bihar_rules_industrial_relations",
    "bihar_rules_social_security": "bihar_rules_social_security",
    "bihar_rules_oshwc": "bihar_rules_oshwc"
}

# =========================================================
# ALIASES
# =========================================================

ALIASES = {

    "osh": "oshwc",
    "oshwc": "oshwc",
    "occupational safety": "oshwc",

    "ir": "industrial_relations",
    "industrial relation": "industrial_relations",
    "industrial relations": "industrial_relations",

    "social security": "social_security",

    "wages code": "code_on_wages",
    "code on wage": "code_on_wages",
    "code on wages": "code_on_wages",

    "bihar wage rule": "bihar_rules_code_on_wages"
}

# =========================================================
# LOAD ALL INDEXES
# =========================================================

indexes = {}
chunks_data = {}

print("\n===================================")
print("LOADING ALL INDEXES")
print("===================================\n")

for category in CATEGORY_MAP.values():

    try:

        index_path = os.path.join(
            INDEX_FOLDER,
            f"{category}.index"
        )

        chunk_path = os.path.join(
            INDEX_FOLDER,
            f"{category}.pkl"
        )

        if os.path.exists(index_path) and os.path.exists(chunk_path):

            indexes[category] = faiss.read_index(index_path)

            with open(chunk_path, "rb") as f:

                chunks_data[category] = pickle.load(f)

            print(f"Loaded: {category}")

        else:

            print(f"Missing: {category}")

    except Exception as e:

        print(f"ERROR loading {category}")
        print(str(e))

print("\n===================================")
print("ALL INDEXES LOADED")
print("===================================\n")

# =========================================================
# EMBEDDING
# =========================================================

def create_embedding(text):

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )

    embedding = response.data[0].embedding

    return np.array([embedding], dtype="float32")

# =========================================================
# SEARCH INDEX
# =========================================================

def search_index(question, category, top_k=8):

    if category not in indexes:

        return "No index found."

    index = indexes[category]

    chunks = chunks_data[category]

    question_embedding = create_embedding(question)

    distances, indices = index.search(question_embedding, top_k)

    results = []

    for i in indices[0]:

        if i < len(chunks):

            results.append(chunks[i])

    return "\n\n".join(results)

# =========================================================
# SEARCH ALL
# =========================================================

def search_all(question):

    combined = ""

    for category in indexes.keys():

        result = search_index(question, category, top_k=3)

        combined += result + "\n\n"

    return combined

# =========================================================
# FORMAT ANSWER
# =========================================================

def generate_answer(question, context, language):

    context = context[:6000]

    if language == "hindi":

        language_instruction = """
Reply ONLY in Hindi.
Use simple legal Hindi.
Use pointwise format.
Use headings.
"""

    else:

        language_instruction = """
Reply ONLY in English.
Use professional legal English.
Use pointwise format.
Use headings.
"""

    prompt = f"""
You are an expert Indian Labour Law AI Assistant.

{language_instruction}

IMPORTANT INSTRUCTIONS:

1. Give structured answers.
2. Use proper headings.
3. Use numbering.
4. Mention Section/Rule references.
5. Explain simply.
6. Avoid unnecessary repetition.
7. Use legal drafting style.
8. If answer is unavailable say:
   "Answer not found in uploaded documents."

USER QUESTION:
{question}

LEGAL CONTEXT:
{context}

NOW GENERATE THE BEST ANSWER.
"""

    response = client.chat.completions.create(

        model="gpt-4.1-mini",

        messages=[
            {
                "role": "system",
                "content": "You are an Indian Labour Law Expert."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0.2
    )

    return response.choices[0].message.content

# =========================================================
# CHAT API
# =========================================================

@app.route("/chat", methods=["POST"])
def chat():

    try:

        data = request.json

        question = data.get("message", "")

        category = data.get("category", "all")

        language = data.get("language", "english")

        if not question:

            return jsonify({
                "response": "Please ask a question."
            })

        # =================================================
        # AUTO ALIAS DETECTION
        # =================================================

        lower_question = question.lower()

        for alias, actual in ALIASES.items():

            if alias in lower_question:

                lower_question += " " + actual

        # =================================================
        # SEARCH
        # =================================================

        if category == "all":

            context = search_all(question)

        else:

            context = search_index(question, category)

        # =================================================
        # GENERATE ANSWER
        # =================================================

        answer = generate_answer(
            question,
            context,
            language
        )

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

    return "JhaGLC AI Multi-Index Backend Running"

# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 10000))

    app.run(
        host="0.0.0.0",
        port=port
    )
