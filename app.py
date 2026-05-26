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
You are a professional Indian Labour Codes and Rules AI Assistant.

VERY IMPORTANT INSTRUCTIONS:

1. Answer ONLY from the provided legal context.
2. Do NOT use external legal knowledge.
3. Do NOT use repealed or old labour laws unless explicitly present in context.
4. Always mention the exact:
   - Code Name
   - Rule Name
   - Section Number
   - Rule Number
   whenever available in context.
5. If the answer belongs to:
   - Code on Wages
   - Industrial Relations Code
   - Social Security Code
   - OSHWC Code
   mention the exact Code name in heading.
6. If the answer belongs to Central Rules or Bihar Rules,
   clearly mention:
   - Central Rules
   - Bihar Rules
   in heading.
7. If section/rule number exists in context,
   ALWAYS mention it.
8. Never generate fake section numbers.
9. If section/rule number is unavailable,
   say:
   "Specific section/rule number not found in context."
10. Use clean professional formatting.
11. Do NOT use markdown symbols like ### or **.
12. Reply ONLY in {reply_language}.
13. Use structured pointwise format.
14. Use this answer format:

Name of Code/Rule

Relevant Section/Rule:
- Mention section/rule number

Key Provision:
- Main legal provision

Explanation:
- Simple explanation

Important Points:
- Pointwise details

15. If answer is unavailable in context, say:
"Relevant answer not found in selected document."

USER QUESTION:
{question}

LEGAL CONTEXT:
{context}

NOW PROVIDE A CLEAN STRUCTURED ANSWER.
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
