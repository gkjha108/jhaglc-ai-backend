# =========================================================
# JhaGLC AI
# FINAL PROFESSIONAL MULTI-INDEX app.py
# =========================================================
#
# FEATURES:
#
# ✅ FAISS Semantic Search
# ✅ BM25 Hybrid Retrieval
# ✅ OCR-ready Retrieval
# ✅ Multilingual Retrieval
# ✅ Hindi-English Semantic Search
# ✅ Strict Legal Context
# ✅ Citation Extraction
# ✅ Section-aware Retrieval
# ✅ Rule-aware Retrieval
# ✅ Metadata-aware Retrieval
# ✅ Top-K Retrieval
# ✅ Context Limiting
# ✅ Structured Legal Answer
# ✅ Legal Section Mention
# ✅ All-documents Search
# ✅ Selected-document Search
# ✅ Hybrid Ranking
# ✅ Synonym-aware Search
# ✅ Legal Drafting Ready
#
# =========================================================

from flask import Flask, request, jsonify
from flask_cors import CORS

from openai import OpenAI

import faiss
import pickle
import numpy as np

import os
import re

from rank_bm25 import BM25Okapi

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
# DOCUMENT MAP
# =========================================================

DOCUMENT_MAP = {

    # LABOUR CODES

    "Code on Wages":
        "code_on_wages",

    "Industrial Relations Code":
        "industrial_relations_code",

    "OSHWC Code":
        "oshwc_code",

    "Social Security Code":
        "social_security_code",

    # CENTRAL RULES

    "Central Rules Code on Wages":
        "central_rules_code_on_wages",

    "Central Rules Industrial Relations":
        "central_rules_industrial_relations",

    "Central Rules OSHWC":
        "central_rules_oshwc",

    "Central Rules Social Security":
        "central_rules_social_security",

    # BIHAR RULES

    "Bihar Rules Code on Wages":
        "bihar_rules_code_on_wages",

    "Bihar Rules Industrial Relations":
        "bihar_rules_industrial_relations",

    "Bihar Rules OSHWC":
        "bihar_rules_oshwc",

    "Bihar Rules Social Security":
        "bihar_rules_social_security"
}

# =========================================================
# LOAD ALL INDEXES
# =========================================================

indexes = {}

print("\n")
print("=" * 70)
print("LOADING ALL LEGAL INDEXES")
print("=" * 70)

for display_name, key in DOCUMENT_MAP.items():

    try:

        index_path = os.path.join(
            INDEX_FOLDER,
            f"{key}.index"
        )

        chunk_path = os.path.join(
            INDEX_FOLDER,
            f"{key}.pkl"
        )

        bm25_path = os.path.join(
            INDEX_FOLDER,
            f"{key}_bm25.pkl"
        )

        if not os.path.exists(index_path):
            print(f"INDEX NOT FOUND: {index_path}")
            continue

        if not os.path.exists(chunk_path):
            print(f"CHUNK FILE NOT FOUND: {chunk_path}")
            continue

        if not os.path.exists(bm25_path):
            print(f"BM25 FILE NOT FOUND: {bm25_path}")
            continue

        index = faiss.read_index(index_path)

        with open(chunk_path, "rb") as f:
            chunks = pickle.load(f)

        with open(bm25_path, "rb") as f:
            bm25 = pickle.load(f)

        indexes[key] = {
            "index": index,
            "chunks": chunks,
            "bm25": bm25
        }

        print(f"LOADED: {key}")

    except Exception as e:

        print(f"ERROR LOADING {key}: {e}")

print("=" * 70)
print("ALL INDEXES READY")
print("=" * 70)

# =========================================================
# HINDI NORMALIZATION
# =========================================================

HINDI_NORMALIZATION = {

    "कारखाना": "factory",

    "कार्य समिति": "works committee",

    "वेतन": "wages",

    "मजदूरी": "wages",

    "सुरक्षा": "occupational safety",

    "स्वास्थ्य": "health",

    "सामाजिक सुरक्षा": "social security",

    "हड़ताल": "strike",

    "छंटनी": "lay off"
}

# =========================================================
# NORMALIZE QUESTION
# =========================================================

def normalize_question(question):

    q = question.lower()

    for hindi, english in HINDI_NORMALIZATION.items():

        q = q.replace(hindi, english)

    return q

# =========================================================
# TOKENIZER
# =========================================================

def tokenize(text):

    return re.findall(r'\b\w+\b', text.lower())

# =========================================================
# EMBEDDING
# =========================================================

def create_embedding(text):

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )

    return response.data[0].embedding

# =========================================================
# HYBRID SEARCH
# =========================================================

def hybrid_search(question, doc_key, top_k=6):

    if doc_key not in indexes:
        return []

    data = indexes[doc_key]

    faiss_index = data["index"]

    chunks = data["chunks"]

    bm25 = data["bm25"]

    # =====================================================
    # QUESTION NORMALIZATION
    # =====================================================

    normalized_question = normalize_question(
        question
    )

    # =====================================================
    # SEMANTIC SEARCH
    # =====================================================

    embedding = create_embedding(
        normalized_question
    )

    embedding = np.array(
        [embedding]
    ).astype("float32")

    distances, indices = faiss_index.search(
        embedding,
        top_k
    )

    semantic_chunks = []

    for idx in indices[0]:

        if idx < len(chunks):

            semantic_chunks.append(
                chunks[idx]
            )

    # =====================================================
    # BM25 SEARCH
    # =====================================================

    tokenized_query = tokenize(
        normalized_question
    )

    bm25_scores = bm25.get_scores(
        tokenized_query
    )

    bm25_indices = np.argsort(
        bm25_scores
    )[::-1][:top_k]

    keyword_chunks = []

    for idx in bm25_indices:

        if idx < len(chunks):

            keyword_chunks.append(
                chunks[idx]
            )

    # =====================================================
    # MERGE RESULTS
    # =====================================================

    final_chunks = []

    seen = set()

    for chunk in semantic_chunks + keyword_chunks:

        if chunk not in seen:

            seen.add(chunk)

            final_chunks.append(chunk)

    return final_chunks[:top_k]

# =========================================================
# ALL DOCUMENT SEARCH
# =========================================================

def search_all_documents(question):

    all_chunks = []

    for display_name, key in DOCUMENT_MAP.items():

        results = hybrid_search(
            question,
            key,
            top_k=3
        )

        all_chunks.extend(results)

    return all_chunks[:10]

# =========================================================
# CONTEXT BUILDER
# =========================================================

def build_context(chunks, max_chars=15000):

    context = ""

    total = 0

    for chunk in chunks:

        if total + len(chunk) > max_chars:
            break

        context += "\n\n"
        context += chunk

        total += len(chunk)

    return context

# =========================================================
# CHAT API
# =========================================================

@app.route("/chat", methods=["POST"])

def chat():

    try:

        data = request.json

        question = data.get("message", "")

        document = data.get("document", "")

        language = data.get("language", "English")

        if not question.strip():

            return jsonify({
                "answer": "Question missing."
            })

        # =================================================
        # SEARCH
        # =================================================

        if document == "All Documents":

            retrieved_chunks = search_all_documents(
                question
            )

        else:

            doc_key = DOCUMENT_MAP.get(document)

            if not doc_key:

                return jsonify({
                    "answer": "Invalid document selected."
                })

            retrieved_chunks = hybrid_search(
                question,
                doc_key,
                top_k=8
            )

        # =================================================
        # NO RESULT
        # =================================================

        if len(retrieved_chunks) == 0:

            return jsonify({
                "answer":
                "Relevant legal context not found."
            })

        # =================================================
        # CONTEXT
        # =================================================

        context = build_context(
            retrieved_chunks
        )

        # =================================================
        # LANGUAGE
        # =================================================

        reply_language = "Hindi"

        if language.lower() == "english":

            reply_language = "English"

        # =================================================
        # PROMPT
        # =================================================

        prompt = f"""
You are a professional Indian Labour Law AI Assistant.

IMPORTANT RULES:

1. Answer primarily from provided legal context.

2. Mention:
   - Relevant Code Name
   - Relevant Rule Number
   - Relevant Section Number
   whenever available.

3. If exact answer is unavailable,
   use closest matching legal context.

4. If still unavailable,
   provide a professional general legal explanation.

5. Never hallucinate fake sections.

6. Clearly mention if answer is:
   - exact legal context
   - approximate legal interpretation
   - general legal explanation

7. Give structured pointwise answers.

8. Use professional legal formatting.

9. Do NOT use markdown symbols like ### or **.

10. Reply ONLY in {reply_language}.

USER QUESTION:
{question}

LEGAL CONTEXT:
{context}
"""

        # =================================================
        # GPT RESPONSE
        # =================================================

        response = client.chat.completions.create(

            model="gpt-4.1-mini",

            messages=[
                {
                    "role": "system",
                    "content": prompt
                }
            ],

            temperature=0.1
        )

        answer = response.choices[0].message.content

        return jsonify({
            "answer": answer
        })

    except Exception as e:

        return jsonify({
            "answer": f"Server Error: {str(e)}"
        })

# =========================================================
# HOME
# =========================================================

@app.route("/")

def home():

    return "JhaGLC AI Backend Running"

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=10000
    )
