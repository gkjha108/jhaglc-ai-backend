import os
import requests
import numpy as np
import faiss

from io import BytesIO

from flask import Flask, request, jsonify
from flask_cors import CORS

from openai import OpenAI
from PyPDF2 import PdfReader

app = Flask(__name__)
CORS(app)

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

PDF_LINKS = [

"https://drive.google.com/uc?id=1B3GWSRsuL_wETMEnCEtmUSnMS5QvcbYK",

"https://drive.google.com/uc?id=1YPnS12f4VeXIhCnEVsjGv_0R2TTm0iBk",

"https://drive.google.com/uc?id=13o86JYqjUWV42CIguV2hTn32k529-Dsb",

"https://drive.google.com/uc?id=1tp0C61Xmgl08U9zDZ3pBVi3P3XuEjq-B",

"https://drive.google.com/uc?id=17UFmxKnrg-s5bcwFyglxl5m__pJk7ygy",

"https://drive.google.com/uc?id=1FCTIOg5QZhETGfVZY24TdWjR7Yd_99Sn",

"https://drive.google.com/uc?id=1ki_UVve5c4-Gdp_OrnE-AMyNMbZk5KiD",

"https://drive.google.com/uc?id=1HoT3NFMM6FscFRCwjSQFBItJ3mBj7d9O",

"https://drive.google.com/uc?id=1K0-5lq_qg_yFe5VzL8jB279MNUXHpovd",

"https://drive.google.com/uc?id=1DcZBxsGUb6L0N6VWmhgYUCcR6ILofScK",

"https://drive.google.com/uc?id=1PD90M3hdfimxTOUsXBBtWg8Y88SQC3-5",

"https://drive.google.com/uc?id=1UTw22UNY1EJrc7X4xogOj6s_oV3eiEHy"

]

all_chunks = []

def split_text(text, chunk_size=1000):

    chunks = []

    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])

    return chunks

print("Loading PDFs...")

for link in PDF_LINKS:

    try:

        response = requests.get(link)

        pdf = PdfReader(BytesIO(response.content))

        text = ""

        for page in pdf.pages:

            extracted = page.extract_text()

            if extracted:
                text += extracted

        chunks = split_text(text)

        all_chunks.extend(chunks)

        print("PDF Loaded")

    except Exception as e:
        print(e)

print("Creating embeddings...")

embeddings = []

for chunk in all_chunks:

    try:

        emb = client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk
        )

        vector = emb.data[0].embedding

        embeddings.append(vector)

    except:
        pass

embedding_matrix = np.array(embeddings).astype("float32")

dimension = len(embedding_matrix[0])

index = faiss.IndexFlatL2(dimension)

index.add(embedding_matrix)

print("FAISS Index Ready")

@app.route("/")
def home():
    return "JhaGLC AI Smart PDF Search Running"

@app.route("/chat", methods=["POST"])
def chat():

    data = request.json

    question = data.get("message", "")

    query_embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=question
    )

    query_vector = np.array(
        [query_embedding.data[0].embedding]
    ).astype("float32")

    D, I = index.search(query_vector, 5)

    relevant_chunks = []

    for idx in I[0]:
        relevant_chunks.append(all_chunks[idx])

    context = "\n\n".join(relevant_chunks)

    SYSTEM_PROMPT = f"""

You are JhaGLC AI.

Your Intelligent Labour Law Assistant.

Use the following labour law context to answer.

{context}

Instructions:

1. Always answer in structured pointwise format

2. Use headings and subheadings

3. Mention:
   - Relevant Section
   - Relevant Rule
   - Practical Meaning
   - Key Provisions

4. For Hindi questions:
   - Answer in Hindi
   - Use simple legal Hindi

5. For English questions:
   - Answer in professional legal English

6. If applicable, compare:
   - Central Rules
   - Bihar Rules

7. Prefer information from provided PDF context only

8. If answer is not available in context, clearly say:
   "Relevant provision not found in uploaded documents."

9. Format answer like:

• Provision
• Explanation
• Practical Impact
• Compliance Requirement

10. Keep answer well structured and readable

"""

    completion = client.chat.completions.create(

        model="gpt-4o-mini",

        messages=[

            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },

            {
                "role": "user",
                "content": question
            }

        ]

    )

    answer = completion.choices[0].message.content

    return jsonify({
        "response": answer
    })
    app.run(
        host="0.0.0.0",
        port=port
    )
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
