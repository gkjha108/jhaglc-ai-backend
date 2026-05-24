import os
import requests
from io import BytesIO

from flask import Flask, request, jsonify
from flask_cors import CORS

from openai import OpenAI

from PyPDF2 import PdfReader

app = Flask(__name__)
CORS(app)

client = OpenAI(
    api_key=os.environ.get(\"OPENAI_API_KEY\")
)

# GOOGLE DRIVE PDF LINKS

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

# LOAD PDF TEXT

all_pdf_text = \"\"

for link in PDF_LINKS:

    try:

        response = requests.get(link)

        pdf = PdfReader(BytesIO(response.content))

        text = \"\"

        for page in pdf.pages:
            text += page.extract_text()

        all_pdf_text += text + \"\\n\\n\"

    except:
        pass

SYSTEM_PROMPT = f\"\"\"

You are JhaGLC AI.

Your Intelligent Labour Law Assistant.

Use the following labour law documents as primary knowledge source:

{all_pdf_text[:120000]}

Instructions:
- Answer only labour law related questions
- Support Hindi and English
- Prefer information from uploaded PDFs
- Explain clearly
- Mention relevant legal provisions when possible
- Give practical interpretation

\"\"\"

@app.route(\"/\")
def home():
    return \"JhaGLC AI PDF Intelligence Running\"

@app.route(\"/chat\", methods=[\"POST\"])
def chat():

    data = request.json

    user_message = data.get(\"message\", \"\")

    completion = client.chat.completions.create(

        model=\"gpt-4o-mini\",

        messages=[

            {
                \"role\": \"system\",
                \"content\": SYSTEM_PROMPT
            },

            {
                \"role\": \"user\",
                \"content\": user_message
            }

        ]

    )

    answer = completion.choices[0].message.content

    return jsonify({
        \"response\": answer
    })

if __name__ == \"__main__\":
    app.run(host=\"0.0.0.0\", port=10000)
