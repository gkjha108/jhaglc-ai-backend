import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI

app = Flask(__name__)
CORS(app)

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

SYSTEM_PROMPT = """
You are JhaGLC AI.

Your Intelligent Labour Law Assistant.

You answer questions related to:

- Labour Codes
- Central Rules
- Bihar Rules
- Labour Compliance
- Occupational Safety
- Social Security
- Industrial Relations

Support Hindi and English.

Always:
1. Explain clearly
2. Give practical interpretation
3. Mention legal context
"""

@app.route("/")
def home():
    return "JhaGLC AI Backend Running"

@app.route("/chat", methods=["POST"])
def chat():

    data = request.json

    user_message = data.get("message", "")

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_message
            }
        ]
    )

    answer = completion.choices[0].message.content

    return jsonify({
        "response": answer
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
