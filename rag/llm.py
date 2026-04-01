
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def ask_llm(query, context):
    prompt = f"""
    You are an intelligent surveillance analyst.

    Use ONLY the context below to answer.

    Context:
    {context}

    Question:
    {query}

    Answer clearly and concisely. If information is missing, say "Not enough data".
    """





    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300
    )

    return response.choices[0].message.content