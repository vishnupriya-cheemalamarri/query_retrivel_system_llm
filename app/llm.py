import requests
from typing import List

def build_prompt(question: str, context_chunks: List[str]) -> str:
    context = "\n".join(context_chunks)
    return f"Context: {context}\n\nQuestion: {question}\nAnswer:"

def generate_answer(question: str, context_chunks: List[str]) -> str:
    prompt = build_prompt(question, context_chunks)

    response = requests.post(
        "https://api-inference.huggingface.co/models/google/flan-t5-small",
        headers={"Accept": "application/json"},
        json={"inputs": prompt}
    )

    try:
        return response.json()[0]["generated_text"]
    except Exception as e:
        return f"Error from model: {response.text}"
