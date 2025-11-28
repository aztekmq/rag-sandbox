import requests
import gradio as gr

OLLAMA_URL = "http://localhost:11434"

def chat_with_ollama(message, history):
    resp = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": "llama3.1:8b",   # or whatever you pulled
            "messages": [
                {"role": "user", "content": message}
            ],
            "stream": False,
        },
        timeout=600,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["message"]["content"]

demo = gr.ChatInterface(chat_with_ollama)
demo.launch()
