import requests
import streamlit as st

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]  # or hardcode temporarily: "gsk_..."

headers = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}

data = {
    "model": "llama3-8b-8192",
    "messages": [{"role": "user", "content": "Hello, Groq!"}],
    "max_tokens": 50
}

try:
    r = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=data)
    print("Status code:", r.status_code)
    print("Response:", r.text)
except Exception as e:
    print("Error:", str(e))