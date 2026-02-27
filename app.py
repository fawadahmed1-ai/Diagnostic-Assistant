import streamlit as st
import os
import torch
import clip
from PIL import Image
import numpy as np
from pinecone import Pinecone
from io import BytesIO
import requests

# ────────────────────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────────────────────

PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")  # Optional

INDEX_NAME = "medigraph"
IMAGE_FOLDER = "data/processed_images"
TOP_K = 12
DEFAULT_THRESHOLD = 0.30

# ────────────────────────────────────────────────────────────────
# Load CLIP (cached)
# ────────────────────────────────────────────────────────────────

@st.cache_resource
def load_clip():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

clip_model, clip_preprocess, device = load_clip()

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# ────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────

def embed_text(text: str):
    text_token = clip.tokenize([text]).to(device)
    with torch.no_grad():
        features = clip_model.encode_text(text_token)
    return features.cpu().numpy().flatten().tolist()

def embed_image(image: Image.Image):
    img_tensor = clip_preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = clip_model.encode_image(img_tensor)
    return features.cpu().numpy().flatten().tolist()

def get_image_path(filename: str):
    return os.path.join(IMAGE_FOLDER, filename)

def generate_explanation(top_match, query_text=""):
    if not GROQ_API_KEY:
        return "LLM explanation unavailable (GROQ_API_KEY missing in secrets.toml)"

    if not top_match:
        return "No top match to explain."

    score = top_match['score']
    meta = top_match['metadata']
    source = meta.get('source', 'unknown')
    content = meta.get('content', 'No description')[:500]

    prompt = f"""
    User query: "{query_text or 'uploaded image'}"
    Top similar case: {source} (similarity {score:.3f})
    Content: {content}

    Give a short, neutral summary (2-3 sentences) of why this case is relevant.
    Do NOT give medical diagnosis or advice.
    """

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "llama-3.1-8b-instant",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 150,
        "temperature": 0.7
    }

    try:
        r = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=data)
        r.raise_for_status()
        return r.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"Explanation unavailable ({str(e)})"

# ────────────────────────────────────────────────────────────────
# UI
# ────────────────────────────────────────────────────────────────

st.set_page_config(page_title="MediGraph - Diagnostic Assistant", layout="wide")

st.title("MediGraph Diagnostic Assistant")
st.markdown("**Prototype**: Search chest X-rays and medical reports using text or image")

# Inputs
query = st.text_input(
    "Describe symptoms, findings or condition:",
    placeholder="e.g. pneumonia, normal chest, cardiomegaly, bilateral infiltrates...",
    key="text_query"
)

threshold = st.slider(
    "Min similarity to show",
    0.0, 1.0, DEFAULT_THRESHOLD, 0.05,
    help="Only show matches ≥ this score (higher = stricter)"
)

uploaded_file = st.file_uploader(
    "Upload your own chest X-ray (PNG/JPG/JPEG)",
    type=["png", "jpg", "jpeg"],
    help="Upload an image to find similar cases"
)

# ────────────────────────────────────────────────────────────────
# Search logic (hybrid text + image)
# ────────────────────────────────────────────────────────────────

query_vector = None
search_source = "nothing"

if uploaded_file is not None:
    image_bytes = uploaded_file.read()
    uploaded_image = Image.open(BytesIO(image_bytes)).convert("RGB")
    st.image(uploaded_image, caption="Uploaded image", width=300)

    with st.spinner("Embedding uploaded image..."):
        image_vector = embed_image(uploaded_image)

    if query:
        with st.spinner("Embedding text query..."):
            text_vector = embed_text(query)
        query_vector = [(t + i) / 2 for t, i in zip(text_vector, image_vector)]
        search_source = "text + uploaded image"
    else:
        query_vector = image_vector
        search_source = "uploaded image"

elif query:
    with st.spinner("Embedding query text..."):
        query_vector = embed_text(query)
    search_source = f'text query: "{query}"'

# Run search
if query_vector is not None:
    with st.spinner(f"Searching database using {search_source}..."):
        results = index.query(
            vector=query_vector,
            top_k=TOP_K * 2,
            include_metadata=True,
            include_values=False
        )

        filtered_matches = [m for m in results['matches'] if m['score'] >= threshold]

        if not filtered_matches:
            st.warning(f"No matches found above {threshold:.2f}")
        else:
            st.success(f"Found {len(filtered_matches)} matches above {threshold:.2f}")

            cols = st.columns(3)
            for i, match in enumerate(filtered_matches):
                score = match['score']
                meta = match['metadata']
                item_type = meta.get('type', 'unknown')
                source = meta.get('source', 'unknown')

                with cols[i % 3]:
                    st.markdown(f"**Match {i+1}** – Similarity: {score:.3f}")

                    if item_type == "text":
                        st.info("**Medical Report**")
                        st.caption(f"File: {source}")
                        content = meta.get('content', 'No full content stored.')
                        st.markdown(content[:800] + "..." if len(content) > 800 else content)
                    else:
                        img_path = get_image_path(source)
                        if os.path.exists(img_path):
                            st.image(
                                img_path,
                                caption=f"Image: {source} (score {score:.3f})",
                                width=300
                            )
                        else:
                            st.error(f"Image not found: {source}")

                    st.markdown("---")

            # LLM explanation
            if filtered_matches:
                top_match = filtered_matches[0]
                with st.expander("AI Summary of Top Match"):
                    explanation = generate_explanation(top_match, query or "uploaded image")
                    st.write(explanation)
else:
    st.info("Enter a description or upload an image to start searching.")