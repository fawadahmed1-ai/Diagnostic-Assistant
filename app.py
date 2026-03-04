import streamlit as st
import os
import torch
import clip
from PIL import Image
import numpy as np
from pinecone import Pinecone
from io import BytesIO
import requests
import base64

# ────────────────────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────────────────────

PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
GROQ_API_KEY     = st.secrets.get("GROQ_API_KEY")

INDEX_NAME       = "medigraph"
IMAGE_FOLDER     = "data/processed_images"
TOP_K            = 12
DEFAULT_THRESHOLD = 0.30

# ────────────────────────────────────────────────────────────────
# Load CLIP
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

def embed_text(text: str) -> list:
    tokens = clip.tokenize([text]).to(device)
    with torch.no_grad():
        emb = clip_model.encode_text(tokens)
    return emb.cpu().numpy().flatten().tolist()

def embed_image(image: Image.Image) -> list:
    tensor = clip_preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = clip_model.encode_image(tensor)
    return emb.cpu().numpy().flatten().tolist()

def get_image_path(filename: str) -> str:
    return os.path.join(IMAGE_FOLDER, filename)

def image_to_base64(img_path):
    try:
        with open(img_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except:
        return None

def generate_explanation(top_match, query_text=""):
    if not GROQ_API_KEY:
        return "LLM summary unavailable (GROQ_API_KEY not set)"

    if not top_match:
        return "No top match to explain."

    score = top_match['score']

    if score < 0.45:
        return "Similarity too low for a meaningful summary (score: {:.3f})."

    meta = top_match['metadata']
    item_type = meta.get('type', 'unknown')
    source = meta.get('source', 'unknown')
    content = meta.get('content', '')

    if item_type == "image" or not content.strip():
        return (
            f"**Top match is a similar chest X-ray** (file: {source}, similarity {score:.3f}). "
            "No report available. Visual similarity suggests shared features — compare side-by-side."
        )

    prompt = f"""
User query: "{query_text or 'uploaded chest X-ray'}"

Top matching case:
- Source: {source}
- Similarity: {score:.3f}
- Content excerpt: {content[:1200]}

Write a concise, neutral 2–4 sentence summary explaining relevance.
Focus on overlapping findings (heart size, lung fields, infiltrates, effusion, cardiomegaly, etc.).
No diagnosis, advice, or speculation.
Factual only.
"""

    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}

    data = {
        "model": "llama-3.1-8b-instant",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 150,
        "temperature": 0.4
    }

    try:
        r = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=data)
        r.raise_for_status()
        return r.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"Summary unavailable ({str(e)})"

# ────────────────────────────────────────────────────────────────
# UI
# ────────────────────────────────────────────────────────────────

st.set_page_config(page_title="MediGraph Diagnostic Assistant", layout="wide")

st.title("MediGraph Diagnostic Assistant")
st.markdown("**Prototype**: Multimodal search of chest X-rays and reports using text or image")

# Controlled input
if "query_value" not in st.session_state:
    st.session_state["query_value"] = ""

col1, col2 = st.columns([4, 1])
with col1:
    query = st.text_input(
        "Describe symptoms, findings or condition:",
        value=st.session_state["query_value"],
        placeholder="e.g. pneumonia, normal chest, cardiomegaly...",
        key="text_query"
    )

with col2:
    threshold = st.slider("Min similarity", 0.0, 1.0, DEFAULT_THRESHOLD, 0.05)

uploaded_file = st.file_uploader("Upload chest X-ray (PNG/JPG/JPEG)", type=["png", "jpg", "jpeg"])

if st.button("Clear Search"):
    st.session_state["query_value"] = ""
    st.rerun()

# ────────────────────────────────────────────────────────────────
# Search Logic
# ────────────────────────────────────────────────────────────────

query_vector = None
search_source = "nothing"
uploaded_image = None

if uploaded_file is not None:
    image_bytes = uploaded_file.read()
    uploaded_image = Image.open(BytesIO(image_bytes)).convert("RGB")
    st.image(uploaded_image, caption="Uploaded image", width="stretch")

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

if query_vector is not None:
    with st.spinner(f"Searching using {search_source}..."):
        # Main query
        main_results = index.query(
            vector=query_vector,
            top_k=TOP_K * 4,
            include_metadata=True,
            include_values=False
        )

        # Force more images with slight threshold boost
        image_results = index.query(
            vector=query_vector,
            top_k=20,  # increased for more images
            filter={"type": "image"},
            include_metadata=True,
            include_values=False
        )

        all_matches = main_results['matches'] + image_results['matches']

        # Deduplicate (keep highest score)
        unique = {}
        for m in all_matches:
            mid = m['id']
            if mid not in unique or m['score'] > unique[mid]['score']:
                unique[mid] = m

        # Filter by threshold
        filtered_matches = [m for m in unique.values() if m['score'] >= threshold]

        if not filtered_matches:
            st.warning(f"No matches ≥ {threshold:.2f}")
        else:
            st.success(f"Found {len(filtered_matches)} matches ≥ {threshold:.2f}")

            # Side-by-side comparison
            if uploaded_image is not None:
                with st.expander("Compare Uploaded Image with Top Matches", expanded=True):
                    cols = st.columns(4)
                    with cols[0]:
                        st.image(uploaded_image, caption="**Your uploaded image**", width="stretch")
                    for i, match in enumerate(filtered_matches[:3]):
                        if match['metadata'].get('type') == 'image':
                            source = match['metadata']['source']
                            img_path = get_image_path(source)
                            if os.path.exists(img_path):
                                with cols[i + 1]:
                                    st.image(img_path, caption=f"Match {i+1} ({source}, score {match['score']:.3f})", width="stretch")

            # Tabs
            tab_all, tab_images = st.tabs(["All Matches", "X-ray Images Only"])

            with tab_all:
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
                                img_base64 = image_to_base64(img_path)
                                if img_base64:
                                    st.markdown(
                                        f'<a href="data:image/png;base64,{img_base64}" target="_blank">'
                                        f'<img src="data:image/png;base64,{img_base64}" style="width:100%; cursor:zoom-in;" /></a>',
                                        unsafe_allow_html=True
                                    )
                                else:
                                    st.image(img_path, caption=f"Image: {source} (score {score:.3f})", width="stretch")
                            else:
                                st.warning(f"Image not found: {source}")

                        st.markdown("---")

            with tab_images:
                image_matches = [m for m in filtered_matches if m['metadata'].get('type') == 'image']
                if not image_matches:
                    st.info("No images in top results. Try lowering threshold or different query.")
                else:
                    cols = st.columns(3)
                    for i, match in enumerate(image_matches):
                        score = match['score']
                        source = match['metadata'].get('source', 'unknown')
                        img_path = get_image_path(source)
                        with cols[i % 3]:
                            if os.path.exists(img_path):
                                img_base64 = image_to_base64(img_path)
                                if img_base64:
                                    st.markdown(
                                        f'<a href="data:image/png;base64,{img_base64}" target="_blank">'
                                        f'<img src="data:image/png;base64,{img_base64}" style="width:100%; cursor:zoom-in;" /></a>',
                                        unsafe_allow_html=True
                                    )
                                else:
                                    st.image(img_path, caption=f"{source} (score {score:.3f})", width="stretch")
                            else:
                                st.warning(f"Missing image: {source}")
                            st.markdown("---")

            # LLM summary
            if filtered_matches:
                top = filtered_matches[0]
                with st.expander("AI Summary of Top Match"):
                    expl = generate_explanation(top, query or "uploaded image")
                    st.write(expl)

else:
    st.info("Enter a description or upload an image to start.")