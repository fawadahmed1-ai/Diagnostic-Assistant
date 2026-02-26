import streamlit as st
import os
import torch
import clip
from PIL import Image
import numpy as np
from pinecone import Pinecone

# ────────────────────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────────────────────

PINECONE_API_KEY = "pcsk_6miYSa_SiscdWtLR8NTmE2THiUkBvicVdapW2K7o9MEXdZjXgvCWxqKJ1JpEGoz8AvFqRP"   # ← YOUR REAL KEY
INDEX_NAME = "medigraph"

IMAGE_FOLDER = "data/processed_images"           # where your resized images are stored

TOP_K = 5                                        # how many results to show

# ────────────────────────────────────────────────────────────────
# Initialization
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
# Functions
# ────────────────────────────────────────────────────────────────

def embed_query(text: str):
    """Embed user query text using CLIP"""
    text_token = clip.tokenize([text]).to(device)
    with torch.no_grad():
        features = clip_model.encode_text(text_token)
    return features.cpu().numpy().flatten().tolist()

def get_image_path(filename: str):
    """Construct full path to the original/resized image"""
    return os.path.join(IMAGE_FOLDER, filename)

# ────────────────────────────────────────────────────────────────
# Streamlit UI
# ────────────────────────────────────────────────────────────────

st.set_page_config(page_title="MediGraph - Basic Diagnostic Assistant", layout="wide")

st.title("MediGraph Diagnostic Assistant")
st.markdown("**Prototype**: Search medical reports and chest X-rays using natural language")

query = st.text_input(
    "Describe symptoms, findings or condition:",
    placeholder="e.g. bilateral infiltrates in lower lobes, pneumonia, normal chest, cardiomegaly...",
    help="Type a description and press Enter"
)

if query:
    with st.spinner("Searching..."):
        # Embed the query
        query_vector = embed_query(query)

        # Query Pinecone
        results = index.query(
            vector=query_vector,
            top_k=TOP_K,
            include_metadata=True,
            include_values=False
        )

        matches = results['matches']

        if not matches:
            st.warning("No relevant results found.")
        else:
            st.success(f"Found {len(matches)} similar items (showing top {TOP_K})")

            # Display results in columns
            cols = st.columns(2) if len(matches) <= 4 else st.columns(3)

            for i, match in enumerate(matches):
                score = match['score']
                metadata = match['metadata']
                item_type = metadata.get('type', 'unknown')
                source = metadata.get('source', 'unknown')

                with cols[i % len(cols)]:
                    st.markdown(f"**Match {i+1}** — Similarity: {score:.3f}")

                    if item_type == "text":
                        st.info(f"**Report excerpt** from {source}")
                        # Show short preview of text
                        st.write(metadata.get('content', 'No content available')[:300] + "...")
                    else:
                        st.image(
                            get_image_path(source),
                            caption=f"Image: {source} (score: {score:.3f})",
                            use_column_width=True
                        )

                    st.markdown("---")

else:
    st.info("Enter a description above to start searching.")