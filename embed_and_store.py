import os
import torch
import clip
from PIL import Image
from pinecone import Pinecone

# ────────────────────────────────────────────────────────────────
# CONFIG - CHANGE THESE VALUES
# ────────────────────────────────────────────────────────────────

PINECONE_API_KEY = "pcsk_6miYSa_SiscdWtLR8NTmE2THiUkBvicVdapW2K7o9MEXdZjXgvCWxqKJ1JpEGoz8AvFqRP"   # ← PASTE YOUR REAL PINECONE API KEY HERE

INDEX_NAME = "medigraph"
DIMENSION = 512                                   # CLIP ViT-B/32 dimension

TEXT_DIR = "data/text"
IMAGE_DIR = "data/processed_images"

# ────────────────────────────────────────────────────────────────
# Initialization
# ────────────────────────────────────────────────────────────────

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load CLIP model (downloads automatically first time)
print("Loading CLIP model...")
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
print("CLIP model loaded.")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if it doesn't exist
if INDEX_NAME not in pc.list_indexes().names():
    print(f"Creating new index: {INDEX_NAME}")
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric="cosine",
        spec={"serverless": {"cloud": "aws", "region": "us-east-1"}}  # ← CHANGE REGION IF DIFFERENT
    )
else:
    print(f"Using existing index: {INDEX_NAME}")

index = pc.Index(INDEX_NAME)

# ────────────────────────────────────────────────────────────────
# Embedding functions (both use CLIP → same dimension)
# ────────────────────────────────────────────────────────────────

def embed_text(text: str):
    """Embed text using CLIP text encoder"""
    text_token = clip.tokenize([text]).to(device)
    with torch.no_grad():
        features = clip_model.encode_text(text_token)
    return features.cpu().numpy().flatten()  # shape: (512,)


def embed_image(image_path: str):
    """Embed image using CLIP image encoder"""
    image = clip_preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        features = clip_model.encode_image(image)
    return features.cpu().numpy().flatten()  # shape: (512,)


# ────────────────────────────────────────────────────────────────
# Main logic
# ────────────────────────────────────────────────────────────────

vectors_to_upsert = []

# 1. Embed text files
if not os.path.exists(TEXT_DIR):
    print(f"Error: Text directory not found → {TEXT_DIR}")
else:
    print(f"Processing text files in: {TEXT_DIR}")
    for filename in os.listdir(TEXT_DIR):
        if filename.lower().endswith(".txt"):
            path = os.path.join(TEXT_DIR, filename)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                vector = embed_text(text)
                vec_id = f"text_{os.path.splitext(filename)[0]}"
                vectors_to_upsert.append({
                    "id": vec_id,
                    "values": vector.tolist(),
                    "metadata": {"type": "text", "source": filename}
                })
                print(f"Embedded text: {vec_id}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

# 2. Embed images
if not os.path.exists(IMAGE_DIR):
    print(f"Error: Image directory not found → {IMAGE_DIR}")
else:
    print(f"Processing images in: {IMAGE_DIR}")
    for filename in os.listdir(IMAGE_DIR):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(IMAGE_DIR, filename)
            try:
                vector = embed_image(path)
                vec_id = f"img_{os.path.splitext(filename)[0]}"
                vectors_to_upsert.append({
                    "id": vec_id,
                    "values": vector.tolist(),
                    "metadata": {"type": "image", "source": filename}
                })
                print(f"Embedded image: {vec_id}")
            except Exception as e:
                print(f"Error processing image {filename}: {e}")

# 3. Upsert to Pinecone
if vectors_to_upsert:
    print(f"Upserting {len(vectors_to_upsert)} vectors...")
    try:
        index.upsert(vectors=vectors_to_upsert)
        print(f"Successfully upserted {len(vectors_to_upsert)} vectors!")
    except Exception as e:
        print(f"Upsert failed: {e}")
else:
    print("No vectors to upsert. Check your data folders.")

print("Script finished.")