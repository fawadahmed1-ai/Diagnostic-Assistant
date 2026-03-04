import os
import torch
import clip
from PIL import Image
from pinecone import Pinecone

# ────────────────────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────────────────────

PINECONE_API_KEY = "pcsk_6miYSa_SiscdWtLR8NTmE2THiUkBvicVdapW2K7o9MEXdZjXgvCWxqKJ1JpEGoz8AvFqRP"

INDEX_NAME = "medigraph"
DIMENSION = 512

TEXT_DIR = "data/text"
IMAGE_DIR = "data/processed_images"

# ────────────────────────────────────────────────────────────────
# Initialization
# ────────────────────────────────────────────────────────────────

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

print("Loading CLIP model...")
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
print("CLIP model loaded.")

pc = Pinecone(api_key=PINECONE_API_KEY)

if INDEX_NAME not in pc.list_indexes().names():
    print(f"Creating index '{INDEX_NAME}'...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric="cosine",
        spec={"serverless": {"cloud": "aws", "region": "us-east-1"}}
    )
else:
    print(f"Using existing index: {INDEX_NAME}")

index = pc.Index(INDEX_NAME)

# ────────────────────────────────────────────────────────────────
# Clear index (safe version)
# ────────────────────────────────────────────────────────────────

try:
    print("Attempting to clear index...")
    index.delete(delete_all=True)
    print("Index cleared successfully.")
except Exception as e:
    print(f"Delete failed (likely namespace empty): {e}")
    print("Continuing with upsert anyway.")

# ────────────────────────────────────────────────────────────────
# Embedding functions
# ────────────────────────────────────────────────────────────────

def embed_text(text: str):
    text_token = clip.tokenize([text]).to(device)
    with torch.no_grad():
        features = clip_model.encode_text(text_token)
    return features.cpu().numpy().flatten()

def embed_image(image_path: str):
    image = clip_preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        features = clip_model.encode_image(image)
    return features.cpu().numpy().flatten()

# ────────────────────────────────────────────────────────────────
# Collect vectors
# ────────────────────────────────────────────────────────────────

vectors = []

# ── Text ─────────────────────────────────────────────────────────
if os.path.exists(TEXT_DIR):
    print(f"Processing text files in: {TEXT_DIR}")
    for filename in os.listdir(TEXT_DIR):
        if filename.lower().endswith(".txt"):
            path = os.path.join(TEXT_DIR, filename)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                vector = embed_text(text)
                vec_id = f"text_{os.path.splitext(filename)[0]}"
                vectors.append({
                    "id": vec_id,
                    "values": vector.tolist(),
                    "metadata": {
                        "type": "text",
                        "source": filename,
                        "content": text
                    }
                })
                print(f"Embedded text: {vec_id} ({len(text)} chars)")
            except Exception as e:
                print(f"Error on text {filename}: {e}")
else:
    print(f"Text directory not found: {TEXT_DIR}")

# ── Images ───────────────────────────────────────────────────────
if os.path.exists(IMAGE_DIR):
    print(f"Processing images in: {IMAGE_DIR}")
    for filename in os.listdir(IMAGE_DIR):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(IMAGE_DIR, filename)
            try:
                vector = embed_image(path)
                vec_id = f"img_{os.path.splitext(filename)[0]}"
                vectors.append({
                    "id": vec_id,
                    "values": vector.tolist(),
                    "metadata": {
                        "type": "image",
                        "source": filename
                    }
                })
                print(f"Embedded image: {vec_id}")
            except Exception as e:
                print(f"Error on image {filename}: {e}")
else:
    print(f"Image directory not found: {IMAGE_DIR}")

# ────────────────────────────────────────────────────────────────
# Upsert in batches
# ────────────────────────────────────────────────────────────────

if vectors:
    print(f"\nUpserting {len(vectors)} vectors in batches of 500...")
    batch_size = 500
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        try:
            index.upsert(vectors=batch)
            print(f"Batch {i//batch_size + 1} successful ({len(batch)} vectors)")
        except Exception as e:
            print(f"Batch {i//batch_size + 1} failed: {e}")
            break
    print("Upsert finished.")
else:
    print("No vectors collected. Check data folders.")

print("\nScript finished.")