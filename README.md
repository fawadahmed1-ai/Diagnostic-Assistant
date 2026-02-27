MediGraph Diagnostic Assistant Prototype: Technical Document
Overview
This document provides a comprehensive, step-by-step summary of the development process for the MediGraph Diagnostic Assistant prototype. MediGraph is a basic Retrieval-Augmented Generation (RAG) application focused on healthcare, specifically assisting with rare condition diagnoses using multimodal data (text reports and medical images like chest X-rays). It leverages Retrieval-Augmented Generation with a graph-based approach, but in this prototype, we emphasized multimodal search using vector embeddings.
The prototype allows users to query symptoms or findings (e.g., "pneumonia") and retrieves similar text reports and X-ray images from a small dataset. It is built using open-source tools and is intended for educational and experimental purposes only—not for real medical use. All development was done on a Windows machine, addressing common setup challenges.
Key technologies:
•	Python 3.12
•	Libraries: torch, clip (OpenAI CLIP for embeddings), sentence-transformers (initially), pinecone (vector database), streamlit (web UI)
•	Data: NIH Chest X-ray sample dataset (36 images) + 3 custom text reports
•	Cloud: Pinecone for vector storage and search
The project folder is C:\Users\FawadAhmed\MediGraph.
Step 1: Environment Setup
•	Installed Python 3.12 from python.org, ensuring "Add Python to PATH" was checked.
•	Created a project folder: C:\Users\FawadAhmed\MediGraph.
•	Set up a virtual environment:
text
python -m venv env
env\Scripts\activate
•	Installed required libraries one by one (CPU-only version for simplicity, no GPU needed):
text
pip install langchain llama-index torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers sentence-transformers neo4j pinecone streamlit pillow
pip install git+https://github.com/openai/CLIP.git
•	Tested installation with a simple script quick_test.py to verify imports and torch version.
Step 2: Data Gathering and Preparation
•	Downloaded a small subset of the NIH Chest X-ray dataset from Kaggle (sample.zip ~5,606 images, but used only 36 for the prototype).
•	Unzipped to data/images.
•	Created data/processed_images for resized versions.
•	Wrote and ran prep_images.py to resize images to 224x224 for CLIP compatibility:
text
import os
from PIL import Image

input_dir = "data/images"
output_dir = "data/processed_images"

os.makedirs(output_dir, exist_ok=True)
for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png', '.jpg')):
        img = Image.open(os.path.join(input_dir, filename)).convert("RGB")
        img.resize((224, 224)).save(os.path.join(output_dir, filename))
•	Created data/text folder with 3 sample report files:
o	report_normal.txt: "Chest X-ray shows clear lungs... Impression: Normal chest radiograph."
o	report_pneumonia.txt: "Bilateral patchy infiltrates... Impression: Community-acquired pneumonia."
o	report_cardiomegaly.txt: "Enlarged cardiac silhouette... Impression: Cardiomegaly."
Step 3: Embedding and Uploading to Pinecone
•	Signed up for Pinecone free tier at pinecone.io and created an API key.
•	Installed/updated Pinecone client: pip install pinecone.
•	Wrote embed_and_store.py to embed data using CLIP (for both text and images to ensure 512 dimensions):
o	Loaded CLIP model.
o	Created Pinecone index with dimension 512, cosine metric.
o	Embedded and upserted text and images.
•	Ran python embed_and_store.py — uploaded ~39 vectors (3 text + 36 images).
Step 4: Building the Streamlit Web App
•	Installed Streamlit: pip install streamlit.
•	Created app.py for the UI:
o	Loads CLIP for query embedding.
o	Connects to Pinecone index.
o	User inputs query → embeds it → queries Pinecone for top 5 matches.
o	Displays text previews and images with similarity scores.
•	Ran the app: streamlit run app.py — opens at http://localhost:8501.
•	Tested queries like "pneumonia" — shows ranked reports and X-rays.
Debugging and Challenges Addressed
•	Windows-specific issues: Git installation for CLIP, PATH setup for Python, activating env.
•	Vector dimension mismatch (384 vs 512) — fixed by using CLIP for text embeddings.
•	Folder paths — moved data and scripts from env subfolder to root.
•	API key errors — regenerated and verified key in dashboard.
•	File not found — ensured data/text and data/processed_images exist in root.
Limitations and Future Enhancements
•	Small dataset — add more images/reports for better accuracy.
•	No graph (Neo4j) yet — future step for entity linking (symptoms → diseases).
•	Basic UI — add image upload, LLM summaries (e.g., via Groq).
•	No production readiness — add authentication, error handling, compliance (e.g., HIPAA for real health data).
•	Performance — use GPU for faster embeddings/queries.
How to Run the Prototype
•	Activate env: env\Scripts\activate
•	Run app: streamlit run app.py
•	Open http://localhost:8501
•	Type query → see results.
This prototype demonstrates core RAG concepts in healthcare. For questions or expansions, refer to the code files.
Last updated: February 26, 2026

# MediGraph Diagnostic Assistant Prototype – Update (February 28, 2026)

## Overview
This document summarizes the progress made on February 28, 2026, building on the initial prototype completed on February 27, 2026.

Today's focus:
- Made image upload fully functional (triggers search with CLIP embedding)
- Added hybrid search (text + image = averaged embeddings)
- Added loading spinner, similarity threshold slider, and improved 3-column grid layout
- Fixed full report text display (via re-upsert with "content" metadata)
- Integrated LLM explanation for top match using Groq API (Llama 3.1 8B Instant)
- Prepared for and deployed the app to Streamlit Community Cloud (public URL)
- Resolved GitHub push issues (secret scanning, .gitignore, secrets exclusion)

All core features from the planned combo are now implemented and tested locally.

## Today's Achievements (Step-by-Step)

1. **Image Upload & Search**
   - Added file_uploader for PNG/JPG/JPEG
   - Preview of uploaded image
   - Automatic CLIP embedding of uploaded image
   - Triggers Pinecone search (same as text query)

2. **Hybrid Text + Image Search**
   - When both text and image are provided → embeddings are averaged
   - Simple but effective multimodal fusion
   - Search source message in spinner ("text + uploaded image", etc.)

3. **UX & Layout Improvements**
   - Loading spinner during embedding & search
   - Similarity threshold slider (0.0–1.0) with live filtering
   - 3-column responsive grid for results
   - Full report text displayed (up to 800 chars, truncated if longer)
   - Captions, info boxes, and markdown separators for clarity

4. **Full Report Text Fix**
   - Re-ran embed_and_store.py with updated metadata:
     "content": text
   - App now shows actual report content instead of "No full content stored."

5. **LLM Explanation (Groq API)**
   - Added generate_explanation function using Groq's Llama 3.1 8B Instant
   - Summarizes top match in 2–3 neutral sentences
   - Shown in expandable section ("AI Summary of Top Match")
   - Handles missing key gracefully

6. **Deployment Preparation & Execution**
   - Created requirements.txt (all dependencies listed)
   - Added .gitignore (excluded .streamlit/secrets.toml)
   - Pushed safe files to public GitHub repo
   - Deployed to Streamlit Community Cloud
   - Added secrets (PINECONE_API_KEY & GROQ_API_KEY) in Cloud settings

## Current Status (as of February 28, 2026 – 02:33 AM PKT)

- App fully functional locally:
  - Text search
  - Image upload + search
  - Hybrid mode
  - Threshold filtering
  - Full report text
  - LLM summary of top match

- Deployed publicly on Streamlit Cloud (URL pending confirmation after secrets added)
- No secrets exposed in GitHub repo

## Remaining / Future Work (Suggestions)

- Scale data: Add 200–500 more chest X-rays + labeled reports
- Refine LLM prompt for more specific/radiology-focused summaries
- UI polish: zoomable images, clear/reset button, dark mode tweaks
- Add confidence visualization (e.g. bar for similarity)
- Explore advanced multimodal models (e.g. CLIP fine-tuning or newer ViT)
- Add basic patient context input (age, gender) for better filtering
- Neo4j graph integration for symptom-disease linking (long-term)

## How to Run Locally

```bash
cd C:\Users\FawadAhmed\MediGraph
env\Scripts\activate
streamlit run app.py
