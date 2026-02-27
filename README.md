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

