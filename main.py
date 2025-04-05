# app.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from typing import List
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load freelancer data
with open("freelancer_profiles_updated.json") as f:
    freelancers = json.load(f)


# Combine text fields for TF-IDF matching
def profile_to_text(profile):
    return " ".join(profile["skills"] + profile["past_projects"]*2)
    


freelancer_texts = [profile_to_text(profile) for profile in freelancers]

# ------------------------------------------------------
# 🧠 Improved TF-IDF Vectorizer for Matching & Ranking
# ------------------------------------------------------
vectorizer = TfidfVectorizer(
    stop_words='english',        # ✅ Remove common/general words
    ngram_range=(1, 2),          # ✅ Capture keywords & bigrams like "web development"
    lowercase=True               # (default) Normalize casing
)

freelancer_vectors = vectorizer.fit_transform(freelancer_texts)
# FastAPI setup
app = FastAPI()
# ✅ New homepage route
@app.get("/")
def root():
    return RedirectResponse(url="/docs")
    


# Request body model
class JobPost(BaseModel):
    description: str


# API endpoint
@app.post("/recommend")
def recommend_freelancers(job: JobPost):
    job_vector = vectorizer.transform([job.description])
    similarities = cosine_similarity(job_vector, freelancer_vectors).flatten()

    top_indices = np.argsort(similarities)[-5:][::-1]
    top_freelancers = [freelancers[i] for i in top_indices]

     # ✅ Prepare response with similarity score
    top_freelancers = []
    for i in top_indices:
        freelancer = freelancers[i].copy()
        freelancer["similarity_score"] = round(float(similarities[i]), 3)
        top_freelancers.append(freelancer)

    return {"recommendations": top_freelancers}
