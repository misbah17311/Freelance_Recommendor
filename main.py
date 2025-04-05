# app.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from typing import List
import json
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
# Load freelancer data
with open("freelancer_profiles_updated.json") as f:
    freelancers = json.load(f)


# Combine text fields for TF-IDF matching
def profile_to_text(profile):
    return " ".join(profile["skills"] + profile["past_projects"]*2)


freelancer_texts = [profile_to_text(f) for f in freelancers]

VEC_PATH = "vectorizer.joblib"
VEC_MATRIX_PATH = "freelancer_vectors.joblib"

if os.path.exists(VEC_PATH) and os.path.exists(VEC_MATRIX_PATH):
    vectorizer = joblib.load(VEC_PATH)
    freelancer_vectors = joblib.load(VEC_MATRIX_PATH)
else:
    vectorizer = TfidfVectorizer(
        stop_words="english", max_df=0.85, min_df=2, max_features=5000
    )
    freelancer_vectors = vectorizer.fit_transform(freelancer_texts)
    joblib.dump(vectorizer, VEC_PATH)
    joblib.dump(freelancer_vectors, VEC_MATRIX_PATH)
# FastAPI setup
app = FastAPI()
@app.get("/")
def root():
    return RedirectResponse(url="/docs")

# Request body model
class JobPost(BaseModel):
    description: str
    budget_in_dollars: int
    timeline_days: int

# API endpoint
@app.post("/recommend")
def recommend_freelancers(job: JobPost):
    job_vector = vectorizer.transform([job.description])
    similarities = cosine_similarity(job_vector, freelancer_vectors).flatten()

    # Filter by budget and timeline
    filtered = []
    for i, freelancer in enumerate(freelancers):
        if freelancer.get("expected_rate_hourly", 0) <= job.budget_in_dollars and freelancer.get("availability_in_days",
                                                                               0) <= job.timeline_days:
            filtered.append((i, similarities[i]))

    if not filtered:
        raise HTTPException(status_code=404, detail="No freelancers match budget/timeline criteria.")

    # Sort filtered freelancers by similarity score
    filtered.sort(key=lambda x: x[1], reverse=True)
    top_indices = [idx for idx, _ in filtered[:5]]

    top_freelancers = []
    for i in top_indices:
        freelancer = freelancers[i].copy()
        freelancer["similarity_score"] = round(float(similarities[i]), 3)
        top_freelancers.append(freelancer)

    return {"recommendations": top_freelancers}
