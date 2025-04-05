# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
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
    return " ".join(profile["skills"] + profile["past_projects"] * 2)

freelancer_texts = [profile_to_text(f) for f in freelancers]

# File paths for saving vectorizer and vectors
VEC_PATH = "vectorizer.joblib"
VEC_MATRIX_PATH = "freelancer_vectors.joblib"

# Load or create vectorizer and matrix
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
app = FastAPI(
    title="Freelancer Recommendation API",
    description="An AI-powered system that suggests top freelancers based on job details (skills, budget, and timeline).",
    version="1.0.0"
)

# Request model
class JobPost(BaseModel):
    description: str
    budget: int
    timeline_days: int

# Main recommendation endpoint
@app.post("/recommend", response_model=None, response_description="Top 5 freelancer matches in plain text")
def recommend_freelancers(job: JobPost):
    job_vector = vectorizer.transform([job.description])
    similarities = cosine_similarity(job_vector, freelancer_vectors).flatten()

    # Filter by budget and timeline
    filtered = []
    for i, freelancer in enumerate(freelancers):
        if freelancer.get("expected_rate", 0) <= job.budget and freelancer.get("availability_in_days", 0) <= job.timeline_days:
            filtered.append((i, similarities[i]))

    if not filtered:
        raise HTTPException(status_code=404, detail="No freelancers match budget/timeline criteria.")

    # Sort by similarity
    filtered.sort(key=lambda x: x[1], reverse=True)
    top_indices = [idx for idx, _ in filtered[:5]]

    # Format output
    result = "🎯 Top Freelancer Matches:\n\n"
    for i, idx in enumerate(top_indices, 1):
        freelancer = freelancers[idx]
        similarity = round(float(similarities[idx]), 3)
        result += (
            f"{i}. {freelancer['name']}\n"
            f"   Skills: {', '.join(freelancer['skills'])}\n"
            f"   Past Projects: {', '.join(freelancer['past_projects'][:2])}...\n"
            f"   Rate: {freelancer['expected_rate']}\n"
            f"   Available in: {freelancer['availability_in_days']} days\n"
            f"   Similarity Score: {similarity}\n\n"
        )

    return result
