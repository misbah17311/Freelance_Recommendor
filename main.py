# app.py
from fastapi import FastAPI, HTTPException
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
    return " ".join(profile["skills"] + profile["past_projects"])


freelancer_texts = [profile_to_text(f) for f in freelancers]

# Vectorize freelancer data
vectorizer = TfidfVectorizer()
freelancer_vectors = vectorizer.fit_transform(freelancer_texts)

# FastAPI setup
app = FastAPI()


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

    return {"recommendations": top_freelancers}
