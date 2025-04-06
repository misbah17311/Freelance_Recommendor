from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
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
for freelancer in freelancers:
    #expected_rate_hourly
    if "expected_rate_hourly" not in freelancer:
        rate_raw = freelancer.get("expected_rate", "0")
        if isinstance(rate_raw, str):
            rate_numeric = ''.join(c for c in rate_raw if c.isdigit() or c == '.')
            freelancer["expected_rate_hourly"] = float(rate_numeric) if rate_numeric else 0
        else:
            freelancer["expected_rate_hourly"] = rate_raw

    #availability
    if "availability_till_next" not in freelancer and "availability_in_days" in freelancer:
        freelancer["availability_till_next"] = freelancer["availability_in_days"]

# Combine text fields for TF-IDF matching
def profile_to_text(profile):
    return " ".join(profile["skills"] + profile["past_projects"] * 2)

freelancer_texts = [profile_to_text(f) for f in freelancers]

# Load or create vectorizer
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

#FastAPI setup
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
def form_ui(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})


@app.post("/recommend", response_class=HTMLResponse)
def recommend_ui(request: Request,
                 description: str = Form(...),
                 budget_in_dollars: int = Form(...),
                 timeline_days: int = Form(...)):

    job_vector = vectorizer.transform([description])
    similarities = cosine_similarity(job_vector, freelancer_vectors).flatten()

    filtered = []
    for i, freelancer in enumerate(freelancers):
        if freelancer.get("expected_rate_hourly", 0) <= budget_in_dollars and freelancer.get("availability_in_days", 0) <= timeline_days:
            filtered.append((i, similarities[i]))

    #Sort by similarity
    filtered.sort(key=lambda x: x[1], reverse=True)

    # New logic: check if all top 5 scores are 0
    if not filtered or all(score == 0 for _, score in filtered[:5]):
        return templates.TemplateResponse("results.html", {
            "request": request,
            "freelancers": [],
            "message": "No matching freelancers found for this job at the moment."
        })

    #top 5 freelancers
    top_indices = [idx for idx, _ in filtered[:5]]
    top_freelancers = []
    for i in top_indices:
        freelancer = freelancers[i].copy()
        freelancer["similarity_score"] = round(float(similarities[i]), 3)
        top_freelancers.append(freelancer)

    return templates.TemplateResponse("results.html", {
        "request": request,
        "freelancers": top_freelancers,
        "message": None
    })
