# 🤖 AI-Powered Freelancer Recommendation System

This project is built as part of the **PeerHire AI/ML Internship Assignment**. It is a full-stack application that leverages **Natural Language Processing (NLP)** and **Machine Learning** to recommend the top 5 most relevant freelancers for a given job post, filtered by **budget** and **timeline**.

---

## 🔍 Problem Statement

Given a job description and client preferences (budget and timeline), recommend a ranked list of the top 5 freelancers best suited for the job based on:

- Skills and past projects
- Availability and expected hourly rate
- Semantic similarity with the job requirements

---

## 💡 Model Selection & Training Process

### 🔧 Data Used

- Dataset: `freelancer_profiles_updated.json` (100 synthetic freelancer profiles)
- Each freelancer has:
  - `skills`
  - `past_projects`
  - `experience_years`
  - `expected_rate`
  - `availability_in_days`
  - `rating`

### 🧠 Approach

1. **Text Preprocessing**:  
   Combined each freelancer's `skills` and `past_projects` into a single string for feature extraction.

2. **Vectorization with TF-IDF**:
   - Used `TfidfVectorizer` from **scikit-learn** to convert freelancer profiles and job descriptions into numeric vectors.
   - Trained only once and saved using `joblib` to avoid re-training on each API call.

3. **Filtering**:
   - Before scoring, freelancers are filtered based on:
     - `expected_rate <= job_budget`
     - `availability_in_days <= job_timeline`

4. **Similarity Calculation**:
   - Used **Cosine Similarity** to measure semantic similarity between the job description and each freelancer's profile.
   - Top 5 freelancers with the highest similarity score are returned.

---

## 🧪 API Functionality & How to Test It

### 🚀 Endpoint 1: Job Form (UI)

- **URL**: `/`
- **Method**: `GET`
- **Description**: Renders a form for submitting job description, budget, and deadline.

### 🚀 Endpoint 2: Get Recommendations

- **URL**: `/recommend`
- **Method**: `POST`
- **Description**: Returns the top 5 freelancer recommendations.

#### 📝 Sample Form Submission

- **Job Description**:  
  `Looking for a developer to build an AI chatbot using NLP and deploy it using Docker.`  
- **Budget**: `80`
- **Timeline**: `15`

#### ✅ Expected Output:

- Webpage displaying cards with the **top 5 freelancers**, each showing:
  - Name
  - Skills
  - Past Projects
  - Expected Rate
  - Availability
  - Rating
  - Similarity Score

---
## 🚀 Deployment (Render)

The project is deployed on **Render**.

### 🔗 Live Demo:
👉 [https://freelance-recommendor.onrender.com](https://freelance-recommendor.onrender.com)

> ⚠️ **Note:** Render may take 1-10 minutes to wake up if idle. Also if it says 502 Bad Gateway then try reloading the link or close the link and open again to remove the error.

## 🛠️ How to Run the Project Locally

 1. Clone the Repository

```bash
git clone https://github.com/your-username/freelancer-recommender.git
cd freelancer-recommender
```
2. Create Virtual Environment (Optional)
```bash
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
```

3. Install Requirements
```bash
pip install -r requirements.txt
```
4. Start the Server
```bash
uvicorn main:app --reload
```
5. Visit the App
#Open your browser and go to:
```bash
👉 http://127.0.0.1:8000/
```
📁 project/
```bash
├── main.py                         # FastAPI app logic
├── freelancer_profiles_updated.json  # Freelancer dataset (sample below)
├── requirements.txt               # Python dependencies
├── vectorizer.joblib              # Saved TF-IDF vectorizer
├── freelancer_vectors.joblib      # Precomputed freelancer vectors
├── templates/
  │   ├── form.html                  # Web form UI
  │   └── results.html               # Results page
├── static/
  |   └── style.css                  # CSS styles
```
Sample Freelancer Entry
```bash
{
  "name": "Bryan Gregory",
  "skills": ["Numpy", "Tensorflow", "Pandas"],
  "past_projects": ["image recognition", "chatbot"],
  "experience_years": 4,
  "expected_rate": "87 $",
  "availability_in_days": 25,
  "rating": 1.7
}
```
## 🛠️ Technologies Used

- **FastAPI** – Web framework for building APIs quickly
- **Uvicorn** – ASGI server for serving FastAPI apps
- **Scikit-learn** – For TF-IDF vectorization and cosine similarity
- **NumPy** – Numerical computing and array handling
- **Jinja2** – Templating engine for rendering HTML pages
- **Python-Multipart** – For handling form data
- **Joblib** – For saving/loading ML models and vectorizers
- **HTML & CSS** – Frontend design for UI (form and results page)

✅ Requirements
See requirements.txt:
```bash
nginx
Copy
Edit
fastapi
uvicorn
scikit-learn
numpy
jinja2
python-multipart
joblib
```
Install with:
```bash
pip install -r requirements.txt
```

🙋‍♂️ Author - 
MD Misbah Ur Rahman

Feel free to fork the repo or raise issues for suggestions/bugs.

  
