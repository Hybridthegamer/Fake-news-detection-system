# Fake News Detection System

An NLP-powered web application that classifies news articles and headlines as **Real** or **Fake** using a fine-tuned DistilBERT transformer model with TF-IDF/Logistic Regression as a fallback baseline. Each prediction includes a confidence score and a LIME-generated feature explanation.

This system was designed and implemented as part of a Final Year Project in Computer Science, following the system analysis and design specifications documented in Chapters 1–3.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Technology Stack and Justification](#technology-stack-and-justification)
4. [System Requirements](#system-requirements)
5. [Setup and Installation (Development)](#setup-and-installation-development)
6. [Dataset Acquisition](#dataset-acquisition)
7. [Training the Models](#training-the-models)
8. [Running the Application](#running-the-application)
9. [API Reference](#api-reference)
10. [Running Tests](#running-tests)
11. [Docker Deployment (Production)](#docker-deployment-production)
12. [Project Structure](#project-structure)

---

## System Overview

The system accepts English-language news text (50–5,000 words) or headlines (5–50 words) and returns:

- **Classification label** — `Real News` or `Fake News`
- **Confidence score** — softmax probability expressed as a percentage
- **Feature explanation** — top three linguistically influential terms identified via LIME
- **Low-confidence flag** — caution indicator when confidence falls below 70%

The backend exposes a REST API consumed by a React.js single-page application. An authenticated admin dashboard provides access to classification history and system statistics.

---

## Architecture

The system adopts a **three-tier client-server architecture** as specified in the design (Section 3.5):

```
┌─────────────────────────────────────────────────────┐
│           Presentation Tier (React.js)              │
│   ArticleInput → ResultPanel → AdminDashboard       │
└──────────────────────┬──────────────────────────────┘
                       │ HTTP/REST (Axios)
┌──────────────────────▼──────────────────────────────┐
│         Application Tier (Flask REST API)           │
│                                                     │
│  POST /api/classify                                 │
│    └─ TextPreprocessor (Algorithm 3.1)              │
│    └─ FakeNewsClassifier                            │
│         ├─ DistilBERT (primary, Algorithm 3.2)      │
│         └─ TF-IDF + LR (baseline, Algorithm 3.3)   │
│    └─ LIME Explainer                                │
│    └─ PredictionLogger                              │
│                                                     │
│  GET  /api/history  (admin, JWT-protected)          │
│  GET  /api/stats    (admin, JWT-protected)          │
│  POST /api/auth/login                               │
└──────────────────────┬──────────────────────────────┘
                       │ SQLAlchemy ORM
┌──────────────────────▼──────────────────────────────┐
│         Data Tier (SQLite / PostgreSQL)             │
│   users │ submissions │ predictions │ audit_log     │
└─────────────────────────────────────────────────────┘
```

---

## Technology Stack and Justification

### Backend — Python + Flask

**Python 3.11** was chosen as the primary implementation language because the entire machine learning and NLP ecosystem (PyTorch, Hugging Face Transformers, scikit-learn, NLTK, spaCy) is Python-native. Attempting to use any other language would require bridging to Python anyway, adding unnecessary complexity.

**Flask 3.0** was selected over Django for the following reasons:
- **Lightweight and unopinionated** — Flask imposes no rigid project structure, making it ideal for research-oriented applications where the ML pipeline is the core component, not the web framework itself.
- **Minimal overhead** — Flask's thin abstraction over WSGI means lower latency per request, which is critical for meeting the ≤5-second response time NFR (NFR-01).
- **REST-first design** — Flask's blueprint system maps cleanly to the three API endpoints specified in the functional requirements, and `flask-jwt-extended` provides robust token-based authentication with minimal boilerplate.
- **Ecosystem fit** — Flask integrates naturally with SQLAlchemy, enabling the ORM layer specified in Section 3.7 without the opinionated ORM lock-in of Django's ORM.

### NLP Pipeline — NLTK + spaCy

**NLTK 3.8** provides the English stop word corpus used in Algorithm 3.1 (step 6). Its stability, comprehensive documentation, and widespread academic adoption made it the natural choice for this component.

**spaCy 3.7** (`en_core_web_sm`) handles tokenisation and lemmatisation (Algorithm 3.1, steps 5 and 7). spaCy was preferred over NLTK's own tokeniser and lemmatiser because:
- Its tokeniser is significantly faster due to Cython-compiled internals, directly supporting NFR-01 (response time ≤5 seconds).
- Its lemmatiser uses lookup tables combined with rule-based analysis, producing more linguistically accurate lemmas than NLTK's Porter Stemmer, which performs stemming rather than true lemmatisation and can distort word meaning.

### Machine Learning — scikit-learn

**scikit-learn 1.3** provides the `TfidfVectorizer` and `LogisticRegression` implementations for Algorithm 3.3 (the TF-IDF + LR baseline). scikit-learn is the de facto standard for classical ML in Python; its consistent API, excellent documentation, and battle-tested implementations make it appropriate for the baseline model. The `TfidfVectorizer` configuration (`max_features=50,000`, `ngram_range=(1,2)`, `sublinear_tf=True`) follows best practices for text classification at this corpus scale.

### Primary Classifier — DistilBERT (Hugging Face Transformers)

**DistilBERT** (`distilbert-base-uncased`) was selected as the primary classifier for the following reasons documented in the literature review (Section 2.4.3):
- It retains **97% of BERT's performance** while being **40% smaller and 60% faster** (Sanh et al., 2022), making it deployable in web application contexts without a dedicated GPU at inference time.
- Transformer-based fine-tuning approaches consistently achieve accuracy above **90%** on binary fake news datasets, outperforming classical and LSTM-based methods (Section 2.4.3).
- The **Hugging Face Transformers library** (v4.36) provides a standardised fine-tuning API that directly maps to Algorithm 3.2, including the `DistilBertForSequenceClassification` head, `AdamW` optimiser, and linear warmup scheduler.
- **PyTorch** was chosen over TensorFlow as the deep learning backend because Hugging Face's primary training utilities (`Trainer`, `AdamW`) are PyTorch-native, and PyTorch's dynamic computation graph provides more flexible debugging during the experimental training phase.

### Explainability — LIME

**LIME** (Local Interpretable Model-Agnostic Explanations) was integrated to address the black-box limitation identified in existing systems (Section 3.3.3). LIME perturbs the input text by masking random subsets of words and observes the resulting classification changes, producing a local linear approximation of the model's decision boundary that identifies the top-N most influential terms. This satisfies FR-05 and directly improves user trust by making the classification decision transparent and auditable by non-technical users such as journalists.

### Database — SQLite / PostgreSQL + SQLAlchemy

**SQLite** is used as the development database due to its zero-configuration setup and file-based storage. **PostgreSQL 15** is specified for production (Section 3.7.1) due to its ACID compliance, support for JSON/JSONB data types (used to store LIME explanation objects), and enterprise-grade reliability.

**SQLAlchemy 2.0** provides the ORM layer, abstracting database operations from business logic and enabling seamless switching between SQLite and PostgreSQL via the `DATABASE_URL` environment variable — directly satisfying NFR-06 (portability).

### Frontend — React.js + Vite

**React.js 18** was specified in the system design (Section 3.5.2) for the following reasons:
- Its **component-based architecture** maps naturally to the UI elements specified in the output design (Section 3.6.2): `ArticleInput`, `ResultPanel`, `AdminDashboard`.
- **Unidirectional data flow** makes the state transitions (input → loading → result) predictable and testable.
- The ecosystem provides `Axios` for async API communication and `react-router-dom` for client-side routing.

**Vite** replaces Create React App as the build tool because it offers significantly faster development server startup and hot module replacement via native ES modules, reducing development cycle friction.

---

## System Requirements

### Minimum (Development / CPU-only inference)

| Component | Requirement |
|-----------|------------|
| OS | Linux, macOS, or Windows (WSL2 recommended) |
| Python | 3.10 or 3.11 |
| Node.js | 18 LTS or 20 LTS |
| RAM | 4 GB (8 GB recommended for DistilBERT loading) |
| Disk | 4 GB free (models + dependencies) |
| CPU | Any modern x86-64 (DistilBERT inference is CPU-viable) |

### Recommended (DistilBERT fine-tuning)

| Component | Requirement |
|-----------|------------|
| GPU | CUDA-compatible (NVIDIA GTX 1070 / RTX equivalent or better) |
| VRAM | 6 GB minimum for batch_size=16 |
| RAM | 16 GB |
| Disk | 10 GB free |

> **Note:** Fine-tuning DistilBERT on the full WELFake dataset (~72,000 articles) takes approximately:
> - **With GPU:** 2–4 hours (4 epochs)
> - **CPU only:** 12–24+ hours
>
> The TF-IDF + Logistic Regression baseline trains in **under 5 minutes** on any modern machine and is the recommended starting point if GPU resources are unavailable.

---

## Setup and Installation (Development)

### 1. Clone the repository

```bash
git clone https://github.com/hybridthegamer/fake-news-detection-system.git
cd fake-news-detection-system
```

### 2. Backend setup

```bash
cd backend

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate          # Linux / macOS
# venv\Scripts\activate           # Windows

# Install Python dependencies
pip install -r requirements.txt

# Download spaCy language model
python -m spacy download en_core_web_sm

# Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

# Copy environment config
cp .env.example .env
# Edit .env — set SECRET_KEY and JWT_SECRET_KEY to random values
```

Generate secure keys with:
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

### 3. Frontend setup

```bash
cd ../frontend
npm install
```

---

## Dataset Acquisition

The system is trained on the **WELFake dataset** (Verma et al., 2021), a unified corpus of 72,134 news articles (35,028 real, 37,106 fake) aggregated from four sources.

**Download from Kaggle:**

1. Create a Kaggle account at [kaggle.com](https://www.kaggle.com)
2. Navigate to: [https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification)
3. Download `WELFake_Dataset.csv`
4. Place the file at:

```
backend/data/WELFake_Dataset.csv
```

**Expected CSV columns:** `Unnamed: 0`, `title`, `text`, `label`  
**Label encoding:** `0` = Real News, `1` = Fake News

---

## Training the Models

Ensure the dataset is in place before running the training script.

```bash
cd backend
source venv/bin/activate   # if not already activated
```

### Option A — Train TF-IDF + Logistic Regression only (fast, ~5 min)

```bash
python scripts/train_model.py --dataset data/WELFake_Dataset.csv --model lr
```

### Option B — Fine-tune DistilBERT only (requires GPU for practical time)

```bash
python scripts/train_model.py --dataset data/WELFake_Dataset.csv --model distilbert --epochs 4 --batch-size 16
```

### Option C — Train both models (recommended)

```bash
python scripts/train_model.py --dataset data/WELFake_Dataset.csv --model all
```

### Evaluate trained models

```bash
python scripts/evaluate_model.py --dataset data/WELFake_Dataset.csv
```

**Expected performance on WELFake test set (80/20 split):**

| Model | Accuracy | F1 (weighted) |
|-------|----------|--------------|
| TF-IDF + Logistic Regression | ~93–95% | ~0.93–0.95 |
| DistilBERT (fine-tuned, 4 epochs) | ~96–98% | ~0.96–0.98 |

These figures are consistent with results reported in the literature (Sections 2.4.1–2.4.3).

---

## Running the Application

### Backend (Flask development server)

```bash
cd backend
source venv/bin/activate

# Create admin user (first time only)
python scripts/seed_admin.py

# Start Flask
python app.py
```

The API will be available at `http://localhost:5000`.

### Frontend (Vite dev server)

In a separate terminal:

```bash
cd frontend
npm run dev
```

The frontend will be available at `http://localhost:5173`.  
API requests are proxied to `localhost:5000` via the Vite dev proxy configuration.

### Admin dashboard

1. Click **Admin Login** in the top navigation
2. Default credentials: `admin` / `admin123`
3. Change the password immediately for any non-development use

---

## API Reference

All API endpoints are prefixed with `/api`.

### `POST /api/classify`

Classify a news article or headline.

**Request body:**
```json
{
  "text": "Your news article or headline text here..."
}
```

**Response (200 OK):**
```json
{
  "label": "Fake",
  "confidence": 94.7,
  "model_used": "DistilBERT",
  "explanation": {
    "top_features": [
      {"feature": "miraculous", "weight": 0.312},
      {"feature": "government", "weight": 0.198},
      {"feature": "suppressed", "weight": 0.145}
    ],
    "summary": "This article was classified as Fake News. Key influencing terms: \"miraculous\"; \"government\"; \"suppressed\"."
  },
  "low_confidence": false,
  "word_count": 87,
  "input_type": "article",
  "processing_time_ms": 312.4
}
```

**Error responses:** `400` (validation), `503` (model not trained)

---

### `POST /api/auth/login`

Authenticate an admin user.

**Request body:**
```json
{"username": "admin", "password": "admin123"}
```

**Response (200 OK):**
```json
{"access_token": "<JWT>", "role": "admin", "username": "admin"}
```

---

### `GET /api/history` *(Admin — Bearer token required)*

Retrieve paginated classification history.

**Query params:** `page` (default: 1), `per_page` (default: 20, max: 100)

---

### `GET /api/stats` *(Admin — Bearer token required)*

Retrieve system statistics (total submissions, fake/real counts, average confidence, model usage).

---

## Running Tests

```bash
cd backend
source venv/bin/activate
pip install pytest

pytest ../tests/ -v
```

The test suite covers:
- `test_preprocessor.py` — Algorithm 3.1 unit tests (lowercase, HTML removal, URL removal, stop word removal, lemmatisation, token length filtering)
- `test_api.py` — Integration tests for all three API blueprints (input validation, authentication, authorisation, XSS sanitisation)

---

## Docker Deployment (Production)

### Prerequisites

- Docker Engine 24+ and Docker Compose v2
- Trained models in `backend/saved_models/`

### 1. Configure environment

```bash
cp backend/.env.example .env
# Edit .env — set SECRET_KEY and JWT_SECRET_KEY
```

### 2. Build and start

```bash
docker compose up --build -d
```

Services:
- **Frontend** → `http://localhost:80`
- **Backend API** → `http://localhost:5000`

### 3. Seed admin user in container

```bash
docker exec -it fakenews_backend python scripts/seed_admin.py
```

### 4. Stop services

```bash
docker compose down
```

### Notes on production deployment

- Replace SQLite with PostgreSQL by setting `DATABASE_URL=postgresql://user:pass@host:5432/fakenews` in your `.env`.
- Serve behind an HTTPS-terminating reverse proxy (e.g., Nginx + Let's Encrypt) to satisfy NFR-04.
- The `saved_models/` directory is volume-mounted so trained models persist across container restarts.

---

## Project Structure

```
fake-news-detection-system/
├── docker-compose.yml
├── .gitignore
├── README.md
│
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── .env.example
│   ├── app.py                    # Flask application factory
│   ├── config.py                 # Dev / Test / Production config classes
│   ├── extensions.py             # SQLAlchemy, JWT, Bcrypt instances
│   │
│   ├── models/
│   │   └── db_models.py          # ORM: User, Submission, Prediction, AuditLog
│   │
│   ├── api/
│   │   ├── auth.py               # POST /api/auth/login
│   │   ├── classify.py           # POST /api/classify  (core endpoint)
│   │   └── history.py            # GET  /api/history, /api/stats  (admin)
│   │
│   ├── nlp/
│   │   ├── preprocessor.py       # TextPreprocessor  (Algorithm 3.1)
│   │   └── classifier.py         # FakeNewsClassifier (Algorithms 3.2 & 3.3)
│   │
│   ├── utils/
│   │   └── prediction_logger.py  # PredictionLogger (DB persistence)
│   │
│   ├── scripts/
│   │   ├── train_model.py        # Model training entry point
│   │   ├── evaluate_model.py     # Model evaluation and reporting
│   │   └── seed_admin.py         # Create default admin user
│   │
│   ├── saved_models/             # Trained model files (git-ignored)
│   └── data/                     # Dataset CSV files (git-ignored)
│
├── frontend/
│   ├── Dockerfile
│   ├── nginx.conf
│   ├── package.json
│   ├── vite.config.js
│   ├── index.html
│   └── src/
│       ├── main.jsx
│       ├── App.jsx               # Root component — state management
│       ├── index.css             # Global styles and CSS variables
│       ├── components/
│       │   ├── Header.jsx        # Navigation bar with admin login
│       │   ├── ArticleInput.jsx  # Text input with word-count validation
│       │   ├── ResultPanel.jsx   # Verdict, confidence bar, LIME features
│       │   ├── AdminDashboard.jsx# Stats and classification history table
│       │   └── LoginModal.jsx    # Admin authentication modal
│       └── services/
│           └── api.js            # Axios API client
│
└── tests/
    ├── conftest.py               # pytest fixtures
    ├── test_preprocessor.py      # Algorithm 3.1 unit tests
    └── test_api.py               # REST API integration tests
```

---

## References

- Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2022). DistilBERT, a distilled version of BERT. *arXiv*. https://doi.org/10.48550/arXiv.1910.01108
- Verma, P. K., Agrawal, P., Amorim, I., & Prodan, R. (2021). WELFake: Word embedding over linguistic features for fake news detection. *IEEE Transactions on Computational Social Systems*, 8(4), 881–893.
- Ribeiro, M. T., Singh, S., & Guestrin, C. (2022). 'Why should I trust you?': Explaining the predictions of any classifier. *Proceedings of KDD 2016*. ACM.
- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers. *NAACL-HLT 2019*. ACL.
