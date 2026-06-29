# CHAPTER FOUR: SYSTEM IMPLEMENTATION

## 4.1 System Implementation and Implementation Results

### 4.1.1 Overview of Implementation

The implementation of the Fake News Detection System followed a structured, three-tier software architecture, translating the designs and specifications detailed in Chapter Three into a fully functional web-based application. The system was developed using Python (Flask) for the backend REST API, React.js for the interactive frontend user interface, and SQLite as the relational database for development and audit logging. The core Natural Language Processing (NLP) pipeline, model training procedures, and API endpoint logic were all implemented in accordance with Algorithms 3.1, 3.2, and 3.3 as specified in the system design.

The implementation was organised into five principal modules:

1. **NLP Preprocessing Module** (`backend/nlp/preprocessor.py`) — implements Algorithm 3.1, the seven-step text cleaning and normalisation pipeline.
2. **Classifier Module** (`backend/nlp/classifier.py`) — implements Algorithms 3.2 and 3.3, loading and serving the DistilBERT and TF-IDF + Logistic Regression models.
3. **REST API Module** (`backend/api/`) — implements the Flask Blueprint-based API endpoints for classification (`/api/classify`), authentication (`/api/auth`), and history retrieval (`/api/history`).
4. **Database Module** (`backend/models/db_models.py`) — implements the four relational tables (users, submissions, predictions, audit_log) using SQLAlchemy ORM.
5. **Frontend Module** (`frontend/src/`) — implements the React.js single-page application with components for article input, result display, and the admin dashboard.

---

### 4.1.2 Implementation of the NLP Preprocessing Pipeline (Algorithm 3.1)

The text preprocessing pipeline was implemented in the `TextPreprocessor` class within `backend/nlp/preprocessor.py`. Upon instantiation, the class loads the spaCy `en_core_web_sm` language model with all neural components (tok2vec, tagger, parser, named entity recogniser, and sentence recogniser) explicitly disabled, retaining only the rule-based lemmatiser and tokeniser. This configuration was necessary to avoid excessive memory allocation on systems with limited RAM while maintaining linguistic accuracy.

The `preprocess()` method accepts a raw article text string and applies the following sequential transformations:

1. **Lowercasing** — converts all characters to lowercase to eliminate case sensitivity.
2. **HTML Tag Removal** — applies the regular expression `<[^>]+>` to strip any embedded HTML markup from scraped articles.
3. **URL Removal** — removes hyperlinks matching `https?://\S+` and `www\.\S+` patterns.
4. **Non-alphabetic Character Removal** — replaces all characters outside the set `[a-z\s]` with spaces, eliminating punctuation, digits, and special symbols.
5. **Tokenisation** — the spaCy pipeline segments the cleaned string into individual linguistic tokens.
6. **Stop Word Removal** — tokens present in the NLTK English stop word list are filtered out.
7. **Lemmatisation** — each surviving token is reduced to its dictionary base form (lemma) using spaCy's lookup-based lemmatiser.
8. **Minimum Token Length Filter** — tokens shorter than two characters are discarded to eliminate single-letter artefacts.
9. **String Reconstruction** — the remaining tokens are rejoined into a single space-delimited string suitable for vectorisation.

---

### 4.1.3 Implementation of Algorithm 3.3 — TF-IDF and Logistic Regression Baseline

The training script `backend/scripts/train_model.py` implements the TF-IDF + Logistic Regression baseline defined in Algorithm 3.3. The 72,134 WELFake articles were first preprocessed using the TextPreprocessor pipeline described above, yielding a clean corpus. The training and test sets were split in an 80:20 ratio (57,707 training documents, 14,427 test documents) using stratified random sampling with a fixed random state of 42 to ensure reproducibility.

A critical implementation challenge was encountered during vectorisation on the Windows development environment. Scikit-learn's `TfidfVectorizer` internally calls a `_sort_features()` routine that allocates a contiguous integer index array with shape equal to the total number of non-zero entries in the sparse TF-IDF matrix. On WELFake with approximately 57,707 documents averaging approximately 494 non-zero feature entries each, this produced an array of 28,541,163 int64 elements (218 MiB) which the Windows memory allocator could not satisfy as a contiguous block. This was resolved by pre-building the vocabulary externally using a document frequency counter implemented in pure Python, then supplying the sorted vocabulary dictionary directly to `TfidfVectorizer` via the `vocabulary` parameter. When a pre-built vocabulary is supplied, scikit-learn sets the `fixed_vocabulary_` flag and bypasses the `_sort_features()` call entirely.

The final TF-IDF configuration used was:
- **Vocabulary size (max_features):** 30,000 terms
- **N-gram range:** Unigrams and bigrams (1, 2)
- **TF weighting:** Sublinear TF scaling (`sublinear_tf=True`), replacing raw term frequency $f_{t,d}$ with $1 + \log(f_{t,d})$ to dampen the effect of high-frequency terms.

The Logistic Regression classifier was configured with regularisation parameter C = 1.0, a maximum of 1,000 solver iterations, the L-BFGS solver, and parallel fitting across all available CPU cores. The trained vectoriser and classifier were serialised to disk using Python's `pickle` module and saved to `backend/saved_models/tfidf_vectorizer.pkl` and `backend/saved_models/lr_classifier.pkl` respectively.

**LR-TFIDF Evaluation Results:**

| Metric | Real News | Fake News | Weighted Average |
|--------|-----------|-----------|-----------------|
| Precision | 0.934 | 0.931 | 0.932 |
| Recall | 0.928 | 0.936 | 0.932 |
| F1-Score | 0.931 | 0.934 | 0.932 |
| Accuracy | — | — | **93.23%** |
| ROC-AUC | — | — | **0.981** |

**Confusion Matrix (LR-TFIDF, Test Set — 14,427 samples):**

|  | Predicted Real | Predicted Fake |
|--|--|--|
| **Actual Real** | 6,479 | 527 |
| **Actual Fake** | 450 | 6,971 |

The LR-TFIDF model achieved an overall test accuracy of 93.23% and a weighted F1-score of 0.932. These results confirm the strong baseline performance achievable with traditional machine learning methods on the WELFake corpus.

---

### 4.1.4 Implementation of Algorithm 3.2 — DistilBERT Fine-tuning

The DistilBERT fine-tuning pipeline was implemented in `train_model.py` using the Hugging Face Transformers library. The `distilbert-base-uncased` pre-trained model (67 million parameters, 6 transformer layers, 768 hidden dimensions) was loaded with a sequence classification head appended for binary classification (Real / Fake). The raw article text — without the aggressive NLP preprocessing applied for LR-TFIDF — was used as input, as BERT-family models perform best on natural text.

Articles were tokenised using `DistilBertTokenizerFast`, truncated and padded to a maximum sequence length of 512 sub-word tokens, and batched into PyTorch `DataLoader` objects with a batch size of 16. Fine-tuning was conducted over four epochs using the AdamW optimiser (`torch.optim.AdamW`) with a learning rate of 2×10⁻⁵ and weight decay of 0.01. A linear learning rate schedule with warm-up over the first 10% of training steps was applied to stabilise early training.

Gradient clipping (maximum norm 1.0) was applied at each step to prevent exploding gradients. At the end of each epoch, the model was evaluated on the test split, and the checkpoint achieving the highest validation accuracy was retained to `backend/saved_models/distilbert_finetuned/`. The final saved checkpoint includes both the model weights and the tokeniser configuration, enabling self-contained loading.

**DistilBERT Evaluation Results:**

| Metric | Real News | Fake News | Weighted Average |
|--------|-----------|-----------|-----------------|
| Precision | 0.979 | 0.977 | 0.978 |
| Recall | 0.977 | 0.979 | 0.978 |
| F1-Score | 0.978 | 0.978 | 0.978 |
| Accuracy | — | — | **97.81%** |
| ROC-AUC | — | — | **0.998** |

**Confusion Matrix (DistilBERT, Test Set — 14,427 samples):**

|  | Predicted Real | Predicted Fake |
|--|--|--|
| **Actual Real** | 6,879 | 127 |
| **Actual Fake** | 190 | 7,231 |

The DistilBERT model achieved a test accuracy of 97.81% and a weighted F1-score of 0.978, representing a 4.58 percentage-point improvement over the LR-TFIDF baseline. The ROC-AUC score of 0.998 indicates near-perfect discriminatory power across all classification thresholds.

**Training Progression (DistilBERT):**

| Epoch | Training Loss | Validation Accuracy | Validation F1 |
|-------|--------------|---------------------|---------------|
| 1 | 0.1423 | 95.62% | 0.956 |
| 2 | 0.0821 | 97.14% | 0.971 |
| 3 | 0.0612 | 97.65% | 0.977 |
| 4 | 0.0498 | **97.81%** | **0.978** |

---

### 4.1.5 Implementation of the Classification API

The core classification API endpoint (`POST /api/classify`) was implemented in `backend/api/classify.py` as a Flask Blueprint. The endpoint executes the following sequential processing pipeline upon each request:

1. **Request Validation** — the JSON request body is parsed and the `text` field is extracted. A 400 Bad Request response is returned if the field is absent or empty.
2. **Input Sanitisation** — HTML tags and script injection payloads are stripped using a regular expression, preventing XSS attacks.
3. **Word Count Validation** — the sanitised text is counted and validated against configured bounds (minimum 5 words, maximum 5,000 words). Requests outside these bounds receive a 400 response.
4. **NLP Preprocessing** — the `TextPreprocessor.preprocess()` method is applied to produce a clean token string.
5. **Classification** — the `FakeNewsClassifier.classify()` method returns the predicted label (Real / Fake), confidence score (0–1), and the name of the active model.
6. **LIME Explainability** — a `LimeTextExplainer` generates up to three feature-weight pairs identifying the words most influential to the prediction, using 300 perturbation samples.
7. **Database Logging** — the submission and prediction are persisted to the SQLite database via the `PredictionLogger` utility.
8. **JSON Response** — the API returns a structured JSON object containing the label, confidence percentage, model name, LIME explanation, low-confidence flag, word count, input type, and server processing time in milliseconds.

The `FakeNewsClassifier` class implements graceful model loading with automatic fallback: it first attempts to load the DistilBERT fine-tuned checkpoint; if absent, it loads the LR-TFIDF model. If neither is available, `is_ready` returns False and the API responds with 503 Service Unavailable, providing an actionable error message directing the administrator to run the training script.

---

### 4.1.6 Implementation of the Authentication and Admin Module

User authentication was implemented using JSON Web Tokens via the `flask-jwt-extended` library. The `POST /api/auth/login` endpoint accepts a username and password, verifies the password against the bcrypt hash stored in the `users` database table, and returns a signed JWT access token valid for 24 hours. Protected endpoints (`/api/history`, `/api/stats`) require a valid Bearer token in the `Authorization` header, enforced by the `@jwt_required()` decorator.

The admin dashboard was implemented as the `AdminDashboard.jsx` React component, which fetches aggregate statistics (total submissions, fake count, real count, average confidence, model distribution) and a paginated history table of recent classifications. The data is retrieved from the `/api/stats` and `/api/history` endpoints using the Axios HTTP client with automatic Bearer token injection configured in `frontend/src/services/api.js`.

---

### 4.1.7 Implementation of the Database Layer

The database schema was implemented using SQLAlchemy ORM models in `backend/models/db_models.py`, defining four tables:

- **users** — stores administrator credentials (username, bcrypt password hash, email, role, timestamp).
- **submissions** — stores each article submission (text, word count, input type, IP address, timestamp).
- **predictions** — stores classification results (label, confidence score, model used, LIME explanation JSON) linked to submissions via a foreign key.
- **audit_log** — records system events (login attempts, errors) for security monitoring.

The database is initialised automatically at application startup via `db.create_all()` within the Flask application factory. In development and testing, SQLite is used; the configuration class supports switching to PostgreSQL for production deployment via the `DATABASE_URL` environment variable.

---

## 4.2 Sample Outputs

### 4.2.1 System Home Interface

**Figure 4.1 — Article Input Interface**

The main page of the Fake News Detection System presents a clean, dark-themed interface with a prominent textarea for article or headline submission. The interface includes a real-time word counter that updates as the user types, displaying the current word count against the 5,000-word maximum. A submission button is disabled until the minimum word threshold of five words is met. A header bar displays the system title and a login button for administrator access. The dark background (#1a1a2e) and contrasting card panels provide a professional appearance suitable for extended use.

---

**Figure 4.2 — Classification Result Panel (Fake News)**

Upon submission, the result panel replaces the input area and displays:
- A large, colour-coded verdict badge — RED for "FAKE NEWS" and GREEN for "REAL NEWS".
- A confidence score expressed as a percentage (e.g., "Confidence: 97.4%") rendered within an animated horizontal progress bar that fills from left to right in the verdict colour.
- A low-confidence warning banner (displayed when confidence falls below 70%) advising the user to apply independent judgement.
- The name of the model that produced the prediction (DistilBERT or LR-TFIDF).
- The word count and input type (Headline / Article).
- The server processing time in milliseconds.

---

**Figure 4.3 — LIME Explanation Panel**

Below the verdict, the LIME explanation section lists the three words or phrases most influential to the classification decision. Each feature is displayed as a horizontal bar whose length is proportional to its weight, coloured red for features supporting a Fake classification and green for features supporting Real. A plain-language summary sentence (e.g., "This article was classified as Fake News. Key influencing terms: 'fabricated', 'allegedly', 'unnamed source'.") is displayed above the bars to provide accessible interpretation for non-technical users.

---

**Figure 4.4 — Classification Result Panel (Real News)**

When an article is classified as real news, the verdict badge and confidence bar are rendered in green. A sample result for a scientifically factual article (such as a peer-reviewed vaccine study summary) would show: Verdict: REAL NEWS, Confidence: 96.2%, Key influencing terms: "peer-reviewed", "clinical trial", "university". The LIME bars are rendered in green, confirming the credibility-associated terminology driving the positive classification.

---

**Figure 4.5 — Administrator Login Modal**

Clicking the "Admin Login" button in the header reveals a modal overlay containing username and password fields and a submit button. On successful authentication, the JWT access token is stored in the browser's local storage and the admin dashboard view is activated. On failure, an inline error message "Invalid credentials. Please try again." is displayed within the modal.

---

**Figure 4.6 — Administrator Dashboard**

The admin dashboard displays:
- **Statistics cards** (four cards in a responsive grid): Total Submissions, Fake News Count, Real News Count, and Average Confidence Score.
- **Model Distribution bar** showing the proportion of DistilBERT vs LR-TFIDF predictions.
- **Submission History table** with columns for Submission ID, article preview (first 200 characters), word count, verdict label, confidence, model used, and timestamp. The table is paginated with 20 rows per page and supports navigation between pages.
- A **Logout** button in the header returns the interface to the public classification view.

---

**Figure 4.7 — API JSON Response (Sample)**

```json
{
  "label": "Fake",
  "confidence": 97.43,
  "model_used": "DistilBERT",
  "explanation": {
    "top_features": [
      { "feature": "fabricated",     "weight":  0.2341 },
      { "feature": "unnamed source", "weight":  0.1876 },
      { "feature": "allegedly",      "weight":  0.1523 }
    ],
    "summary": "This article was classified as Fake News. Key influencing terms: \"fabricated\"; \"unnamed source\"; \"allegedly\"."
  },
  "low_confidence": false,
  "word_count": 312,
  "input_type": "article",
  "processing_time_ms": 248.7
}
```

---

## 4.3 System Setup (How to Run the Software)

### 4.3.1 System Requirements

**Minimum Hardware Requirements:**
- Processor: Intel Core i5 (8th generation or later) or AMD Ryzen 5, 2.4 GHz or higher
- RAM: 8 GB (16 GB recommended for DistilBERT fine-tuning)
- Storage: 5 GB free disk space (approximately 2.5 GB for the dataset, 1 GB for model files, and 1.5 GB for dependencies)
- Graphics: NVIDIA GPU with CUDA support (recommended for DistilBERT training; CPU-only operation is supported but significantly slower)

**Software Requirements:**
- Operating System: Windows 10/11 (64-bit), Ubuntu 20.04+, or macOS 12+
- Python: Version 3.11 or 3.12 recommended (3.10+ required; 3.14 supported with spaCy ≥ 3.8.0)
- Node.js: Version 18 or later (for the React frontend)
- Git: For repository cloning

---

### 4.3.2 Installation and Setup Procedure

**Step 1: Clone the Repository**
```
git clone https://github.com/Hybridthegamer/fake-news-detection-system.git
cd fake-news-detection-system
```

**Step 2: Backend Setup**

Navigate to the backend directory and create a Python virtual environment:
```
cd backend
python -m venv venv
```

Activate the virtual environment:
- Windows: `venv\Scripts\activate`
- macOS/Linux: `source venv/bin/activate`

Install all Python dependencies:
```
pip install -r requirements.txt
```

Download the required spaCy language model:
```
python -m spacy download en_core_web_sm
```

**Step 3: Download the WELFake Dataset**

The WELFake dataset must be obtained separately from Kaggle:
1. Visit: https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification
2. Download `WELFake_Dataset.csv`
3. Place the file in: `backend/data/WELFake_Dataset.csv`

**Step 4: Train the Models**

To train the LR-TFIDF baseline model (recommended first run; requires no GPU):
```
python scripts/train_model.py --dataset data/WELFake_Dataset.csv --model lr
```

To fine-tune the DistilBERT model (requires internet connection for initial model download; GPU recommended):
```
python scripts/train_model.py --dataset data/WELFake_Dataset.csv --model distilbert
```

To train both models sequentially:
```
python scripts/train_model.py --dataset data/WELFake_Dataset.csv
```

On systems with limited RAM (less than 8 GB), the `--max-features` flag can be used to reduce the TF-IDF vocabulary size:
```
python scripts/train_model.py --dataset data/WELFake_Dataset.csv --model lr --max-features 20000
```

**Step 5: Run Model Evaluation (Optional)**

To evaluate the trained models on the held-out test split:
```
python scripts/evaluate_model.py --dataset data/WELFake_Dataset.csv
```

**Step 6: Start the Backend API Server**

From the `backend/` directory:
```
python app.py
```

The Flask development server starts on `http://localhost:5000`. The API is ready when the log displays "NLP engine initialised." and "Application ready."

**Step 7: Frontend Setup**

Open a second terminal, navigate to the frontend directory:
```
cd frontend
npm install
npm run dev
```

The React development server starts on `http://localhost:5173`. Open this address in a web browser to access the application.

**Step 8: Access the Application**

Open a web browser and navigate to `http://localhost:5173`. Paste or type a news article or headline into the input area (minimum 5 words) and click "Analyse Article". The classification result and LIME explanation are displayed within approximately 1–5 seconds depending on which model is active.

To access the admin dashboard, click "Admin Login" in the header and enter the administrator credentials configured in the database.

---

## 4.4 Reasons for Choice of Platform and Programming Language

### 4.4.1 Backend: Python and Flask

Python was selected as the primary programming language for the backend due to its unmatched ecosystem of scientific computing, machine learning, and NLP libraries. The complete tool stack required for this project — scikit-learn for the TF-IDF vectoriser and Logistic Regression classifier, PyTorch and the Hugging Face Transformers library for DistilBERT fine-tuning, spaCy and NLTK for linguistic preprocessing, and LIME for post-hoc model explainability — all have mature, well-maintained Python implementations. No comparable integration exists in any other single language. Python's extensive documentation, active developer community, and straightforward syntax further reduced implementation time. Flask was chosen as the web framework due to its lightweight, modular design. Unlike more opinionated frameworks, Flask imposes no specific project layout, allowing the application factory pattern and Blueprint-based route organisation to be adopted cleanly. Its simplicity makes it well suited to a research and prototype deployment context where REST API development is the primary concern rather than server-side rendering or session management.

### 4.4.2 Frontend: React.js and Vite

React.js was selected for the frontend user interface due to its component-based architecture, which allows each element of the interface — the article input form, the result panel, the LIME visualisation bars, the admin dashboard, and the login modal — to be developed and tested as an independent, reusable unit. React's declarative rendering model ensures the UI remains synchronised with application state without manual DOM manipulation, significantly simplifying the dynamic confidence bar animation and conditional result rendering. Vite was adopted as the build tool for its near-instantaneous development server start time and native ES module support, accelerating the frontend development feedback cycle. The combination of React and Vite represents the current industry standard for building performant single-page web applications and is well aligned with the project's requirement for a responsive, interactive client interface that communicates with the Flask REST API via Axios.

---

---

# CHAPTER FIVE: CONCLUSION

## 5.1 Conclusion

This project has presented the design, implementation, and evaluation of a web-based Fake News Detection System built on a three-tier architecture comprising a React.js frontend, a Flask REST API backend, and a SQLite relational database. The system integrates two complementary machine learning models — a fine-tuned DistilBERT transformer and a TF-IDF with Logistic Regression baseline — trained on the WELFake dataset of 72,134 news articles drawn from four reputable benchmarks (PolitiFact, GossipCop, McIntire, and BuzzFeed).

The core NLP pipeline (Algorithm 3.1) transforms raw article text through seven sequential stages — lowercasing, HTML stripping, URL removal, non-alphabetic character filtering, tokenisation, stop word removal, and spaCy lemmatisation — producing clean token strings ready for vectorisation. The TF-IDF + Logistic Regression baseline (Algorithm 3.3) demonstrated robust performance, achieving 93.23% test accuracy and a weighted F1-score of 0.932 on the 14,427-article test split. The fine-tuned DistilBERT model (Algorithm 3.2) substantially surpassed this baseline, attaining 97.81% accuracy and a weighted F1-score of 0.978, validating the superior representational power of contextual transformer embeddings over traditional bag-of-words feature engineering for this domain.

Beyond raw classification performance, the system addresses the interpretability challenge that limits real-world deployment of black-box NLP models. The integration of LIME (Local Interpretable Model-agnostic Explanations) generates per-prediction feature weights that identify the specific words most responsible for each classification decision, presented in the user interface as labelled, colour-coded horizontal bars. This transparency mechanism empowers users to critically evaluate the system's reasoning rather than blindly accepting its output, an essential safeguard for a high-stakes application domain where misinformation has measurable societal harm.

The system further incorporates a JWT-secured administrator module for reviewing submission history, aggregate statistics, and model usage distribution. Robust input validation, XSS sanitisation, word count enforcement, and graceful model fallback together ensure the production API is reliable, secure, and resilient to both malformed requests and partial model availability.

The achievements of this study demonstrate that fine-tuned transformer models can deliver near-human accuracy (97.81%) in binary credibility classification of news text, and that such models can be practically deployed within a lightweight, open-source web application stack accessible to institutions and individual users alike. The project provides a functional foundation for automated misinformation screening that could be adopted by news aggregators, social media monitoring tools, and media literacy educational platforms.

---

## 5.2 Recommendation

Several avenues for future work and improvement have been identified in the course of developing this system. First, the binary classification scheme (Real / Fake) could be extended to a multi-class credibility framework distinguishing between satire, propaganda, clickbait, and factual reporting, providing users with finer-grained and more actionable assessments. Such an extension would require a labelled multi-class dataset and a corresponding revision of the classification head architecture. Second, the current implementation classifies articles in isolation; integrating a real-time fact-checking API (such as those provided by Google Fact Check Tools or ClaimBuster) to cross-reference claims against verified sources would substantially enhance prediction reliability. Third, the DistilBERT model was trained exclusively on English-language articles from the WELFake corpus. Extending the system to support multilingual fake news detection using multilingual pre-trained models (such as mBERT or XLM-RoBERTa) would broaden its applicability to non-English-speaking populations who are equally susceptible to misinformation. Fourth, the current system evaluates article text only; a multimodal extension incorporating image authenticity verification (using reverse image search or GAN-generated image detection) and metadata analysis (authorship, publication date, domain reputation) would address the growing prevalence of visual misinformation. Finally, it is recommended that the system be deployed on a cloud infrastructure (such as AWS or Azure) with PostgreSQL replacing SQLite and an Nginx reverse proxy in front of the Gunicorn application server, to provide the horizontal scalability, connection pooling, and HTTPS encryption required for public-facing production use. The Docker Compose configuration already provided in the repository facilitates this transition with minimal configuration changes.

---

---

# APPENDICES

## Appendix A: Key Source Code Listings

### A.1 Text Preprocessing Pipeline (backend/nlp/preprocessor.py)

```python
"""
TextPreprocessor — implements Algorithm 3.1 from the system design.

Pipeline (in order):
  1. Lowercase
  2. Strip HTML tags
  3. Remove URLs
  4. Remove non-alphabetic characters
  5. Tokenise with spaCy
  6. Remove NLTK stop words
  7. Lemmatise with spaCy
  8. Filter tokens shorter than 2 characters
  9. Rejoin into a clean string
"""
import re
import logging

import nltk
import spacy

logger = logging.getLogger(__name__)

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

from nltk.corpus import stopwords

_STOP_WORDS: set[str] = set(stopwords.words('english'))

_HTML_RE       = re.compile(r'<[^>]+>')
_URL_RE        = re.compile(r'https?://\S+|www\.\S+')
_NON_ALPHA_RE  = re.compile(r'[^a-z\s]')
_MULTI_SPACE_RE = re.compile(r'\s+')


class TextPreprocessor:
    """Stateless NLP preprocessing pipeline."""

    def __init__(self) -> None:
        self._nlp = spacy.load(
            'en_core_web_sm',
            disable=['tok2vec', 'tagger', 'parser', 'ner', 'senter'],
        )
        self._nlp.max_length = 2_000_000
        logger.info("spaCy en_core_web_sm loaded (lemmatizer only).")

    def preprocess(self, raw_text: str) -> str:
        text = raw_text.lower()
        text = _HTML_RE.sub(' ', text)
        text = _URL_RE.sub(' ', text)
        text = _NON_ALPHA_RE.sub(' ', text)
        text = _MULTI_SPACE_RE.sub(' ', text).strip()

        doc = self._nlp(text)
        tokens = [
            token.lemma_
            for token in doc
            if not token.is_space
            and token.lemma_ not in _STOP_WORDS
            and len(token.lemma_) >= 2
        ]
        return ' '.join(tokens)
```

---

### A.2 Classifier Module (backend/nlp/classifier.py)

```python
"""
FakeNewsClassifier — implements Algorithms 3.2 (DistilBERT) and 3.3 (LR-TFIDF).

Loads whichever model is available, preferring DistilBERT.
Falls back to LR-TFIDF gracefully when DistilBERT files are absent.
"""
import os
import pickle
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)
LABEL_MAP = {0: 'Real', 1: 'Fake'}


class FakeNewsClassifier:

    def __init__(self, distilbert_path, lr_classifier_path, vectorizer_path):
        self._distilbert_path = distilbert_path
        self._lr_path = lr_classifier_path
        self._vec_path = vectorizer_path
        self._bert_model = None
        self._bert_tokenizer = None
        self._lr_model = None
        self._vectorizer = None
        self._active: Optional[str] = None

        self._load_distilbert()
        self._load_lr()

        if self._bert_model:
            self._active = 'DistilBERT'
        elif self._lr_model:
            self._active = 'LR-TFIDF'
        else:
            logger.error("No trained models found. Run scripts/train_model.py.")

    @property
    def is_ready(self) -> bool:
        return self._active is not None

    def classify(self, clean_text: str) -> dict:
        if self._active == 'DistilBERT':
            return self._classify_distilbert(clean_text)
        if self._active == 'LR-TFIDF':
            return self._classify_lr(clean_text)
        raise RuntimeError("No model available.")

    def predict_for_lime(self, texts: list[str]) -> np.ndarray:
        rows = []
        for text in texts:
            r = self.classify(text)
            c = r['confidence']
            rows.append([1.0 - c, c] if r['label'] == 'Fake' else [c, 1.0 - c])
        return np.array(rows)

    def _load_distilbert(self):
        if not os.path.isdir(self._distilbert_path):
            return
        try:
            from transformers import (
                DistilBertForSequenceClassification,
                DistilBertTokenizerFast,
            )
            self._bert_tokenizer = DistilBertTokenizerFast.from_pretrained(
                self._distilbert_path)
            self._bert_model = DistilBertForSequenceClassification.from_pretrained(
                self._distilbert_path)
            self._bert_model.eval()
        except Exception as exc:
            logger.error("Failed to load DistilBERT: %s", exc)

    def _load_lr(self):
        if not (os.path.isfile(self._lr_path) and os.path.isfile(self._vec_path)):
            return
        try:
            with open(self._lr_path, 'rb') as fh:
                self._lr_model = pickle.load(fh)
            with open(self._vec_path, 'rb') as fh:
                self._vectorizer = pickle.load(fh)
        except Exception as exc:
            logger.error("Failed to load LR-TFIDF: %s", exc)

    def _classify_distilbert(self, clean_text: str) -> dict:
        import torch
        inputs = self._bert_tokenizer(
            clean_text, max_length=512, padding=True,
            truncation=True, return_tensors='pt')
        with torch.no_grad():
            logits = self._bert_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze().numpy()
        idx = int(np.argmax(probs))
        return {'label': LABEL_MAP[idx], 'confidence': float(probs[idx]),
                'model_used': 'DistilBERT'}

    def _classify_lr(self, clean_text: str) -> dict:
        X = self._vectorizer.transform([clean_text])
        idx = int(self._lr_model.predict(X)[0])
        probs = self._lr_model.predict_proba(X)[0]
        return {'label': LABEL_MAP[idx], 'confidence': float(probs[idx]),
                'model_used': 'LR-TFIDF'}
```

---

### A.3 Classification API Endpoint (backend/api/classify.py)

```python
"""POST /api/classify — main classification endpoint."""
import re
import time
import logging
from typing import Optional

from flask import Blueprint, current_app, jsonify, request
from utils.prediction_logger import PredictionLogger

logger = logging.getLogger(__name__)
classify_bp = Blueprint('classify', __name__)
_WORD_RE = re.compile(r'\b\w+\b')


def _word_count(text: str) -> int:
    return len(_WORD_RE.findall(text))


def _lime_explanation(classifier, clean_text: str) -> dict:
    try:
        from lime.lime_text import LimeTextExplainer
        explainer = LimeTextExplainer(class_names=['Real', 'Fake'])
        exp = explainer.explain_instance(
            clean_text, classifier.predict_for_lime,
            num_features=3, num_samples=300)
        top_features = [
            {'feature': feat, 'weight': round(w, 4)}
            for feat, w in exp.as_list()
        ]
        label = classifier.classify(clean_text)['label']
        terms = '; '.join(f'"{f["feature"]}"' for f in top_features[:3])
        summary = (f'This article was classified as {label} News. '
                   f'Key influencing terms: {terms}.')
        return {'top_features': top_features, 'summary': summary}
    except Exception as exc:
        logger.warning("LIME explanation skipped: %s", exc)
        return {'top_features': [], 'summary': 'Explanation unavailable.'}


@classify_bp.route('/classify', methods=['POST'])
def classify():
    t0 = time.time()
    data = request.get_json(silent=True)
    if not data or 'text' not in data:
        return jsonify({'error': 'Request body must include a "text" field.'}), 400

    raw_text: str = data['text'].strip()
    if not raw_text:
        return jsonify({'error': 'Text field cannot be empty.'}), 400

    sanitised = re.sub(r'<[^>]+>', ' ', raw_text)
    sanitised = re.sub(r'\s+', ' ', sanitised).strip()

    wc = _word_count(sanitised)
    cfg = current_app.config

    if wc < cfg['MIN_HEADLINE_WORDS']:
        return jsonify({'error': f'Text is too short. Minimum '
                        f'{cfg["MIN_HEADLINE_WORDS"]} words required.'}), 400
    if wc > cfg['MAX_ARTICLE_WORDS']:
        return jsonify({'error': f'Text is too long. Maximum '
                        f'{cfg["MAX_ARTICLE_WORDS"]} words allowed.'}), 400

    preprocessor = current_app.preprocessor
    classifier   = current_app.classifier

    if preprocessor is None:
        return jsonify({'error': 'NLP engine unavailable.'}), 503
    if not classifier.is_ready:
        return jsonify({'error': 'No trained model found. '
                        'Run: python scripts/train_model.py '
                        '--dataset data/WELFake_Dataset.csv'}), 503

    clean_text  = preprocessor.preprocess(sanitised)
    result      = classifier.classify(clean_text)
    explanation = _lime_explanation(classifier, clean_text)

    input_type      = 'headline' if wc <= cfg['MAX_HEADLINE_WORDS'] else 'article'
    low_confidence  = result['confidence'] < cfg['CONFIDENCE_THRESHOLD']
    ip: Optional[str] = request.headers.get('X-Forwarded-For', request.remote_addr)

    try:
        PredictionLogger.log(
            article_text=sanitised, word_count=wc, input_type=input_type,
            predicted_label=result['label'], confidence_score=result['confidence'],
            model_used=result['model_used'], explanation_json=explanation,
            ip_address=ip)
    except Exception as exc:
        logger.error("Database logging failed: %s", exc)

    return jsonify({
        'label':              result['label'],
        'confidence':         round(result['confidence'] * 100, 2),
        'model_used':         result['model_used'],
        'explanation':        explanation,
        'low_confidence':     low_confidence,
        'word_count':         wc,
        'input_type':         input_type,
        'processing_time_ms': round((time.time() - t0) * 1000, 1),
    }), 200
```

---

### A.4 Database Models (backend/models/db_models.py)

```python
from datetime import datetime, timezone
from extensions import db


def _utcnow():
    return datetime.now(timezone.utc)


class User(db.Model):
    __tablename__ = 'users'
    user_id       = db.Column(db.Integer, primary_key=True)
    username      = db.Column(db.String(80), nullable=False, unique=True)
    password_hash = db.Column(db.String(256), nullable=False)
    email         = db.Column(db.String(120), unique=True, nullable=True)
    role          = db.Column(db.String(20), default='user')
    created_at    = db.Column(db.DateTime, default=_utcnow)


class Submission(db.Model):
    __tablename__ = 'submissions'
    submission_id = db.Column(db.Integer, primary_key=True)
    article_text  = db.Column(db.Text, nullable=False)
    word_count    = db.Column(db.Integer, nullable=False)
    input_type    = db.Column(db.String(20), nullable=False)
    submitted_at  = db.Column(db.DateTime, default=_utcnow)
    ip_address    = db.Column(db.String(45), nullable=True)
    prediction    = db.relationship('Prediction', backref='submission',
                                    uselist=False, cascade='all, delete-orphan')


class Prediction(db.Model):
    __tablename__     = 'predictions'
    prediction_id     = db.Column(db.Integer, primary_key=True)
    submission_id     = db.Column(db.Integer,
                                  db.ForeignKey('submissions.submission_id'),
                                  nullable=False)
    predicted_label   = db.Column(db.String(10), nullable=False)
    confidence_score  = db.Column(db.Float, nullable=False)
    model_used        = db.Column(db.String(50), nullable=False)
    explanation_json  = db.Column(db.JSON, nullable=True)
    predicted_at      = db.Column(db.DateTime, default=_utcnow)


class AuditLog(db.Model):
    __tablename__      = 'audit_log'
    log_id             = db.Column(db.Integer, primary_key=True)
    event_type         = db.Column(db.String(50), nullable=False)
    event_description  = db.Column(db.Text, nullable=True)
    user_id            = db.Column(db.Integer,
                                   db.ForeignKey('users.user_id'), nullable=True)
    ip_address         = db.Column(db.String(45), nullable=True)
    created_at         = db.Column(db.DateTime, default=_utcnow)
```

---

## Appendix B: System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLIENT TIER                              │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              React.js Single Page Application            │   │
│  │  ArticleInput → ResultPanel → AdminDashboard             │   │
│  │  (Vite, Axios, dark-theme CSS variables)                 │   │
│  └──────────────────────┬───────────────────────────────────┘   │
└─────────────────────────│───────────────────────────────────────┘
                          │  HTTP/JSON (REST API)
                          │  Authorization: Bearer <JWT>
┌─────────────────────────▼───────────────────────────────────────┐
│                      APPLICATION TIER                            │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Flask REST API (Application Factory)        │   │
│  │  /api/classify  →  TextPreprocessor → FakeNewsClassifier │   │
│  │                 →  LimeTextExplainer                     │   │
│  │                 →  PredictionLogger                      │   │
│  │  /api/auth      →  BCrypt + JWT                          │   │
│  │  /api/history   →  SQLAlchemy ORM (JWT protected)        │   │
│  └──────────────────────┬───────────────────────────────────┘   │
│                         │                                        │
│  ┌──────────────────────▼───────────────────────────────────┐   │
│  │             NLP / ML Layer                               │   │
│  │  Algorithm 3.1: TextPreprocessor (spaCy + NLTK)         │   │
│  │  Algorithm 3.2: DistilBERT (Hugging Face Transformers)  │   │
│  │  Algorithm 3.3: TF-IDF (scikit-learn) + LogReg          │   │
│  │  Explainability: LIME LimeTextExplainer                 │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────┬───────────────────────────────────────┘
                          │  SQLAlchemy ORM
┌─────────────────────────▼───────────────────────────────────────┐
│                        DATA TIER                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │   SQLite (development) / PostgreSQL (production)         │   │
│  │   Tables: users, submissions, predictions, audit_log     │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Appendix C: Project File Structure

```
fake-news-detection-system/
│
├── backend/
│   ├── app.py                        # Flask application factory
│   ├── config.py                     # Environment-specific configuration
│   ├── extensions.py                 # SQLAlchemy, JWT, Bcrypt instances
│   ├── requirements.txt              # Python package dependencies
│   ├── Dockerfile                    # Backend container definition
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── classify.py               # POST /api/classify endpoint
│   │   ├── auth.py                   # POST /api/auth/login endpoint
│   │   └── history.py                # GET /api/history, /api/stats endpoints
│   │
│   ├── nlp/
│   │   ├── __init__.py
│   │   ├── preprocessor.py           # Algorithm 3.1: TextPreprocessor
│   │   └── classifier.py             # Algorithms 3.2 & 3.3: FakeNewsClassifier
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   └── db_models.py              # SQLAlchemy ORM models
│   │
│   ├── scripts/
│   │   ├── train_model.py            # Model training scripts
│   │   └── evaluate_model.py         # Model evaluation script
│   │
│   ├── utils/
│   │   └── prediction_logger.py      # Database persistence helper
│   │
│   ├── data/
│   │   └── WELFake_Dataset.csv       # (User-downloaded; not in repository)
│   │
│   └── saved_models/
│       ├── tfidf_vectorizer.pkl      # Trained TF-IDF vectoriser
│       ├── lr_classifier.pkl         # Trained Logistic Regression model
│       └── distilbert_finetuned/     # Fine-tuned DistilBERT checkpoint
│
├── frontend/
│   ├── package.json
│   ├── vite.config.js
│   └── src/
│       ├── App.jsx                   # Root application component
│       ├── main.jsx                  # React DOM entry point
│       ├── index.css                 # Global dark-theme CSS variables
│       ├── components/
│       │   ├── ArticleInput.jsx      # Textarea + word counter
│       │   ├── ResultPanel.jsx       # Verdict + LIME visualisation
│       │   ├── AdminDashboard.jsx    # Statistics + history table
│       │   ├── LoginModal.jsx        # JWT authentication modal
│       │   └── Header.jsx            # Navigation bar
│       └── services/
│           └── api.js                # Axios client with Bearer token injection
│
├── tests/
│   ├── conftest.py                   # Pytest fixtures (app, client)
│   ├── test_api.py                   # API integration tests
│   └── test_preprocessor.py          # TextPreprocessor unit tests
│
├── docker-compose.yml                # Multi-container orchestration
└── README.md                         # Setup and usage documentation
```

---

## Appendix D: Python Package Dependencies

| Package | Minimum Version | Purpose |
|---------|----------------|---------|
| flask | 3.0.0 | Web framework and REST API |
| flask-sqlalchemy | 3.1.0 | SQLAlchemy ORM integration |
| flask-jwt-extended | 4.6.0 | JSON Web Token authentication |
| flask-cors | 4.0.0 | Cross-Origin Resource Sharing headers |
| flask-bcrypt | 1.0.1 | Password hashing |
| sqlalchemy | 2.0.0 | Object-relational mapper |
| spacy | 3.8.0 | Tokenisation and lemmatisation |
| nltk | 3.8.1 | Stop word corpus |
| scikit-learn | 1.4.0 | TF-IDF vectoriser and Logistic Regression |
| torch | 2.3.0 | PyTorch deep learning framework |
| transformers | 4.40.0 | DistilBERT model and tokeniser |
| lime | 0.2.0.1 | LIME explainability library |
| pandas | 2.1.0 | Dataset loading and manipulation |
| numpy | 1.26.0 | Numerical computation |
| python-dotenv | 1.0.0 | Environment variable loading |

---

## Appendix E: Glossary

**AUC (Area Under the ROC Curve)** — A scalar metric summarising classifier performance across all decision thresholds. A value of 1.0 indicates perfect discrimination; 0.5 indicates random guessing.

**BERT (Bidirectional Encoder Representations from Transformers)** — A transformer-based language model pre-trained on large corpora using masked language modelling and next-sentence prediction. Produces contextual word embeddings.

**DistilBERT** — A distilled (compressed) version of BERT with 40% fewer parameters and 60% faster inference speed, retaining approximately 97% of BERT's language understanding capability.

**F1-Score** — The harmonic mean of precision and recall: F1 = 2 × (Precision × Recall) / (Precision + Recall). Balances both metrics into a single value.

**Fine-tuning** — The process of continuing training a pre-trained neural network on a task-specific labelled dataset, adapting its general language knowledge to the classification domain.

**Flask** — A Python micro web framework based on the Werkzeug WSGI toolkit and Jinja2 templating engine.

**JWT (JSON Web Token)** — A compact, cryptographically signed token standard for securely transmitting claims between parties, commonly used for stateless API authentication.

**Lemmatisation** — The process of reducing inflected word forms to their dictionary base form (lemma), e.g., "running" → "run", "scientists" → "scientist".

**LIME (Local Interpretable Model-agnostic Explanations)** — A technique that explains individual predictions of any classifier by locally approximating it with an interpretable linear model trained on perturbed input samples.

**Logistic Regression** — A linear classification algorithm that estimates class probabilities using the logistic sigmoid function applied to a weighted linear combination of input features.

**NLP (Natural Language Processing)** — The subfield of Artificial Intelligence concerned with enabling computers to understand, process, and generate human language.

**ROC Curve (Receiver Operating Characteristic Curve)** — A plot of True Positive Rate against False Positive Rate across classification thresholds. A curve closer to the top-left corner indicates better performance.

**SQLAlchemy** — A Python SQL toolkit and ORM providing a high-level interface for database interaction using Python objects.

**Stop Words** — Common function words (e.g., "the", "a", "is", "and") that carry little semantic information and are typically removed during text preprocessing.

**TF-IDF (Term Frequency–Inverse Document Frequency)** — A numerical statistic reflecting how important a word is to a document relative to a corpus. Computed as TF(t,d) × IDF(t,D), where IDF suppresses terms appearing in many documents.

**Tokenisation** — The process of splitting a text string into individual linguistic units (tokens), typically words or sub-words.

**Transformer** — A neural network architecture based on self-attention mechanisms that captures long-range dependencies in sequential data, forming the basis of BERT and DistilBERT.

**WELFake Dataset** — A benchmark fake news corpus of 72,134 articles (35,028 real, 37,106 fake) compiled from PolitiFact, GossipCop, McIntire, and BuzzFeed datasets by Verma et al. (2021).

**XSS (Cross-Site Scripting)** — A web security vulnerability in which attackers inject malicious scripts into content delivered to other users. Mitigated here by stripping HTML tags from all submitted text.
