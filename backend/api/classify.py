"""
POST /api/classify — main classification endpoint.

Processing pipeline (Algorithm 3.1 → 3.2 / 3.3 from the system design):
  1. Validate and sanitise input
  2. Preprocess text (TextPreprocessor)
  3. Classify (FakeNewsClassifier — DistilBERT or LR-TFIDF fallback)
  4. Generate LIME feature explanations
  5. Log to database
  6. Return JSON response
"""
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
            clean_text,
            classifier.predict_for_lime,
            num_features=3,
            num_samples=300,
        )
        top_features = [
            {'feature': feat, 'weight': round(w, 4)}
            for feat, w in exp.as_list()
        ]
        label = classifier.classify(clean_text)['label']
        terms = '; '.join(f'"{f["feature"]}"' for f in top_features[:3])
        summary = (
            f'This article was classified as {label} News. '
            f'Key influencing terms: {terms}.'
        )
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

    # Strip HTML/script tags (XSS prevention)
    sanitised = re.sub(r'<[^>]+>', ' ', raw_text)
    sanitised = re.sub(r'\s+', ' ', sanitised).strip()

    wc = _word_count(sanitised)
    cfg = current_app.config

    if wc < cfg['MIN_HEADLINE_WORDS']:
        return jsonify({
            'error': f'Text is too short. Minimum {cfg["MIN_HEADLINE_WORDS"]} words required.'
        }), 400
    if wc > cfg['MAX_ARTICLE_WORDS']:
        return jsonify({
            'error': f'Text is too long. Maximum {cfg["MAX_ARTICLE_WORDS"]} words allowed.'
        }), 400

    preprocessor = current_app.preprocessor
    classifier = current_app.classifier

    if preprocessor is None:
        return jsonify({'error': 'NLP engine unavailable. Check server logs.'}), 503
    if not classifier.is_ready:
        return jsonify({
            'error': (
                'No trained model found. '
                'Run: python scripts/train_model.py --dataset data/WELFake_Dataset.csv'
            )
        }), 503

    clean_text = preprocessor.preprocess(sanitised)
    result = classifier.classify(clean_text)
    explanation = _lime_explanation(classifier, clean_text)

    input_type = 'headline' if wc <= cfg['MAX_HEADLINE_WORDS'] else 'article'
    low_confidence = result['confidence'] < cfg['CONFIDENCE_THRESHOLD']

    ip: Optional[str] = request.headers.get('X-Forwarded-For', request.remote_addr)

    try:
        PredictionLogger.log(
            article_text=sanitised,
            word_count=wc,
            input_type=input_type,
            predicted_label=result['label'],
            confidence_score=result['confidence'],
            model_used=result['model_used'],
            explanation_json=explanation,
            ip_address=ip,
        )
    except Exception as exc:
        logger.error("Database logging failed: %s", exc)

    return jsonify({
        'label': result['label'],
        'confidence': round(result['confidence'] * 100, 2),
        'model_used': result['model_used'],
        'explanation': explanation,
        'low_confidence': low_confidence,
        'word_count': wc,
        'input_type': input_type,
        'processing_time_ms': round((time.time() - t0) * 1000, 1),
    }), 200
