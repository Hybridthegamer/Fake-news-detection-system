"""
FakeNewsClassifier — implements Algorithms 3.2 and 3.3.

Primary model  : DistilBERT fine-tuned on WELFake (distilbert-base-uncased)
Baseline model : TF-IDF (50k features, bigrams) + Logistic Regression

The classifier loads whichever model is available, preferring DistilBERT.
It falls back to LR-TFIDF gracefully when DistilBERT files are absent.
"""
import os
import pickle
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

LABEL_MAP = {0: 'Real', 1: 'Fake'}


class FakeNewsClassifier:

    def __init__(
        self,
        distilbert_path: str,
        lr_classifier_path: str,
        vectorizer_path: str,
    ) -> None:
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
            logger.warning("DistilBERT unavailable — falling back to LR-TFIDF.")
        else:
            logger.error("No trained models found. Run scripts/train_model.py.")

    # ------------------------------------------------------------------ #
    # Public interface                                                     #
    # ------------------------------------------------------------------ #

    @property
    def is_ready(self) -> bool:
        return self._active is not None

    @property
    def active_model_name(self) -> Optional[str]:
        return self._active

    def classify(self, clean_text: str) -> dict:
        """
        Returns:
            {'label': 'Real'|'Fake', 'confidence': float 0–1, 'model_used': str}
        """
        if self._active == 'DistilBERT':
            return self._classify_distilbert(clean_text)
        if self._active == 'LR-TFIDF':
            return self._classify_lr(clean_text)
        raise RuntimeError("No model available. Run scripts/train_model.py first.")

    def predict_for_lime(self, texts: list[str]) -> np.ndarray:
        """
        Batch predict probabilities for LIME.
        Returns shape (n, 2) where col-0 = P(Real), col-1 = P(Fake).
        """
        rows = []
        for text in texts:
            r = self.classify(text)
            c = r['confidence']
            rows.append([1.0 - c, c] if r['label'] == 'Fake' else [c, 1.0 - c])
        return np.array(rows)

    # ------------------------------------------------------------------ #
    # Private loaders                                                      #
    # ------------------------------------------------------------------ #

    def _load_distilbert(self) -> None:
        if not os.path.isdir(self._distilbert_path):
            return
        try:
            from transformers import (
                DistilBertForSequenceClassification,
                DistilBertTokenizerFast,
            )
            self._bert_tokenizer = DistilBertTokenizerFast.from_pretrained(
                self._distilbert_path
            )
            self._bert_model = DistilBertForSequenceClassification.from_pretrained(
                self._distilbert_path
            )
            self._bert_model.eval()
            logger.info("DistilBERT loaded from %s", self._distilbert_path)
        except Exception as exc:
            logger.error("Failed to load DistilBERT: %s", exc)

    def _load_lr(self) -> None:
        if not (os.path.isfile(self._lr_path) and os.path.isfile(self._vec_path)):
            return
        try:
            with open(self._lr_path, 'rb') as fh:
                self._lr_model = pickle.load(fh)
            with open(self._vec_path, 'rb') as fh:
                self._vectorizer = pickle.load(fh)
            logger.info("LR-TFIDF model loaded.")
        except Exception as exc:
            logger.error("Failed to load LR-TFIDF: %s", exc)

    # ------------------------------------------------------------------ #
    # Private classifiers                                                  #
    # ------------------------------------------------------------------ #

    def _classify_distilbert(self, clean_text: str) -> dict:
        import torch

        inputs = self._bert_tokenizer(
            clean_text,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors='pt',
        )
        with torch.no_grad():
            logits = self._bert_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze().numpy()
        idx = int(np.argmax(probs))
        return {
            'label': LABEL_MAP[idx],
            'confidence': float(probs[idx]),
            'model_used': 'DistilBERT',
        }

    def _classify_lr(self, clean_text: str) -> dict:
        X = self._vectorizer.transform([clean_text])
        idx = int(self._lr_model.predict(X)[0])
        probs = self._lr_model.predict_proba(X)[0]
        return {
            'label': LABEL_MAP[idx],
            'confidence': float(probs[idx]),
            'model_used': 'LR-TFIDF',
        }
