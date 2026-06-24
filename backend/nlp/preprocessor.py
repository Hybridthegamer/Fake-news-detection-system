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

_HTML_RE = re.compile(r'<[^>]+>')
_URL_RE = re.compile(r'https?://\S+|www\.\S+')
_NON_ALPHA_RE = re.compile(r'[^a-z\s]')
_MULTI_SPACE_RE = re.compile(r'\s+')


class TextPreprocessor:
    """Stateless NLP preprocessing pipeline."""

    def __init__(self) -> None:
        try:
            # Disable all neural components — we only need the lookup-based
            # lemmatizer and tokenizer. tok2vec/tagger trigger heavy Cython
            # allocations on large texts that can fail on Windows.
            self._nlp = spacy.load(
                'en_core_web_sm',
                disable=['tok2vec', 'tagger', 'parser', 'ner', 'senter'],
            )
            self._nlp.max_length = 2_000_000  # handle long articles
            logger.info("spaCy en_core_web_sm loaded (lemmatizer only).")
        except OSError:
            raise RuntimeError(
                "spaCy model 'en_core_web_sm' not found. "
                "Run: python -m spacy download en_core_web_sm"
            )

    def preprocess(self, raw_text: str) -> str:
        """Return a cleaned, lemmatised token string ready for vectorisation."""
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
