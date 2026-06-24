#!/usr/bin/env python3
"""
Train the Fake News Detection models.

Implements Algorithms 3.2 (DistilBERT) and 3.3 (TF-IDF + LR) from the design.

Usage:
    # Train both models (recommended first run)
    python scripts/train_model.py --dataset data/WELFake_Dataset.csv

    # Train LR-TFIDF baseline only (fast, no GPU needed)
    python scripts/train_model.py --dataset data/WELFake_Dataset.csv --model lr

    # Fine-tune DistilBERT only
    python scripts/train_model.py --dataset data/WELFake_Dataset.csv --model distilbert

Dataset:
    Download WELFake from Kaggle:
    https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification
    Place the CSV at: backend/data/WELFake_Dataset.csv
"""
import argparse
import logging
import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

# Make backend/ importable when running from scripts/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nlp.preprocessor import TextPreprocessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, 'saved_models')
TFIDF_PATH = os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')
LR_PATH = os.path.join(MODEL_DIR, 'lr_classifier.pkl')
DISTILBERT_PATH = os.path.join(MODEL_DIR, 'distilbert_finetuned')


# ------------------------------------------------------------------ #
# Dataset loading                                                      #
# ------------------------------------------------------------------ #

def load_dataset(path: str) -> pd.DataFrame:
    logger.info("Loading dataset: %s", path)
    df = pd.read_csv(path)
    logger.info("Columns: %s", list(df.columns))

    if 'label' not in df.columns:
        raise ValueError("Dataset must have a 'label' column (0 = Real, 1 = Fake).")

    if 'title' in df.columns and 'text' in df.columns:
        df['combined'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
    elif 'text' in df.columns:
        df['combined'] = df['text'].fillna('')
    else:
        raise ValueError("Dataset must have a 'text' column.")

    df = pd.DataFrame(df[['combined', 'label']].dropna())
    # Cast to object dtype to avoid PyArrow-backed strings failing on long texts
    df['combined'] = df['combined'].astype(object)
    df['label'] = df['label'].astype(int)

    real_n = (df['label'] == 0).sum()
    fake_n = (df['label'] == 1).sum()
    logger.info("Loaded %d samples  —  Real: %d  |  Fake: %d", len(df), real_n, fake_n)
    return df


def preprocess_corpus(texts: list[str], preprocessor: TextPreprocessor) -> list[str]:
    logger.info("Preprocessing %d documents...", len(texts))
    cleaned = []
    for i, t in enumerate(texts):
        if i % 5000 == 0 and i:
            logger.info("  %d / %d done", i, len(texts))
        cleaned.append(preprocessor.preprocess(str(t)))
    logger.info("Preprocessing complete.")
    return cleaned


# ------------------------------------------------------------------ #
# Algorithm 3.3 — TF-IDF + Logistic Regression                        #
# ------------------------------------------------------------------ #

def train_lr_tfidf(
    X_train_clean: list[str],
    y_train: list[int],
    X_test_clean: list[str],
    y_test: list[int],
) -> tuple[float, float]:
    logger.info("=== Training TF-IDF + Logistic Regression ===")

    vectorizer = TfidfVectorizer(
        max_features=50_000,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    X_tr = vectorizer.fit_transform(X_train_clean)
    X_te = vectorizer.transform(X_test_clean)

    clf = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs', n_jobs=-1)
    clf.fit(X_tr, y_train)

    y_pred = clf.predict(X_te)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    logger.info("LR-TFIDF — Accuracy: %.4f  |  Weighted F1: %.4f", acc, f1)
    logger.info("\n%s", classification_report(y_test, y_pred, target_names=['Real', 'Fake']))

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(TFIDF_PATH, 'wb') as fh:
        pickle.dump(vectorizer, fh)
    with open(LR_PATH, 'wb') as fh:
        pickle.dump(clf, fh)

    logger.info("Saved TF-IDF vectorizer → %s", TFIDF_PATH)
    logger.info("Saved LR classifier    → %s", LR_PATH)
    return acc, f1


# ------------------------------------------------------------------ #
# Algorithm 3.2 — DistilBERT fine-tuning                              #
# ------------------------------------------------------------------ #

def train_distilbert(
    X_train: list[str],
    y_train: list[int],
    X_test: list[str],
    y_test: list[int],
    epochs: int = 4,
    batch_size: int = 16,
) -> tuple[float, float]:
    logger.info("=== Fine-tuning DistilBERT ===")
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
        from transformers import (
            DistilBertForSequenceClassification,
            DistilBertTokenizerFast,
            get_linear_schedule_with_warmup,
        )
    except ImportError as exc:
        logger.error("PyTorch / Transformers not installed: %s", exc)
        return 0.0, 0.0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("Device: %s", device)

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased', num_labels=2
    )
    model.to(device)

    class _NewsDataset(Dataset):
        def __init__(self, texts, labels):
            self.enc = tokenizer(
                list(texts),
                max_length=512,
                padding=True,
                truncation=True,
            )
            self.labels = list(labels)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            item = {k: torch.tensor(v[idx]) for k, v in self.enc.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

    train_loader = DataLoader(_NewsDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(_NewsDataset(X_test, y_test), batch_size=batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )

    best_acc = 0.0
    final_preds, final_labels = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            running_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        avg_loss = running_loss / len(train_loader)

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(**batch).logits
                preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
                trues.extend(batch['labels'].cpu().numpy())

        val_acc = accuracy_score(trues, preds)
        val_f1 = f1_score(trues, preds, average='weighted')
        logger.info(
            "Epoch %d/%d — Loss: %.4f  |  Val Acc: %.4f  |  Val F1: %.4f",
            epoch, epochs, avg_loss, val_acc, val_f1,
        )

        if val_acc > best_acc:
            best_acc = val_acc
            final_preds, final_labels = preds, trues
            os.makedirs(DISTILBERT_PATH, exist_ok=True)
            model.save_pretrained(DISTILBERT_PATH)
            tokenizer.save_pretrained(DISTILBERT_PATH)
            logger.info("  ✓ Checkpoint saved (acc = %.4f)", val_acc)

    best_f1 = f1_score(final_labels, final_preds, average='weighted') if final_preds else 0.0
    logger.info(
        "DistilBERT — Best Val Accuracy: %.4f  |  F1: %.4f", best_acc, best_f1
    )
    logger.info("\n%s", classification_report(final_labels, final_preds, target_names=['Real', 'Fake']))
    return best_acc, best_f1


# ------------------------------------------------------------------ #
# Entry point                                                          #
# ------------------------------------------------------------------ #

def main() -> None:
    parser = argparse.ArgumentParser(description='Train Fake News Detection models.')
    parser.add_argument(
        '--model', choices=['all', 'lr', 'distilbert'], default='all',
        help='Which model(s) to train (default: all)',
    )
    parser.add_argument(
        '--dataset', default='data/WELFake_Dataset.csv',
        help='Path to WELFake CSV (default: data/WELFake_Dataset.csv)',
    )
    parser.add_argument('--test-size', type=float, default=0.2, help='Test split ratio')
    parser.add_argument('--epochs', type=int, default=4, help='DistilBERT fine-tuning epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='DistilBERT batch size')
    args = parser.parse_args()

    dataset_path = args.dataset
    if not os.path.isabs(dataset_path):
        dataset_path = os.path.join(ROOT, dataset_path)
    if not os.path.isfile(dataset_path):
        logger.error("Dataset not found: %s", dataset_path)
        logger.error(
            "Download WELFake from https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification"
            " and place it at backend/data/WELFake_Dataset.csv"
        )
        sys.exit(1)

    df = load_dataset(dataset_path)
    X_raw = list(df['combined'])   # list() avoids PyArrow ChunkedArray.to_numpy()
    y = list(df['label'])

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=args.test_size, random_state=42, stratify=y
    )

    if args.model in ('all', 'lr'):
        preprocessor = TextPreprocessor()
        X_train_clean = preprocess_corpus(X_train_raw, preprocessor)
        X_test_clean = preprocess_corpus(X_test_raw, preprocessor)
        train_lr_tfidf(X_train_clean, y_train, X_test_clean, y_test)

    if args.model in ('all', 'distilbert'):
        train_distilbert(
            X_train_raw, y_train, X_test_raw, y_test,
            epochs=args.epochs, batch_size=args.batch_size,
        )

    logger.info("=== Training complete. ===")


if __name__ == '__main__':
    main()
