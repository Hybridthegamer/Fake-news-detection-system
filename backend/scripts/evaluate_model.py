#!/usr/bin/env python3
"""
Evaluate trained models on a test split of the dataset.

Usage:
    python scripts/evaluate_model.py --dataset data/WELFake_Dataset.csv
    python scripts/evaluate_model.py --dataset data/WELFake_Dataset.csv --model lr
"""
import argparse
import logging
import os
import pickle
import sys

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nlp.preprocessor import TextPreprocessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, 'saved_models')


def evaluate_lr(X_test_clean, y_test):
    tfidf_path = os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')
    lr_path = os.path.join(MODEL_DIR, 'lr_classifier.pkl')

    if not os.path.isfile(tfidf_path) or not os.path.isfile(lr_path):
        logger.error("LR-TFIDF model not found. Run train_model.py --model lr first.")
        return

    with open(tfidf_path, 'rb') as fh:
        vectorizer = pickle.load(fh)
    with open(lr_path, 'rb') as fh:
        clf = pickle.load(fh)

    X = vectorizer.transform(X_test_clean)
    y_pred = clf.predict(X)
    y_proba = clf.predict_proba(X)[:, 1]

    _print_report("LR-TFIDF", y_test, y_pred, y_proba)


def evaluate_distilbert(X_test_raw, y_test):
    distilbert_path = os.path.join(MODEL_DIR, 'distilbert_finetuned')
    if not os.path.isdir(distilbert_path):
        logger.error("DistilBERT model not found. Run train_model.py --model distilbert first.")
        return

    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
        from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
    except ImportError as exc:
        logger.error("torch/transformers not installed: %s", exc)
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = DistilBertTokenizerFast.from_pretrained(distilbert_path)
    model = DistilBertForSequenceClassification.from_pretrained(distilbert_path)
    model.to(device)
    model.eval()

    class _DS(Dataset):
        def __init__(self, texts, labels):
            self.enc = tokenizer(list(texts), max_length=512, padding=True, truncation=True)
            self.labels = list(labels)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            item = {k: torch.tensor(v[idx]) for k, v in self.enc.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

    loader = DataLoader(_DS(X_test_raw, y_test), batch_size=32)
    preds, trues, probas = [], [], []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            import torch as _t
            p = _t.softmax(logits, dim=-1).cpu().numpy()
            preds.extend(p.argmax(axis=1).tolist())
            probas.extend(p[:, 1].tolist())
            trues.extend(batch['labels'].cpu().numpy().tolist())

    _print_report("DistilBERT", trues, preds, probas)


def _print_report(name, y_true, y_pred, y_proba=None):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n{'='*60}")
    print(f"  {name} Evaluation Report")
    print(f"{'='*60}")
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  F1 (weighted): {f1:.4f}")
    if y_proba is not None:
        auc = roc_auc_score(y_true, y_proba)
        print(f"  ROC-AUC   : {auc:.4f}")
    print(f"\nConfusion Matrix (rows=True, cols=Pred):")
    print(f"              Real    Fake")
    print(f"  True Real:  {cm[0,0]:5d}   {cm[0,1]:5d}")
    print(f"  True Fake:  {cm[1,0]:5d}   {cm[1,1]:5d}")
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['Real', 'Fake']))


def main():
    parser = argparse.ArgumentParser(description='Evaluate Fake News Detection models.')
    parser.add_argument('--model', choices=['all', 'lr', 'distilbert'], default='all')
    parser.add_argument('--dataset', default='data/WELFake_Dataset.csv')
    parser.add_argument('--test-size', type=float, default=0.2)
    args = parser.parse_args()

    dataset_path = args.dataset
    if not os.path.isabs(dataset_path):
        dataset_path = os.path.join(ROOT, dataset_path)

    df = pd.read_csv(dataset_path)
    if 'title' in df.columns and 'text' in df.columns:
        df['combined'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
    else:
        df['combined'] = df['text'].fillna('')
    df = df[['combined', 'label']].dropna()
    df['combined'] = df['combined'].astype(object)  # avoid PyArrow on long strings
    df['label'] = df['label'].astype(int)

    _, X_test_raw, _, y_test = train_test_split(
        list(df['combined']), list(df['label']),
        test_size=args.test_size, random_state=42, stratify=list(df['label'])
    )

    if args.model in ('all', 'lr'):
        preprocessor = TextPreprocessor()
        X_test_clean = [preprocessor.preprocess(t) for t in X_test_raw]
        evaluate_lr(X_test_clean, y_test)

    if args.model in ('all', 'distilbert'):
        evaluate_distilbert(X_test_raw, y_test)


if __name__ == '__main__':
    main()
