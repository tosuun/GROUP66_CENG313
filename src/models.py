"""Baseline training, pairwise prediction and cross-validation."""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline


def prepare_training_texts(pairs):
    texts, labels = [], []
    for _, row in pairs.iterrows():
        texts.extend([row["text_1"], row["text_2"]])
        labels.extend([int(row["real_text_id"] == 1),
                       int(row["real_text_id"] == 2)])
    return texts, labels


def build_tfidf_model():
    return Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42))
    ])


def majority_position(pairs):
    return int(pairs["real_text_id"].mode().iloc[0])


def predict_shorter(pairs, tie_position):
    predictions = []
    for _, row in pairs.iterrows():
        first, second = len(row["text_1"]), len(row["text_2"])
        if first < second:
            predictions.append(1)
        elif second < first:
            predictions.append(2)
        else:
            predictions.append(tie_position)
    return predictions


def predict_pairs(model, pairs, tie_position):
    predictions = []
    real_class_index = list(model.classes_).index(1)
    for _, row in pairs.iterrows():
        probabilities = model.predict_proba([row["text_1"], row["text_2"]])
        first, second = probabilities[:, real_class_index]
        if first > second:
            predictions.append(1)
        elif second > first:
            predictions.append(2)
        else:
            predictions.append(tie_position)
    return predictions


def cross_validate_baselines(pairs, n_splits=5, random_state=42):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    results = []
    for fold, (train_indices, val_indices) in enumerate(
        cv.split(pairs, pairs["real_text_id"]), start=1
    ):
        train = pairs.iloc[train_indices]
        validation = pairs.iloc[val_indices]
        texts, labels = prepare_training_texts(train)
        model = build_tfidf_model()
        model.fit(texts, labels)
        position = majority_position(train)
        actual = validation["real_text_id"]
        results.append({
            "Fold": fold,
            "Majority position": accuracy_score(actual, [position] * len(validation)),
            "Shorter text": accuracy_score(actual, predict_shorter(validation, position)),
            "TF-IDF + Logistic Regression": accuracy_score(
                actual, predict_pairs(model, validation, position)
            )
        })
    return pd.DataFrame(results).set_index("Fold")
