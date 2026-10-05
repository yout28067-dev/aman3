"""Train and use a phishing-email detector based on text and URL signals.

Expected CSV columns: ``text`` and ``label``. Labels may be phishing/safe,
phishing/legitimate, or 1/0.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

SUSPICIOUS_WORDS = (
    "urgent", "verify", "password", "suspended", "account", "login", "click",
    "bank", "security alert", "confirm", "limited time", "gift card", "bitcoin",
)
URL_PATTERN = re.compile(r"(?:https?://|www\.)[^\s<>()]+", re.IGNORECASE)
IP_URL_PATTERN = re.compile(r"https?://(?:\d{1,3}\.){3}\d{1,3}", re.IGNORECASE)
SUSPICIOUS_TLDS = (".zip", ".top", ".xyz", ".click", ".ru", ".tk")
SHORTENERS = ("bit.ly", "tinyurl.com", "t.co", "goo.gl", "shorturl.at")


def url_and_keyword_features(texts: pd.Series) -> np.ndarray:
    """Return numerical cues that complement TF-IDF email text features."""
    rows = []
    for raw_text in texts.fillna("").astype(str):
        text = raw_text.lower()
        urls = URL_PATTERN.findall(text)
        rows.append([
            len(urls),
            len(IP_URL_PATTERN.findall(text)),
            sum(tld in url for url in urls for tld in SUSPICIOUS_TLDS),
            sum(shortener in url for url in urls for shortener in SHORTENERS),
            sum(text.count(word) for word in SUSPICIOUS_WORDS),
            text.count("!"),
            text.count("$"),
            sum(character.isupper() for character in raw_text) / max(len(raw_text), 1),
        ])
    return np.asarray(rows, dtype=float)


def normalise_labels(labels: pd.Series) -> np.ndarray:
    values = labels.astype(str).str.strip().str.lower()
    phishing = {"phishing", "1", "true", "malicious", "spam"}
    safe = {"safe", "legitimate", "ham", "0", "false", "benign"}
    unknown = set(values) - phishing - safe
    if unknown:
        raise ValueError(f"Unrecognised label(s): {', '.join(sorted(unknown))}")
    return values.isin(phishing).astype(int).to_numpy()


def build_model() -> Pipeline:
    features = ColumnTransformer([
        ("words", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True), "text"),
        ("characters", TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=1, sublinear_tf=True), "text"),
        ("signals", FunctionTransformer(url_and_keyword_features, validate=False), "text"),
    ])
    return Pipeline([
        ("features", features),
        ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
    ])


def prepare_frame(data: pd.DataFrame) -> pd.DataFrame:
    required = {"text", "label"}
    if not required.issubset(data.columns):
        raise ValueError("Dataset must contain columns named 'text' and 'label'.")
    frame = data.dropna(subset=["text", "label"]).copy()
    frame["text"] = frame["text"].astype(str)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a phishing email classifier.")
    parser.add_argument("--data", default="data/emails.csv", help="CSV containing text,label columns")
    parser.add_argument("--predict", help="Classify one email instead of only evaluating")
    parser.add_argument("--matrix", default="confusion_matrix.png", help="Output PNG for the confusion matrix")
    args = parser.parse_args()

    dataset = pd.read_csv(args.data)
    frame = prepare_frame(dataset)
    y = normalise_labels(frame["label"])
    if len(np.unique(y)) != 2:
        raise ValueError("The dataset needs at least one phishing and one safe email.")

    train, test, y_train, y_test = train_test_split(
        frame[["text"]], y, test_size=0.25, random_state=42, stratify=y
    )
    model = build_model()
    model.fit(train, y_train)
    predictions = model.predict(test)

    print(f"Accuracy: {accuracy_score(y_test, predictions):.1%}")
    print("\nConfusion matrix (rows: actual; columns: predicted; Safe, Phishing):")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))
    print("\n" + classification_report(y_test, predictions, target_names=["Safe", "Phishing"], zero_division=0))

    display = ConfusionMatrixDisplay.from_predictions(
        y_test, predictions, display_labels=["Safe", "Phishing"], cmap="Blues", colorbar=False
    )
    display.ax_.set_title("Phishing Email Detection")
    plt.tight_layout()
    plt.savefig(args.matrix, dpi=160)
    print(f"Confusion-matrix image saved to: {Path(args.matrix).resolve()}")

    if args.predict:
        candidate = pd.DataFrame({"text": [args.predict]})
        probability = model.predict_proba(candidate)[0, 1]
        label = "Phishing" if probability >= 0.5 else "Safe"
        print(f"\nPrediction: {label} (phishing probability: {probability:.1%})")


if __name__ == "__main__":
    main()
