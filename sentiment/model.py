"""
Feature extraction, classifiers, evaluation, explanation and persistence.

Every model is one scikit-learn Pipeline:  TextCleaner -> vectorizer -> classifier
so cleaning, vocabulary and weights are saved together in a single file.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, log_loss, precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from .preprocess import TextCleaner

VECTORIZERS = {"count": "CountVectorizer", "tfidf": "TF-IDF"}
CLASSIFIERS = {"nb": "Naive Bayes", "lr": "Logistic Regression"}
NGRAMS = {(1, 1): "unigrams", (1, 2): "uni+bigrams"}
LOW_CONFIDENCE = 0.65  # below this the CLI flags the prediction as uncertain


@dataclass(frozen=True)
class Config:
    vectorizer: str               # "count" | "tfidf"
    classifier: str               # "nb" | "lr"
    ngram: tuple                  # (1, 1) | (1, 2)
    strength: float = 1.0         # Logistic Regression C, or Naive Bayes alpha
    keep_stopwords: bool = False  # stopwords can carry sentiment in short texts ("i'm", "you")
    balanced: bool = False        # weight classes equally (LR class_weight, NB uniform prior)

    @property
    def name(self) -> str:
        base = f"{VECTORIZERS[self.vectorizer]} + {CLASSIFIERS[self.classifier]} ({NGRAMS[self.ngram]})"
        extras = []
        if self.strength != 1.0:
            extras.append(f"{'C' if self.classifier == 'lr' else 'alpha'}={self.strength:g}")
        if self.keep_stopwords:
            extras.append("stopwords kept")
        if self.balanced:
            extras.append("balanced classes")
        return base + (", " + ", ".join(extras) if extras else "")


# Stage 1: the 8 feature/classifier combinations with default settings
ALL_CONFIGS = [Config(v, c, n) for v in VECTORIZERS for c in CLASSIFIERS for n in NGRAMS]

# Stage 2: settings tried on the best stage-1 combination
STRENGTHS = {"lr": (1.0, 4.0, 10.0),   # C: higher = less regularisation
             "nb": (1.0, 0.3, 0.1)}    # alpha: lower = less smoothing


IMBALANCE_THRESHOLD = 0.6   # largest class above 60 % of the data -> also try class balancing


def is_imbalanced(labels) -> bool:
    values, counts = np.unique(np.asarray(labels), return_counts=True)
    return counts.max() / counts.sum() > IMBALANCE_THRESHOLD


def tuning_grid(base: Config, imbalanced: bool = False) -> list:
    """Variants of `base` with other regularisation strengths and with stopwords kept.
    On imbalanced data, each variant is also tried with balanced class weights."""
    balance_options = (False, True) if imbalanced else (False,)
    grid = [Config(base.vectorizer, base.classifier, base.ngram, s, keep, bal)
            for s in STRENGTHS[base.classifier] for keep in (False, True) for bal in balance_options]
    return [c for c in grid if c != base]


def build_pipeline(config: Config, remove_emoticons: bool = True, seed: int = 42, calibrate: int = 0) -> Pipeline:
    """calibrate=k > 0 wraps the classifier in sigmoid calibration fitted with k-fold
    cross-validation, so a reported 80 % confidence is right about 80 % of the time."""
    # Text is already cleaned and space-joined, so tokens are just non-space runs.
    # min_df=2 drops words seen only once (mostly typos and usernames).
    common = dict(token_pattern=r"\S+", ngram_range=config.ngram, min_df=2)
    if config.vectorizer == "tfidf":
        vectorizer = TfidfVectorizer(sublinear_tf=True, **common)
    else:
        vectorizer = CountVectorizer(**common)
    if config.classifier == "nb":
        # fit_prior=False: every class gets the same prior instead of its share of the data
        clf = MultinomialNB(alpha=config.strength, fit_prior=not config.balanced)
    else:
        clf = LogisticRegression(C=config.strength, max_iter=3000, random_state=seed,
                                 class_weight="balanced" if config.balanced else None)
    if calibrate:
        # ensemble=False: one classifier trained on all the data, calibrated on out-of-fold predictions
        clf = CalibratedClassifierCV(clf, method="sigmoid", cv=calibrate, ensemble=False)
    return Pipeline([
        ("clean", TextCleaner(remove_emoticons=remove_emoticons, remove_stopwords=not config.keep_stopwords)),
        ("vectorize", vectorizer),
        ("classify", clf),
    ])


def cross_validate(config, X, y, folds=5, seed=42, remove_emoticons=True, n_jobs=1) -> dict:
    """Macro-F1 with stratified k-fold cross-validation on the training set only."""
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = cross_val_score(build_pipeline(config, remove_emoticons, seed), X, y,
                             cv=cv, scoring="f1_macro", n_jobs=n_jobs)
    return {"config": config, "cv_f1_mean": float(scores.mean()), "cv_f1_std": float(scores.std())}


def cv_log_loss(config, X, y, folds=5, seed=42, remove_emoticons=True, n_jobs=1, calibrate=False) -> float:
    """Cross-validated log loss on the training set, with or without calibration.
    Used to decide whether calibration actually helps a model (it does not always)."""
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    pipe = build_pipeline(config, remove_emoticons, seed, calibrate=folds if calibrate else 0)
    return float(-cross_val_score(pipe, X, y, cv=cv, scoring="neg_log_loss", n_jobs=n_jobs).mean())


def calibration_error(y_true, proba, classes, bins: int = 10) -> float:
    """Expected calibration error: average gap between confidence and actual accuracy
    over confidence bins (0 = perfectly calibrated)."""
    proba = np.asarray(proba)
    conf = proba.max(axis=1)
    correct = np.asarray(classes)[proba.argmax(axis=1)] == np.asarray(y_true)
    edges = np.linspace(0, 1, bins + 1)
    error = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf > lo) & (conf <= hi)
        if in_bin.any():
            error += in_bin.mean() * abs(correct[in_bin].mean() - conf[in_bin].mean())
    return float(error)


def evaluate(pipeline, X_test, y_test) -> dict:
    proba = pipeline.predict_proba(X_test)
    labels = list(pipeline.classes_)
    y_pred = np.asarray(labels)[proba.argmax(axis=1)]
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        # accuracy of always predicting the most common class: the bar a model must beat
        "baseline_accuracy": float(max(np.mean(np.array(y_test) == c) for c in labels)),
        "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "log_loss": float(log_loss(y_test, proba, labels=labels)),
        "calibration_error": calibration_error(y_test, proba, labels),
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=labels).tolist(),
        "report": classification_report(y_test, y_pred, labels=labels, output_dict=True, zero_division=0),
    }


# --------------------------------------------------------------------------- #
#  Prediction + explanation
# --------------------------------------------------------------------------- #
def predict(pipeline, text: str) -> dict:
    proba = pipeline.predict_proba([text])[0]
    classes = list(pipeline.classes_)
    best = int(np.argmax(proba))
    tokens = pipeline.named_steps["clean"].tokenize(text)
    return {
        "text": text,
        "label": classes[best],
        "confidence": float(proba[best]),
        "probabilities": {c: float(p) for c, p in zip(classes, proba)},
        "tokens": tokens,
        "low_confidence": bool(proba[best] < LOW_CONFIDENCE or not tokens),
        "explanation": explain(pipeline, text, classes[best]),
    }


def base_classifier(pipeline):
    """The Naive Bayes / Logistic Regression model, unwrapped from calibration if present."""
    clf = pipeline.named_steps["classify"]
    if isinstance(clf, CalibratedClassifierCV):
        clf = clf.calibrated_classifiers_[0].estimator
    return clf


def explain(pipeline, text: str, label: str, top: int = 5) -> list:
    """Which words pushed the prediction towards (+) or away from (-) `label`.

    contribution = (weight of the feature for this class) x (feature value in this text)
    For Logistic Regression the weight is the coefficient; for Naive Bayes it is the
    log-probability of the word in this class minus its average over the other classes.
    """
    cleaned = pipeline.named_steps["clean"].transform([text])
    vec = pipeline.named_steps["vectorize"]
    clf = base_classifier(pipeline)
    x = vec.transform(cleaned).tocoo()
    if x.nnz == 0:
        return []
    classes = list(clf.classes_)
    k = classes.index(label)
    if isinstance(clf, LogisticRegression):
        if clf.coef_.shape[0] == 1:          # binary: one row, pointing at classes_[1]
            weights = clf.coef_[0] * (1 if k == 1 else -1)
        else:
            weights = clf.coef_[k]
    else:
        logp = clf.feature_log_prob_
        others = np.delete(logp, k, axis=0).mean(axis=0)
        weights = logp[k] - others
    names = vec.get_feature_names_out()
    contribs = [(names[j], float(weights[j] * v)) for j, v in zip(x.col, x.data)]
    contribs.sort(key=lambda c: abs(c[1]), reverse=True)
    return contribs[:top]


def top_features(pipeline, n: int = 15) -> dict:
    """Most indicative words per class (binary models)."""
    vec = pipeline.named_steps["vectorize"]
    clf = base_classifier(pipeline)
    names = vec.get_feature_names_out()
    classes = list(clf.classes_)
    if isinstance(clf, LogisticRegression) and clf.coef_.shape[0] == 1:
        w = clf.coef_[0]
    elif isinstance(clf, MultinomialNB) and len(classes) == 2:
        w = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
    else:
        return {}
    order = np.argsort(w)
    return {
        classes[1]: [(names[i], float(w[i])) for i in order[::-1][:n]],
        classes[0]: [(names[i], float(-w[i])) for i in order[:n]],
    }


# --------------------------------------------------------------------------- #
#  Saving and loading
# --------------------------------------------------------------------------- #
def save_model(pipeline, path, metadata: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "metadata": {**metadata, "saved_at": datetime.now().isoformat(timespec="seconds")}}, path)
    return path


def load_model(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No trained model at {path}. Train one first: python train.py")
    bundle = joblib.load(path)
    return bundle["pipeline"], bundle["metadata"]
