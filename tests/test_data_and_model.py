"""Tests for data loading, model building, evaluation, explanation and saving."""

import numpy as np
import pandas as pd
import pytest

from sentiment.data import load_csv, load_dataset
from sentiment.model import (ALL_CONFIGS, Config, build_pipeline, cross_validate, evaluate,
                             explain, load_model, predict, save_model, top_features)


# --------------------------------------------------------------------------- #
#  Data loading
# --------------------------------------------------------------------------- #
def test_load_csv_with_custom_columns(toy_csv):
    df = load_csv(toy_csv, text_col="review", label_col="sentiment")
    assert list(df.columns) == ["text", "label"]
    assert set(df.label) == {"positive", "negative"}


def test_load_csv_normalises_labels(tmp_path):
    p = tmp_path / "d.csv"
    pd.DataFrame({"text": ["a b", "c d", None], "label": [" Positive", "NEGATIVE ", "positive"]}).to_csv(p, index=False)
    df = load_csv(p)
    assert sorted(df.label) == ["negative", "positive"]   # stripped, lowercased, NaN row dropped


def test_load_csv_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_csv(tmp_path / "missing.csv")
    p = tmp_path / "bad.csv"
    pd.DataFrame({"body": ["x"], "label": ["positive"]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="not in"):
        load_csv(p)
    p2 = tmp_path / "oneclass.csv"
    pd.DataFrame({"text": ["x", "y"], "label": ["positive", "positive"]}).to_csv(p2, index=False)
    with pytest.raises(ValueError, match="two different classes"):
        load_csv(p2)


def test_duplicates_are_dropped(tmp_path):
    p = tmp_path / "dup.csv"
    pd.DataFrame({"text": ["same", "same", "other"], "label": ["positive", "positive", "negative"]}).to_csv(p, index=False)
    assert len(load_dataset(csv=str(p))) == 2


def test_unknown_dataset():
    with pytest.raises(ValueError, match="Unknown dataset"):
        load_dataset("yelp")


def test_bundled_tweets_csv():
    from sentiment.data import LOCAL_FILES
    assert LOCAL_FILES["tweets"].exists(), "data/tweets.csv is missing"
    df = load_dataset("tweets")
    assert df.label.value_counts().to_dict() == {"positive": 5000, "negative": 5000}
    assert df.text.str.len().min() > 0


def test_bundled_reviews_csv():
    df = load_dataset("reviews")
    assert df.label.value_counts().to_dict() == {"positive": 1000, "negative": 1000}


# --------------------------------------------------------------------------- #
#  Models
# --------------------------------------------------------------------------- #
def test_eight_configurations():
    names = {c.name for c in ALL_CONFIGS}
    assert len(ALL_CONFIGS) == 8 == len(names)
    assert "TF-IDF + Logistic Regression (uni+bigrams)" in names
    assert "CountVectorizer + Naive Bayes (unigrams)" in names


@pytest.mark.parametrize("cfg", ALL_CONFIGS, ids=lambda c: c.name)
def test_every_pipeline_learns_the_toy_data(cfg, toy_df):
    pipe = build_pipeline(cfg).fit(toy_df.text.tolist(), toy_df.label.tolist())
    m = evaluate(pipe, toy_df.text.tolist(), toy_df.label.tolist())
    assert m["accuracy"] > 0.95
    assert set(m) >= {"accuracy", "precision_macro", "recall_macro", "f1_macro", "confusion_matrix"}


def test_cross_validate_returns_mean_and_std(toy_df):
    r = cross_validate(Config("count", "nb", (1, 1)), toy_df.text.tolist(), toy_df.label.tolist(), folds=3)
    assert 0 <= r["cv_f1_std"] < r["cv_f1_mean"] <= 1


def test_confusion_matrix_counts_add_up(trained, toy_df):
    m = evaluate(trained, toy_df.text.tolist(), toy_df.label.tolist())
    assert np.array(m["confusion_matrix"]).sum() == len(toy_df)
    assert m["labels"] == ["negative", "positive"]


@pytest.mark.parametrize("text, label", [
    ("I love this phone, amazing!", "positive"),
    ("the worst, terrible service", "negative"),
])
def test_predict(trained, text, label):
    r = predict(trained, text)
    assert r["label"] == label
    assert abs(sum(r["probabilities"].values()) - 1) < 1e-9
    assert r["confidence"] == max(r["probabilities"].values())


def test_negation_marking_reaches_model(trained):
    # The toy data has no "not_" features, but "not" pulls nothing; check marking reaches the model
    tokens = predict(trained, "not happy at all")["tokens"]
    assert tokens == ["not", "not_happy"]


def test_unknown_words_are_low_confidence(trained):
    r = predict(trained, "zxqv blorp")
    assert r["low_confidence"] is True
    assert r["explanation"] == []


def test_explanation_signs(trained):
    expl = dict(explain(trained, "I love this but the service was terrible", "positive", top=10))
    assert expl["love"] > 0          # supports positive
    assert expl["terrible"] < 0      # works against positive


@pytest.mark.parametrize("clf", ["nb", "lr"])
def test_explanation_for_both_classifiers(clf, toy_df):
    pipe = build_pipeline(Config("count", clf, (1, 1))).fit(toy_df.text.tolist(), toy_df.label.tolist())
    expl = dict(explain(pipe, "amazing day", "positive"))
    assert expl["amazing"] > 0


def test_top_features(trained):
    feats = top_features(trained, 5)
    pos_words = {w for w, _ in feats["positive"]}
    neg_words = {w for w, _ in feats["negative"]}
    assert pos_words & {"love", "great", "amazing", "happy", "excellent", "wonderful", "fantastic", "enjoyed"}
    assert neg_words & {"hate", "terrible", "awful", "sad", "horrible", "boring", "worst", "disappointed"}


def test_save_and_load_roundtrip(trained, tmp_path):
    path = save_model(trained, tmp_path / "m.joblib", {"name": "toy"})
    pipe, meta = load_model(path)
    assert meta["name"] == "toy" and "saved_at" in meta
    assert pipe.predict(["I love it"])[0] == trained.predict(["I love it"])[0]


def test_load_missing_model(tmp_path):
    with pytest.raises(FileNotFoundError, match="python train.py"):
        load_model(tmp_path / "nope.joblib")


def test_prepare_removes_empty_duplicates_and_conflicts():
    from sentiment.data import prepare
    from sentiment.preprocess import TextCleaner
    df = pd.DataFrame({"text": ["Great day!", "great day :)", "@bob :)", "thanks", "Thanks!!", "awful"],
                       "label": ["positive", "positive", "positive", "positive", "negative", "negative"]})
    out, stats = prepare(df, TextCleaner())
    assert out.text.tolist() == ["Great day!", "awful"]
    assert stats == {"loaded": 6, "empty_after_cleaning": 1, "conflicting_labels": 2,
                     "near_duplicates": 1, "kept": 2}



def test_tuning_grid():
    from sentiment.model import Config, tuning_grid
    lr = Config("tfidf", "lr", (1, 2))
    grid = tuning_grid(lr)
    assert len(grid) == 5 and lr not in grid
    assert {c.strength for c in grid} == {1.0, 4.0, 10.0}
    assert any(c.keep_stopwords for c in grid)
    nb = tuning_grid(Config("count", "nb", (1, 1)))
    assert {c.strength for c in nb} == {1.0, 0.3, 0.1}


def test_config_names_show_settings():
    from sentiment.model import Config
    assert Config("tfidf", "lr", (1, 2)).name == "TF-IDF + Logistic Regression (uni+bigrams)"
    assert Config("tfidf", "lr", (1, 2), 4.0, True).name == \
        "TF-IDF + Logistic Regression (uni+bigrams), C=4, stopwords kept"
    assert Config("count", "nb", (1, 1), 0.3).name == "CountVectorizer + Naive Bayes (unigrams), alpha=0.3"


def test_settings_reach_the_pipeline(toy_df):
    from sentiment.model import Config, build_pipeline
    p = build_pipeline(Config("count", "nb", (1, 1), 0.3, True))
    assert p.named_steps["classify"].alpha == 0.3
    assert p.named_steps["clean"].remove_stopwords is False
    p = build_pipeline(Config("tfidf", "lr", (1, 1), 10.0))
    assert p.named_steps["classify"].C == 10.0


def test_bundled_imdb():
    df = load_dataset("imdb")
    assert len(df) > 49000                       # 50,000 minus exact duplicate reviews
    assert set(df.label) == {"positive", "negative"}



def test_bundled_airline_drops_neutral():
    from sentiment.data import load_airline
    full = load_airline(include_neutral=True)
    assert full.label.value_counts().to_dict() == {"negative": 9178, "neutral": 3099, "positive": 2363}
    assert set(load_dataset("airline").label) == {"negative", "positive"}


def test_emoticons_kept_only_where_labels_are_human():
    from sentiment.data import EMOTICONS_ARE_SIGNAL
    assert "airline" in EMOTICONS_ARE_SIGNAL
    assert "tweets" not in EMOTICONS_ARE_SIGNAL      # NLTK tweets were labelled by emoticons


def test_baseline_and_balanced_accuracy(trained, toy_df):
    m = evaluate(trained, toy_df.text.tolist(), toy_df.label.tolist())
    assert m["baseline_accuracy"] == 0.5               # toy data is balanced
    assert 0 <= m["balanced_accuracy"] <= 1


def test_baseline_on_imbalanced_labels(trained):
    m = evaluate(trained, ["I love it"] * 3 + ["awful, terrible"], ["positive"] * 3 + ["negative"])
    assert m["baseline_accuracy"] == 0.75


def test_imbalance_detection():
    from sentiment.model import is_imbalanced
    assert not is_imbalanced(["a", "b"] * 50)
    assert is_imbalanced(["neg"] * 81 + ["pos"] * 19)


def test_balancing_only_tried_on_imbalanced_data():
    from sentiment.model import Config, tuning_grid
    base = Config("count", "nb", (1, 2))
    assert not any(c.balanced for c in tuning_grid(base))
    grid = tuning_grid(base, imbalanced=True)
    assert len(grid) == 11 and any(c.balanced for c in grid)


def test_balanced_settings_reach_the_pipeline():
    from sentiment.model import Config, build_pipeline
    assert build_pipeline(Config("count", "nb", (1, 1), balanced=True)).named_steps["classify"].fit_prior is False
    assert build_pipeline(Config("tfidf", "lr", (1, 1), balanced=True)).named_steps["classify"].class_weight == "balanced"
    assert Config("tfidf", "lr", (1, 1), balanced=True).name.endswith("balanced classes")


def test_airline3_has_three_classes():
    assert set(load_dataset("airline3").label) == {"negative", "neutral", "positive"}


def test_three_class_prediction_and_explanation():
    from sentiment.model import Config, build_pipeline, explain
    texts = ["great flight thanks", "love the crew", "awful delay", "worst service ever",
             "flight leaves at noon", "what time is boarding"] * 10
    labels = ["positive", "positive", "negative", "negative", "neutral", "neutral"] * 10
    pipe = build_pipeline(Config("count", "lr", (1, 1))).fit(texts, labels)
    r = predict(pipe, "what time is the flight")
    assert r["label"] == "neutral" and set(r["probabilities"]) == {"negative", "neutral", "positive"}
    assert explain(pipe, "awful delay", "negative")[0][1] > 0


# --------------------------------------------------------------------------- #
#  General 3-class data and probability calibration
# --------------------------------------------------------------------------- #
def test_bundled_tweeteval_keeps_official_test_apart():
    from sentiment.data import load_tweeteval
    train = load_tweeteval()
    test = load_tweeteval(("test",))
    assert len(train) == 47615 and len(test) == 12284
    assert set(train.label) == set(test.label) == {"negative", "neutral", "positive"}


def test_general_dataset_mixes_sources():
    from sentiment.data import GENERAL_IMDB_SAMPLE
    df = load_dataset("general")
    assert set(df.source) == {"tweeteval", "airline", "imdb"}
    assert (df.source == "imdb").sum() <= GENERAL_IMDB_SAMPLE
    assert set(df.label) == {"negative", "neutral", "positive"}
    assert not set(df[df.source == "imdb"].label) & {"neutral"}


def test_calibration_error():
    from sentiment.model import calibration_error
    classes = ["negative", "positive"]
    proba = np.array([[0.1, 0.9]] * 10)
    assert calibration_error(["positive"] * 9 + ["negative"], proba, classes) == pytest.approx(0.0)
    assert calibration_error(["negative"] * 10, proba, classes) == pytest.approx(0.9)


def test_calibrated_pipeline_predicts_and_explains(toy_df):
    from sklearn.calibration import CalibratedClassifierCV
    from sentiment.model import base_classifier
    for clf in ("nb", "lr"):
        pipe = build_pipeline(Config("count", clf, (1, 1)), calibrate=3).fit(toy_df.text, toy_df.label)
        assert isinstance(pipe.named_steps["classify"], CalibratedClassifierCV)
        assert not isinstance(base_classifier(pipe), CalibratedClassifierCV)
        r = predict(pipe, "I love this, amazing")
        assert r["label"] == "positive"
        assert r["explanation"] and r["explanation"][0][1] > 0
        assert top_features(pipe)["positive"]
        m = evaluate(pipe, toy_df.text, toy_df.label)
        assert 0 <= m["calibration_error"] <= 1 and m["log_loss"] > 0
