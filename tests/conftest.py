"""Shared fixtures. Tests use a small synthetic dataset, so no download is needed."""

import random
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

POS_WORDS = ["love", "great", "amazing", "happy", "excellent", "wonderful", "fantastic", "enjoyed"]
NEG_WORDS = ["hate", "terrible", "awful", "sad", "horrible", "boring", "worst", "disappointed"]
SUBJECTS = ["this phone", "the movie", "my day", "the service", "this song", "the food", "our trip"]
TEMPLATES = ["I {w} {s}", "{s} was {w}", "really {w}, {s}", "{s} is so {w}!", "honestly {w} about {s}"]


def make_toy_df(n_per_class=120, seed=0) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    for label, words in (("positive", POS_WORDS), ("negative", NEG_WORDS)):
        # every template x word x subject combination is a distinct sentence (280 per class)
        combos = [(t, w, s) for t in TEMPLATES for w in words for s in SUBJECTS]
        for t, w, s in rng.sample(combos, n_per_class):
            rows.append((t.format(w=w, s=s), label))
    return pd.DataFrame(rows, columns=["text", "label"])


@pytest.fixture
def toy_df():
    return make_toy_df()


@pytest.fixture
def toy_csv(tmp_path, toy_df):
    path = tmp_path / "toy.csv"
    toy_df.rename(columns={"text": "review", "label": "sentiment"}).to_csv(path, index=False)
    return path


@pytest.fixture
def trained(toy_df):
    from sentiment.model import Config, build_pipeline
    pipe = build_pipeline(Config("tfidf", "lr", (1, 2)))
    pipe.fit(toy_df.text.tolist(), toy_df.label.tolist())
    return pipe
