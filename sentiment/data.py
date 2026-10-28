"""
Loading labelled text data.

Built-in datasets, included as CSV files in data/ (see data/README.md):
  imdb     data/imdb_reviews.csv.gz 50,000 IMDb reviews, 25,000 positive / 25,000 negative (Maas et al., 2011)
  airline  data/airline_tweets.csv  14,640 airline tweets labelled by people; neutral dropped -> 11,541 (Kaggle)
  airline3 the same file with the 3,099 neutral tweets kept: positive / negative / neutral
  general  positive / negative / neutral on everyday text: TweetEval sentiment (train + validation,
           47,615 general tweets) + airline3 + a 15,000-review sample of IMDb
  tweets   data/tweets.csv          10,000 tweets, 5,000 positive / 5,000 negative (NLTK twitter_samples)
  reviews  data/movie_reviews.csv    2,000 movie reviews, 1,000 positive / 1,000 negative (NLTK movie_reviews)

If a file is missing, tweets and reviews are downloaded through NLTK, IMDb from IMDB_URL and
airline tweets from AIRLINE_URL.

Or bring your own CSV with a text column and a label column.
"""

from pathlib import Path

import pandas as pd

DATASETS = ("general", "imdb", "airline", "airline3", "tweets", "reviews")
# Airline tweets were labelled by people, so emoticons are genuine signal and kept by default.
# (The NLTK tweets were labelled BY their emoticons: keeping them there leaks the label.
#  Other datasets keep the safe default of removing them.)
EMOTICONS_ARE_SIGNAL = {"airline", "airline3", "general"}
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LOCAL_FILES = {"tweets": DATA_DIR / "tweets.csv", "reviews": DATA_DIR / "movie_reviews.csv",
               "imdb": DATA_DIR / "imdb_reviews.csv.gz", "airline": DATA_DIR / "airline_tweets.csv",
               "tweeteval": DATA_DIR / "tweeteval_sentiment.csv.gz"}
# Kaggle-format copy of the Stanford Large Movie Review Dataset (columns: review, sentiment)
IMDB_URL = "https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv"
# Kaggle "Twitter US Airline Sentiment" (Tweets.csv, columns include text and airline_sentiment)
AIRLINE_URL = "https://raw.githubusercontent.com/likarajo/usairline_sentiment/master/Tweets.csv"
# TweetEval (Barbieri et al., 2020), sentiment task from SemEval-2017 Task 4: one text file per split
TWEETEVAL_URL = "https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/sentiment/{split}_{part}.txt"
GENERAL_IMDB_SAMPLE = 15000   # enough reviews to cover that domain without drowning out the tweets


def _read_local(name: str):
    path = LOCAL_FILES[name]
    if path.exists():
        # keep_default_na=False: a text such as "NA" or "null" must stay text, not become NaN
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    return None


def _ensure_corpus(name: str):
    import nltk
    try:
        nltk.data.find(f"corpora/{name}")
    except LookupError:
        print(f"Downloading NLTK corpus '{name}' (one time only)...")
        if not nltk.download(name, quiet=True):
            raise RuntimeError(
                f"Could not download the NLTK corpus '{name}'. Check your internet connection, "
                f"or run: python -m nltk.downloader {name}")


def load_tweets() -> pd.DataFrame:
    local = _read_local("tweets")
    if local is not None:
        return local
    _ensure_corpus("twitter_samples")
    from nltk.corpus import twitter_samples as ts
    pos = ts.strings("positive_tweets.json")
    neg = ts.strings("negative_tweets.json")
    return pd.DataFrame({"text": pos + neg,
                         "label": ["positive"] * len(pos) + ["negative"] * len(neg)})


def load_imdb() -> pd.DataFrame:
    local = _read_local("imdb")
    if local is not None:
        return local
    print("Downloading the IMDb dataset (66 MB, one time only)...")
    df = pd.read_csv(IMDB_URL).rename(columns={"review": "text", "sentiment": "label"})
    df.to_csv(LOCAL_FILES["imdb"], index=False, compression={"method": "gzip", "mtime": 0})
    return df


def load_airline(include_neutral: bool = False) -> pd.DataFrame:
    """Airline tweets labelled by human annotators (Feb 2015). Neutral tweets are dropped by
    default so the task matches the other datasets (positive vs negative)."""
    df = _read_local("airline")
    if df is None:
        print("Downloading the airline tweets dataset (3.4 MB, one time only)...")
        raw = pd.read_csv(AIRLINE_URL)
        df = raw[["text", "airline_sentiment"]].rename(columns={"airline_sentiment": "label"})
        df.to_csv(LOCAL_FILES["airline"], index=False)
    if not include_neutral:
        df = df[df["label"] != "neutral"]
    return df.reset_index(drop=True)


def load_tweeteval(splits=("train", "val")) -> pd.DataFrame:
    """General English tweets labelled positive / negative / neutral by people.
    The official "test" split is kept apart and only used as an extra benchmark."""
    df = _read_local("tweeteval")
    if df is None:
        print("Downloading the TweetEval sentiment dataset (one time only)...")
        import urllib.request
        names = {"0": "negative", "1": "neutral", "2": "positive"}
        parts = []
        for split in ("train", "val", "test"):
            read = lambda part: urllib.request.urlopen(TWEETEVAL_URL.format(split=split, part=part)).read().decode("utf-8")  # noqa: E731
            texts = read("text").rstrip("\n").split("\n")
            labels = [names[x.strip()] for x in read("labels").rstrip("\n").split("\n")]
            parts.append(pd.DataFrame({"text": texts, "label": labels, "split": split}))
        df = pd.concat(parts, ignore_index=True)
        df.to_csv(LOCAL_FILES["tweeteval"], index=False, compression="gzip")
    return df[df["split"].isin(splits)].drop(columns="split").reset_index(drop=True)


def load_general(seed: int = 42) -> pd.DataFrame:
    """Mixed-domain 3-class data with a `source` column, so results can be reported per source."""
    tweets = load_tweeteval().assign(source="tweeteval")
    airline = load_airline(include_neutral=True).assign(source="airline")
    imdb = load_imdb().drop_duplicates(subset="text")
    imdb = imdb.sample(n=min(GENERAL_IMDB_SAMPLE, len(imdb)), random_state=seed).assign(source="imdb")
    return pd.concat([tweets, airline, imdb], ignore_index=True)


def load_reviews() -> pd.DataFrame:
    local = _read_local("reviews")
    if local is not None:
        return local
    _ensure_corpus("movie_reviews")
    from nltk.corpus import movie_reviews as mr
    rows = [(mr.raw(fid), cat) for cat in ("pos", "neg") for fid in mr.fileids(cat)]
    df = pd.DataFrame(rows, columns=["text", "label"])
    df["label"] = df["label"].map({"pos": "positive", "neg": "negative"})
    return df


def load_csv(path, text_col: str = "text", label_col: str = "label") -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    missing = [c for c in (text_col, label_col) if c not in df.columns]
    if missing:
        raise ValueError(f"Column(s) {missing} not in {path.name}. Available: {list(df.columns)}")
    df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})
    df = df.dropna()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    if df["label"].nunique() < 2:
        raise ValueError("The label column needs at least two different classes.")
    return df.reset_index(drop=True)


def load_dataset(name: str = "tweets", csv: str = None, text_col="text", label_col="label") -> pd.DataFrame:
    if csv:
        df = load_csv(csv, text_col, label_col)
    elif name == "imdb":
        df = load_imdb()
    elif name == "airline":
        df = load_airline()
    elif name == "airline3":           # positive / negative / neutral
        df = load_airline(include_neutral=True)
    elif name == "general":            # positive / negative / neutral, several domains
        df = load_general()
    elif name == "tweets":
        df = load_tweets()
    elif name == "reviews":
        df = load_reviews()
    else:
        raise ValueError(f"Unknown dataset '{name}'. Choose one of {DATASETS} or pass --csv.")
    # Exact duplicates would let the same text appear in both train and test
    return df.drop_duplicates(subset="text").reset_index(drop=True)


def prepare(df: pd.DataFrame, cleaner) -> tuple:
    """Remove texts that would distort training or evaluation.

    Uses the same cleaner as the model, so "Thanks for the follow!! :)" and
    "thanks for the follow" count as the same text.
      - empty after cleaning (only emoticons, mentions or links): nothing to learn from
      - same cleaned text with different labels: ambiguous, all copies dropped
      - same cleaned text repeated: keep one copy, otherwise copies can land in
        both train and test and inflate the test score
    """
    df = df.copy()
    df["_clean"] = cleaner.transform(df["text"].tolist())
    stats = {"loaded": len(df)}
    empty = df["_clean"] == ""
    stats["empty_after_cleaning"] = int(empty.sum())
    df = df[~empty]
    n_labels = df.groupby("_clean")["label"].transform("nunique")
    conflict = n_labels > 1
    stats["conflicting_labels"] = int(conflict.sum())
    df = df[~conflict]
    dup = df.duplicated("_clean")
    stats["near_duplicates"] = int(dup.sum())
    df = df[~dup]
    stats["kept"] = len(df)
    return df.drop(columns="_clean").reset_index(drop=True), stats
