# Datasets

All files are CSV with a `text` and a `label` column (`positive` / `negative`, plus `neutral` where the source has it), so training works offline.

| File | Rows | Labels | Source |
|---|---|---|---|
| `imdb_reviews.csv.gz` | 50,000 | 25,000 positive, 25,000 negative | Stanford Large Movie Review Dataset (IMDb), gzip-compressed (26 MB) |
| `airline_tweets.csv` | 14,640 | 9,178 negative, 3,099 neutral, 2,363 positive (`airline` drops neutral, `airline3` keeps it) | Twitter US Airline Sentiment (Kaggle), tweets about six US airlines, February 2015 |
| `tweeteval_sentiment.csv.gz` | 59,899 | train 45,615 / validation 2,000 / test 12,284; 11,377 negative, 27,479 neutral, 21,043 positive | TweetEval sentiment task (SemEval-2017 Task 4), general English tweets; extra `split` column |
| `tweets.csv` | 10,000 | 5,000 positive, 5,000 negative | NLTK `twitter_samples` (English tweets, 2015) |
| `movie_reviews.csv` | 2,000 | 1,000 positive, 1,000 negative | NLTK `movie_reviews` (Polarity dataset v2.0, reviews from the IMDb archive of rec.arts.movies.reviews) |

## Things to know before using them

- **Tweet labels come from emoticons.** The corpus documentation says they "were collected in July 2015 by searching against" lists of positive emoticons (`:)`, `:D`, `<3`…) and negative ones (`:(`, `:'(`…), so about 97 % of positive tweets contain a smiley and 99 % of negative tweets contain a frown. Any model that keeps emoticons reads the label back (99 % accuracy). `TextCleaner` removes them by default.
- **Airline tweets come in two versions:** `--dataset airline` (positive vs negative, neutral dropped) and `--dataset airline3` (all three labels).
- **The `general` dataset** (`--dataset general`) combines TweetEval train + validation, all airline tweets (neutral kept) and a fixed random sample of 15,000 IMDb reviews (seed 42). The TweetEval **test** split is never used for training; `train.py` reports it as a separate benchmark.
- **Airline tweets were labelled by people**, so their emoticons are genuine signal and are kept for this dataset. 81 % of the positive/negative tweets are negative: report macro-F1 and balanced accuracy, not only accuracy.
- **The tweet file contains near-duplicate spam** (for example "followed thanks followed" 51 times) and some identical texts with both labels. `train.py` removes these before splitting; the raw file is left unchanged.
- **IMDb reviews contain HTML line breaks** (`<br />` in 29,200 of the 50,000 reviews); `TextCleaner` strips HTML tags. 418 reviews are exact duplicates, which `load_dataset` removes before anything else.
- **Movie reviews are long** (746 words on average, median 696) and already lowercased, with punctuation separated by spaces.

## Sources and citation

- IMDb: Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng and Christopher Potts, *Learning Word Vectors for Sentiment Analysis*, Proceedings of ACL 2011. Original: https://ai.stanford.edu/~amaas/data/sentiment/. The file here is the widely used single-CSV version of the same 50,000 labelled reviews (train and test combined).
- Airline tweets: "Twitter US Airline Sentiment", released by CrowdFlower (now part of Appen) in its *Data for Everyone* collection and published on Kaggle. Contributors labelled each tweet as positive, negative or neutral. The file here keeps only the `text` and `airline_sentiment` (renamed `label`) columns of the original `Tweets.csv`. See the Kaggle page for the licence; it is used here for non-commercial, educational purposes.
- TweetEval: Francesco Barbieri, Jose Camacho-Collados, Leonardo Neves and Luis Espinosa-Anke, *TweetEval: Unified Benchmark and Comparative Evaluation for Tweet Classification*, Findings of EMNLP 2020; sentiment data from Sara Rosenthal, Noura Farra and Preslav Nakov, *SemEval-2017 Task 4: Sentiment Analysis in Twitter*. Files from https://github.com/cardiffnlp/tweeteval (labels 0/1/2 mapped to negative/neutral/positive). The repository says TweetEval is released without restrictions, but the original datasets' licences and Twitter's terms still apply.
- Tweets: NLTK Twitter samples corpus, distributed with the [NLTK data collection](https://www.nltk.org/nltk_data/).
- Movie reviews: Bo Pang and Lillian Lee, *A Sentimental Education: Sentiment Analysis Using Subjectivity Summarization Based on Minimum Cuts*, Proceedings of ACL 2004. Distributed with NLTK as `movie_reviews`.

The tweet and movie_reviews files are exact exports of the NLTK corpora. These datasets are included for educational use in this internship project. Tweet text remains subject to the original platform's terms.

## Regenerating the files

```python
import nltk
from sentiment.data import LOCAL_FILES
# delete data/*.csv first, then the loaders fall back to NLTK:
from sentiment.data import load_tweets, load_reviews
load_tweets().to_csv(LOCAL_FILES["tweets"], index=False)
load_reviews().to_csv(LOCAL_FILES["reviews"], index=False)
```
