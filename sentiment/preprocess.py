"""
Text preprocessing: cleaning and tokenisation.

TextCleaner is a scikit-learn transformer, so the exact cleaning settings are
saved inside the trained model and applied identically at prediction time.
"""

import html
import re
import unicodedata

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Negations carry sentiment ("not good"), so they must survive stopword removal.
NEGATIONS = {"no", "not", "nor", "never", "none", "nothing", "nobody", "neither", "nowhere", "cannot"}
STOPWORDS = frozenset(ENGLISH_STOP_WORDS) - NEGATIONS

URL_RE = re.compile(r"https?://\S+|www\.\S+")
# Real HTML tags only (IMDb reviews contain "<br />"); "<3" and "a < b" are left alone
HTML_TAG_RE = re.compile(r"</?\s*(?:br|p|i|b|em|strong|div|span|a)\b[^>]*>", re.I)
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
# Western emoticons such as :) :-( ;D =P :/ <3 and emoji. The NLTK tweet corpus was
# labelled *from* these emoticons, so keeping them leaks the answer into the input.
_EYES = r"(?:(?<!\w)[xX8]|[:;=])"                      # x/X/8 only count as eyes when not inside a word
_NOSE = r"[\-o\*']?"
_MOUTH = r"(?:[\)\]\(\[\|@/\\]+|[dDpP3](?![a-zA-Z0-9]))"  # letter mouths must not start a word: "xD" yes, "explain" no
EMOTICON_RE = re.compile(
    _EYES + _NOSE + _MOUTH                                # :)  :-(  ;D  =P  xD  :/  :3
    + r"|[\)\]\(\[]+" + _NOSE + r"[:;=](?!\w)"          # (:  ):  reversed faces
    + r"|<3"
    + r"|[\U0001F300-\U0001FAFF\u2600-\u27BF]"           # emoji
)
ELONGATED_RE = re.compile(r"(\w)\1{2,}")            # "soooo" -> "soo"
CONTRACTION_RE = re.compile(r"n't\b")               # "don't" -> "do not"
CLAUSE_RE = re.compile(r"[.,!?;:]+|\bbut\b")          # "not bad, but..." ends the negation
TOKEN_RE = re.compile(r"emo_[a-z_]+|[a-z]+(?:'[a-z]+)?")

# Readable names for kept emoticons (only used when remove_emoticons=False)
EMOTICON_NAMES = {")": "smile", "]": "smile", "D": "grin", "d": "grin", "(": "frown", "[": "frown",
                  "P": "tongue", "p": "tongue", "/": "skeptical", "\\": "skeptical", "|": "neutral",
                  "@": "shock", "3": "kiss"}


def emoticon_token(match) -> str:
    """':)' -> ' emo_smile ', ';)' -> ' emo_wink_smile ', emoji -> ' emo_<unicode name> '."""
    e = match.group(0)
    if e == "<3":
        return " emo_heart "
    if len(e) == 1 and ord(e) > 0x2000:
        name = unicodedata.name(e, "emoji").lower()
        return " emo_" + re.sub(r"[^a-z]+", "_", name).strip("_") + " "
    if e[0] in ")]([":                                   # reversed face: "(:" smiles, "):" frowns
        mouth = "smile" if e[0] in "([" else "frown"
    else:
        mouth = next((EMOTICON_NAMES[c] for c in e if c in EMOTICON_NAMES), "face")
    return f" emo_{'wink_' if ';' in e else ''}{mouth} "


def fold_accents(text: str) -> str:
    """'BELİEVE' -> 'believe', 'café' -> 'cafe'. Without this, the combining dot that
    lowercasing adds to Turkish 'İ' splits words into fragments like 'beli' + 'eve'."""
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


class TextCleaner(BaseEstimator, TransformerMixin):
    """Clean raw text and return space-joined tokens.

    Steps (each one can be switched off):
      1. unescape HTML entities (&amp; -> &) and remove HTML tags (<br />)
      2. remove URLs, @mentions; keep hashtag words (#happy -> happy)
      3. remove emoticons and emoji (on by default, see EMOTICON_RE)
      4. lowercase, fold accents (é -> e, İ -> i), expand n't -> not, shorten elongated words
      5. tokenise into words, drop stopwords (negations are kept)
      6. negation marking: words after "not/no/never..." get a not_ prefix
         until punctuation or `negation_scope` words ("not happy" -> not not_happy)
      7. optional Porter stemming
    """

    def __init__(self, remove_emoticons=True, remove_stopwords=True, mark_negation=True,
                 negation_scope=3, stem=False, min_token_len=2):
        self.remove_emoticons = remove_emoticons
        self.mark_negation = mark_negation
        self.negation_scope = negation_scope
        self.remove_stopwords = remove_stopwords
        self.stem = stem
        self.min_token_len = min_token_len

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [" ".join(self.tokenize(text)) for text in X]

    def tokenize(self, text) -> list:
        text = html.unescape(str(text))
        text = HTML_TAG_RE.sub(" ", text)
        text = URL_RE.sub(" ", text)
        text = MENTION_RE.sub(" ", text)
        text = HASHTAG_RE.sub(r" \1 ", text)
        # An emoticon ends a clause like punctuation does: "don't like this :( so sad" must not negate "sad".
        # A removed one is replaced by " . " so the negation scope still stops there.
        text = EMOTICON_RE.sub(" . " if self.remove_emoticons else emoticon_token, text)
        text = fold_accents(text.lower().replace("’", "'"))
        text = CONTRACTION_RE.sub(" not", text)
        text = ELONGATED_RE.sub(r"\1\1", text)
        if self.mark_negation:
            tokens = []
            for clause in CLAUSE_RE.split(text):   # negation never crosses punctuation
                tokens += self._negate(self._words(clause))
        else:
            tokens = self._words(text)
        if self.stem:
            tokens = [_stem(t) for t in tokens]
        return tokens

    def _words(self, text) -> list:
        tokens = [t for t in TOKEN_RE.findall(text) if len(t) >= self.min_token_len or t in NEGATIONS]
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in STOPWORDS]
        return tokens

    def _negate(self, tokens) -> list:
        out, left = [], 0
        for t in tokens:
            if t in NEGATIONS:
                out.append(t)
                left = self.negation_scope
            elif t.startswith("emo_"):            # a kept emoticon also ends the scope
                out.append(t)
                left = 0
            elif left:
                out.append("not_" + t)
                left -= 1
            else:
                out.append(t)
        return out


_STEMMER = None


def _stem(token: str) -> str:
    global _STEMMER
    if _STEMMER is None:
        from nltk.stem import PorterStemmer  # needs no downloaded data
        _STEMMER = PorterStemmer()
    prefix = "not_" if token.startswith("not_") else ""
    return prefix + _STEMMER.stem(token[len(prefix):])
