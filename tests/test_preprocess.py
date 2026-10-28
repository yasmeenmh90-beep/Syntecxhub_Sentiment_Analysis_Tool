"""Tests for text cleaning and tokenisation."""

import pytest

from sentiment.preprocess import STOPWORDS, TextCleaner


@pytest.fixture
def clean():
    return TextCleaner().tokenize


def test_urls_mentions_and_hashtags(clean):
    assert clean("@bob check https://x.co/abc and www.site.com #Happy") == ["check", "happy"]


def test_html_entities(clean):
    assert clean("fish &amp; chips") == ["fish", "chips"]


def test_lowercase_and_elongation(clean):
    assert clean("SOOOO GOOOOD") == ["soo", "good"]


def test_contractions_become_negations(clean):
    assert clean("I don't like it") == ["not", "not_like"]


def test_negations_survive_stopword_removal():
    for word in ("not", "no", "never", "nothing"):
        assert word not in STOPWORDS
    assert "the" in STOPWORDS


def test_empty_and_symbol_only_text(clean):
    assert clean("") == []
    assert clean("!!! ??? 123") == []


# --------------------------------------------------------------------------- #
#  Emoticons (the tweet labels were made from them, so they must be removed)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text", [":)", ":-)", ":(", ";D", "=P", ":/", "<3", "xD", "XD", "(:", "):", "😊", ":p"])
def test_emoticons_are_removed(clean, text):
    assert clean(f"good {text} day") == ["good", "day"]


@pytest.mark.parametrize("text, expected", [
    ("I expect to explain the sixpack", ["expect", "explain", "sixpack"]),
    ("my word: hello", ["word", "hello"]),
    ("meet at 3:30", ["meet"]),
    ("Dear Paul, Darn", ["dear", "paul", "darn"]),
    ("boxdrop and xperia", ["boxdrop", "xperia"]),
])
def test_emoticon_pattern_does_not_damage_words(clean, text, expected):
    """Regression test: an earlier pattern turned "explain" into "lain"."""
    assert clean(text) == expected


def test_emoticons_kept_as_named_tokens():
    keep = TextCleaner(remove_emoticons=False).tokenize
    assert keep("great :) bad :( ;) <3 (:") == ["great", "emo_smile", "bad", "emo_frown",
                                                "emo_wink_smile", "emo_heart", "emo_smile"]
    assert keep("ok 😊")[1].startswith("emo_smiling_face")


def test_emoticon_glued_to_words(clean):
    assert clean("great:)thanks") == ["great", "thanks"]


# --------------------------------------------------------------------------- #
#  Negation marking
# --------------------------------------------------------------------------- #
def test_negation_marks_following_words(clean):
    assert clean("not happy with this at all") == ["not", "not_happy"]


def test_negation_stops_at_punctuation(clean):
    assert clean("not bad, actually fun") == ["not", "not_bad", "actually", "fun"]


def test_negation_stops_at_but(clean):
    assert clean("I don't like it but the music is great") == ["not", "not_like", "music", "great"]


def test_negation_scope_limit():
    tok = TextCleaner(negation_scope=2).tokenize
    assert tok("never liked movie plot actors") == ["never", "not_liked", "not_movie", "plot", "actors"]


def test_negation_can_be_switched_off():
    assert TextCleaner(mark_negation=False).tokenize("not happy") == ["not", "happy"]


def test_stemming_keeps_negation_prefix():
    assert TextCleaner(stem=True).tokenize("not loving these movies") == ["not", "not_love", "not_movi"]


def test_transform_returns_joined_strings():
    assert TextCleaner().transform(["Great day!", "not good"]) == ["great day", "not not_good"]


def test_cleaner_settings_are_sklearn_params():
    c = TextCleaner(remove_emoticons=False, negation_scope=5)
    assert c.get_params()["remove_emoticons"] is False
    assert c.get_params()["negation_scope"] == 5


def test_accent_folding_keeps_words_whole(clean):
    """Regression test: Turkish dotted İ used to split 'BELİEVE' into 'beli' + 'eve'."""
    assert clean("I BELİEVE HE WİLL FOLLOW") == ["believe", "follow"]
    assert clean("café naïve") == ["cafe", "naive"]



def test_html_tags_removed_but_hearts_kept():
    assert TextCleaner().tokenize("Great film.<br /><br />Loved it") == ["great", "film", "loved"]
    assert TextCleaner(remove_emoticons=False).tokenize("i <3 it") == ["emo_heart"]
    assert TextCleaner().tokenize("<b>Bold</b> acting") == ["bold", "acting"]


def test_emoticon_ends_negation_scope():
    text = "I don't like this :( so disappointed"
    assert TextCleaner().tokenize(text) == ["not", "not_like", "disappointed"]
    assert TextCleaner(remove_emoticons=False).tokenize(text) == ["not", "not_like", "emo_frown", "disappointed"]
