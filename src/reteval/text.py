"""Tokenizer for the lexical retrievers (TF-IDF and BM25)."""

import re
from functools import lru_cache

import Stemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

TOKEN = re.compile(r"[a-z0-9]+")


@lru_cache(maxsize=1)
def _stemmer() -> Stemmer.Stemmer:
    return Stemmer.Stemmer("english")


def tokenize(text: str, stem: bool = True) -> list[str]:
    """Lowercase, split into words, drop stop words, and optionally stem ("cells" -> "cell")."""
    words = [w for w in TOKEN.findall(text.lower()) if w not in ENGLISH_STOP_WORDS]
    return _stemmer().stemWords(words) if stem else words
