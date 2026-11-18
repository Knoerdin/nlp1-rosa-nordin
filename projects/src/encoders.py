"""Feature representations (text -> vector) for Project 1.

All encoders follow part1.clslib.Encoder: len(encoder) is the dimensionality and
encoder(text) returns a dense numpy vector. part1.loglin / part1.smlp convert
the vectors to sparse matrices themselves.
"""
import numpy as np
from tokenizers import Tokenizer

from part1.clslib import Encoder


class TokenIndicator(Encoder):
    """1 if a token occurs in the text, else 0.

    Same as part1.clslib.TokenIndicator, but __len__ uses self.tokenizer
    (the course version accidentally reads a global variable `tokenizer`).
    """

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self._dim = tokenizer.get_vocab_size()

    def __len__(self) -> int:
        return self._dim

    def __call__(self, x: str) -> np.ndarray:
        vector = np.zeros(self._dim)
        vector[list(set(self.tokenizer.encode(x).ids))] = 1
        return vector


class BagOfWords(Encoder):
    """Token counts: how often each token occurs in the text."""

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self._dim = tokenizer.get_vocab_size()

    def __len__(self) -> int:
        return self._dim

    def __call__(self, x: str) -> np.ndarray:
        return np.bincount(self.tokenizer.encode(x).ids, minlength=self._dim).astype(float)


class TfIdf(Encoder):
    """
    TF-IDF: token counts (as BagOfWords) weighted by inverse document frequency, L2-normalised.

        idf(t)  = ln((1 + N) / (1 + df(t))) + 1        (smoothed, as in scikit-learn)
        x       = counts(text) * idf,  then  x / ||x||

    Tokens that occur in almost every comment ("the", ".") get a low weight, rare and
    informative tokens a high one; the normalisation makes long and short comments comparable.

    Leakage: the IDF weights are *fitted* on data, so they must come from the training part of
    each fold only. Use fit_per_fold() to get one fitted encoder per fold; every CV function
    here accepts such a {fold: encoder} dict wherever it accepts an encoder.
    """

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self._dim = tokenizer.get_vocab_size()
        self.idf = None

    def __len__(self) -> int:
        return self._dim

    def fit(self, texts) -> "TfIdf":
        df = np.zeros(self._dim)
        n = 0
        for text in texts:
            df[list(set(self.tokenizer.encode(text).ids))] += 1
            n += 1
        self.idf = np.log((1 + n) / (1 + df)) + 1
        return self

    def __call__(self, x: str) -> np.ndarray:
        if self.idf is None:
            raise RuntimeError("TfIdf must be fitted first: TfIdf(tokenizer).fit(training_texts)")
        vector = np.bincount(self.tokenizer.encode(x).ids, minlength=self._dim) * self.idf
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector


def fit_per_fold(folds, make_encoder) -> dict:
    """
    {fold: encoder fitted on that fold's training texts only}.

    make_encoder: () -> an unfitted encoder with a .fit(texts) method, e.g. lambda: TfIdf(tokenizer)
    """
    return {fold: make_encoder().fit([text for text, _ in split["training"]]) for fold, split in folds.items()}
