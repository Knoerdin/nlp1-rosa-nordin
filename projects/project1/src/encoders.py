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


# TODO: our third representation, e.g. TF-IDF, n-grams, normalised counts, embeddings.
# Watch out for leakage: anything *fitted* on data (like IDF weights) must be
# fitted on the training folds only, not on the whole dataset.
class ThirdRepresentation(Encoder):
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self._dim = tokenizer.get_vocab_size()

    def __len__(self) -> int:
        return self._dim

    def __call__(self, x: str) -> np.ndarray:
        raise NotImplementedError
