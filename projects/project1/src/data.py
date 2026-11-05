"""Loading our dataset.

Everything downstream (part1.experiment, loglin, smlp) expects a list of
(text: str, label: int) pairs with 0-based integer labels, plus a
part1.clslib.NamedClasses describing the label names.
"""
from part1.clslib import NamedClasses
import os

def load_pairs(split: str = "train"):
    """
    Return (pairs, outcome_space) for the given split.

    TODO: fill in once we have chosen a dataset, e.g. with HuggingFace:

        from datasets import load_dataset
        ds = load_dataset("<name>", split=split)
        names = ds.features["label"].names
        pairs = list(zip(ds["text"], ds["label"]))
        return pairs, NamedClasses(names)
    """
    from datasets import load_dataset
    data = load_dataset("go_emotions", split=split)
    
    raise NotImplementedError("Choose a dataset first (see 01_data_exploration.ipynb)")


def tokenizer_corpus():
    """
    Return a list of raw texts to train the BPE tokeniser on.

    If our dataset is small, use a larger corpus of similar text here
    (the tutorial default is part1.experiment.download_ag_news_texts()).
    """
    raise NotImplementedError
