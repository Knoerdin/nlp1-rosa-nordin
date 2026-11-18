from pathlib import Path

import pandas as pd

from part1.clslib import NamedClasses

DATA_DIR = Path(__file__).resolve().parent / "datasets" / "go-emotions"
MIN_RATERS = 2

EMOTIONS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
    "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
    "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism",
    "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral",
]

# Mappings from the official GoEmotions repo (data/ekman_mapping.json, data/sentiment_mapping.json)
EKMAN = {
    "anger": ["anger", "annoyance", "disapproval"],
    "disgust": ["disgust"],
    "fear": ["fear", "nervousness"],
    "joy": ["joy", "amusement", "approval", "excitement", "gratitude", "love", "optimism",
            "relief", "pride", "admiration", "desire", "caring"],
    "sadness": ["sadness", "disappointment", "embarrassment", "grief", "remorse"],
    "surprise": ["surprise", "realization", "confusion", "curiosity"],
    "neutral": ["neutral"],
}
SENTIMENT = {
    "positive": ["amusement", "excitement", "joy", "love", "desire", "optimism", "caring",
                 "pride", "admiration", "gratitude", "relief", "approval"],
    "negative": ["fear", "nervousness", "remorse", "embarrassment", "disappointment", "sadness",
                 "grief", "disgust", "anger", "annoyance", "disapproval"],
    "ambiguous": ["realization", "surprise", "curiosity", "confusion"],
    "neutral": ["neutral"],
}


def load_raw() -> pd.DataFrame:
    """All rater-level rows from the three CSVs."""
    files = sorted(DATA_DIR.glob("goemotions_*.csv"))
    if not files:
        raise FileNotFoundError(f"No goemotions_*.csv in {DATA_DIR}; download them from Kaggle")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def load_comments(min_raters: int = MIN_RATERS) -> pd.DataFrame:
    """
    One row per comment: id, text, n_raters, labels (list of emotion names chosen by
    >= min_raters raters). Comments where no emotion reaches the threshold are dropped.
    """
    raw = load_raw()
    raw = raw[~raw["example_very_unclear"]]
    votes = raw.groupby("id")[EMOTIONS].sum()
    meta = raw.groupby("id").agg(text=("text", "first"), n_raters=("rater_id", "size"))
    labels = votes.apply(lambda row: [e for e in EMOTIONS if row[e] >= min_raters], axis=1)
    comments = meta.assign(labels=labels).reset_index()
    return comments[comments["labels"].str.len() > 0].reset_index(drop=True)


def load_pairs(label_mode: str = "fine", min_raters: int = MIN_RATERS):
    """
    Return (pairs, outcome_space, stats).

    pairs: list of (text, label_id) with 0-based integer labels, ready for
           part1.experiment.cross_validation_split
    outcome_space: part1.clslib.NamedClasses with the class names
    stats: how many comments we kept / dropped as multi-label
    """
    if label_mode == "fine":
        class_names, to_class = EMOTIONS, {e: e for e in EMOTIONS}
    elif label_mode in ("ekman", "sentiment"):
        mapping = EKMAN if label_mode == "ekman" else SENTIMENT
        class_names = list(mapping)
        to_class = {fine: coarse for coarse, fines in mapping.items() for fine in fines}
    else:
        raise ValueError(f"unknown label_mode={label_mode!r}")

    comments = load_comments(min_raters)
    outcome_space = NamedClasses(class_names)
    pairs, dropped = [], 0
    for text, labels in zip(comments["text"], comments["labels"]):
        classes = {to_class[e] for e in labels}
        if len(classes) == 1:
            pairs.append((text, outcome_space.index(classes.pop())))
        else:
            dropped += 1

    stats = {"label_mode": label_mode, "min_raters": min_raters,
             "comments_with_label": len(comments), "kept": len(pairs), "dropped_multilabel": dropped}
    return pairs, outcome_space, stats


def tokenizer_corpus():
    """
    Raw texts to train the BPE tokeniser on: every unique GoEmotions comment.

    Note: this includes the comments that end up in the validation folds. The tokeniser
    never sees labels, but if we want to be strict we can train it per fold on the training
    part only. ~58k short comments may also be small for BPE; the tutorial's AG News
    (part1.experiment.download_ag_news_texts) is bigger but is news, not Reddit language.
    """
    return load_raw()["text"].drop_duplicates().tolist()
