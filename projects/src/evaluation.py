"""
Evaluation and comparison helpers for Project 2 (embed-and-pool vs. LSTM vs. Project 1).

Reads saved results only; nothing here trains a model. Two kinds of saved results:

1. Fold scores: results/<name>.json, written by `save_results` in embedding.ipynb
   (same format as Project 1's results_io.save_cv). Enough for the overview, the learning
   curves and the Bayesian fold-level comparison.
2. Per-comment predictions: results/predictions/<name>.npz, written by `save_predictions` below.
   Needed for macro-F1, per-class scores, confusion matrices, the paired McNemar test and
   the error analysis. Call it right after training, with the dict `cross_validate` returns:

       from src import evaluation
       evaluation.save_predictions(name, out, folds)

All systems use Project 1's folds (same pairs, k and seed), so every comment is predicted
exactly once per system, by the model that did not see it during training.
"""
import json
import re
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator, PercentFormatter
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from part1.analysis import BenavoliCorrelatedT

from .config import REPO_ROOT

PROJECTS_DIR = REPO_ROOT / "projects"
RESULTS_DIRS = {"P1": PROJECTS_DIR / "project1" / "results", "P2": PROJECTS_DIR / "project2" / "results"}
PREDICTIONS_DIR = RESULTS_DIRS["P2"] / "predictions"
FIGURES_DIR = PROJECTS_DIR / "project2" / "figures"


# ---------------------------------------------------------------- saving predictions

def save_predictions(name, out, folds, directory=None):
    """
    Store each fold's best-epoch validation predictions together with the comment ids
    (positions in `pairs`), so systems can be compared comment by comment.

    out:   the dict returned by cross_validate / run_experiment; out["predictions"][fold] follows
           the order of folds[fold]["validation_ids"] (the validation loader does not shuffle)
    folds: the dict from part1.experiment.cross_validation_split
    """
    arrays = {}
    for fold, preds in out["predictions"].items():
        ids = np.asarray(folds[fold]["validation_ids"], dtype=int)
        preds = np.asarray(np.concatenate([np.ravel(p) for p in preds]) if isinstance(preds, list) else preds,
                           dtype=int).ravel()
        if len(ids) != len(preds):
            raise ValueError(f"fold {fold}: {len(preds)} predictions for {len(ids)} validation comments")
        arrays[f"ids_{fold}"], arrays[f"pred_{fold}"] = ids, preds
    directory = directory or PREDICTIONS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.npz"
    np.savez_compressed(path, **arrays)
    return path


def load_predictions(name, n_items, directory=None):
    """-> (pred, fold): two int arrays of length n_items, indexed like `pairs`."""
    data = np.load((directory or PREDICTIONS_DIR) / f"{name}.npz")
    pred, fold = np.full(n_items, -1), np.full(n_items, -1)
    for key in data.files:
        if key.startswith("ids_"):
            f = int(key[4:])
            pred[data[key]], fold[data[key]] = data[f"pred_{f}"], f
    if (pred < 0).any():
        raise ValueError(f"{name}: {(pred < 0).sum()} comments have no prediction (different folds/dataset?)")
    return pred, fold


def systems_with_predictions(directory=None):
    return sorted(p.stem for p in (directory or PREDICTIONS_DIR).glob("*.npz"))


# ---------------------------------------------------------------- fold-level results

def load_records():
    """{name: record} for the main results of Project 1 and Project 2 (subfolders like sweeps/ are left out)."""
    records = {}
    for project, folder in RESULTS_DIRS.items():
        for path in sorted(folder.glob("*.json")):
            r = json.loads(path.read_text())
            r["project"] = project
            name = r["name"] if r["name"] not in records else f"{project}:{r['name']}"
            records[name] = r
    return records


def fold_scores(records):
    """{name: [accuracy per fold]}, folds in order (input for BenavoliCorrelatedT)."""
    return {n: [r["accuracies"][f] for f in sorted(r["accuracies"], key=int)] for n, r in records.items()}


def overview(records):
    """One row per system: CV accuracy (best epoch per fold), spread over folds, size and best epoch."""
    rows = {}
    for name, r in records.items():
        acc = np.array(list(r["accuracies"].values()))
        best = r.get("best_epochs")
        rows[name] = {"project": r["project"], "cv acc": acc.mean(), "std": acc.std(), "min": acc.min(),
                      "max": acc.max(), "folds": len(acc), "params": r.get("n_params"),
                      "best epoch (mean)": np.mean([e + 1 for e in best.values()]) if best else None}
    return pd.DataFrame(rows).T.sort_values("cv acc", ascending=False)


def data_frame(records):
    """Long format (model, accuracy, run, fold) for part1.analysis.plot_box_and_points."""
    return pd.DataFrame([[n, a, 0, int(f)] for n, r in records.items() for f, a in r["accuracies"].items()],
                        columns=["model", "accuracy", "run", "fold"])


def _curve(curves, split, metric):
    """
    {fold: [epoch metrics]} -> array [folds, epochs]. With early stopping the folds can stop at
    different epochs; the missing epochs are NaN (use np.nanmean & co.).
    """
    rows = [[ep[f"{split}_metrics"][metric] for ep in curves[f]] for f in sorted(curves, key=int)]
    out = np.full((len(rows), max(len(r) for r in rows)), np.nan)
    for i, r in enumerate(rows):
        out[i, :len(r)] = r
    return out


def plot_learning_curves(records, names, metric="accuracy"):
    """
    Train (dashed) and validation (solid) curves per system: mean over the folds still training
    at that epoch, band = min-max over folds. The dot marks the best mean validation epoch.
    """
    names = [n for n in names if "curves" in records[n]]
    fig, axes = plt.subplots(1, len(names), figsize=(4.5 * len(names), 3.6), sharey=True, squeeze=False)
    for ax, name in zip(axes[0], names):
        for split, style in (("train", "--"), ("val", "-")):
            c = _curve(records[name]["curves"], split, metric)
            mean = np.nanmean(c, 0)
            ep = np.arange(1, c.shape[1] + 1)
            ax.plot(ep, mean, style, marker="o", ms=3, label=split)
            ax.fill_between(ep, np.nanmin(c, 0), np.nanmax(c, 0), alpha=0.2)
            if split == "val":
                best = np.nanargmax(mean) if metric == "accuracy" else np.nanargmin(mean)
                ax.plot(ep[best], mean[best], "o", ms=9, mec="white", mew=2, color="C1")
        ax.set(title=name, xlabel="epoch")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if metric == "accuracy":
            ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.grid(alpha=0.3)
        ax.legend()
    axes[0, 0].set_ylabel(metric)
    fig.tight_layout()
    return fig


def overfitting_table(records, names):
    """Where does validation accuracy peak, and how far apart are train and val when training stopped?"""
    rows = {}
    for name in names:
        if "curves" not in records[name]:
            continue
        curves = records[name]["curves"]
        v, t = _curve(curves, "val", "accuracy"), _curve(curves, "train", "accuracy")
        last = [len(curves[f]) - 1 for f in sorted(curves, key=int)]
        v_end = np.mean([v[i, e] for i, e in enumerate(last)])
        t_end = np.mean([t[i, e] for i, e in enumerate(last)])
        rows[name] = {"epochs run (mean)": np.mean(last) + 1, "best epoch": np.nanargmax(np.nanmean(v, 0)) + 1,
                      "best val acc": np.nanmax(np.nanmean(v, 0)), "val acc at end": v_end,
                      "train acc at end": t_end, "train - val (end)": t_end - v_end}
    return pd.DataFrame(rows).T


def sweep_table(records, project="P2"):
    """
    One row per experiment: the settings that differ between experiments (from the saved meta)
    next to the CV accuracy. Settings that are the same everywhere are left out.
    """
    rows = []
    for name, r in records.items():
        if r["project"] != project:
            continue
        acc = np.array(list(r["accuracies"].values()))
        rows.append({"name": name, **r.get("meta", {}), "cv acc": acc.mean(), "std": acc.std(),
                     "params": r.get("n_params")})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).set_index("name")
    keep = [c for c in df.columns if c in ("cv acc", "std", "params") or df[c].astype(str).nunique() > 1]
    return df[keep].sort_values("cv acc", ascending=False)


def plot_data_size(records, families):
    """
    families: {label: {train fraction: experiment name}}. CV accuracy (± std over folds) against the
    amount of training data; missing experiments are skipped.
    """
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for label, runs in families.items():
        points = [(frac, np.array(list(records[n]["accuracies"].values())))
                  for frac, n in sorted(runs.items()) if n in records]
        if points:
            fracs = [p[0] for p in points]
            ax.errorbar(fracs, [a.mean() for _, a in points], yerr=[a.std() for _, a in points],
                        marker="o", capsize=3, label=label)
    ax.set(xlabel="share of the training folds used", ylabel="CV accuracy")
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def seed_table(records, families):
    """
    families: {label: [experiment names that differ only in their seed]}. How much does the CV accuracy
    move between seeds, compared with the spread over folds?
    """
    rows = {}
    for label, names in families.items():
        means = [np.mean(list(records[n]["accuracies"].values())) for n in names if n in records]
        stds = [np.std(list(records[n]["accuracies"].values())) for n in names if n in records]
        if means:
            rows[label] = {"seeds": len(means), "cv acc (mean over seeds)": np.mean(means),
                           "std over seeds": np.std(means), "range over seeds": np.ptp(means),
                           "std over folds (mean)": np.mean(stds)}
    return pd.DataFrame(rows).T


def bayesian_comparisons(scores, comparisons, rope=0.01, plot=True):
    """
    Benavoli correlated t-test (course code) for each (A, B) pair, on fold accuracies.
    mean diff = B - A. Pairs with a missing system are skipped (e.g. LSTM not trained yet).
    """
    rows = []
    for a, b in comparisons:
        if a not in scores or b not in scores:
            print(f"skipped {a} vs {b}: no results yet")
            continue
        test = BenavoliCorrelatedT(scores[a], scores[b], names=[a, b])
        p_a, p_tie, p_b = test.probs(rope)
        rows.append({"A": a, "B": b, "mean diff (B-A)": test.mean_diff, "std": test.std,
                     "P(A better)": p_a, "P(equivalent)": p_tie, "P(B better)": p_b})
        if plot:
            test.plot(rope)
            plt.show()
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- comment-level results

def metrics_table(labels, preds, folds):
    """Accuracy and macro-/weighted-F1 per system (pooled over all comments) plus macro-F1 mean ± std over folds."""
    rows = {}
    for name, p in preds.items():
        per_fold = [f1_score(labels[folds[name] == f], p[folds[name] == f], average="macro", zero_division=0)
                    for f in np.unique(folds[name])]
        rows[name] = {"accuracy": (p == labels).mean(),
                      "macro F1": f1_score(labels, p, average="macro", zero_division=0),
                      "weighted F1": f1_score(labels, p, average="weighted", zero_division=0),
                      "macro F1 fold mean": np.mean(per_fold), "macro F1 fold std": np.std(per_fold)}
    return pd.DataFrame(rows).T.sort_values("macro F1", ascending=False)


def macro_f1_per_fold(labels, preds, folds):
    """{name: [macro-F1 per fold]}: another input for bayesian_comparisons (better metric for 28 imbalanced classes)."""
    return {n: [f1_score(labels[folds[n] == f], p[folds[n] == f], average="macro", zero_division=0)
                for f in np.unique(folds[n])] for n, p in preds.items()}


def per_class_f1(labels, preds, class_names):
    """Classes x systems F1, plus the class frequency (support)."""
    k = len(class_names)
    table = {n: precision_recall_fscore_support(labels, p, labels=range(k), zero_division=0)[2]
             for n, p in preds.items()}
    df = pd.DataFrame(table, index=class_names)
    df.insert(0, "support", np.bincount(labels, minlength=k))
    return df.sort_values("support", ascending=False)


def plot_per_class_f1(f1_table):
    systems = [c for c in f1_table.columns if c != "support"]
    fig, ax = plt.subplots(figsize=(12, 4))
    x, w = np.arange(len(f1_table)), 0.8 / len(systems)
    for i, s in enumerate(systems):
        ax.bar(x + i * w - 0.4 + w / 2, f1_table[s], w, label=s)
    ax.set_xticks(x, [f"{c} ({n})" for c, n in zip(f1_table.index, f1_table["support"])], rotation=60, ha="right")
    ax.set(ylabel="F1", title="F1 per emotion (support in brackets, most frequent first)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_confusion(labels, pred, class_names, title=""):
    """Row-normalised confusion matrix (each row: where the comments of that gold class went)."""
    cm = confusion_matrix(labels, pred, labels=range(len(class_names)), normalize="true")
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)), class_names, rotation=90)
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set(xlabel="predicted", ylabel="gold", title=title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig


def top_confusions(labels, pred, class_names, n=15):
    """Most frequent (gold -> predicted) mistakes."""
    wrong = labels != pred
    counts = Counter(zip(labels[wrong], pred[wrong]))
    support = np.bincount(labels, minlength=len(class_names))
    return pd.DataFrame([{"gold": class_names[g], "predicted": class_names[p], "count": c,
                          "share of gold": c / support[g]} for (g, p), c in counts.most_common(n)])


def mcnemar(labels, pred_a, pred_b):
    """
    Paired, comment-level test: on how many comments is only A right vs. only B right?
    Exact McNemar (binomial test on the discordant comments). Complements the fold-level Bayesian test.
    """
    ca, cb = pred_a == labels, pred_b == labels
    only_a, only_b = int((ca & ~cb).sum()), int((~ca & cb).sum())
    p = stats.binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return {"both right": int((ca & cb).sum()), "only A right": only_a, "only B right": only_b,
            "both wrong": int((~ca & ~cb).sum()), "acc A": ca.mean(), "acc B": cb.mean(), "p-value": p}


def mcnemar_table(labels, preds, comparisons):
    rows = []
    for a, b in comparisons:
        if a in preds and b in preds:
            rows.append({"A": a, "B": b, **mcnemar(labels, preds[a], preds[b])})
        else:
            print(f"skipped {a} vs {b}: no saved predictions")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- error analysis

NEGATION = re.compile(r"\b(not|no|never|nothing|nobody|none|nor|neither|cannot|without)\b|n['’]t\b")
EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿]")


def glove_vocabulary(path):
    """The words of a GloVe .txt file (all GloVe Twitter sizes share one vocabulary, so the 25d file is enough)."""
    with open(path, encoding="utf-8") as f:
        return {line.split(" ", 1)[0] for line in f}


def text_features(pairs, folds, tokenizer, embedding_vocab, rare_below=5):
    """
    Features that may explain why a comment is hard, one row per comment (indexed like `pairs`):
      n_tokens        length
      oov_rate        share of tokens without a GloVe vector (the encoders only see <unk> for those)
      rare_rate       share of tokens seen < rare_below times in the training part of the comment's fold
      negation        contains a negation (word order matters here, which pooling ignores)
      has_name        contains the [NAME] placeholder
      exclamations / questions / has_emoji
    """
    tokens = [tokenizer.tokens(text) for text, _ in pairs]
    rare_rate = np.zeros(len(pairs))
    for split in folds.values():
        counts = Counter(t for i in split["training_ids"] for t in tokens[i])
        for i in split["validation_ids"]:
            rare_rate[i] = np.mean([counts[t] < rare_below for t in tokens[i]]) if tokens[i] else 0.0
    texts = [text for text, _ in pairs]
    return pd.DataFrame({
        "n_tokens": [len(t) for t in tokens],
        "oov_rate": [np.mean([w not in embedding_vocab for w in t]) if t else 0.0 for t in tokens],
        "rare_rate": rare_rate,
        "negation": [int(bool(NEGATION.search(x.lower()))) for x in texts],
        "has_name": [int("[NAME]" in x) for x in texts],
        "exclamations": [x.count("!") for x in texts],
        "questions": [x.count("?") for x in texts],
        "has_emoji": [int(bool(EMOJI.search(x))) for x in texts],
    })


def accuracy_by_bins(features, correct, feature, bins):
    """Accuracy per system within bins of one feature (correct: {system: bool array})."""
    groups = pd.cut(features[feature], bins=bins, include_lowest=True)
    table = pd.DataFrame({name: pd.Series(c.astype(float)).groupby(groups, observed=False).mean()
                          for name, c in correct.items()})
    table.insert(0, "comments", groups.value_counts(sort=False).values)
    return table


def plot_accuracy_by_bins(table, feature):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    systems = [c for c in table.columns if c != "comments"]
    labels = [str(i) for i in table.index]
    for s in systems:
        ax.plot(labels, table[s], marker="o", label=s)
    ax.set(xlabel=feature, ylabel="accuracy")
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.tick_params(axis="x", rotation=30)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def l1_correctness_weights(features, correct, C=0.1):
    """
    Per system, an L1 logistic regression from standardised features to 'prediction correct'
    (the sklearn alternative from the tutorial). Weights near 0: feature irrelevant;
    < 0: the system struggles as the feature grows; > 0: the system does better.
    """
    x = features.to_numpy(dtype=float)
    x = (x - x.mean(0)) / (x.std(0) + 1e-9)
    weights = {}
    for name, c in correct.items():
        clf = LogisticRegression(penalty="l1", solver="liblinear", C=C).fit(x, c.astype(int))
        weights[name] = clf.coef_[0]
    return pd.DataFrame(weights, index=features.columns)


def bda_inputs(features, correct, sample=3000, seed=0):
    """
    Long-format inputs for part2.analysis.fit_bda_model on a random subsample of comments
    (the same comments for every system, so the comparison stays paired). MCMC is slow on
    all 45k comments x systems, hence the subsample.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(features), size=min(sample, len(features)), replace=False)
    names = list(correct)
    sys_id = np.repeat(np.arange(len(names)), len(idx))
    x = np.vstack([features.to_numpy(dtype=float)[idx]] * len(names))
    c = np.concatenate([correct[n][idx].astype(int) for n in names])
    return names, sys_id, x, c


def save(fig, name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=200, bbox_inches="tight")
