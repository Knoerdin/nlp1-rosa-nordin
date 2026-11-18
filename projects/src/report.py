"""Small plotting helpers for 03_analysis (overfitting and the rank sweep)."""
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator, PercentFormatter

from .config import FIGURES_DIR, PROJECT_DIR, RESULTS_DIR

REP_COLOR = {"indicator": "#2a78d6", "bow": "#eb6834", "tfidf": "#1baf7a"}
RANK_COLORS = ["#86b6ef", "#5598e7", "#3987e5", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]  # light = low rank


def val_curve(curves: dict, metric="accuracy") -> np.ndarray:
    """{fold: [epoch metrics]} -> array [folds, epochs] of the validation metric."""
    return np.array([[ep["val_metrics"][metric] for ep in curves[f]] for f in sorted(curves, key=int)])


def course_mlp_curves_from_log(notebook=PROJECT_DIR / "02_experiments.ipynb") -> dict:
    """
    part1.experiment does not store the per-epoch metrics of the course MLP, but part1.smlp prints
    them. Read them back from the output of 02: 'Epoch 01' starts a new fold, and the
    'mlp__<rep> OrderedDict(...)' line printed after the 5 folds names the experiment.
    """
    pat = re.compile(r"Epoch (\d+) \| train loss: ([\d.]+) \| train acc: ([\d.]+) \| "
                     r"val loss: ([\d.]+) \| val acc: ([\d.]+)")
    cells = json.loads(Path(notebook).read_text())["cells"]
    text = "".join("".join(o.get("text", "")) for c in cells if c["cell_type"] == "code"
                   for o in c.get("outputs", []))
    out, folds = {}, []
    for line in text.splitlines():
        if m := pat.search(line):
            ep, tl, ta, vl, va = int(m[1]), *map(float, m.groups()[1:])
            if ep == 1:
                folds.append([])
            folds[-1].append({"train_metrics": {"loss": tl, "accuracy": ta},
                              "val_metrics": {"loss": vl, "accuracy": va}})
        elif m := re.match(r"(mlp__\w+) OrderedDict", line):
            out[m[1]] = {str(i): f for i, f in enumerate(folds)}
            folds = []
    return out


def plot_val_accuracy(groups: dict):
    """
    groups: {panel title: {representation: curves}}. Mean validation accuracy over the folds per
    epoch; the dot marks the best epoch, after which the model starts to overfit.
    """
    fig, axes = plt.subplots(1, len(groups), figsize=(5 * len(groups), 3.6), sharey=True, squeeze=False)
    for ax, (title, curve_sets) in zip(axes[0], groups.items()):
        for rep, curves in curve_sets.items():
            v = val_curve(curves).mean(0)
            ep = np.arange(1, len(v) + 1)
            ax.plot(ep, v, marker="o", ms=3, color=REP_COLOR.get(rep), label=rep)
            ax.plot(ep[v.argmax()], v.max(), "o", ms=9, color=REP_COLOR.get(rep), mec="white", mew=2)
        ax.set(title=title, xlabel="epoch")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.grid(alpha=0.3)
        ax.legend()
    axes[0, 0].set_ylabel("validation accuracy")
    fig.tight_layout()
    return fig


def load_sweep(subdir="sweeps/rank") -> pd.DataFrame:
    """One row per (representation, rank) of the rank sweep."""
    rows = []
    for path in (RESULTS_DIR / subdir).glob("*.json"):
        r = json.loads(path.read_text())
        acc = np.array(list(r["accuracies"].values()))
        rows.append({"representation": r["name"].split("__")[1], "rank": r["meta"]["rank"] or "full",
                     "_order": r["meta"]["rank"] or 10**9, "params": r["n_params"],
                     "acc": acc, "cv acc": acc.mean(), "cv std": acc.std(), "curves": r["curves"]})
    return pd.DataFrame(rows).sort_values(["representation", "_order"]).reset_index(drop=True)


def plot_rank_curves(sweep: pd.DataFrame):
    """Validation accuracy per epoch, one line per rank, one panel per representation."""
    reps = list(dict.fromkeys(sweep["representation"]))
    fig, axes = plt.subplots(1, len(reps), figsize=(4.5 * len(reps), 3.6), sharey=True, squeeze=False)
    for ax, rep in zip(axes[0], reps):
        for color, (_, row) in zip(RANK_COLORS, sweep[sweep["representation"] == rep].iterrows()):
            v = val_curve(row["curves"]).mean(0)
            ax.plot(np.arange(1, len(v) + 1), v, color=color, label=f"r = {row['rank']}")
        ax.set(title=rep, xlabel="epoch")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("validation accuracy")
    axes[0, -1].legend(loc="center left", bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
    return fig


def plot_compression_tradeoff(sweep: pd.DataFrame):
    """Best-checkpoint CV accuracy (± std over folds) against the number of parameters."""
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for rep, sub in sweep.groupby("representation", sort=False):
        ax.errorbar(sub["params"], sub["cv acc"], yerr=sub["cv std"], marker="o", capsize=3,
                    color=REP_COLOR.get(rep), label=rep)
    for _, row in sweep.drop_duplicates("rank").iterrows():
        ax.annotate(f"r={row['rank']}", (row["params"], sweep["cv acc"].max() + 0.012),
                    ha="center", fontsize=8)
    ax.set_xscale("log")
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set(xlabel="trainable parameters (log scale)", ylabel="CV accuracy (best epoch)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def save(fig, name):
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=200, bbox_inches="tight")
