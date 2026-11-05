"""Save / load cross-validation results so analysis never needs retraining."""
import json

from .config import RESULTS_DIR


def save_cv(name: str, cv_output: dict, **meta):
    """
    Store the per-fold scores from part1.experiment.cross_validation_experiment.

    name: e.g. "loglin__bow"  (architecture__representation)
    meta: anything worth remembering (hyperparameters, vocab size, seed, ...)
    """
    record = {
        "name": name,
        "accuracies": {str(fold): float(acc) for fold, acc in cv_output["accuracies"].items()},
        "meta": meta,
    }
    path = RESULTS_DIR / f"{name}.json"
    path.write_text(json.dumps(record, indent=2))
    return path


def _records():
    for path in sorted(RESULTS_DIR.glob("*.json")):
        yield json.loads(path.read_text())


def for_dataframe() -> dict:
    """{name: [run]} in the format part1.analysis.make_data_frame expects (one run per system)."""
    return {r["name"]: [{"accuracies": {int(f): a for f, a in r["accuracies"].items()}}] for r in _records()}


def fold_scores() -> dict:
    """{name: [score per fold]} — the input part1.analysis.BenavoliCorrelatedT expects."""
    return {r["name"]: [r["accuracies"][f] for f in sorted(r["accuracies"], key=int)] for r in _records()}
