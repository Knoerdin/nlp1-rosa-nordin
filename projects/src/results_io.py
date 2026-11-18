"""Save / load cross-validation results so analysis never needs retraining."""
import json

from .config import RESULTS_DIR


def save_cv(name: str, cv_output: dict, subdir: str | None = None, **meta):
    """
    Store the per-fold scores from part1.experiment.cross_validation_experiment
    (or src.models.run_experiments, which also stores the per-epoch curves and best epochs).

    name:   e.g. "loglin__bow"  (architecture__representation)
    subdir: e.g. "sweeps/rank" for side experiments. Only files directly in results/ are part of
            the main comparison (fold_scores / for_dataframe); subfolders are left out of it.
    meta:   anything worth remembering (hyperparameters, vocab size, seed, ...)
    """
    record = {
        "name": name,
        "accuracies": {str(fold): float(acc) for fold, acc in cv_output["accuracies"].items()},
        "meta": meta,
    }
    if "curves" in cv_output:
        record["n_params"] = cv_output["n_params"]
        record["best_epochs"] = {str(f): int(e) for f, e in cv_output["best_epochs"].items()}
        record["curves"] = {str(f): ep for f, ep in cv_output["curves"].items()}
    folder = RESULTS_DIR / subdir if subdir else RESULTS_DIR
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.json"
    path.write_text(json.dumps(record, indent=2))
    return path


def load(name: str, subdir: str | None = None) -> dict:
    """Read one saved record back (e.g. to rebuild models.epoch_table without retraining)."""
    folder = RESULTS_DIR / subdir if subdir else RESULTS_DIR
    return json.loads((folder / f"{name}.json").read_text())


def _records():
    for path in sorted(RESULTS_DIR.glob("*.json")):
        yield json.loads(path.read_text())


def for_dataframe() -> dict:
    """{name: [run]} in the format part1.analysis.make_data_frame expects (one run per system)."""
    return {r["name"]: [{"accuracies": {int(f): a for f, a in r["accuracies"].items()}}] for r in _records()}


def fold_scores() -> dict:
    """{name: [score per fold]} — the input part1.analysis.BenavoliCorrelatedT expects."""
    return {r["name"]: [r["accuracies"][f] for f in sorted(r["accuracies"], key=int)] for r in _records()}
