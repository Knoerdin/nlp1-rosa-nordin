"""Paths and shared settings for Project 1."""
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]   # projects/project1
REPO_ROOT = PROJECT_DIR.parents[1]                  # nlp1-rosa-nordin

RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = PROJECT_DIR / "figures"
MODELS_DIR = PROJECT_DIR / "models"                 # git-ignored
DATA_CACHE = PROJECT_DIR / "data" / "cache"         # git-ignored

for _d in (RESULTS_DIR, FIGURES_DIR, MODELS_DIR, DATA_CACHE):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 42
K_FOLDS = 5          # TODO: decide (and justify) k
