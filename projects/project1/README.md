# Project 1 — Text classification

Compare text classifiers on our dataset: 3 representations × 3 architectures,
evaluated with k-fold cross-validation and a Bayesian comparison with a ROPE.

Full assignment text, deliverables and deadline: [assignment.md](assignment.md)

## Where everything is

| Path | What's in it |
|---|---|
| `assignment.md` | Original assignment text + course/Canvas rules (deliverables, deadline, oral exam) |
| `01_data_exploration.ipynb` | Loading the dataset, statistics, label distribution, examples |
| `02_experiments.ipynb` | Tokeniser, encoders, cross-validation runs; writes scores to `results/` |
| `03_analysis.ipynb` | Reads `results/`: box plots, Bayesian/ROPE comparisons, error analysis |
| `src/config.py` | Paths, random seed, number of folds |
| `src/data.py` | Loading our dataset as `(text, label_id)` pairs + tokeniser training corpus |
| `src/encoders.py` | Feature representations: token indicators, bag-of-words, our third one |
| `src/models.py` | Our third architecture |
| `src/results_io.py` | Saving / loading cross-validation scores (JSON) |
| `results/` | One JSON per experiment, named `<architecture>__<representation>.json` (committed) |
| `figures/` | Plots for slides and blog (committed) |
| `models/` | Trained checkpoints (git-ignored, regenerate by rerunning `02`) |
| `slides/` | Slide outline and the final PDF |

## Course code we use (`part1/` in the repo root)

| Module | Used for |
|---|---|
| `experiment` | `train_tokenizer`, `cross_validation_split`, `cross_validation_experiment` |
| `loglin` | Logistic regression |
| `smlp` | One-hidden-layer MLP (`SparseMLP`) and its training loop |
| `analysis` | `make_data_frame`, `plot_box_and_points`, `BenavoliCorrelatedT` (Bayesian test with ROPE) |
| `clslib` | `NamedClasses`, `Encoder` base class |

Note: `part1.clslib.TokenIndicator` reads a global `tokenizer` variable in
`__len__`; `src/encoders.py` has a fixed copy, use that one.

## Running

Open the notebooks from this folder (JupyterLab / VS Code do this by default).
The first cell of each notebook makes `part1` and `src` importable. Environment
setup is in the root `README.md`.
