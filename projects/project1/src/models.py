import json
import multiprocessing as mp
import os
import time
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse
import torch
import torch.nn as nn
from scipy.sparse import csr_matrix
from tqdm.auto import tqdm

from part1 import smlp


class SparseLowRankInput(nn.Module):
    """First layer for sparse inputs: x -> x @ A @ B + b (rank r), or x @ W + b if rank is None."""

    def __init__(self, input_dim: int, output_dim: int, rank: int | None):
        super().__init__()
        self.rank = rank
        first_dim = output_dim if rank is None else rank
        # A (or the full W) multiplies the sparse input, so it is a plain parameter (as in SparseMLP)
        self.A = nn.Parameter(torch.empty(input_dim, first_dim))
        nn.init.xavier_uniform_(self.A)
        self.B = None if rank is None else nn.Linear(rank, output_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x):
        h = torch.sparse.mm(x, self.A)
        if self.B is not None:
            h = self.B(h)
        return h + self.bias


class LowRankLinear(nn.Module):
    """Dense layer with W = A @ B of rank r (two thin Linear layers); a normal Linear if rank is None."""

    def __init__(self, in_dim: int, out_dim: int, rank: int | None):
        super().__init__()
        if rank is None or rank >= min(in_dim, out_dim):   # a "low" rank this big saves nothing
            self.net = nn.Linear(in_dim, out_dim)
        else:
            self.net = nn.Sequential(nn.Linear(in_dim, rank, bias=False), nn.Linear(rank, out_dim))

    def forward(self, x):
        return self.net(x)


class DeepLowRankMLP(nn.Module):
    """
    Sparse input -> [low-rank layer -> ReLU -> dropout] x len(hidden_dims) -> linear -> log-softmax.

    input_dim:   vocabulary size (len(encoder))
    hidden_dims: widths of the hidden layers, e.g. [256, 256, 256, 256]
    num_classes: number of labels
    rank:        rank r of every factorised weight matrix; None = full-rank (a plain deep MLP)
    dropout:     dropout after every hidden layer
    residual:    add skip connections (h + f(h)) between hidden layers of equal width;
                 these make many-layer networks much easier to train
    """

    def __init__(self, input_dim, hidden_dims, num_classes, rank=32, dropout=0.2, residual=True):
        super().__init__()
        self.residual = residual
        self.input_layer = SparseLowRankInput(input_dim, hidden_dims[0], rank)
        self.hidden = nn.ModuleList(
            LowRankLinear(d_in, d_out, rank) for d_in, d_out in zip(hidden_dims[:-1], hidden_dims[1:])
        )
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Sequential(nn.Linear(hidden_dims[-1], num_classes), nn.LogSoftmax(dim=-1))

    def forward(self, x):
        h = self.dropout(self.activation(self.input_layer(x)))
        for layer in self.hidden:
            out = self.dropout(self.activation(layer(h)))
            h = h + out if (self.residual and out.shape == h.shape) else out
        return self.output(h)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



# ---------------------------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------------------------
# The same code runs on macOS (Apple Silicon), Linux and Windows:
#   device="auto" -> CUDA if an NVIDIA GPU is available, otherwise CPU.
#   On a Mac we deliberately stay on the CPU: torch.sparse.mm works on "mps", but for a model
#   this small it was no faster than the CPU (measured ~0.5 ms vs ~0.4 ms per step), and sparse
#   support on MPS is incomplete. Pass device="mps" explicitly to try it anyway.
# The real speed-up on a many-core CPU comes from training folds/experiments in parallel (n_jobs).

def resolve_device(device="auto") -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def default_n_jobs(n_tasks: int, device) -> int:
    """Training jobs run in parallel: up to half the cores on CPU (2 threads each), 1 on a GPU."""
    if resolve_device(device).type != "cpu":
        return 1   # several processes on one GPU mostly fight over it
    return max(1, min(n_tasks, (os.cpu_count() or 1) // 2))


# ---------------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------------

def encode_sparse(texts, encoder) -> csr_matrix:
    """Encode texts row by row straight into a sparse matrix (never builds a dense N x V matrix)."""
    return scipy.sparse.vstack([csr_matrix(encoder(t)) for t in texts], format="csr", dtype=np.float32)


def csr_to_torch(x: csr_matrix) -> torch.Tensor:
    """SciPy CSR -> torch sparse COO. CSR rows are sorted and duplicate-free, so no coalesce() needed."""
    x = x.tocoo()
    indices = torch.from_numpy(np.vstack([x.row, x.col]).astype(np.int64))
    values = torch.from_numpy(x.data.astype(np.float32, copy=False))
    return torch.sparse_coo_tensor(indices, values, size=x.shape, is_coalesced=True,
                                   check_invariants=False)   # built from valid CSR; skip the check


class SparseBatches:
    """
    Drop-in replacement for DataLoader(SparseDataset, collate_fn=sparse_collate), yielding the same
    (sparse X, y) batches. Instead of stacking batch_size single rows per step, it slices the CSR
    matrix once per batch, which removes most of the per-step Python overhead.
    """

    def __init__(self, X: csr_matrix, y, batch_size: int, shuffle: bool, seed: int | None = None):
        self.X, self.y = X.tocsr(), torch.as_tensor(np.asarray(y), dtype=torch.long)
        self.batch_size, self.shuffle = batch_size, shuffle
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return -(-self.X.shape[0] // self.batch_size)

    def __iter__(self):
        n = self.X.shape[0]
        order = self.rng.permutation(n) if self.shuffle else np.arange(n)
        for start in range(0, n, self.batch_size):
            idx = order[start:start + self.batch_size]
            yield csr_to_torch(self.X[idx]), self.y[idx]


# ---------------------------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------------------------

def fit(X_train, y_train, X_val, y_val, num_classes, save_to, model_args: dict,
        batch_size=64, learning_rate=1e-3, weight_decay=1e-4, num_epochs=10, device="auto",
        seed=None, verbose=True):
    """
    Train DeepLowRankMLP on already-encoded sparse matrices, with the same procedure as
    part1.smlp.train_neural_network_model: always train all num_epochs (no early stopping, so
    the curves show the overfitting), and keep the checkpoint with the lowest validation loss.

    Returns (metrics per epoch, best epoch, validation predictions of the best model).
    """
    device = resolve_device(device)
    if seed is not None:
        torch.manual_seed(seed)
    train_loader = SparseBatches(X_train, y_train, batch_size, shuffle=True, seed=seed)
    val_loader = SparseBatches(X_val, y_val, batch_size, shuffle=False)

    args = dict(input_dim=X_train.shape[1], num_classes=num_classes, **model_args)
    Path(save_to).parent.mkdir(parents=True, exist_ok=True)
    Path(f"{save_to}.json").write_text(json.dumps(args, indent=2))

    model = DeepLowRankMLP(**args).to(device)
    log = print if verbose else (lambda *a, **k: None)
    log(f"DeepLowRankMLP: {count_parameters(model):,} trainable parameters on {device}")
    criterion = nn.NLLLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    metrics, best_epoch, best_loss, predictions = [], None, float("inf"), None
    for epoch in range(num_epochs):
        train_metrics = smlp.train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics, preds = smlp.evaluate(model, val_loader, criterion, device)
        metrics.append({"train_metrics": train_metrics, "val_metrics": val_metrics})
        log(f"Epoch {epoch + 1:02d} | train loss {train_metrics['loss']:.4f} acc {train_metrics['accuracy']:.4f}"
            f" | val loss {val_metrics['loss']:.4f} acc {val_metrics['accuracy']:.4f}")
        if val_metrics["loss"] <= best_loss:
            best_loss, best_epoch, predictions = val_metrics["loss"], epoch, preds
            torch.save(model.state_dict(), f"{save_to}.pt")

    return metrics, best_epoch, predictions


def train_model(training_set, validation_set, num_classes, encoder, save_to, model_args: dict, **train_args):
    """Same as fit(), but takes (text, label) pairs and an encoder, like part1.smlp."""
    t0 = time.time()
    X_train = encode_sparse([t for t, _ in training_set], encoder)
    X_val = encode_sparse([t for t, _ in validation_set], encoder)
    print(f"encoding took {time.time() - t0:.1f}s")
    return fit(X_train, [int(y) for _, y in training_set], X_val, [int(y) for _, y in validation_set],
               num_classes, save_to, model_args, **train_args)


def load(path, device="cpu"):
    """Rebuild a trained DeepLowRankMLP from {path}.json + {path}.pt."""
    device = resolve_device(device)
    args = json.loads(Path(f"{path}.json").read_text())
    model = DeepLowRankMLP(**args)
    model.load_state_dict(torch.load(f"{path}.pt", map_location=device))
    return model.to(device)


# ---------------------------------------------------------------------------------------------
# Cross-validation (one experiment, or many at once, e.g. a sweep over ranks)
# ---------------------------------------------------------------------------------------------
# Every (experiment, fold) pair is one independent training job. All jobs of all experiments
# go into one process pool, so a sweep keeps every core busy instead of waiting for the
# slowest fold of each experiment. Each encoder's matrix is sent to every worker once
# (pool initializer); a job only carries row indices.

_WORKER_X = {}


def _init_worker(X_by_encoder, threads):
    _WORKER_X.update(X_by_encoder)
    torch.set_num_threads(threads)


def _run_job(job):
    """One fold of one experiment. Module-level so 'spawn' workers can import it."""
    name, fold, enc_key, rows, labels, fit_args = job
    X = _WORKER_X[enc_key]
    metrics, best_epoch, preds = fit(X[rows["training"]], labels["training"],
                                     X[rows["validation"]], labels["validation"], **fit_args, verbose=False)
    return name, fold, metrics, best_epoch, preds


def run_experiments(experiments: dict, data_splits, outcome_space, save_root, train_args=dict(), n_jobs=None):
    """
    k-fold CV for several DeepLowRankMLP configurations at once.

    experiments: {name: {"encoder": encoder, "model_args": {...}}}; the encoder may also be a
                 {fold: encoder} dict, for encoders fitted per fold (encoders.fit_per_fold)
    save_root:   checkpoints go to {save_root}/{name}/fold-{k}/model
    n_jobs:      parallel worker processes. None = half the cores on CPU, 1 on a GPU; 1 = no pool.

    Returns {name: cv_output}, where cv_output has the keys of
    part1.experiment.cross_validation_experiment ("models", "predictions", "accuracies"; so
    results_io.save_cv works unchanged) plus "curves" (per fold: the metrics of every epoch),
    "best_epochs" (per fold) and "n_params". Prints nothing but one progress bar.
    """
    folds = list(data_splits.keys())
    device = train_args.get("device", "auto")

    # 1. encode every distinct text once per distinct encoder
    row_of = {}
    for fold in folds:
        for part in ("training", "validation"):
            for text, _ in data_splits[fold][part]:
                row_of.setdefault(text, len(row_of))
    texts = list(row_of)
    def encoder_for(exp, fold):
        return exp["encoder"][fold] if isinstance(exp["encoder"], dict) else exp["encoder"]

    encoders = {id(encoder_for(e, f)): encoder_for(e, f) for e in experiments.values() for f in folds}
    X_by_encoder = {key: encode_sparse(texts, enc) for key, enc in tqdm(encoders.items(), desc="encoding", leave=False)}

    rows = {fold: {p: np.array([row_of[t] for t, _ in data_splits[fold][p]]) for p in ("training", "validation")}
            for fold in folds}
    labels = {fold: {p: [int(y) for _, y in data_splits[fold][p]] for p in ("training", "validation")}
              for fold in folds}
    jobs = [(name, fold, id(encoder_for(exp, fold)), rows[fold], labels[fold],
             dict(num_classes=len(outcome_space), save_to=f"{save_root}/{name}/fold-{fold}/model",
                  model_args=exp["model_args"], **train_args))
            for name, exp in experiments.items() for fold in folds]

    # 2. train all jobs
    n_jobs = default_n_jobs(len(jobs), device) if n_jobs is None else max(1, min(n_jobs, len(jobs)))
    threads = max(1, (os.cpu_count() or 1) // n_jobs)
    bar = tqdm(total=len(jobs), desc=f"{len(experiments)} experiments x {len(folds)} folds", leave=False)
    results = {}
    if n_jobs == 1:
        _init_worker(X_by_encoder, torch.get_num_threads())
        for job in jobs:
            name, fold, *res = _run_job(job)
            results[name, fold] = res
            bar.update()
    else:
        # 'spawn' works on macOS, Windows and Linux alike (and is safe with CUDA)
        with ProcessPoolExecutor(max_workers=n_jobs, mp_context=mp.get_context("spawn"),
                                 initializer=_init_worker, initargs=(X_by_encoder, threads)) as pool:
            for name, fold, *res in pool.map(_run_job, jobs):
                results[name, fold] = res
                bar.update()
    bar.close()
    _WORKER_X.clear()

    # 3. collect per experiment
    outputs = {}
    for name, exp in experiments.items():
        out = {k: OrderedDict() for k in ("models", "predictions", "accuracies", "curves", "best_epochs")}
        for fold in folds:
            metrics, best_epoch, preds = results[name, fold]
            out["models"][fold] = f"{save_root}/{name}/fold-{fold}/model"
            out["predictions"][fold] = preds
            out["accuracies"][fold] = metrics[best_epoch]["val_metrics"]["accuracy"]
            out["curves"][fold] = metrics
            out["best_epochs"][fold] = best_epoch
        out["n_params"] = count_parameters(DeepLowRankMLP(X_by_encoder[id(encoder_for(exp, folds[0]))].shape[1],
                                                          num_classes=len(outcome_space), **exp["model_args"]))
        outputs[name] = out
    return outputs


def cross_validation_experiment(data_splits, outcome_space, encoder, save_to, model_args: dict,
                                train_args=dict(), n_jobs=None):
    """Single-experiment version of run_experiments (checkpoints in {save_to}/fold-{k}/model)."""
    save_to = Path(save_to)
    return run_experiments({save_to.name: dict(encoder=encoder, model_args=model_args)}, data_splits,
                           outcome_space, save_to.parent, train_args, n_jobs)[save_to.name]


# ---------------------------------------------------------------------------------------------
# Compact summaries (instead of one block of epoch lines per fold)
# ---------------------------------------------------------------------------------------------

def epoch_table(cv_output) -> pd.DataFrame:
    """
    One row per epoch: train/val loss and accuracy averaged over the k folds (± std across folds).
    The '*' marks the epoch with the lowest mean validation loss; after it, train loss keeps
    falling while val loss rises = overfitting.
    """
    curves = cv_output["curves"]
    stat = {f"{split} {m}": np.array([[ep[f"{split}_metrics"][m] for ep in curves[f]] for f in curves])
            for split in ("train", "val") for m in ("loss", "accuracy")}
    best = int(stat["val loss"].mean(axis=0).argmin())
    table = pd.DataFrame({col: [f"{mu:.3f} ± {sd:.3f}" for mu, sd in zip(v.mean(axis=0), v.std(axis=0))]
                          for col, v in stat.items()})
    table.index = [f"{e + 1}{' *' if e == best else ''}" for e in range(len(table))]
    table.index.name = "epoch"
    return table


def summary_row(cv_output, **columns) -> dict:
    """One line per experiment, for comparing many runs (e.g. ranks) in one DataFrame."""
    acc = np.array(list(cv_output["accuracies"].values()))
    curves = cv_output["curves"]
    last = lambda split, m: np.mean([curves[f][-1][f"{split}_metrics"][m] for f in curves])
    return {**columns,
            "params": cv_output["n_params"],
            "cv acc (best ckpt)": f"{acc.mean():.4f} ± {acc.std():.4f}",
            "best epoch": " ".join(str(e + 1) for e in cv_output["best_epochs"].values()),
            "last-epoch train acc": round(last("train", "accuracy"), 4),
            "last-epoch val acc": round(last("val", "accuracy"), 4),
            "last-epoch val loss": round(last("val", "loss"), 4)}
