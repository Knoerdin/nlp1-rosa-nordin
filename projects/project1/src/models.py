import json
import time
from collections import OrderedDict
from pathlib import Path

import scipy.sparse
import torch
import torch.nn as nn
from scipy.sparse import csr_matrix
from torch.utils.data import DataLoader
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


def encode_sparse(texts, encoder) -> csr_matrix:
    """Encode texts row by row straight into a sparse matrix (never builds a dense N x V matrix)."""
    return scipy.sparse.vstack([csr_matrix(encoder(t)) for t in texts]).tocsr()


def train_model(training_set, validation_set, num_classes, encoder, save_to, model_args: dict,
                batch_size=64, learning_rate=1e-3, weight_decay=1e-4, num_epochs=10, device="cpu"):
    """
    Same procedure as part1.smlp.train_neural_network_model, but for DeepLowRankMLP:
    train for num_epochs, keep the checkpoint with the lowest validation loss.

    Returns (metrics per epoch, best epoch, validation predictions of the best model).
    """
    device = torch.device(device)
    t0 = time.time()
    X_train = encode_sparse([t for t, _ in training_set], encoder)
    X_val = encode_sparse([t for t, _ in validation_set], encoder)
    y_train = [int(y) for _, y in training_set]
    y_val = [int(y) for _, y in validation_set]

    train_loader = DataLoader(smlp.SparseDataset(X_train, y_train), batch_size=batch_size,
                              shuffle=True, collate_fn=smlp.sparse_collate)
    val_loader = DataLoader(smlp.SparseDataset(X_val, y_val), batch_size=batch_size,
                            shuffle=False, collate_fn=smlp.sparse_collate)

    args = dict(input_dim=len(encoder), num_classes=num_classes, **model_args)
    Path(save_to).parent.mkdir(parents=True, exist_ok=True)
    Path(f"{save_to}.json").write_text(json.dumps(args, indent=2))

    model = DeepLowRankMLP(**args).to(device)
    print(f"DeepLowRankMLP: {count_parameters(model):,} trainable parameters "
          f"(encoding took {time.time() - t0:.1f}s)")
    criterion = nn.NLLLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    metrics, best_epoch, best_loss, predictions = [], None, float("inf"), None
    for epoch in tqdm(range(num_epochs)):
        train_metrics = smlp.train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics, preds = smlp.evaluate(model, val_loader, criterion, device)
        metrics.append({"train_metrics": train_metrics, "val_metrics": val_metrics})
        print(f"Epoch {epoch + 1:02d} | train loss {train_metrics['loss']:.4f} acc {train_metrics['accuracy']:.4f}"
              f" | val loss {val_metrics['loss']:.4f} acc {val_metrics['accuracy']:.4f}")
        if val_metrics["loss"] <= best_loss:
            best_loss, best_epoch, predictions = val_metrics["loss"], epoch, preds
            torch.save(model.state_dict(), f"{save_to}.pt")

    return metrics, best_epoch, predictions


def load(path, device="cpu"):
    """Rebuild a trained DeepLowRankMLP from {path}.json + {path}.pt."""
    args = json.loads(Path(f"{path}.json").read_text())
    model = DeepLowRankMLP(**args)
    model.load_state_dict(torch.load(f"{path}.pt", map_location=device))
    return model.to(device)


def cross_validation_experiment(data_splits, outcome_space, encoder, save_to, model_args: dict, train_args=dict()):
    """
    k-fold CV for DeepLowRankMLP. Returns the same dict as
    part1.experiment.cross_validation_experiment, so results_io.save_cv works unchanged.
    """
    models, predictions, accuracies = OrderedDict(), OrderedDict(), OrderedDict()
    for fold in tqdm(data_splits.keys()):
        path = f"{save_to}/fold-{fold}/model"
        metrics, best_epoch, predictions[fold] = train_model(
            data_splits[fold]["training"], data_splits[fold]["validation"], len(outcome_space),
            encoder, path, model_args, **train_args,
        )
        models[fold] = path
        accuracies[fold] = metrics[best_epoch]["val_metrics"]["accuracy"]
    return {"models": models, "predictions": predictions, "accuracies": accuracies}
