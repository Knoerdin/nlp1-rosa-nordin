"""Our third architecture.

Starting point: part1.smlp.SparseMLP (one hidden layer, ReLU, dropout).
Ideas: more hidden layers, a different activation, different dropout / width.

part1.smlp.train_neural_network_model builds a SparseMLP internally, so to train
a different nn.Module either copy that function here with a `model` argument,
or make a small variant of part1.experiment.cross_validation_experiment that
calls our own training function.
"""
import torch.nn as nn


class ThirdArchitecture(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], num_classes: int, dropout: float = 0.2):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError
