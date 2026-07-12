"""Phenotype classifier head for the TCRI Pyro module (:class:`~tcri.model._module.TCRIModule`)."""
import torch.nn as nn


class PhenotypeClassifier(nn.Module):
    def __init__(self, n_latent, classifier_hidden, P, num_layers=3, dropout_rate=0.1, temperature=1.0):
        super(PhenotypeClassifier, self).__init__()
        layers = []
        input_dim = n_latent
        for _ in range(num_layers):
            layers.append(nn.Linear(input_dim, classifier_hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            input_dim = classifier_hidden
        layers.append(nn.Linear(classifier_hidden, P))
        self.mlp = nn.Sequential(*layers)
        self.temperature = temperature  # Add temperature parameter

    def forward(self, x):
        logits = self.mlp(x)
        return logits / self.temperature
