"""Stable import path for MOMENT's custom PEFT wrapper and saved adapters."""

from torch import nn


class MomentClassifier(nn.Module):
    """Give PEFT a dict config without changing MOMENT's Namespace config."""

    def __init__(self, moment: nn.Module):
        super().__init__()
        self.moment = moment
        self.config = {"model_type": "moment", "tie_word_embeddings": False}

    def forward(self, **kwargs):
        return self.moment(**kwargs)
