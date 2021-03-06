from typing import NamedTuple

import torch
from pytorch_trainer import report
from torch import nn, Tensor
from torch.nn.functional import cross_entropy

from voice_vector.config import ModelConfig, NetworkConfig
from voice_vector.network.adacos import AdaCos
from voice_vector.network.predictor import create_predictor, Predictor


class Networks(NamedTuple):
    predictor: Predictor
    tail: AdaCos


def create_network(config: NetworkConfig):
    return Networks(
        predictor=create_predictor(config),
        tail=AdaCos(
            num_features=config.hidden_size if config.feature_size is None else config.feature_size,
            num_classes=config.out_size,
        ),
    )


def accuracy(output: Tensor, target: Tensor):
    with torch.no_grad():
        indexes = torch.argmax(output, dim=1)
        correct = torch.eq(indexes, target).view(-1)
        return correct.float().mean()


class Model(nn.Module):
    def __init__(
            self,
            model_config: ModelConfig,
            networks: Networks
    ) -> None:
        super().__init__()
        self.model_config = model_config
        self.predictor = networks.predictor
        self.tail = networks.tail

    def __call__(
            self,
            input: Tensor,
            target: Tensor,
    ):
        feature = self.predictor(input)
        output = self.tail(feature, target)

        loss = cross_entropy(output, target)

        # report
        values = dict(
            loss=loss,
            accuracy=accuracy(output, target),
        )
        if not self.training:
            weight = input.shape[0]
            values = {key: (l, weight) for key, l in values.items()}  # add weight
        report(values, self)

        return loss
