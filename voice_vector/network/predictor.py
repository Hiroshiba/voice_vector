from typing import Optional

import torch.nn.functional as F
from torch import nn, Tensor

from voice_vector.config import NetworkConfig


class Predictor(nn.Module):
    def __init__(
            self,
            in_size: int,
            hidden_size: int,
            layer_num: int,
            feature_size: Optional[int],
    ):
        super().__init__()
        self.layer_num = layer_num

        self.layers = nn.ModuleList([
            nn.utils.weight_norm(
                nn.Linear(
                    in_features=in_size if i == 0 else hidden_size,
                    out_features=hidden_size,
                )
            )
            for i in range(layer_num)
        ])

        if feature_size is None:
            self.feature_layer = None
        else:
            self.feature_layer = nn.utils.weight_norm(
                nn.Linear(
                    in_features=hidden_size,
                    out_features=feature_size,
                )
            )

    def forward(
            self,
            h: Tensor,
    ):
        for i in range(self.layer_num):
            h = self.layers[i](h)
            if i != self.layer_num - 1:
                h = F.relu(h)

        if self.feature_layer is not None:
            h = F.relu(h)
            h = self.feature_layer(h)
        feature = F.normalize(h)
        return feature


def create_predictor(config: NetworkConfig):
    return Predictor(
        in_size=config.in_size,
        hidden_size=config.hidden_size,
        feature_size=config.feature_size,
        layer_num=config.layer_num,
    )
