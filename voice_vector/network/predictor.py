import torch.nn.functional as F
from torch import nn, Tensor

from voice_vector.config import NetworkConfig


class Predictor(nn.Module):
    def __init__(
            self,
            in_size: int,
            hidden_size: int,
            layer_num: int,
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

    def forward(
            self,
            h: Tensor,
    ):
        for i in range(self.layer_num):
            h = self.layers[i](h)
            if i != self.layer_num - 1:
                h = F.relu(h)

        feature = F.normalize(h)
        return feature


def create_predictor(config: NetworkConfig):
    return Predictor(
        in_size=config.in_size,
        hidden_size=config.hidden_size,
        layer_num=config.layer_num,
    )
