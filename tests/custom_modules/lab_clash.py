import torch
from torch import nn


class Other(nn.Module):
    """lab_units と同じ名前の部品 (重なりの誤りを確かめるため)"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        入力をそのまま返す関数

        Parameters
        ----------
        x: torch.Tensor
            入力

        Returns
        ----------
        y: torch.Tensor
            入力
        """
        y = x

        return y


UNITS = [{"cls": Other, "name": "lab.Scale"}]
