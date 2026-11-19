from typing import Callable

import torch
from torch import nn


class Doubler(nn.Module):
    """入力を 2 倍にする部品"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        入力を 2 倍にする関数

        Parameters
        ----------
        x: torch.Tensor
            入力

        Returns
        ----------
        y: torch.Tensor
            2 倍にした値
        """
        y = x * 2

        return y


def register_units(register: Callable) -> None:
    """
    部品を登録する関数 (起動の設定から読み込むときに呼ばれる)

    Parameters
    ----------
    register: Callable
        register_unit と同じ引数を受け取る登録の関数

    Returns
    ----------
    None
    """
    register(Doubler, name="hook.Doubler")
