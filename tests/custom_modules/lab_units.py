import torch
from torch import nn


class ScaleBlock(nn.Module):
    """入力を決めた倍にする部品"""

    def __init__(self, factor: float = 2.0):
        # 親クラスのコンストラクタを呼び出す
        super(ScaleBlock, self).__init__()

        # 倍率を保存
        self.factor = factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        入力を倍にする関数

        Parameters
        ----------
        x: torch.Tensor
            入力

        Returns
        ----------
        y: torch.Tensor
            倍にした値
        """
        y = x * self.factor

        return y


class SplitPair(nn.Module):
    """最後の次元を半分ずつに分ける部品"""

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        最後の次元を半分ずつに分ける関数

        Parameters
        ----------
        x: torch.Tensor
            入力

        Returns
        ----------
        pair: tuple[torch.Tensor, torch.Tensor]
            前の半分と後の半分
        """
        half = x.shape[-1] // 2
        pair = (x[..., :half], x[..., half:])

        return pair


# 起動の設定から読み込む部品の宣言 (クラスか register_unit の引数の辞書)
UNITS = [
    {"cls": ScaleBlock, "name": "lab.Scale",
     "notes": {"factor": {"description": "倍率"}}},
    {"cls": SplitPair, "name": "lab.SplitPair", "outputs": ["first", "second"]}
]
