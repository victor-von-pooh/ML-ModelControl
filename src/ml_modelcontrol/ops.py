from typing import Any

import torch
from torch import nn


class Add(nn.Module):
    def __init__(self):
        # 親クラスのコンストラクタを呼び出す
        super(Add, self).__init__()

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        """
        全ての入力を足し合わせる関数

        Parameters
        ----------
        *inputs: torch.Tensor
            足し合わせるテンソル (ブロードキャストできる形)

        Returns
        ----------
        total: torch.Tensor
            入力の和
        """
        # 順に足し合わせ
        total = inputs[0]
        for value in inputs[1:]:
            total = total + value

        return total


class Multiply(nn.Module):
    def __init__(self):
        # 親クラスのコンストラクタを呼び出す
        super(Multiply, self).__init__()

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        """
        全ての入力を要素ごとに掛け合わせる関数

        Parameters
        ----------
        *inputs: torch.Tensor
            掛け合わせるテンソル (ブロードキャストできる形)

        Returns
        ----------
        product: torch.Tensor
            入力の積
        """
        # 順に掛け合わせ
        product = inputs[0]
        for value in inputs[1:]:
            product = product * value

        return product


class Concat(nn.Module):
    def __init__(self, dim: int = 1):
        # 親クラスのコンストラクタを呼び出す
        super(Concat, self).__init__()

        # つなぐ次元を保存
        self.dim = dim

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        """
        全ての入力を決めた次元でつなぐ関数

        Parameters
        ----------
        *inputs: torch.Tensor
            つなぐテンソル (つなぐ次元以外は同じ形)

        Returns
        ----------
        joined: torch.Tensor
            つないだテンソル
        """
        joined = torch.cat(inputs, dim=self.dim)

        return joined


class Stack(nn.Module):
    def __init__(self, dim: int = 1):
        # 親クラスのコンストラクタを呼び出す
        super(Stack, self).__init__()

        # 重ねる次元を保存
        self.dim = dim

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        """
        全ての入力を新しい次元で重ねる関数

        Parameters
        ----------
        *inputs: torch.Tensor
            重ねるテンソル (全て同じ形)

        Returns
        ----------
        stacked: torch.Tensor
            重ねたテンソル
        """
        stacked = torch.stack(inputs, dim=self.dim)

        return stacked


class Reshape(nn.Module):
    def __init__(self, shape: list):
        # 親クラスのコンストラクタを呼び出す
        super(Reshape, self).__init__()

        # バッチの次元を除いた形を保存
        self.shape = [int(size) for size in shape]

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        バッチの次元を残して形を変える関数

        Parameters
        ----------
        input: torch.Tensor
            形を変えるテンソル

        Returns
        ----------
        reshaped: torch.Tensor
            (バッチ, *shape) の形のテンソル. shape の -1 は残りの大きさ
        """
        reshaped = input.reshape(input.shape[0], *self.shape)

        return reshaped


class Permute(nn.Module):
    def __init__(self, dims: list):
        # 親クラスのコンストラクタを呼び出す
        super(Permute, self).__init__()

        # 並べ替えた後の次元の順 (バッチの次元も含む) を保存
        self.dims = [int(dim) for dim in dims]

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        次元の順を並べ替える関数

        Parameters
        ----------
        input: torch.Tensor
            並べ替えるテンソル

        Returns
        ----------
        permuted: torch.Tensor
            並べ替えたテンソル
        """
        permuted = input.permute(*self.dims)

        return permuted


class Transpose(nn.Module):
    def __init__(self, dim0: int = 1, dim1: int = 2):
        # 親クラスのコンストラクタを呼び出す
        super(Transpose, self).__init__()

        # 入れ替える 2 つの次元を保存
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        2 つの次元を入れ替える関数

        Parameters
        ----------
        input: torch.Tensor
            入れ替えるテンソル

        Returns
        ----------
        swapped: torch.Tensor
            入れ替えたテンソル
        """
        swapped = input.transpose(self.dim0, self.dim1)

        return swapped


class Mean(nn.Module):
    def __init__(self, dim: int = 1, keepdim: bool = False):
        # 親クラスのコンストラクタを呼び出す
        super(Mean, self).__init__()

        # 平均を取る次元と, その次元を残すかを保存
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        決めた次元で平均を取る関数

        Parameters
        ----------
        input: torch.Tensor
            平均を取るテンソル

        Returns
        ----------
        mean: torch.Tensor
            平均
        """
        mean = input.mean(dim=self.dim, keepdim=self.keepdim)

        return mean


class Select(nn.Module):
    def __init__(self, index: int = 0):
        # 親クラスのコンストラクタを呼び出す
        super(Select, self).__init__()

        # 取り出す要素の番号を保存
        self.index = index

    def forward(self, input: Any) -> Any:
        """
        タプルやリストの出力から, 決めた番号の要素を取り出す関数

        Parameters
        ----------
        input: Any
            タプルかリスト (nn.LSTM の出力など)

        Returns
        ----------
        item: Any
            取り出した要素
        """
        item = input[self.index]

        return item


class GetAttr(nn.Module):
    def __init__(self, name: str = "last_hidden_state"):
        # 親クラスのコンストラクタを呼び出す
        super(GetAttr, self).__init__()

        # 取り出す属性かキーの名前を保存
        self.name = name

    def forward(self, input: Any) -> Any:
        """
        出力から, 決めた名前の属性かキーの値を取り出す関数

        Hugging Face のモデルの出力 (last_hidden_state など) や辞書に使う

        Parameters
        ----------
        input: Any
            属性を持つ値か辞書

        Returns
        ----------
        value: Any
            取り出した値
        """
        # 辞書はキーで, それ以外は属性で取り出し
        if isinstance(input, dict):
            value = input[self.name]
        else:
            value = getattr(input, self.name)

        return value


class Index(nn.Module):
    def __init__(self, dim: int = 1, index: int = 0):
        # 親クラスのコンストラクタを呼び出す
        super(Index, self).__init__()

        # 取り出す次元と番号を保存
        self.dim = dim
        self.index = index

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        決めた次元の決めた番号を取り出す関数 (その次元は無くなる)

        文の先頭のトークン ([CLS]) の表現 (dim 1, index 0) などに使う

        Parameters
        ----------
        input: torch.Tensor
            取り出すテンソル

        Returns
        ----------
        item: torch.Tensor
            取り出したテンソル
        """
        item = input.select(self.dim, self.index)

        return item


class Positions(nn.Module):
    def __init__(self, start: int = 0):
        # 親クラスのコンストラクタを呼び出す
        super(Positions, self).__init__()

        # 最初の位置の番号を保存
        self.start = start

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        (バッチ, 長さ, ...) の入力から, 位置の番号 (start, start + 1, ...)
        を (バッチ, 長さ) の整数で作る関数

        位置の埋め込み (nn.Embedding) の入力に使う

        Parameters
        ----------
        input: torch.Tensor
            2 次元以上のテンソル (値は使わず, 形と置き場だけを使う)

        Returns
        ----------
        positions: torch.Tensor
            (バッチ, 長さ) の int64

        Raises
        ----------
        ValueError
            入力が 1 次元以下の場合
        """
        # 位置の番号をバッチの数だけ並べたテンソル (後の層が書き換えても
        # よいように連続したメモリ)
        if input.dim() < 2:
            raise ValueError(
                "2 次元以上 (バッチ, 長さ, ...) の入力を入れてください"
            )
        length = input.shape[1]
        line = torch.arange(
            self.start, self.start + length, device=input.device
        )
        positions = line.unsqueeze(0).repeat(input.shape[0], 1)

        return positions


class Equal(nn.Module):
    def __init__(self, value: float = 0.0):
        # 親クラスのコンストラクタを呼び出す
        super(Equal, self).__init__()

        # 比べる値を保存
        self.value = value

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        値が決めた値に等しい所を True にする関数

        詰め物の番号 (0) から, 注意で見ない位置の印 (nn.MultiheadAttention
        の key_padding_mask) を作るのに使う

        Parameters
        ----------
        input: torch.Tensor
            比べるテンソル

        Returns
        ----------
        mask: torch.Tensor
            同じ形の bool のテンソル
        """
        mask = input == self.value

        return mask
