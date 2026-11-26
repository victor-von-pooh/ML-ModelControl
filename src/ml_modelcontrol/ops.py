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


class Max(nn.Module):
    def __init__(self, dim: int = 1, keepdim: bool = False):
        # 親クラスのコンストラクタを呼び出す
        super(Max, self).__init__()

        # 最大値を取る次元と, その次元を残すかを保存
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        決めた次元で最大値を取る関数

        Parameters
        ----------
        input: torch.Tensor
            最大値を取るテンソル

        Returns
        ----------
        largest: torch.Tensor
            最大値 (どの位置かの番号は返さない)
        """
        largest = input.amax(dim=self.dim, keepdim=self.keepdim)

        return largest


class CumSum(nn.Module):
    def __init__(self, dim: int = 1):
        # 親クラスのコンストラクタを呼び出す
        super(CumSum, self).__init__()

        # 足していく次元を保存
        self.dim = dim

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        決めた次元で前から順に足していく関数 (累積和)

        負にならない差を足していき, 下がらない並びを作るときなどに使う
        (分位点が交わらない出し方)

        Parameters
        ----------
        input: torch.Tensor
            足していくテンソル

        Returns
        ----------
        total: torch.Tensor
            同じ形の累積和
        """
        total = torch.cumsum(input, dim=self.dim)

        return total


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


class Exp(nn.Module):
    def __init__(self):
        # 親クラスのコンストラクタを呼び出す
        super(Exp, self).__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        要素ごとに指数を取る関数

        対数分散から分散を作るときなどに使う

        Parameters
        ----------
        input: torch.Tensor
            指数を取るテンソル

        Returns
        ----------
        output: torch.Tensor
            同じ形の指数のテンソル
        """
        output = torch.exp(input)

        return output


class Scale(nn.Module):
    def __init__(self, factor: float = 1.0):
        # 親クラスのコンストラクタを呼び出す
        super(Scale, self).__init__()

        # 掛ける定数を保存
        self.factor = factor

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        入力に決めた定数を掛ける関数

        対数分散を半分にする (0.5) ときや, 類似度を温度で割る (1 / 温度)
        ときに使う

        Parameters
        ----------
        input: torch.Tensor
            掛けるテンソル

        Returns
        ----------
        output: torch.Tensor
            定数を掛けたテンソル
        """
        output = input * self.factor

        return output


class Chunk(nn.Module):
    def __init__(self, chunks: int = 2, index: int = 0, dim: int = 1):
        # 親クラスのコンストラクタを呼び出す
        super(Chunk, self).__init__()

        # 分ける数, 取り出す番号, 分ける次元を保存
        self.chunks = chunks
        self.index = index
        self.dim = dim

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        決めた次元を chunks 個に等分し, index 番目を取り出す関数

        出力の口は部品ごとに決まっているので, 分けた全てを返さず 1 つだけを
        返す. VAE の符号化器の出力から平均 (index 0) と対数分散 (index 1)
        を取り出すときなどに使う

        Parameters
        ----------
        input: torch.Tensor
            分けるテンソル (分ける次元の大きさは chunks で割り切れる)

        Returns
        ----------
        piece: torch.Tensor
            取り出したテンソル (分ける次元の大きさは元の 1 / chunks)

        Raises
        ----------
        ValueError
            分ける数か番号が範囲の外の場合と, 割り切れない場合
        """
        # 分ける数と番号の確認
        if self.chunks < 1:
            raise ValueError(
                f"分ける数 chunks は 1 以上にしてください ({self.chunks})"
            )
        if not -self.chunks <= self.index < self.chunks:
            raise ValueError(
                f"番号 index {self.index} は -{self.chunks} 以上 "
                f"{self.chunks - 1} 以下にしてください"
            )

        # 割り切れる場合だけ, 等分した index 番目の取り出し
        size = input.shape[self.dim]
        if size % self.chunks != 0:
            raise ValueError(
                f"次元 {self.dim} の大きさ {size} は chunks {self.chunks} "
                "で割り切れません"
            )
        width = size // self.chunks
        start = (self.index % self.chunks) * width
        piece = input.narrow(self.dim, start, width)

        return piece


class L2Normalize(nn.Module):
    def __init__(self, dim: int = -1, eps: float = 1e-12):
        # 親クラスのコンストラクタを呼び出す
        super(L2Normalize, self).__init__()

        # 正規化する次元と, 割り算を安定させる小さな数を保存
        self.dim = dim
        self.eps = eps

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        決めた次元の L2 ノルムが 1 になるように割る関数

        対照学習の埋め込みや, コサイン類似度の前処理に使う

        Parameters
        ----------
        input: torch.Tensor
            正規化するテンソル

        Returns
        ----------
        output: torch.Tensor
            同じ形の正規化したテンソル (割るノルムの下限は eps)
        """
        norm = input.norm(p=2, dim=self.dim, keepdim=True).clamp_min(self.eps)
        output = input / norm

        return output


class MatMul(nn.Module):
    def __init__(self):
        # 親クラスのコンストラクタを呼び出す
        super(MatMul, self).__init__()

    def forward(
        self, input: torch.Tensor, other: torch.Tensor
    ) -> torch.Tensor:
        """
        2 つの入力の行列の積を取る関数 (torch.matmul と同じ規則)

        2 つの埋め込みの類似度の表 ((N, D) と (D, M) から (N, M)) などに
        使う. 右の入力の転置は op.Transpose で行う

        Parameters
        ----------
        input: torch.Tensor
            左の行列
        other: torch.Tensor
            右の行列

        Returns
        ----------
        product: torch.Tensor
            行列の積
        """
        product = torch.matmul(input, other)

        return product


class Detach(nn.Module):
    def __init__(self):
        # 親クラスのコンストラクタを呼び出す
        super(Detach, self).__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        値はそのままに, 手前の層に勾配を流さないようにする関数

        目標の側の枝 (BYOL や SimSiam) や, 量子化の勾配の素通しなどに使う

        Parameters
        ----------
        input: torch.Tensor
            切り離すテンソル

        Returns
        ----------
        output: torch.Tensor
            同じ値で勾配を持たないテンソル
        """
        output = input.detach()

        return output


class SinusoidalEmbedding(nn.Module):
    def __init__(self, embedding_dim: int, max_period: float = 10000.0):
        # 親クラスのコンストラクタを呼び出す
        super(SinusoidalEmbedding, self).__init__()

        # 埋め込みの大きさと, 最も長い周期を保存
        self.embedding_dim = embedding_dim
        self.max_period = max_period

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        時刻 (位置) の値を, 周期の違う cos と sin を並べた埋め込みにする関数

        拡散モデルの時刻の埋め込みに使う. 前半が cos, 後半が sin で,
        周波数は 1 から 1 / max_period に向けて等比で並べる (1 / max_period
        は含まない). 出力は入力の型に依らず float32 (embedding_dim が奇数の
        場合は最後に 0 を 1 列足す)

        Parameters
        ----------
        input: torch.Tensor
            時刻のテンソル (例: (バッチ,) の整数か小数)

        Returns
        ----------
        embedding: torch.Tensor
            (*入力の形, embedding_dim) の float32 のテンソル
        """
        # 周波数の作成 (half 個. 1 から 1 / max_period に向けて等比)
        half = self.embedding_dim // 2
        exponents = torch.arange(
            half, dtype=torch.float32, device=input.device
        ) / max(half, 1)
        frequencies = float(self.max_period) ** (-exponents)

        # 時刻と周波数の積の cos と sin を並べ, 奇数なら 0 の列を追加
        angles = input.float().unsqueeze(-1) * frequencies
        embedding = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
        if self.embedding_dim % 2 == 1:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[..., :1])], dim=-1
            )

        return embedding


class Reparameterize(nn.Module):
    def __init__(self, sample_in_eval: bool = False):
        # 親クラスのコンストラクタを呼び出す
        super(Reparameterize, self).__init__()

        # 評価の形でも乱数を引くかを保存
        self.sample_in_eval = sample_in_eval

    def forward(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        平均と対数分散から, 再パラメータ化で潜在変数を引く関数 (VAE)

        z = mu + exp(logvar / 2) * eps (eps は標準正規分布の乱数) で,
        勾配は mu と logvar に流れる. 評価の形 (eval()) では,
        sample_in_eval が False なら乱数を引かずに mu を返す

        Parameters
        ----------
        mu: torch.Tensor
            平均
        logvar: torch.Tensor
            対数分散 (mu と同じ形)

        Returns
        ----------
        z: torch.Tensor
            潜在変数 (mu と同じ形)
        """
        # 評価の形で乱数を引かない場合は平均のまま
        if not self.training and not self.sample_in_eval:
            z = mu
        else:
            std = torch.exp(0.5 * logvar)
            z = mu + torch.randn_like(std) * std

        return z
