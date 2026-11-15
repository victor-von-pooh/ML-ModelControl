import inspect
import re
from typing import Callable, Optional, Union

from torch import nn

from ml_modelcontrol import ops
from ml_modelcontrol.external import (
    load_geometric_units, load_huggingface_units
)
from ml_modelcontrol.libraries import load_library_units
from ml_modelcontrol.registry import (
    REGISTRY, ParamSpec, PortSpec, UnitSpec, apply_notes, describe_params,
    needs_module, params_from_init, ports_from_forward, register
)

# 部品にしない torch.nn のクラス (入れ物と, 他の部品の土台のクラス)
EXCLUDED_TORCH = {
    "Module", "Sequential", "ModuleList", "ModuleDict", "ParameterList",
    "ParameterDict", "Container", "RNNBase", "RNNCellBase", "DataParallel",
    "AdaptiveLogSoftmaxWithLoss"
}

# RNN の系統の引数 (*args で受け取るので, シグネチャから取り出せない)
RNN_PARAMS = [
    ParamSpec("input_size", "int", required=True,
              description="入力の特徴量の数", minimum=1),
    ParamSpec("hidden_size", "int", required=True,
              description="隠れ状態の大きさ", minimum=1),
    ParamSpec("num_layers", "int", 1, description="重ねる層の数", minimum=1),
    ParamSpec("bias", "bool", True, description="バイアスを使うか"),
    ParamSpec("batch_first", "bool", False,
              description="入力の形を (バッチ, 系列, 特徴量) にするか"),
    ParamSpec("dropout", "float", 0.0,
              description="層の間のドロップアウトの確率",
              minimum=0.0, maximum=1.0),
    ParamSpec("bidirectional", "bool", False, description="双方向にするか")
]

# RNN の系統の出力 (出力の系列と, 最後の隠れ状態. LSTM はセルの状態も)
RNN_OUTPUTS = [
    PortSpec("output", "全ての時刻の出力", path=[0]),
    PortSpec("h_n", "最後の隠れ状態", path=[1])
]
LSTM_OUTPUTS = [
    PortSpec("output", "全ての時刻の出力", path=[0]),
    PortSpec("h_n", "最後の隠れ状態", path=[1, 0]),
    PortSpec("c_n", "最後のセルの状態", path=[1, 1])
]

# よく使う torch.nn の層の説明と範囲
TORCH_NOTES = {
    "Linear": {
        "description": "全結合層. 入力の最後の次元を out_features に変換する",
        "params": {
            "in_features": {"description": "入力の特徴量の数", "minimum": 1},
            "out_features": {"description": "出力の特徴量の数", "minimum": 1},
            "bias": {"description": "バイアスを使うか"}
        }
    },
    "Conv1d": {
        "description": "1 次元の畳み込み層. 入力は (バッチ, チャネル, 長さ)",
        "params": {
            "in_channels": {"description": "入力のチャネル数", "minimum": 1},
            "out_channels": {"description": "出力のチャネル数", "minimum": 1},
            "kernel_size": {"description": "カーネルの大きさ"}
        }
    },
    "Conv2d": {
        "description": (
            "2 次元の畳み込み層. 入力は (バッチ, チャネル, 高さ, 幅)"
        ),
        "params": {
            "in_channels": {"description": "入力のチャネル数", "minimum": 1},
            "out_channels": {"description": "出力のチャネル数", "minimum": 1},
            "kernel_size": {"description": "カーネルの大きさ"},
            "stride": {"description": "カーネルを動かす幅"},
            "padding": {"description": "周りに足す幅 (same も可)"}
        }
    },
    "ConvTranspose2d": {
        "description": "2 次元の転置畳み込み層. 画像を大きくするのに使う"
    },
    "MaxPool2d": {
        "description": "2 次元の最大値プーリング",
        "params": {"kernel_size": {"description": "プーリングの窓の大きさ"}}
    },
    "AvgPool2d": {"description": "2 次元の平均プーリング"},
    "AdaptiveAvgPool2d": {
        "description": "出力の大きさを決めた 2 次元の平均プーリング",
        "params": {"output_size": {"description": "出力の高さと幅"}}
    },
    "BatchNorm1d": {
        "description": "バッチ正規化 (入力は (バッチ, 特徴量) か (バッチ, 特徴量, 長さ))",
        "params": {"num_features": {"description": "特徴量の数", "minimum": 1}}
    },
    "BatchNorm2d": {
        "description": "画像のバッチ正規化 (チャネルごと)",
        "params": {"num_features": {"description": "チャネル数", "minimum": 1}}
    },
    "LayerNorm": {
        "description": "層正規化. 最後の次元 (normalized_shape) で正規化する",
        "params": {"normalized_shape": {"description": "正規化する次元の大きさ"}}
    },
    "Dropout": {
        "description": "ドロップアウト. 学習時に一部の値を 0 にする",
        "params": {
            "p": {"description": "0 にする確率", "minimum": 0.0, "maximum": 1.0}
        }
    },
    "ReLU": {"description": "活性化関数 ReLU (負の値を 0 にする)"},
    "GELU": {"description": "活性化関数 GELU (Transformer でよく使う)"},
    "SiLU": {"description": "活性化関数 SiLU (Swish)"},
    "Tanh": {"description": "活性化関数 tanh"},
    "Sigmoid": {"description": "活性化関数 sigmoid (0 から 1 に変換)"},
    "LeakyReLU": {"description": "活性化関数 LeakyReLU (負の値に小さな傾き)"},
    "Softmax": {
        "description": "softmax (dim の次元の和を 1 にする)",
        "params": {"dim": {"description": "和を 1 にする次元"}}
    },
    "Flatten": {
        "description": "start_dim から end_dim までの次元を 1 つにまとめる"
    },
    "Identity": {"description": "入力をそのまま返す"},
    "Embedding": {
        "description": "埋め込み層. 整数の番号をベクトルに変換する",
        "params": {
            "num_embeddings": {"description": "番号の数 (語彙の大きさ)",
                               "minimum": 1},
            "embedding_dim": {"description": "ベクトルの大きさ", "minimum": 1}
        }
    },
    "LSTM": {"description": "LSTM. 出力は全ての時刻の出力と最後の状態"},
    "GRU": {"description": "GRU. 出力は全ての時刻の出力と最後の隠れ状態"},
    "RNN": {"description": "単純な RNN. 出力は全ての時刻の出力と最後の隠れ状態"},
    "MultiheadAttention": {
        "description": "マルチヘッドの注意機構. 出力は注意の結果と重み",
        "params": {
            "embed_dim": {"description": "埋め込みの大きさ", "minimum": 1},
            "num_heads": {"description": "ヘッドの数 (embed_dim を割り切る)",
                          "minimum": 1}
        }
    },
    "TransformerEncoderLayer": {
        "description": "Transformer のエンコーダーの 1 層 (注意と全結合)",
        "params": {
            "d_model": {"description": "埋め込みの大きさ", "minimum": 1},
            "nhead": {"description": "ヘッドの数 (d_model を割り切る)",
                      "minimum": 1}
        }
    }
}

# 手で書いた説明の無い torch.nn の層の短い説明 (部品の一覧に出す)
TORCH_DESCRIPTIONS = {
    "AlphaDropout": "SELU 向けのドロップアウト (平均と分散を保つ)",
    "FeatureAlphaDropout": "チャンネルごとの AlphaDropout",
    "Dropout1d": "チャンネルごとに 0 にするドロップアウト (1 次元)",
    "Dropout2d": "チャンネルごとに 0 にするドロップアウト (2 次元)",
    "Dropout3d": "チャンネルごとに 0 にするドロップアウト (3 次元)",
    "BatchNorm3d": "バッチ正規化 (3 次元. 入力は (N, C, D, H, W))",
    "SyncBatchNorm": "複数の GPU で統計をそろえるバッチ正規化",
    "GroupNorm": "チャンネルをグループに分けて正規化する",
    "InstanceNorm1d": "サンプルとチャンネルごとの正規化 (1 次元)",
    "InstanceNorm2d": "サンプルとチャンネルごとの正規化 (2 次元)",
    "InstanceNorm3d": "サンプルとチャンネルごとの正規化 (3 次元)",
    "RMSNorm": "二乗平均平方根で割る正規化",
    "LocalResponseNorm": "近くのチャンネルで割る局所応答正規化",
    "CrossMapLRN2d": "チャンネル方向の局所応答正規化 (2 次元)",
    "Bilinear": "2 つの入力の双線形変換 (x1^T A x2 + b)",
    "Conv3d": "3 次元の畳み込み層",
    "ConvTranspose1d": "1 次元の転置畳み込み層 (大きさを広げる)",
    "ConvTranspose3d": "3 次元の転置畳み込み層 (大きさを広げる)",
    "Fold": "区画の列を画像に戻す (Unfold の逆)",
    "Unfold": "画像を区画に切り出して列にする",
    "Unflatten": "dim の次元を unflattened_size の形に分ける",
    "PixelShuffle": "チャンネルを空間に並べ替えて解像度を上げる",
    "PixelUnshuffle": "空間をチャンネルに並べ替えて解像度を下げる",
    "ChannelShuffle": "チャンネルをグループの間で入れ替える",
    "Upsample": "補間で大きさを広げる (mode で補間の方法)",
    "UpsamplingBilinear2d": "双線形補間で大きさを広げる (2 次元)",
    "UpsamplingNearest2d": "最近傍で大きさを広げる (2 次元)",
    "EmbeddingBag": "単語の埋め込みを袋ごとに合計か平均する",
    "CosineSimilarity": "2 つの入力のコサイン類似度",
    "PairwiseDistance": "2 つの入力の距離 (p ノルム)",
    "GRUCell": "GRU の 1 時刻分",
    "LSTMCell": "LSTM の 1 時刻分 (出力は h と c)",
    "RNNCell": "単純な RNN の 1 時刻分",
    "Transformer": "Transformer のエンコーダーとデコーダーの全体",
    "TransformerDecoderLayer": "Transformer のデコーダーの 1 層",
    "TransformerEncoder": "エンコーダーの層を重ねたもの",
    "TransformerDecoder": "デコーダーの層を重ねたもの",
    "CELU": "活性化関数 CELU",
    "ELU": "活性化関数 ELU (負の側をなめらかに飽和)",
    "SELU": "活性化関数 SELU (自己正規化)",
    "GLU": "ゲート付き線形ユニット (半分をシグモイドのゲートにする)",
    "Hardshrink": "絶対値が lambd 以下を 0 にする",
    "Softshrink": "絶対値を lambd だけ縮め, 以下は 0 にする",
    "Tanhshrink": "x - tanh(x)",
    "Hardsigmoid": "活性化関数 Hardsigmoid (シグモイドの区分線形の近似)",
    "Hardswish": "活性化関数 Hardswish",
    "Hardtanh": "min_val と max_val の範囲に切る活性化関数",
    "ReLU6": "活性化関数 ReLU6 (0 から 6 に切る)",
    "PReLU": "負の側の傾きを学習する ReLU",
    "RReLU": "負の側の傾きを学習のときに乱数で決める ReLU",
    "Threshold": "threshold 以下を value に置き換える",
    "Mish": "活性化関数 Mish",
    "LogSigmoid": "シグモイドの対数",
    "LogSoftmax": "dim の次元のソフトマックスの対数",
    "Softmax2d": "各位置でチャンネル方向のソフトマックス",
    "Softmin": "dim の次元のソフトミン (小さいほど大きい確率)",
    "Softplus": "活性化関数 Softplus (ReLU のなめらかな近似)",
    "Softsign": "x / (1 + |x|)",
    "FractionalMaxPool2d": "割合で大きさを決める最大値プーリング (2 次元)",
    "FractionalMaxPool3d": "割合で大きさを決める最大値プーリング (3 次元)",
}

# 次元の数だけが違う層の説明 (クラス名の頭, 説明)
TORCH_FAMILY_DESCRIPTIONS = [
    ("AdaptiveAvgPool", "出力の大きさを決める平均値プーリング"),
    ("AdaptiveMaxPool", "出力の大きさを決める最大値プーリング"),
    ("AvgPool", "平均値プーリング"),
    ("MaxPool", "最大値プーリング"),
    ("MaxUnpool", "最大値プーリングの位置に値を戻す (逆)"),
    ("LPPool", "p 乗平均のプーリング"),
    ("CircularPad", "周期的に (反対側の値で) 端を埋める"),
    ("ConstantPad", "決めた値で端を埋める"),
    ("ReflectionPad", "端で折り返した値で端を埋める"),
    ("ReplicationPad", "端の値の繰り返しで端を埋める"),
    ("ZeroPad", "0 で端を埋める")
]


def default_description(name: str) -> str:
    """
    手で書いた説明の無い torch.nn の層の説明を返す関数

    Parameters
    ----------
    name: str
        クラスの名前

    Returns
    ----------
    description: str
        説明 (無い場合は空). 次元の数だけが違う層は, 頭の説明に次元を付加
    """
    if name in TORCH_DESCRIPTIONS:
        return TORCH_DESCRIPTIONS[name]
    for head, text in TORCH_FAMILY_DESCRIPTIONS:
        rest = name[len(head):]
        if name.startswith(head) and rest in ("1d", "2d", "3d"):
            return f"{text} ({rest[0]} 次元)"
    description = ""

    return description


# 型ヒントでは選択肢が分からない, 文字の引数の選択肢 (クラス, 引数)
PARAM_CHOICES = {
    ("Transformer", "activation"): ["relu", "gelu"],
    ("TransformerEncoderLayer", "activation"): ["relu", "gelu"],
    ("TransformerDecoderLayer", "activation"): ["relu", "gelu"],
    ("EmbeddingBag", "mode"): ["sum", "mean", "max"],
    ("GELU", "approximate"): ["none", "tanh"],
    ("RNNCell", "nonlinearity"): ["tanh", "relu"],
    ("Upsample", "mode"): [
        "nearest", "nearest-exact", "linear", "bilinear", "bicubic",
        "trilinear", "area"
    ]
}

# 演算の部品 (名前, クラス, 説明, 入力が可変の数か)
OPS = [
    ("Add", ops.Add, "全ての入力を足し合わせる (残差のつなぎ方など)", True),
    ("Multiply", ops.Multiply, "全ての入力を要素ごとに掛け合わせる", True),
    ("Concat", ops.Concat, "全ての入力を dim の次元でつなぐ", True),
    ("Stack", ops.Stack, "全ての入力を新しい dim の次元で重ねる", True),
    ("Reshape", ops.Reshape, "バッチの次元を残して形を shape にする", False),
    ("Permute", ops.Permute, "次元の順を dims に並べ替える (バッチも含む)",
     False),
    ("Transpose", ops.Transpose, "2 つの次元を入れ替える", False),
    ("Mean", ops.Mean, "dim の次元で平均を取る", False),
    ("Select", ops.Select,
     "index 番目を取り出す (タプルやリストは要素, テンソルは最初の次元)",
     False),
    ("GetAttr", ops.GetAttr, "出力から name の属性かキーの値を取り出す",
     False)
]

# 演算の引数の説明
OP_PARAM_NOTES = {
    "Add": {}, "Multiply": {},
    "Concat": {"dim": {"description": "つなぐ次元 (負の数は後ろから)"}},
    "Stack": {"dim": {"description": "新しく作る次元の位置"}},
    "Reshape": {
        "shape": {"description": "バッチの次元を除いた形 (例: [16, 4]. -1 は残り)"}
    },
    "Permute": {
        "dims": {"description": "並べ替えた次元の順 (バッチも含む. 例: [0, 2, 1])"}
    },
    "Transpose": {
        "dim0": {"description": "入れ替える 1 つ目の次元"},
        "dim1": {"description": "入れ替える 2 つ目の次元"}
    },
    "Mean": {
        "dim": {"description": "平均を取る次元"},
        "keepdim": {"description": "平均を取った次元を大きさ 1 で残すか"}
    },
    "Select": {
        "index": {"description": "取り出す番号 (負の数は後ろから. 例: LSTM の "
                                 "h_n から最後の層は -1)"}
    },
    "GetAttr": {
        "name": {"description": "取り出す属性かキーの名前 (例: last_hidden_state)"}
    }
}

# 組み込みの部品を読み込んだかどうかの印
LOADED = {"done": False}


def load_builtin_units() -> None:
    """
    組み込みの部品 (torch.nn の層と演算) を登録簿に読み込む関数

    最初に呼ばれたときだけ読み込む

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 2 回目以降は処理なし
    if LOADED["done"]:
        return
    LOADED["done"] = True

    # torch.nn の層, 演算, 外部のライブラリの部品の登録
    for cls in torch_unit_classes():
        register_torch_unit(cls)
    for name, cls, description, variadic in OPS:
        register_op(name, cls, description, variadic)
    load_geometric_units()
    load_huggingface_units()
    load_library_units()


def torch_unit_classes() -> list[type]:
    """
    部品にする torch.nn のクラスを集める関数

    Parameters
    ----------
    None

    Returns
    ----------
    classes: list[type]
        nn.Module のクラス (入れ物, 土台, 損失関数を除く)
    """
    # torch.nn の公開された名前から, nn.Module のクラスを選択
    classes = []
    for name in sorted(dir(nn)):
        cls = getattr(nn, name)
        if (
            not inspect.isclass(cls) or not issubclass(cls, nn.Module)
            or name.startswith(("_", "Lazy")) or name in EXCLUDED_TORCH
            or issubclass(cls, nn.modules.loss._Loss)
        ):
            continue
        classes.append(cls)

    return classes


def make_builder(cls: type) -> Callable[..., nn.Module]:
    """
    引数の値からクラスの部品を作る関数を作る関数

    Parameters
    ----------
    cls: type
        nn.Module のクラス

    Returns
    ----------
    builder: Callable[..., nn.Module]
        引数を受け取って部品を作る関数
    """
    def builder(**params: object) -> nn.Module:
        """
        引数の値から部品を作る関数

        Parameters
        ----------
        **params: object
            クラスに渡す引数

        Returns
        ----------
        module: nn.Module
            作った部品
        """
        module = cls(**params)

        return module

    return builder


def register_torch_unit(cls: type) -> UnitSpec:
    """
    torch.nn のクラスを部品として登録する関数

    Parameters
    ----------
    cls: type
        torch.nn のクラス

    Returns
    ----------
    spec: UnitSpec
        登録した部品
    """
    # シグネチャから引数と入力の口を取り出し, 分類は定義したファイルの名前
    params, extra = params_from_init(cls)
    inputs, forward_params = ports_from_forward(cls)
    category = cls.__module__.rsplit(".", 1)[-1]
    outputs = [PortSpec("out")]

    # RNN の系統は引数と出力を手で指定 (RNN は活性化関数, LSTM は射影も)
    if cls.__name__ in ("RNN", "LSTM", "GRU"):
        params, extra = [
            ParamSpec(**vars(param)) for param in RNN_PARAMS
        ], False
        if cls.__name__ == "RNN":
            params.append(ParamSpec(
                "nonlinearity", "str", "tanh", description="活性化関数",
                choices=["tanh", "relu"]
            ))
        if cls.__name__ == "LSTM":
            params.append(ParamSpec(
                "proj_size", "int", 0,
                description="出力の射影の大きさ (0 なら射影しない)", minimum=0
            ))
        source = LSTM_OUTPUTS if cls.__name__ == "LSTM" else RNN_OUTPUTS
        outputs = [PortSpec(**vars(port)) for port in source]
    if cls.__name__ == "MultiheadAttention":
        outputs = [
            PortSpec("output", "注意の結果", path=[0]),
            PortSpec("weights", "注意の重み (need_weights が false なら None)",
                     path=[1])
        ]

    # 部品の作成と, 手で書いた説明の反映
    spec = UnitSpec(
        name=f"nn.{cls.__name__}", category=category, source="torch.nn",
        params=params, inputs=inputs, outputs=outputs,
        forward_params=forward_params, extra_params=extra,
        builder=make_builder(cls),
        code_name=f"nn.{cls.__name__}", code_import="from torch import nn"
    )
    apply_notes(spec, TORCH_NOTES.get(cls.__name__, {}))
    if not spec.description:
        spec.description = default_description(cls.__name__)
    describe_params(spec, cls)
    for param in spec.params:
        choices = PARAM_CHOICES.get((cls.__name__, param.name))
        if choices is not None:
            param.kind = "str"
            param.choices = list(choices)

    # 別の部品を引数に取るクラス (TransformerEncoder など) は今は対象外
    if needs_module(cls):
        spec.available = False
        spec.reason = (
            "引数に別の部品 (nn.Module) が必要なので, 今は定義から作れません"
        )
    register(spec)

    return spec


def register_op(
    name: str, cls: type, description: str, variadic: bool
) -> UnitSpec:
    """
    演算のクラスを部品として登録する関数

    Parameters
    ----------
    name: str
        演算の名前 (Add など)
    cls: type
        ops のクラス
    description: str
        説明
    variadic: bool
        入力が可変の数か

    Returns
    ----------
    spec: UnitSpec
        登録した部品
    """
    # 可変の数の入力は xs, それ以外は input の口
    params, _ = params_from_init(cls)
    inputs = (
        [PortSpec("xs", "つなぐ値 (書いた順)", variadic=True)]
        if variadic else [PortSpec("input")]
    )
    spec = UnitSpec(
        name=f"op.{name}", category="op", source="ml_modelcontrol",
        description=description, params=params, inputs=inputs,
        outputs=[PortSpec("out")], builder=make_builder(cls),
        code_name=f"ops.{name}", code_import="from ml_modelcontrol import ops"
    )
    apply_notes(spec, {"params": OP_PARAM_NOTES.get(name, {})})
    register(spec)

    return spec


def registered_names() -> list[str]:
    """
    登録した部品の名前を返す関数

    Parameters
    ----------
    None

    Returns
    ----------
    names: list[str]
        名前の順の部品の名前
    """
    # 組み込みの部品を読み込んでから一覧を作成
    load_builtin_units()
    names = sorted(REGISTRY)

    return names


# 組み込みの部品の名前の頭 (自作の部品には使えない. 版が上がって同じ名前の
# 部品が増えた場合に重ならないため)
BUILTIN_PREFIXES = {
    "nn", "op", "pyg", "hf", "tv", "timm", "audio", "einops", "kornia",
    "monai", "smp", "sub"
}

# 自作の部品の名前 (識別子を . でつないだもの)
UNIT_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*")


def output_ports(
    outputs: Optional[Union[list[str], dict[str, list]]]
) -> list[PortSpec]:
    """
    自作の部品の出力の口の指定から, 出力の口を作る関数

    Parameters
    ----------
    outputs: Optional[Union[list[str], dict[str, list]]]
        口の名前のリスト (戻り値のタプルの順), 口の名前ごとの取り出し方の
        辞書, None なら out の 1 つ

    Returns
    ----------
    ports: list[PortSpec]
        出力の口

    Raises
    ----------
    ValueError
        口が無い場合と, 取り出し方が番号と名前の並びでない場合
    """
    # 指定が無い場合は戻り値そのもの, リストはタプルの要素の番号
    if outputs is None:
        return [PortSpec("out")]
    if not isinstance(outputs, dict):
        for port in outputs:
            if not isinstance(port, str) or not port.isidentifier():
                raise ValueError(
                    f"出力の口の名前 {port!r} は識別子 (英字か _ で始まる "
                    "英数字と _) にしてください"
                )
        if len(set(outputs)) != len(outputs):
            raise ValueError(f"出力の口の名前が重なっています: {outputs}")
        ports = [
            PortSpec(port, path=[index]) for index, port in enumerate(outputs)
        ]
    else:
        ports = []
        for port, path in outputs.items():
            if not isinstance(port, str) or not port.isidentifier():
                raise ValueError(
                    f"出力の口の名前 {port!r} は識別子 (英字か _ で始まる "
                    "英数字と _) にしてください"
                )
            steps = path if isinstance(path, (list, tuple)) else [path]
            if not all(
                isinstance(step, (int, str)) and not isinstance(step, bool)
                for step in steps
            ):
                raise ValueError(
                    f"出力の口 {port} の取り出し方は, 番号か名前の並びにして"
                    f"ください (今は {path!r})"
                )
            ports.append(PortSpec(port, path=list(steps)))
    if not ports:
        raise ValueError("outputs に 1 つ以上の口を書いてください")

    return ports


def register_unit(
    cls: type, name: Optional[str] = None, description: str = "",
    category: str = "custom",
    outputs: Optional[Union[list[str], dict[str, list]]] = None,
    notes: Optional[dict] = None, overwrite: bool = False
) -> UnitSpec:
    """
    自作の nn.Module のクラスを部品として登録する関数

    引数は __init__ の, 入力の口は forward のシグネチャから取り出す.
    モデルの定義からは登録できない (Python のコードから呼ぶ)

    Parameters
    ----------
    cls: type
        nn.Module のクラス
    name: Optional[str] = None
        部品の名前, None の場合はクラス名
    description: str = ""
        説明
    category: str = "custom"
        分類
    outputs: Optional[Union[list[str], dict[str, list]]] = None
        出力の口. リストはタプルを返す場合の口の名前 (戻り値の順), 辞書は
        口の名前ごとの取り出し方 (番号か, 属性かキーの名前の並び. 例:
        {"logits": ["logits"], "last": ["hidden_states", -1]}),
        None なら out の 1 つ
    notes: Optional[dict] = None
        引数の説明と範囲 (params の名前ごとの description, minimum など)
    overwrite: bool = False
        同じ名前の自作の部品を上書きするか (組み込みの部品は上書き不可)

    Returns
    ----------
    spec: UnitSpec
        登録した部品
    """
    # nn.Module のクラスだけを受け付け
    if not inspect.isclass(cls) or not issubclass(cls, nn.Module):
        raise TypeError(f"{cls!r} は nn.Module のクラスではありません")
    load_builtin_units()

    # 名前は識別子を . でつないだもの. 組み込みの部品と, 上書きしない場合の
    # 同じ名前の自作の部品は不可
    unit_name = name or cls.__name__
    if not UNIT_NAME.fullmatch(unit_name):
        raise ValueError(
            f"部品の名前 {unit_name!r} は, 英字か _ で始まる英数字と _ を . で"
            "つないだ名前にしてください (例: my.Block)"
        )
    if unit_name.split(".")[0] in BUILTIN_PREFIXES:
        raise ValueError(
            f"部品の名前 {unit_name} は組み込みの部品の頭 "
            f"({unit_name.split('.')[0]}.) で始まるので使えません. 別の頭"
            "にしてください (例: my.Block)"
        )
    existing = REGISTRY.get(unit_name)
    if existing is not None and existing.source != "custom":
        raise ValueError(
            f"部品 {unit_name} は組み込みの部品なので登録できません. 別の名前"
            "にしてください"
        )
    if existing is not None and not overwrite:
        raise ValueError(
            f"部品 {unit_name} は登録済みです (overwrite=True で上書き可)"
        )
    # 説明を省いた場合は, そのクラス自身の docstring の 1 行目 (親の
    # nn.Module の説明は使わない)
    if description == "":
        description = inspect.cleandoc(
            cls.__dict__.get("__doc__") or ""
        ).split("\n")[0]

    # 引数, 入力の口, 出力の口を決めて登録
    params, extra = params_from_init(cls)
    inputs, forward_params = ports_from_forward(cls)
    ports = output_ports(outputs)
    spec = UnitSpec(
        name=unit_name, category=category, source="custom",
        description=description, params=params, inputs=inputs, outputs=ports,
        forward_params=forward_params, extra_params=extra,
        builder=make_builder(cls), code_name=cls.__name__,
        code_import=f"from {cls.__module__} import {cls.__name__}"
    )
    apply_notes(spec, {"params": notes or {}})
    describe_params(spec, cls)
    register(spec)

    return spec
