import inspect
from typing import Callable, Optional, Union

from torch import nn

from ml_modelcontrol import ops
from ml_modelcontrol.external import (
    load_geometric_units, load_huggingface_units
)
from ml_modelcontrol.libraries import load_library_units
from ml_modelcontrol.registry import (
    REGISTRY, ParamSpec, PortSpec, UnitSpec, apply_notes, needs_module,
    params_from_init, ports_from_forward, register
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
    ("Select", ops.Select, "タプルの出力から index 番目を取り出す", False),
    ("GetAttr", ops.GetAttr, "出力から name の属性かキーの値を取り出す",
     False)
]

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
        ports = [
            PortSpec(port, path=[index]) for index, port in enumerate(outputs)
        ]
    else:
        ports = []
        for port, path in outputs.items():
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

    # 組み込みの部品と, 上書きしない場合の同じ名前の自作の部品は不可
    unit_name = name or cls.__name__
    existing = REGISTRY.get(unit_name)
    if existing is not None and (
        existing.source != "custom" or not overwrite
    ):
        raise ValueError(
            f"部品 {unit_name} は登録済みです (自作の部品は overwrite=True で"
            "上書き可)"
        )

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
    register(spec)

    return spec
