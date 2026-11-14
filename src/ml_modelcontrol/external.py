import importlib
import importlib.util
import inspect
import warnings
from typing import Any, Optional

import torch
from torch import nn

from ml_modelcontrol.registry import (
    ParamSpec, PortSpec, UnitSpec, apply_notes, needs_module,
    params_from_init, ports_from_forward, register
)

# ライブラリが無い場合も一覧に出す, PyTorch Geometric のよく使う部品
GEOMETRIC_COMMON = [
    "GCNConv", "GATConv", "SAGEConv", "GINConv", "GraphConv",
    "TransformerConv", "MeanAggregation", "SumAggregation", "MaxAggregation"
]

# 部品にしない PyTorch Geometric のクラス (土台, CUDA だけのクラス)
EXCLUDED_GEOMETRIC = {"MessagePassing", "Aggregation", "MultiAggregation"}

# よく使う PyTorch Geometric の部品の説明
GEOMETRIC_NOTES = {
    "GCNConv": {
        "description": (
            "グラフ畳み込み層 (GCN). x は (ノード数, 特徴量), edge_index は "
            "(2, エッジ数) の int64"
        ),
        "params": {
            "in_channels": {"description": "入力の特徴量の数", "minimum": 1},
            "out_channels": {"description": "出力の特徴量の数", "minimum": 1}
        }
    },
    "GATConv": {
        "description": "注意機構によるグラフ畳み込み層 (GAT)",
        "params": {
            "out_channels": {"description": "ヘッドごとの出力の特徴量の数",
                             "minimum": 1},
            "heads": {"description": "ヘッドの数", "minimum": 1}
        }
    },
    "SAGEConv": {"description": "近傍を集約するグラフ畳み込み層 (GraphSAGE)"},
    "GraphConv": {"description": "重み付きの近傍の和によるグラフ畳み込み層"},
    "TransformerConv": {"description": "Transformer の注意によるグラフ畳み込み層"},
    "MeanAggregation": {
        "description": (
            "ノードの特徴量をグラフごとに平均する (index にグラフの番号 batch "
            "をつなぐ. つながない場合は全体の平均)"
        )
    },
    "SumAggregation": {"description": "ノードの特徴量をグラフごとに合計する"},
    "MaxAggregation": {"description": "ノードの特徴量をグラフごとの最大値にする"}
}

# Hugging Face のモデルの種類 (部品の名前, transformers のクラス, 説明, 出力の口)
HUGGINGFACE_MODELS = [
    (
        "AutoModel", "AutoModel",
        "Hugging Face のモデルの本体 (BERT など). 出力は最後の層の隠れ状態",
        [("last_hidden_state", "最後の層の隠れ状態 (バッチ, 系列, 隠れ層)"),
         ("pooler_output", "文全体の表現 (モデルによっては None)")]
    ),
    (
        "AutoModelForSequenceClassification",
        "AutoModelForSequenceClassification",
        "Hugging Face の文の分類のモデル. 出力は分類のロジット",
        [("logits", "分類のロジット (バッチ, num_labels)")]
    )
]

# Hugging Face のモデルの共通の引数
HUGGINGFACE_PARAMS = [
    ParamSpec("model_name", "str", required=True,
              description="モデルの名前 (例: bert-base-uncased) かローカルのパス"),
    ParamSpec("pretrained", "bool", True,
              description="学習済みの重みを読むか (false なら設定だけから作る)"),
    ParamSpec("revision", "str", "main", description="モデルの版 (ブランチかコミット)"),
    ParamSpec("local_files_only", "bool", False,
              description="手元のファイルだけを使い, ネットワークを使わないか")
]


# Hugging Face のモデルを, 定義の口で呼べるようにする入れ物
class HuggingFaceModel(nn.Module):
    def __init__(
        self, kind: str, model_name: str, pretrained: bool = True,
        revision: str = "main", local_files_only: bool = False,
        num_labels: Optional[int] = None
    ):
        # 親クラスのコンストラクタを呼び出す
        super(HuggingFaceModel, self).__init__()

        # transformers のクラスで, 重みを読むか設定だけから作成
        # 外部のコード (trust_remote_code) は使わない
        transformers = importlib.import_module("transformers")
        model_cls = getattr(transformers, kind)
        options: dict[str, Any] = {
            "revision": revision, "local_files_only": local_files_only,
            "trust_remote_code": False
        }
        extra = {} if num_labels is None else {"num_labels": num_labels}
        if pretrained:
            self.model = model_cls.from_pretrained(
                model_name, **options, **extra
            )
        else:
            config = transformers.AutoConfig.from_pretrained(
                model_name, **options, **extra
            )
            self.model = model_cls.from_config(config)

    def forward(
        self, input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> Any:
        """
        単語の番号から, モデルの出力を求める関数

        Parameters
        ----------
        input_ids: torch.Tensor
            単語の番号 (バッチ, 系列) の int64
        attention_mask: Optional[torch.Tensor] = None
            注意のマスク (1 が使う位置), None の場合は全て使う
        token_type_ids: Optional[torch.Tensor] = None
            文の区別の番号, None の場合は渡さない

        Returns
        ----------
        outputs: Any
            transformers のモデルの出力 (last_hidden_state などを持つ)
        """
        # 渡した値だけをキーワードで受け渡し
        kwargs = {"input_ids": input_ids}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        outputs = self.model(**kwargs)

        return outputs


def has_module(name: str) -> bool:
    """
    ライブラリが入っているかを, 読み込まずに確かめる関数

    Parameters
    ----------
    name: str
        ライブラリの名前

    Returns
    ----------
    found: bool
        入っていれば True
    """
    found = importlib.util.find_spec(name) is not None

    return found


def load_geometric_units() -> None:
    """
    PyTorch Geometric の畳み込み, 集約, 正規化の層を登録簿に読み込む関数

    ライブラリが無い場合は, よく使う部品を使えないものとして登録する

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # ライブラリが無い場合は理由付きで使えない部品にして終了
    if not has_module("torch_geometric"):
        for name in GEOMETRIC_COMMON:
            register(UnitSpec(
                name=f"pyg.{name}", category="graph", source="torch_geometric",
                available=False,
                reason="PyTorch Geometric (torch_geometric) が入っていません"
            ))
        return

    # 読み込みの警告 (PyTorch の古い機能の案内など) は表示しない
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        geometric = importlib.import_module("torch_geometric.nn")
    groups = [
        ("graph_conv", geometric.conv), ("graph_aggr", geometric.aggr),
        ("graph_norm", geometric.norm)
    ]
    for category, module in groups:
        for name in getattr(module, "__all__", []):
            cls = getattr(module, name, None)
            if (
                not inspect.isclass(cls) or not issubclass(cls, nn.Module)
                or name in EXCLUDED_GEOMETRIC or name.startswith("CuGraph")
                or name.startswith("Hetero")
            ):
                continue
            register_geometric_unit(cls, category)


def register_geometric_unit(cls: type, category: str) -> UnitSpec:
    """
    PyTorch Geometric のクラスを部品として登録する関数

    Parameters
    ----------
    cls: type
        torch_geometric.nn のクラス
    category: str
        分類 (graph_conv, graph_aggr, graph_norm)

    Returns
    ----------
    spec: UnitSpec
        登録した部品
    """
    # シグネチャから引数と口を取り出し, 作る関数を用意
    params, extra = params_from_init(cls)
    inputs, forward_params = ports_from_forward(cls)

    def builder(**values: object) -> nn.Module:
        """
        引数の値から部品を作る関数

        Parameters
        ----------
        **values: object
            クラスに渡す引数

        Returns
        ----------
        module: nn.Module
            作った部品
        """
        module = cls(**values)

        return module

    spec = UnitSpec(
        name=f"pyg.{cls.__name__}", category=category,
        source="torch_geometric", params=params, inputs=inputs,
        outputs=[PortSpec("out")], forward_params=forward_params,
        extra_params=extra, builder=builder, code_name=cls.__name__,
        code_import=f"from torch_geometric.nn import {cls.__name__}"
    )
    apply_notes(spec, GEOMETRIC_NOTES.get(cls.__name__, {}))

    # 別の部品を引数に取るクラス (GINConv など) は今は対象外
    if needs_module(cls):
        spec.available = False
        spec.reason = (
            "引数に別の部品 (nn.Module) が必要なので, 今は定義から作れません"
        )
    register(spec)

    return spec


def load_huggingface_units() -> None:
    """
    Hugging Face のモデルの部品を登録簿に読み込む関数

    ライブラリが無い場合は, 使えないものとして登録する

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 種類ごとに, 共通の引数と口で登録
    available = has_module("transformers")
    for name, kind, description, outputs in HUGGINGFACE_MODELS:
        params = [ParamSpec(**vars(param)) for param in HUGGINGFACE_PARAMS]
        if name == "AutoModelForSequenceClassification":
            params.append(ParamSpec(
                "num_labels", "int", 2, description="分類のクラスの数",
                minimum=1
            ))
        register(UnitSpec(
            name=f"hf.{name}", category="huggingface", source="transformers",
            description=description, params=params,
            inputs=[
                PortSpec("input_ids", "単語の番号 (バッチ, 系列) の int64"),
                PortSpec("attention_mask", "注意のマスク (1 が使う位置)",
                         optional=True),
                PortSpec("token_type_ids", "文の区別の番号", optional=True)
            ],
            outputs=[
                PortSpec(port, text, path=[port]) for port, text in outputs
            ],
            available=available,
            reason="" if available else (
                "Hugging Face の transformers が入っていません"
            ),
            builder=make_huggingface_builder(kind),
            code_name="HuggingFaceModel",
            code_import=(
                "from ml_modelcontrol.external import HuggingFaceModel"
            ),
            code_args=[kind]
        ))


def make_huggingface_builder(kind: str) -> Any:
    """
    Hugging Face のモデルの種類ごとに, 部品を作る関数を作る関数

    Parameters
    ----------
    kind: str
        transformers のクラスの名前 (AutoModel など)

    Returns
    ----------
    builder: Any
        引数を受け取って HuggingFaceModel を作る関数
    """
    def builder(**values: Any) -> nn.Module:
        """
        引数の値から Hugging Face のモデルの部品を作る関数

        Parameters
        ----------
        **values: Any
            model_name, pretrained, revision, local_files_only, num_labels

        Returns
        ----------
        module: nn.Module
            作った部品
        """
        module = HuggingFaceModel(kind, **values)

        return module

    return builder
