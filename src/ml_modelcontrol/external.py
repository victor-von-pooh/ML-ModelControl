import importlib
import inspect
import warnings
from typing import Any, Optional

import torch
from torch import nn

from ml_modelcontrol.libraries import has_module, register_class
from ml_modelcontrol.registry import (
    ParamSpec, PortSpec, UnitSpec, apply_notes, register
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

# Hugging Face のモデルの種類 (部品の名前, transformers のクラス, 入力
# (text は文, image は画像), 説明, 出力の口, 分類のクラスの数を持つか)
HUGGINGFACE_MODELS = [
    (
        "AutoModel", "AutoModel", "text",
        (
            "Hugging Face の文のモデルの本体 (BERT など). 出力は最後の層の"
            "隠れ状態"
        ),
        [("last_hidden_state", "最後の層の隠れ状態 (バッチ, 系列, 隠れ層)"),
         ("pooler_output", "文全体の表現 (モデルによっては None)")],
        False
    ),
    (
        "AutoModelForSequenceClassification",
        "AutoModelForSequenceClassification", "text",
        "Hugging Face の文の分類のモデル. 出力は分類のロジット",
        [("logits", "分類のロジット (バッチ, num_labels)")], True
    ),
    (
        "AutoModelForTokenClassification", "AutoModelForTokenClassification",
        "text",
        "Hugging Face の単語ごとの分類のモデル (固有表現の抽出など)",
        [("logits", "単語ごとのロジット (バッチ, 系列, num_labels)")], True
    ),
    (
        "AutoModelForMaskedLM", "AutoModelForMaskedLM", "text",
        "Hugging Face の穴埋めの言語モデル (BERT など)",
        [("logits", "語彙ごとのロジット (バッチ, 系列, 語彙)")], False
    ),
    (
        "AutoModelForCausalLM", "AutoModelForCausalLM", "text",
        "Hugging Face の次の単語を予測する言語モデル (GPT など)",
        [("logits", "語彙ごとのロジット (バッチ, 系列, 語彙)")], False
    ),
    (
        "AutoModelForQuestionAnswering", "AutoModelForQuestionAnswering",
        "text",
        "Hugging Face の抜き出し型の質問応答のモデル",
        [("start_logits", "答えの始まりの位置のロジット (バッチ, 系列)"),
         ("end_logits", "答えの終わりの位置のロジット (バッチ, 系列)")],
        False
    ),
    (
        "AutoVisionModel", "AutoModel", "image",
        (
            "Hugging Face の画像のモデルの本体 (ViT など. transformers の "
            "AutoModel に画像を渡す)"
        ),
        [("last_hidden_state", "最後の層の隠れ状態 (バッチ, 系列, 隠れ層)"),
         ("pooler_output", "画像全体の表現 (モデルによっては None)")],
        False
    ),
    (
        "AutoModelForImageClassification", "AutoModelForImageClassification",
        "image",
        "Hugging Face の画像の分類のモデル. 出力は分類のロジット",
        [("logits", "分類のロジット (バッチ, num_labels)")], True
    )
]

# 入力の種類ごとの口
HUGGINGFACE_INPUTS = {
    "text": [
        PortSpec("input_ids", "単語の番号 (バッチ, 系列) の int64"),
        PortSpec("attention_mask", "注意のマスク (1 が使う位置)",
                 optional=True),
        PortSpec("token_type_ids", "文の区別の番号", optional=True)
    ],
    "image": [
        PortSpec("pixel_values", "画像 (バッチ, 3, 高さ, 幅)")
    ]
}

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
        num_labels: Optional[int] = None,
        ignore_mismatched_sizes: bool = False
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
                model_name, **options, **extra,
                ignore_mismatched_sizes=ignore_mismatched_sizes
            )
        else:
            config = transformers.AutoConfig.from_pretrained(
                model_name, **options, **extra
            )
            self.model = model_cls.from_config(config)

    def forward(
        self, input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None
    ) -> Any:
        """
        単語の番号か画像から, モデルの出力を求める関数

        Parameters
        ----------
        input_ids: Optional[torch.Tensor] = None
            単語の番号 (バッチ, 系列) の int64 (文のモデルの場合)
        attention_mask: Optional[torch.Tensor] = None
            注意のマスク (1 が使う位置), None の場合は全て使う
        token_type_ids: Optional[torch.Tensor] = None
            文の区別の番号, None の場合は渡さない
        pixel_values: Optional[torch.Tensor] = None
            画像 (バッチ, 3, 高さ, 幅) (画像のモデルの場合)

        Returns
        ----------
        outputs: Any
            transformers のモデルの出力 (last_hidden_state などを持つ)
        """
        # 渡した値だけをキーワードで受け渡し
        values = {
            "input_ids": input_ids, "attention_mask": attention_mask,
            "token_type_ids": token_type_ids, "pixel_values": pixel_values
        }
        kwargs = {
            key: value for key, value in values.items() if value is not None
        }
        outputs = self.model(**kwargs)

        return outputs


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
    # ライブラリが無いか読み込めない場合は, 理由付きで使えない部品にして終了
    # 読み込みの警告 (PyTorch の古い機能の案内など) は表示しない
    reason = "PyTorch Geometric (torch_geometric) が入っていません"
    geometric = None
    if has_module("torch_geometric"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                geometric = importlib.import_module("torch_geometric.nn")
        except Exception as error:
            reason = (
                "PyTorch Geometric を読み込めません: "
                f"{type(error).__name__}: {error}"
            )
    if geometric is None:
        for name in GEOMETRIC_COMMON:
            spec = UnitSpec(
                name=f"pyg.{name}", category="graph",
                source="torch_geometric", available=False, reason=reason
            )
            apply_notes(spec, {
                "description": GEOMETRIC_NOTES.get(name, {}).get(
                    "description", ""
                )
            })
            register(spec)
        return
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
            try:
                register_geometric_unit(cls, category)
            except Exception:
                continue


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
    # 他のライブラリのクラスと同じ方法で登録
    spec = register_class(
        cls, f"pyg.{cls.__name__}", category, "torch_geometric",
        cls.__name__, f"from torch_geometric.nn import {cls.__name__}",
        GEOMETRIC_NOTES.get(cls.__name__, {})
    )

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
    for name, kind, input_kind, description, outputs, labels in (
        HUGGINGFACE_MODELS
    ):
        params = [ParamSpec(**vars(param)) for param in HUGGINGFACE_PARAMS]
        if labels:
            params += [
                ParamSpec(
                    "num_labels", "int", 2, description="分類のクラスの数",
                    minimum=1
                ),
                ParamSpec(
                    "ignore_mismatched_sizes", "bool", False,
                    description=(
                        "学習済みの分類の層と num_labels が違う場合に, 分類の"
                        "層を作り直すか"
                    )
                )
            ]
        register(UnitSpec(
            name=f"hf.{name}", category="huggingface", source="transformers",
            description=description, params=params,
            inputs=[
                PortSpec(**vars(port))
                for port in HUGGINGFACE_INPUTS[input_kind]
            ],
            outputs=[
                PortSpec(port, text, path=[port]) for port, text in outputs
            ],
            available=available,
            reason="" if available else (
                "Hugging Face の transformers が入っていません (pip install "
                "\"ml-modelcontrol[huggingface]\")"
            ),
            builder=make_huggingface_builder(kind),
            code_name="HuggingFaceModel",
            code_import=(
                "from ml_modelcontrol.external import HuggingFaceModel"
            ),
            code_args=[kind], offline={"pretrained": False}
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
