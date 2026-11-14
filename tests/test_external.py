import warnings
from pathlib import Path

import pytest
import torch

from ml_modelcontrol import (
    build_model, describe_units, export_python, get_unit, infer, validate
)
from test_core import make_spec

# GCN と GAT でノードの特徴量を求め, グラフごとに平均するモデルの定義
GRAPH = """
inputs:
  x: {shape: [5, 4]}
  edge_index: {shape: [2, 6], dtype: int64}
  batch: {shape: [5], dtype: int64}
nodes:
  conv1:
    unit: pyg.GCNConv
    params: {in_channels: 4, out_channels: 8}
    inputs: {x: x, edge_index: edge_index}
  act: {unit: nn.ReLU, inputs: conv1}
  conv2:
    unit: pyg.GATConv
    params: {in_channels: 8, out_channels: 3, heads: 2}
    inputs: {x: act, edge_index: edge_index}
  pool: {unit: pyg.MeanAggregation, inputs: {x: conv2, index: batch}}
outputs:
  y: pool
"""


def run_exported(code: str, inputs: tuple) -> tuple:
    """
    書き出したコードを実行してモデルを作る関数

    Parameters
    ----------
    code: str
        export_python のコード
    inputs: tuple
        使わない (呼び出し側の値の形を示すため)

    Returns
    ----------
    model: tuple
        書き出したクラスのモデルと, 読み込んだ名前の空間
    """
    # 名前の空間でコードを実行してクラスを取り出し
    namespace: dict = {}
    exec(compile(code, "<exported>", "exec"), namespace)
    model = namespace["Model"]()

    return model, namespace


def test_export_matches_build() -> None:
    """
    書き出したコードのモデルが, 組み立てたモデルと同じ state_dict のキーと
    同じ出力を持つことを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 同じ重みを読み込んで同じ入力を通すと, 同じ出力
    spec = make_spec()
    built = build_model(spec, seed=0)
    code = export_python(spec)
    exported, _ = run_exported(code, ())
    assert list(exported.state_dict()) == list(built.state_dict())
    exported.load_state_dict(built.state_dict())
    x = torch.randn(3, 5, 6)
    expected = built(x)
    actual = exported(x)
    for key in expected:
        assert torch.allclose(actual[key], expected[key])
    assert "self.nodes['head'](input=lstm[0])" in code


def test_geometric_units() -> None:
    """
    PyTorch Geometric の部品で, グラフのモデルを組み立てて形を推定できる
    ことを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # ライブラリが無い場合は使えない部品として一覧に出る
    pytest.importorskip("torch_geometric")
    gcn = get_unit("pyg.GCNConv")
    assert gcn.available
    assert [port.name for port in gcn.inputs] == [
        "x", "edge_index", "edge_weight"
    ]
    assert not get_unit("pyg.GINConv").available

    # 形の推定 (ノードの数はそのまま, 平均で 1 つのグラフ)
    spec = make_spec(GRAPH)
    assert validate(spec) == []
    result = infer(spec)
    assert result["ok"]
    assert result["nodes"]["conv2"]["outputs"]["out"]["shape"] == [5, 6]
    assert result["outputs"]["y"]["shape"] == [1, 6]

    # 書き出したコードも同じ出力
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        built = build_model(spec, seed=0)
        exported, _ = run_exported(export_python(spec), ())
    exported.load_state_dict(built.state_dict())
    x = torch.randn(5, 4)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 0], [1, 2, 3, 4, 0, 2]])
    batch = torch.tensor([0, 0, 0, 1, 1])
    built.eval()
    exported.eval()
    assert torch.allclose(
        built(x, edge_index, batch), exported(x, edge_index, batch)
    )


def test_huggingface_units(tmp_path: Path) -> None:
    """
    手元に保存した小さな Hugging Face のモデルを部品として読み込み, 形を
    推定できることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 小さな BERT を保存し, ネットワークを使わずに読み込む定義
    transformers = pytest.importorskip("transformers")
    config = transformers.BertConfig(
        vocab_size=50, hidden_size=8, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=16
    )
    transformers.BertModel(config).save_pretrained(tmp_path)
    spec = make_spec(f"""
inputs:
  ids: {{shape: [2, 5], dtype: int64}}
  mask: {{shape: [2, 5], dtype: int64, fill: ones}}
nodes:
  bert:
    unit: hf.AutoModel
    params: {{model_name: "{tmp_path}", local_files_only: true}}
    inputs: {{input_ids: ids, attention_mask: mask}}
  head:
    unit: nn.Linear
    params: {{in_features: 8, out_features: 2}}
    inputs: "bert:last_hidden_state"
outputs:
  y: head
""")
    result = infer(spec)
    assert result["ok"], result
    assert result["outputs"]["y"]["shape"] == [2, 5, 2]
    code = export_python(spec)
    assert "HuggingFaceModel(" in code and "'AutoModel'," in code

    # 外部のコードは定義から有効にできない (引数に無い)
    names = [param.name for param in get_unit("hf.AutoModel").params]
    assert "trust_remote_code" not in names
    assert {"hf.AutoModel"} <= {unit["name"] for unit in describe_units()}


def test_review_fixes_external(tmp_path: Path) -> None:
    """
    書き出しの名前の扱い, 部品を作れない場合の形の推定, プーラーの無い
    Hugging Face のモデルを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # モデルの名前はコードとして解釈されない
    spec = make_spec()
    spec["name"] = 'a"""\nPWNED = 1\n"""'
    namespace: dict = {}
    exec(compile(export_python(spec), "<exported>", "exec"), namespace)
    assert "PWNED" not in namespace

    # 読み込む名前と重なるノードとクラスの名前は書き出さない
    spec = make_spec()
    spec["nodes"]["nn"] = spec["nodes"].pop("act")
    spec["nodes"]["add"]["inputs"] = {"xs": ["x", "nn"]}
    with pytest.raises(ValueError, match="重なる"):
        export_python(spec)
    with pytest.raises(ValueError, match="重なる"):
        export_python(make_spec(), class_name="nn")

    # 確認は通るが作れない引数 (割り切れないヘッドの数) は, そのノードの
    # エラー
    spec = make_spec("""
inputs:
  q: {shape: [2, 4, 6]}
nodes:
  fc: {unit: nn.Linear, params: {in_features: 6, out_features: 6}, inputs: q}
  att:
    unit: nn.MultiheadAttention
    params: {embed_dim: 6, num_heads: 4, batch_first: true}
    inputs: {query: fc, key: fc, value: fc}
outputs:
  y: "att:output"
""")
    result = infer(spec)
    assert not result["ok"]
    assert result["nodes"]["att"]["status"] == "error"
    assert result["nodes"]["fc"]["status"] == "skipped"

    # プーラーの無いモデルも, last_hidden_state の形を推定できる
    transformers = pytest.importorskip("transformers")
    config = transformers.DistilBertConfig(
        vocab_size=50, dim=8, n_layers=1, n_heads=2, hidden_dim=16
    )
    transformers.DistilBertModel(config).save_pretrained(tmp_path)
    spec = make_spec(f"""
inputs:
  ids: {{shape: [2, 5], dtype: int64}}
nodes:
  bert:
    unit: hf.AutoModel
    params: {{model_name: "{tmp_path}", local_files_only: true}}
    inputs: {{input_ids: ids}}
outputs:
  y: "bert:last_hidden_state"
""")
    result = infer(spec)
    assert result["ok"], result
    assert result["nodes"]["bert"]["outputs"]["pooler_output"] == {
        "kind": "none"
    }
    assert "unchecked_model" in {issue["code"] for issue in validate(spec)}
