import json
from pathlib import Path

import pytest
import torch
import yaml
from torch import nn

from ml_modelcontrol import (
    SpecError, build_model, describe_units, dump_spec, get_unit, infer,
    load_spec, register_unit, validate
)
from ml_modelcontrol.infer import count_params

# 残差のつなぎ方と LSTM を持つモデルの定義
RESIDUAL = """
format_version: 1
name: residual
description: 残差のつなぎ方と LSTM を持つモデル
inputs:
  x: {shape: [3, 5, 6]}
nodes:
  fc1:
    unit: nn.Linear
    params: {in_features: 6, out_features: 6}
    inputs: {input: x}
    ui: {x: 10, y: 20}
  act: {unit: nn.ReLU, inputs: fc1}
  add: {unit: op.Add, inputs: {xs: [x, act]}}
  lstm:
    unit: nn.LSTM
    params: {input_size: 6, hidden_size: 4, batch_first: true}
    inputs: {input: add}
  head:
    unit: nn.Linear
    params: {in_features: 4, out_features: 2}
    inputs: "lstm:output"
outputs:
  logits: head
  h_n: "lstm:h_n"
"""


def make_spec(text: str = RESIDUAL) -> dict:
    """
    YAML の文字からモデルの定義を作る関数

    Parameters
    ----------
    text: str = RESIDUAL
        YAML の文字

    Returns
    ----------
    spec: dict
        モデルの定義
    """
    spec = load_spec(yaml.safe_load(text))

    return spec


def codes(spec: dict) -> set[str]:
    """
    確認の結果の識別子を集める関数

    Parameters
    ----------
    spec: dict
        モデルの定義

    Returns
    ----------
    found: set[str]
        誤りと注意の code
    """
    found = {issue["code"] for issue in validate(spec)}

    return found


def test_registry_lists_units() -> None:
    """
    登録簿が torch.nn の層と演算を, JSON にできる形で返すことを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 一覧は JSON にでき, 入れ物, 損失関数, 遅延初期化の層は対象外
    units = describe_units()
    json.dumps(units)
    names = {unit["name"] for unit in units}
    assert {"nn.Linear", "nn.Conv2d", "nn.LSTM", "op.Add"} <= names
    assert not {"nn.Sequential", "nn.MSELoss", "nn.LazyLinear"} & names

    # LSTM の出力の口と取り出し方, 注意機構の設定の値
    lstm = get_unit("nn.LSTM")
    assert [(port.name, port.path) for port in lstm.outputs] == [
        ("output", [0]), ("h_n", [1, 0]), ("c_n", [1, 1])
    ]
    attention = get_unit("nn.MultiheadAttention")
    assert [port.name for port in attention.inputs][:3] == [
        "query", "key", "value"
    ]
    assert "need_weights" in [param.name for param in attention.forward_params]
    linear = get_unit("nn.Linear")
    assert linear.params[0].description == "入力の特徴量の数"
    assert get_unit("op.Add").inputs[0].variadic


def test_spec_round_trip(tmp_path: Path) -> None:
    """
    モデルの定義を YAML と JSON に保存して読み戻せることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 保存して読み戻すと同じ定義 (ノードの順と ui も保持)
    spec = make_spec()
    for suffix in (".yaml", ".json"):
        path = tmp_path / f"model{suffix}"
        dump_spec(spec, path)
        loaded = load_spec(path)
        assert loaded == spec
        assert list(loaded["nodes"]) == list(spec["nodes"])
    text = dump_spec(spec)
    assert text.index("format_version") < text.index("nodes")

    # ライブラリより新しい版は読み込み不可
    spec["format_version"] = 99
    with pytest.raises(ValueError, match="format_version 99"):
        load_spec(spec)


def test_validate_reports_all_issues() -> None:
    """
    確認が, 最初の誤りで止めずにノードごとの誤りと注意を返すことを確かめる
    関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 正しい定義は誤りも注意も無い
    assert validate(make_spec()) == []

    # 様々な誤りを入れた定義
    spec = make_spec()
    nodes = spec["nodes"]
    spec["extra"] = 1
    nodes["fc1"]["params"]["out_feature"] = 3
    nodes["fc1"]["params"].pop("in_features")
    nodes["act"]["inputs"] = "missing"
    nodes["head"]["inputs"] = "lstm"
    nodes["type"] = {"unit": "nn.Nope"}
    nodes["drop"] = {"unit": "nn.Dropout", "params": {"p": 2.0}, "inputs": "x"}
    nodes["att"] = {
        "unit": "nn.MultiheadAttention",
        "params": {"embed_dim": 6, "num_heads": 2}, "inputs": "x"
    }
    found = codes(spec)
    assert {
        "unknown_key", "unknown_param", "missing_param", "unknown_ref",
        "ambiguous_ref", "reserved_name", "unknown_unit", "invalid_value",
        "ambiguous_inputs", "unused_node"
    } <= found

    # 誤りにノードの名前と場所
    issues = validate(spec)
    value = next(issue for issue in issues if issue["node"] == "drop")
    assert value["location"] == "params.p"
    assert "1.0 以下" in value["message"]


def test_validate_cycle_and_names() -> None:
    """
    循環, 入力と同じ名前, 口の誤りを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 2 つのノードが互いにつながる循環
    spec = make_spec()
    spec["nodes"]["fc1"]["inputs"] = {"input": "act"}
    assert "cycle" in codes(spec)

    # 入力と同じ名前のノード, 知らない口, 足りない口, 入力の型
    spec = make_spec()
    spec["nodes"]["x"] = {"unit": "nn.ReLU", "inputs": {"inputz": "fc1"}}
    spec["inputs"]["x"]["dtype"] = "float8"
    found = codes(spec)
    assert {
        "duplicate_name", "unknown_port", "missing_input", "invalid_dtype"
    } <= found


def test_build_model_matches_manual() -> None:
    """
    組み立てたモデルが, 同じ部品を手で計算した結果と一致することを確かめる
    関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 同じ種で 2 回組み立てると同じ重み, state_dict のキーはノードの名前
    model = build_model(make_spec(), seed=0)
    again = build_model(make_spec(), seed=0)
    for key, value in model.state_dict().items():
        assert torch.equal(value, again.state_dict()[key])
    assert "fc1.weight" in [key.removeprefix("nodes.") for key in (
        model.state_dict()
    )]

    # 手で同じ計算をした結果と一致 (入力は位置でも名前でも可)
    x = torch.randn(3, 5, 6)
    out = model(x)
    nodes = model.nodes
    hidden = x + torch.relu(nodes["fc1"](x))
    output, (h_n, _) = nodes["lstm"](hidden)
    assert torch.allclose(out["logits"], nodes["head"](output))
    assert torch.allclose(out["h_n"], h_n)
    assert torch.allclose(model(x=x)["logits"], out["logits"])

    # 誤りのある定義は組み立てずにエラー
    spec = make_spec()
    spec["nodes"]["fc1"]["unit"] = "nn.Nope"
    with pytest.raises(SpecError, match="nn.Nope"):
        build_model(spec)


def test_forward_params_and_optional_ports() -> None:
    """
    呼び出すときの設定の値と, 省略した口の後ろの口をキーワードで渡すことを
    確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 注意機構で key_padding_mask を省略し attn_mask をつなぎ, 重みは返さない設定
    spec = make_spec("""
inputs:
  q: {shape: [2, 4, 6]}
  mask: {shape: [4, 4], dtype: bool}
nodes:
  att:
    unit: nn.MultiheadAttention
    params: {embed_dim: 6, num_heads: 2, batch_first: true}
    forward_params: {need_weights: false}
    inputs: {query: q, key: q, value: q, attn_mask: mask}
outputs:
  out: "att:output"
  weights: "att:weights"
""")
    result = infer(spec)
    assert result["ok"]
    assert result["nodes"]["att"]["outputs"]["output"]["shape"] == [2, 4, 6]
    assert result["outputs"]["weights"] == {"kind": "none"}


def test_infer_shapes_errors_and_params() -> None:
    """
    形の推定が, ノードごとの形, パラメータ数, 失敗したノードとその先を返す
    ことを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 形とパラメータ数 (Linear 6x6+6, LSTM, Linear 4x2+2)
    result = infer(make_spec())
    assert result["ok"]
    assert result["nodes"]["head"]["outputs"]["out"]["shape"] == [3, 5, 2]
    assert result["outputs"]["h_n"]["shape"] == [1, 3, 4]
    lstm = 4 * (6 * 4 + 4 * 4 + 4 + 4)
    assert result["params"] == 42 + lstm + 10

    # 形の合わない層は失敗し, その先は計算なし
    spec = make_spec()
    spec["nodes"]["lstm"]["params"]["input_size"] = 7
    result = infer(spec)
    assert not result["ok"]
    assert result["nodes"]["lstm"]["status"] == "error"
    assert result["nodes"]["head"]["status"] == "skipped"
    assert result["nodes"]["fc1"]["status"] == "ok"

    # 上限を超えるモデルは計算せずにパラメータ数だけ
    result = infer(make_spec(), max_params=10)
    assert result["estimated_only"]
    assert result["params"] == 42 + lstm + 10
    assert result["nodes"] == {}


def test_infer_int_inputs_and_shared_params() -> None:
    """
    整数の入力 (単語の番号) と, 共有した重みを 1 回だけ数えることを確かめる
    関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 埋め込み層に通し番号の入力
    spec = make_spec("""
inputs:
  ids: {shape: [2, 5], dtype: int64, fill: arange}
nodes:
  emb:
    unit: nn.Embedding
    params: {num_embeddings: 10, embedding_dim: 3}
    inputs: ids
outputs:
  out: emb
""")
    result = infer(spec)
    assert result["nodes"]["emb"]["outputs"]["out"]["shape"] == [2, 5, 3]
    assert result["params"] == 30

    # 同じ重みは, 別の部品から数えても 1 回だけ
    shared = nn.Linear(2, 2)
    other = nn.Linear(2, 2)
    other.weight = shared.weight
    seen: set[int] = set()
    assert count_params(shared, seen) == (6, 6)
    assert count_params(other, seen) == (2, 2)
    shared.bias.requires_grad_(False)
    assert count_params(shared, set()) == (6, 4)


# 2 つの値を返す自作の部品
class SplitHalf(nn.Module):
    def __init__(self, dim: int = 1):
        # 親クラスのコンストラクタを呼び出す
        super(SplitHalf, self).__init__()

        # 分ける次元を保存
        self.dim = dim

    def forward(self, input: torch.Tensor) -> tuple:
        """
        入力を決めた次元で半分に分ける関数

        Parameters
        ----------
        input: torch.Tensor
            分けるテンソル

        Returns
        ----------
        halves: tuple
            前半と後半
        """
        halves = input.chunk(2, dim=self.dim)

        return halves


def test_register_custom_unit() -> None:
    """
    自作の部品を登録して, 出力の口ごとにつなげることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 2 つの出力の口を持つ部品として登録
    unit = register_unit(
        SplitHalf, name="custom.SplitHalf", outputs=["first", "second"],
        notes={"dim": {"description": "分ける次元"}}
    )
    assert unit.params[0].description == "分ける次元"
    spec = make_spec("""
inputs:
  x: {shape: [2, 6]}
nodes:
  split: {unit: custom.SplitHalf, inputs: x}
  add: {unit: op.Add, inputs: {xs: ["split:first", "split:second"]}}
outputs:
  y: add
""")
    x = torch.arange(12.0).reshape(2, 6)
    out = build_model(spec)(x)
    assert torch.equal(out, x[:, :3] + x[:, 3:])
    with pytest.raises(TypeError):
        register_unit(int)


# *args の後ろとキーワードだけの口を持つ自作の部品
class ScaledSum(nn.Module):
    def __init__(self):
        # 親クラスのコンストラクタを呼び出す
        super(ScaledSum, self).__init__()

    def forward(
        self, *xs: torch.Tensor, scale: torch.Tensor = None
    ) -> torch.Tensor:
        """
        全ての入力の和に, 決めた値を掛ける関数

        Parameters
        ----------
        *xs: torch.Tensor
            足し合わせるテンソル
        scale: torch.Tensor = None
            掛ける値, None の場合は掛けない

        Returns
        ----------
        total: torch.Tensor
            入力の和 (scale がある場合は掛けた値)
        """
        # 和を求めて掛け算
        total = sum(xs)
        if scale is not None:
            total = total * scale

        return total


def test_review_fixes(tmp_path: Path) -> None:
    """
    引数の種類, 口の渡し方, 必須の null, YAML の数, 定義のファイルの確認,
    組み込みの部品の上書き, 形の無い入力を確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # Conv の padding の same と組の stride, 1e-5 の書き方の eps
    path = tmp_path / "conv.yaml"
    path.write_text(
        "inputs:\n  x: {shape: [2, 1, 5, 5]}\n"
        "nodes:\n  conv:\n    unit: nn.Conv2d\n"
        "    params: {in_channels: 1, out_channels: 2, kernel_size: [3, 3],"
        " padding: same, stride: [1, 1]}\n    inputs: x\n"
        "  norm:\n    unit: nn.BatchNorm2d\n"
        "    params: {num_features: 2, eps: 1e-5}\n    inputs: conv\n"
        "outputs:\n  y: norm\n", encoding="utf-8"
    )
    assert validate(path) == []
    assert infer(path)["outputs"]["y"]["shape"] == [2, 2, 5, 5]

    # *args の後ろの口はキーワードで受け渡し
    register_unit(ScaledSum, name="custom.ScaledSum", overwrite=True)
    spec = make_spec("""
inputs:
  x: {shape: [2, 3]}
  s: {shape: [2, 3], fill: ones}
nodes:
  sum: {unit: custom.ScaledSum, inputs: {xs: [x, x], scale: s}}
outputs:
  y: sum
""")
    x = torch.randn(2, 3)
    assert torch.allclose(build_model(spec)(x, x * 0 + 2), x * 4)

    # 必須の引数の null, 新しい版, 組み込みの部品の上書き
    spec = make_spec()
    spec["nodes"]["fc1"]["params"]["in_features"] = None
    assert "missing_param" in codes(spec)
    spec["format_version"] = 99
    assert validate(spec)[0]["code"] == "invalid_spec"
    with pytest.raises(ValueError, match="登録済み"):
        register_unit(ScaledSum, name="nn.Linear")

    # 形の無い入力がある場合は計算なし
    spec = make_spec()
    spec["inputs"]["x"].pop("shape")
    result = infer(spec)
    assert not result["ok"]
    assert result["nodes"] == {}
    assert "no_shape" in {issue["code"] for issue in result["issues"]}
