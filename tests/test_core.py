import sys
import json
from pathlib import Path

import pytest
import torch
import yaml
from torch import nn

from ml_modelcontrol import (
    SpecError, build_model, describe_units, dump_spec, export_python,
    get_unit, infer, load_spec, register_unit, validate
)
from ml_modelcontrol.infer import count_params
from ml_modelcontrol.registry import REGISTRY

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

    # 版を書かない定義は版 1 として読む
    unversioned = make_spec()
    del unversioned["format_version"]
    assert load_spec(unversioned)["format_version"] == 1


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
    nodes["spare"] = {"unit": "nn.ReLU", "inputs": "x"}
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


# 辞書で値を返す自作の部品
class Stats(nn.Module):
    def forward(self, input: torch.Tensor) -> dict:
        """
        最後の次元の平均と, 最大値と位置を求める関数

        Parameters
        ----------
        input: torch.Tensor
            入力

        Returns
        ----------
        stats: dict
            mean (平均) と max (torch.max の結果. values と indices)
        """
        stats = {"mean": input.mean(dim=-1), "max": input.max(dim=-1)}

        return stats


def test_register_custom_output_paths() -> None:
    """
    自作の部品の出力の口を, キーと属性の並びで取り出せることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # キーと, キーの後の属性で取り出す口
    register_unit(
        Stats, name="custom.Stats",
        outputs={"mean": ["mean"], "argmax": ["max", "indices"]},
        overwrite=True
    )
    spec = make_spec("""
inputs:
  x: {shape: [2, 3, 4]}
nodes:
  stats: {unit: custom.Stats, inputs: x}
outputs:
  mean: "stats:mean"
  argmax: "stats:argmax"
""")
    assert validate(spec) == []
    result = infer(spec)
    assert result["outputs"]["mean"]["shape"] == [2, 3]
    assert result["outputs"]["argmax"]["dtype"] == "int64"

    # 書き出したコードも, 名前付きのタプルの属性を取り出して同じ値
    code = export_python(spec)
    assert "_pick(_pick(stats, 'max'), 'indices')" in code
    namespace: dict = {}
    exec(compile(code, "<exported>", "exec"), namespace)
    x = torch.randn(2, 3, 4)
    expected = build_model(spec)(x)
    actual = namespace["Model"]()(x)
    assert torch.equal(actual["argmax"], expected["argmax"])
    assert torch.allclose(actual["mean"], expected["mean"])

    # 取り出し方が番号と名前でない場合と, 口の名前が識別子でない場合の失敗
    with pytest.raises(ValueError, match="取り出し方"):
        register_unit(
            Stats, name="custom.Bad", outputs={"x": [1.5]}, overwrite=True
        )
    with pytest.raises(ValueError, match="1 つ以上"):
        register_unit(Stats, name="custom.Empty", outputs={})
    with pytest.raises(ValueError, match="識別子"):
        register_unit(
            Stats, name="custom.BadPort", outputs={"a:b": ["mean"]}
        )


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
    with pytest.raises(ValueError, match="組み込みの部品"):
        register_unit(ScaledSum, name="nn.Linear")

    # 形の無い入力がある場合は計算なし
    spec = make_spec()
    spec["inputs"]["x"].pop("shape")
    result = infer(spec)
    assert not result["ok"]
    assert result["nodes"] == {}
    assert "no_shape" in {issue["code"] for issue in result["issues"]}


def test_input_checks() -> None:
    """
    入力の型の書き間違い (リスト) と, 大きすぎる入力の例を確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 型と値の作り方にリストを書いても, 例外ではなく誤り
    spec = make_spec()
    spec["inputs"]["x"]["dtype"] = ["float32"]
    spec["inputs"]["x"]["fill"] = {"a": 1}
    assert {"invalid_dtype", "invalid_fill"} <= codes(spec)

    # 要素数が上限を超える入力の例は計算せずに注意
    spec = make_spec()
    spec["inputs"]["x"]["shape"] = [100000, 100000, 6]
    result = infer(spec)
    assert not result["ok"]
    assert result["nodes"] == {}
    assert "input_too_large" in {issue["code"] for issue in result["issues"]}


def test_param_choices() -> None:
    """
    型ヒントで選択肢が分からない文字の引数にも, 選択肢が付くことを確かめる
    関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 選択肢の付いた引数と, 選択肢に無い値の誤り
    layer = get_unit("nn.TransformerEncoderLayer")
    activation = next(
        param for param in layer.params if param.name == "activation"
    )
    assert (activation.kind, activation.choices) == ("str", ["relu", "gelu"])
    spec = make_spec("""
inputs:
  x: {shape: [2, 5, 8]}
nodes:
  enc:
    unit: nn.TransformerEncoderLayer
    params: {d_model: 8, nhead: 2, batch_first: true, activation: swish}
    inputs: x
outputs:
  y: enc
""")
    assert [
        (issue["code"], issue["location"]) for issue in validate(spec)
    ] == [("invalid_value", "params.activation")]
    spec["nodes"]["enc"]["params"]["activation"] = "gelu"
    assert infer(spec)["outputs"]["y"]["shape"] == [2, 5, 8]


def test_export_line_width() -> None:
    """
    書き出すコードの行が 79 文字に収まり, 折り返しても実行できることを
    確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 引数の多い部品と長い引数で, 作る式と呼び出しの両方の折り返し
    spec = make_spec("""
inputs:
  source_tokens: {shape: [2, 5, 32]}
nodes:
  encoder_block:
    unit: nn.TransformerEncoderLayer
    params: {d_model: 32, nhead: 4, dim_feedforward: 64, dropout: 0.0,
             batch_first: true, norm_first: true, layer_norm_eps: 0.00001}
    inputs: {src: source_tokens}
    forward_params: {is_causal: false}
outputs:
  y: encoder_block
""")
    code = export_python(spec)
    assert max(len(line) for line in code.splitlines()) <= 79
    namespace: dict = {}
    exec(compile(code, "<exported>", "exec"), namespace)
    built = build_model(spec, seed=0).eval()
    exported = namespace["Model"]().eval()
    exported.load_state_dict(built.state_dict())
    x = torch.randn(2, 5, 32)
    assert torch.allclose(built(x), exported(x))


def test_infer_without_outputs() -> None:
    """
    出力をまだ書いていない組み立ての途中でも, ノードごとの形を推定する
    ことを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 出力が無いことは誤りのまま, ノードの形とパラメータ数は求める
    spec = make_spec("""
inputs:
  x: {shape: [2, 8]}
nodes:
  fc: {unit: nn.Linear, params: {in_features: 8, out_features: 3}, inputs: x}
outputs: {}
""")
    result = infer(spec)
    assert not result["ok"]
    assert "no_outputs" in {issue["code"] for issue in result["issues"]}
    assert result["nodes"]["fc"]["outputs"]["out"]["shape"] == [2, 3]
    assert result["params"] == 27

    # 引数の誤りのあるノードは, 確認の誤りの文を持つエラー
    spec["nodes"]["fc"]["params"]["out_features"] = 0
    node = infer(spec)["nodes"]["fc"]
    assert node["status"] == "error"
    assert "out_features" in node["error"]


def test_extra_param_names() -> None:
    """
    一覧に無い引数の名前は, Python のキーワード引数として書ける名前だけに
    することを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # **kwargs を持つ自作の部品に, 予約語と識別子でない名前の引数
    class Scaled(nn.Module):
        def __init__(self, scale: float = 1.0, **options: object):
            # 親クラスのコンストラクタを呼び出す
            super(Scaled, self).__init__()
            self.scale = scale

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            入力を scale 倍する関数

            Parameters
            ----------
            x: torch.Tensor
                入力

            Returns
            ----------
            y: torch.Tensor
                scale 倍した値
            """
            y = x * self.scale

            return y

    register_unit(Scaled, name="custom.Scaled", overwrite=True)
    spec = make_spec("""
inputs:
  x: {shape: [2, 3]}
nodes:
  s: {unit: custom.Scaled, inputs: x}
outputs:
  y: s
""")
    for key in ("class", "a-b"):
        spec["nodes"]["s"]["params"] = {key: 1}
        assert "invalid_param_name" in codes(spec)
    spec["nodes"]["s"]["params"] = {"gain": 1}
    assert codes(spec) == set()


def test_partial_infer() -> None:
    """
    誤りのあるノードとその先だけを除き, 残りのノードの形を推定することを
    確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # つないでいないノード (複製した直後など) と, 部品を作れないノード
    spec = make_spec("""
inputs:
  x: {shape: [2, 8]}
nodes:
  fc: {unit: nn.Linear, params: {in_features: 8, out_features: 6}, inputs: x}
  act: {unit: nn.ReLU, inputs: fc}
  loose: {unit: nn.Linear, params: {in_features: 6, out_features: 2}}
  att:
    unit: nn.MultiheadAttention
    params: {embed_dim: 6, num_heads: 4}
    inputs: {query: act, key: act, value: act}
  after: {unit: nn.ReLU, inputs: "att:output"}
outputs:
  y: act
  z: after
""")
    result = infer(spec)
    assert not result["ok"]
    nodes = result["nodes"]
    assert nodes["fc"]["status"] == "ok"
    assert nodes["act"]["outputs"]["out"]["shape"] == [2, 6]
    assert nodes["loose"]["status"] == "error"
    assert nodes["att"]["status"] == "error"
    assert nodes["after"]["status"] == "skipped"
    assert result["outputs"]["y"]["shape"] == [2, 6]
    assert "z" not in result["outputs"]
    assert result["params"] == 54

    # 文字でない出力の参照と口を書かない複数の出力は, 計算せずに出力のエラー
    spec["outputs"] = {"y": "act", "bad": 5}
    result = infer(spec)
    assert result["outputs"]["bad"]["kind"] == "error"
    assert result["outputs"]["y"]["shape"] == [2, 6]

    # 確認は通るが作れない部品だけの誤りでも ok ではなく, 作れる部分の
    # パラメータ数で上限を確かめる
    spec = make_spec("""
inputs:
  x: {shape: [1, 4, 8, 8]}
nodes:
  conv:
    unit: nn.Conv2d
    params: {in_channels: 3, out_channels: 4, kernel_size: 1, groups: 2}
    inputs: x
  big:
    unit: nn.Linear
    params: {in_features: 8, out_features: 3000}
    inputs: x
outputs:
  y: big
""")
    result = infer(spec)
    assert not result["ok"]
    assert result["nodes"]["conv"]["status"] == "error"
    assert list(result["nodes"]) == ["conv", "big"]
    limited = infer(spec, max_params=1000)
    assert limited["estimated_only"]
    assert limited["params"] == 27000


def test_export_name_warning() -> None:
    """
    書き出すコードで使う名前のノードに, 確認で注意を出すことを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 確認の時点で注意 (組み立てはできる), 書き出しは失敗
    spec = make_spec()
    spec["nodes"]["result"] = spec["nodes"].pop("act")
    spec["nodes"]["add"]["inputs"] = {"xs": ["x", "result"]}
    assert "export_name" in codes(spec)
    build_model(spec)
    with pytest.raises(ValueError, match="重なる"):
        export_python(spec)


def test_validate_messages() -> None:
    """
    書き間違いの候補, 循環の先の注意, 誤りの先の注意の抑え方, 省略形を
    確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 部品, 引数, つなぎ先の書き間違いに近い候補
    spec = make_spec("""
inputs:
  x: {shape: [2, 8]}
nodes:
  fc: {unit: nn.Linaer, inputs: x}
  fc2:
    unit: nn.Linear
    params: {in_features: 8, out_feature: 3}
    inputs: x
  add: {unit: op.Add, inputs: fc22}
  a: {unit: nn.ReLU, inputs: b}
  b: {unit: nn.ReLU, inputs: a}
  c: {unit: nn.ReLU, inputs: b}
outputs:
  y: c
""")
    issues = {(item["code"], item["node"]): item for item in validate(spec)}
    assert "nn.Linear" in issues[("unknown_unit", "fc")]["message"]
    assert "out_features" in issues[("unknown_param", "fc2")]["message"]
    assert "fc2" in issues[("unknown_ref", "add")]["message"]

    # 循環に入るノードだけが誤りで, 循環の先は注意
    assert ("cycle", "a") in issues and ("cycle", "b") in issues
    assert ("after_cycle", "c") in issues
    assert ("cycle", "c") not in issues

    # 可変の数の口だけの部品は, つなぎ先の文字 1 つでも書ける
    spec = make_spec("""
inputs:
  x: {shape: [2, 8]}
nodes:
  add: {unit: op.Add, inputs: x}
outputs:
  y: add
""")
    assert [issue["code"] for issue in validate(spec)] == ["single_input"]

    # nodes がリストの場合は種類の誤り
    spec = make_spec()
    spec["nodes"] = [1, 2]
    assert "invalid_nodes" in codes(spec)


def test_runtime_checks(tmp_path: Path) -> None:
    """
    実行時の誤りのノードの名前, 自作の部品の無いキー, register_unit の値,
    読めないファイルの扱いを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 組み立てたモデルの部品の中の誤りへの, ノードの名前の付加
    model = build_model(make_spec(), seed=0)
    with pytest.raises(RuntimeError) as caught:
        model(torch.randn(3, 5, 7))
    notes = getattr(caught.value, "__notes__", [])
    assert sys.version_info < (3, 11) or any("ノード" in note for note in notes)

    # 自作の部品の辞書に無いキーは, None ではなく誤り
    register_unit(
        Stats, name="custom.StatsMissing",
        outputs={"zzz": ["nokey"], "mean": ["mean"]}, overwrite=True
    )
    spec = make_spec("""
inputs:
  x: {shape: [2, 3, 4]}
nodes:
  s: {unit: custom.StatsMissing, inputs: x}
outputs:
  y: "s:zzz"
""")
    node = infer(spec)["nodes"]["s"]
    assert node["status"] == "error" and "nokey" in node["error"]
    with pytest.raises(ValueError, match="ノード s"):
        build_model(spec)(torch.randn(2, 3, 4))

    # 組み込みの部品の頭の名前は使えない
    with pytest.raises(ValueError, match="組み込みの部品の頭"):
        register_unit(Stats, name="nn.MyStats")

    # register_unit の名前と口の確認, docstring の説明
    with pytest.raises(ValueError, match="名前"):
        register_unit(Stats, name="bad name!")
    with pytest.raises(ValueError, match="重なって"):
        register_unit(Stats, name="custom.Dup", outputs=["a", "a"])
    with pytest.raises(ValueError, match="識別子"):
        register_unit(Stats, name="custom.Hyphen", outputs=["out-1", "x"])

    # 説明を省いた場合の, そのクラス自身の docstring の 1 行目
    unit = register_unit(SplitHalf, name="custom.Doc", overwrite=True)
    assert unit.description == ""
    class Documented(nn.Module):
        """
        説明付きの自作の部品

        2 行目以降は使わない
        """

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            入力をそのまま返す関数

            Parameters
            ----------
            x: torch.Tensor
                入力

            Returns
            ----------
            x: torch.Tensor
                入力
            """
            return x

    unit = register_unit(Documented, name="custom.Doc2", overwrite=True)
    assert unit.description == "説明付きの自作の部品"

    # 登録した部品の片付け (他のテストの部品の一覧に残さない)
    for name in ("custom.StatsMissing", "custom.Doc", "custom.Doc2"):
        REGISTRY.pop(name, None)

    # 読めない YAML と, パスの代わりの YAML の文字は ValueError
    broken = tmp_path / "broken.yaml"
    broken.write_text("nodes: [", encoding="utf-8")
    assert validate(broken)[0]["code"] == "invalid_spec"
    with pytest.raises(ValueError, match="YAML の文字"):
        load_spec("format_version: 1\nnodes: {}")


def test_any_param_types() -> None:
    """
    種類が任意の引数も, 型ヒントから分かる種類 (int | tuple など) で値を
    確かめることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 型ヒントから取り出した種類
    conv = {param.name: param for param in get_unit("nn.Conv2d").params}
    assert conv["kernel_size"].types == ["int", "list"]
    norm = {param.name: param for param in get_unit("nn.BatchNorm2d").params}
    assert norm["momentum"].types == ["float", "null"]

    # 種類に合わない値は計算する前に誤り, 合う値は通る
    spec = make_spec("""
inputs:
  x: {shape: [1, 3, 8, 8]}
nodes:
  c:
    unit: nn.Conv2d
    params: {in_channels: 3, out_channels: 4, kernel_size: three}
    inputs: x
  b:
    unit: nn.BatchNorm2d
    params: {num_features: 4, momentum: x}
    inputs: c
outputs:
  y: b
""")
    locations = {
        issue["location"] for issue in validate(spec)
        if issue["code"] == "invalid_value"
    }
    assert locations == {"params.kernel_size", "params.momentum"}
    spec["nodes"]["c"]["params"]["kernel_size"] = [3, 3]
    spec["nodes"]["b"]["params"]["momentum"] = 0.1
    assert validate(spec) == []


def test_docstring_params():
    """
    docstring の Google 形式, NumPy 形式, Sphinx 形式から引数の説明を取り出す
    ことと, 名前ごとの説明 (日本語) を優先することを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    from ml_modelcontrol.registry import (
        UnitSpec, ParamSpec, describe_params, parse_doc_params
    )

    google = parse_doc_params(
        "Block.\n\nArgs:\n    width (int): Width of the\n        block.\n"
        "    gain: Output gain.\n\nReturns:\n    out: not a param.\n"
    )
    assert google == {"width": "Width of the block.", "gain": "Output gain."}
    numpy_style = parse_doc_params(
        "Block.\n\nParameters\n----------\nwidth : int\n    Width.\n"
        "a, b : float\n    Pair.\n\nReturns\n-------\nout : Tensor\n    No.\n"
    )
    assert numpy_style == {"width": "Width.", "a": "Pair.", "b": "Pair."}
    sphinx = parse_doc_params(":param int width: Width\n    of block.\n")
    assert sphinx == {"width": "Width of block."}

    # 引数の字下げの読めない行は前の引数に混ぜず, 入れ子の括弧の型と
    # *args も読み, reST の書き方は中の文字だけ
    assert parse_doc_params(
        "Args:\n    a: first\n    Note that x\n    b: second"
    ) == {"a": "first", "b": "second"}
    assert parse_doc_params(
        "Args:\n    a: first\n    c (Callable[(int), int]): third\n"
    ) == {"a": "first", "c": "third"}
    assert parse_doc_params(
        "Parameters\n----------\n*args : tuple\n    pos\nx : int\n    X.\n"
    ) == {"args": "pos", "x": "X."}
    assert parse_doc_params("Args:\n    a: If :obj:`True`, uses ``x``.") == {
        "a": "If True, uses x."
    }

    # 数式は命令と波括弧を除いた読める文字
    assert parse_doc_params(
        "Args:\n    a: computes :math:`\\mathbf{\\hat{A}} + 2\\mathbf{I}`."
    ) == {"a": "computes A + 2I."}

    # 名前ごとの説明があればそれ, 無ければ docstring
    class Documented(nn.Module):
        """
        Args:
            in_channels (int): channels in.
            depth (int): number of layers.
        """

    spec = UnitSpec(
        name="test.Documented", category="test", source="test",
        params=[
            ParamSpec("in_channels", "int", None),
            ParamSpec("depth", "int", None)
        ]
    )
    describe_params(spec, Documented)
    assert [param.description for param in spec.params] == [
        "入力のチャンネル数", "number of layers."
    ]


def test_unused_input_message_and_combination():
    """
    出力に届かない入力の注意の文と, tv.Model の重みとクラスの数の組み合わせ
    の誤りを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # つないだ先が出力に届かない入力と, どこにもつながない入力
    spec = {
        "inputs": {"x": {"shape": [2, 4]}, "z": {"shape": [2, 4]}},
        "nodes": {"a": {"unit": "nn.ReLU", "inputs": "x"}},
    }
    messages = {
        issue["location"]: issue["message"] for issue in validate(spec)
        if issue["code"] == "unused_input"
    }
    assert messages == {
        "inputs.x": "入力 x はつないだ先が出力に届きません",
        "inputs.z": "入力 z は使われていません"
    }

    # 学習済みの重みのクラスの数と違う num_classes は誤り
    if not get_unit("tv.Model").available:
        return
    spec = {
        "inputs": {"x": {"shape": [2, 3, 32, 32]}},
        "nodes": {"m": {
            "unit": "tv.Model", "inputs": "x",
            "params": {
                "name": "resnet18", "weights": "DEFAULT", "num_classes": 10
            }
        }},
        "outputs": {"y": "m"}
    }
    codes = [issue["code"] for issue in validate(spec)]
    assert "invalid_combination" in codes
    spec["nodes"]["m"]["params"]["weights"] = None
    assert "invalid_combination" not in [
        issue["code"] for issue in validate(spec)
    ]


def test_single_input_and_doc_summary():
    """
    まとめる演算のつなぎ先が 1 つの注意と, docstring の最初の文の部品の説明
    を確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    from ml_modelcontrol.registry import docstring_summary

    spec = {
        "inputs": {"x": {"shape": [2, 4]}},
        "nodes": {"add": {"unit": "op.Add", "inputs": {"xs": ["x"]}}},
        "outputs": {"y": "add"}
    }
    assert [issue["code"] for issue in validate(spec)] == ["single_input"]
    spec["nodes"]["add"]["inputs"] = {"xs": ["x", "x"]}
    assert validate(spec) == []

    class Documented(nn.Module):
        """
        The `"GCN" <https://example.com>`_ layer with :math:`x^2` from
        :obj:`torch`. More text.

        Args:
            a: not part of the summary.
        """

    assert docstring_summary(Documented) == (
        "The GCN layer with from torch."
    )


def test_error_shows_incoming_values():
    """
    計算に失敗したノードの誤りの文に, 入った値の形と型を添えることを確かめる
    関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    spec = {
        "inputs": {"x": {"shape": [4, 16]}},
        "nodes": {"fc": {
            "unit": "nn.Linear", "inputs": "x",
            "params": {"in_features": 8, "out_features": 2}
        }},
        "outputs": {"y": "fc"}
    }
    error = infer(spec)["nodes"]["fc"]["error"]
    assert "(入った値: input [4, 16] float32)" in error


def test_error_shape_hints():
    """
    形と型の食い違いの誤りに, 日本語の直し方を添えることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    spec = {
        "inputs": {"x": {"shape": [4, 16]}, "i": {"shape": [2, 3, 8, 8]}},
        "nodes": {
            "fc": {
                "unit": "nn.Linear", "inputs": "x",
                "params": {"in_features": 8, "out_features": 2}
            },
            "emb": {
                "unit": "nn.Embedding", "inputs": "x",
                "params": {"num_embeddings": 10, "embedding_dim": 3}
            },
            "conv": {
                "unit": "nn.Conv2d", "inputs": "i",
                "params": {
                    "in_channels": 1, "out_channels": 4, "kernel_size": 3
                }
            }
        },
        "outputs": {"y": "fc", "z": "emb", "w": "conv"}
    }
    nodes = infer(spec)["nodes"]
    assert "最後の次元 16 と in_features 8 が合いません" in nodes["fc"]["error"]
    assert "番号 (int64) を入れてください" in nodes["emb"]["error"]
    assert "in_channels を 3 にする" in nodes["conv"]["error"]


def test_token_ops() -> None:
    """
    位置の番号, 等しい所の印, 決めた次元の取り出しの演算を確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    from ml_modelcontrol import ops

    # 位置の番号はバッチの数だけ並べた (バッチ, 長さ) の整数
    ids = torch.tensor([[5, 6, 0], [7, 0, 0]])
    positions = ops.Positions()(ids)
    assert positions.tolist() == [[0, 1, 2], [0, 1, 2]]
    assert positions.dtype == torch.int64
    assert ops.Positions(start=1)(torch.zeros(1, 2, 4))[0].tolist() == [1, 2]

    # 詰め物 (0) の印と, 先頭の取り出し
    assert ops.Equal(0)(ids).tolist() == [
        [False, False, True], [False, True, True]
    ]
    hidden = torch.arange(24.0).reshape(2, 3, 4)
    assert torch.equal(ops.Index(1, 0)(hidden), hidden[:, 0])

    # 部品から組んだ BERT の例は, 詰め物を変えても先頭の出力が変わらない
    from ml_modelcontrol import load_example

    model = build_model(load_example("tiny_bert"), seed=0).eval()
    short = torch.tensor([[3, 4, 5, 0, 0]])
    longer = torch.tensor([[3, 4, 5, 0, 0, 0, 0]])
    with torch.no_grad():
        assert torch.allclose(model(short), model(longer), atol=1e-5)
