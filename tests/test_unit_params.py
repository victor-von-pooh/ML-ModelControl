from typing import Any

import pytest
import torch
import yaml
from torch import nn

from ml_modelcontrol import (
    build_model, describe_units, dump_spec, export_python, infer, load_spec,
    register_unit, validate
)
from ml_modelcontrol.registry import (
    REGISTRY, ParamSpec, PortSpec, UnitSpec, get_unit, register
)

# 層を重ねるエンコーダー (引数に層と正規化の部品を書く)
ENCODER = """
format_version: 3
inputs:
  x: {shape: [2, 5, 16]}
nodes:
  enc:
    unit: nn.TransformerEncoder
    params:
      encoder_layer:
        unit: nn.TransformerEncoderLayer
        params: {d_model: 16, nhead: 4, dim_feedforward: 32, dropout: 0.0,
                 batch_first: true}
      num_layers: 2
      norm: {unit: nn.LayerNorm, params: {normalized_shape: 16}}
      enable_nested_tensor: false
    inputs: {src: x}
outputs:
  y: enc
"""


class Residual(nn.Module):
    """
    引数の部品の出力を入力に足す部品 (テストの自作の部品)
    """

    def __init__(self, inner: nn.Module, scale: float = 1.0):
        # 親クラスのコンストラクタを呼び出す
        super(Residual, self).__init__()

        # 中の部品と倍率を保存
        self.inner = inner
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        中の部品の出力を倍にして入力に足す関数

        Parameters
        ----------
        x: torch.Tensor
            入力

        Returns
        ----------
        y: torch.Tensor
            入力と中の部品の出力の和
        """
        y = x + self.scale * self.inner(x)

        return y


@pytest.fixture
def residual() -> Any:
    """
    Residual を自作の部品として登録し, 終わったら外すフィクスチャ

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    register_unit(Residual, name="test.Residual", overwrite=True)
    yield
    REGISTRY.pop("test.Residual", None)


def issues_of(spec: dict) -> list[tuple[str, str]]:
    """
    確認の誤りの識別子と場所の組を集める関数

    Parameters
    ----------
    spec: dict
        モデルの定義

    Returns
    ----------
    found: list[tuple[str, str]]
        誤りの code と location の組
    """
    found = [
        (issue["code"], issue["location"]) for issue in validate(spec)
        if issue["severity"] == "error"
    ]

    return found


def test_unit_param_listed() -> None:
    """
    部品だけを受け取る引数を持つクラスが使える部品になり, 引数の種類が
    unit になることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 層の引数は型ヒントのクラスを候補に持ち, 省ける正規化は null が既定
    units = {unit["name"]: unit for unit in describe_units()}
    for name, layer, cls in (
        ("nn.TransformerEncoder", "encoder_layer", "TransformerEncoderLayer"),
        ("nn.TransformerDecoder", "decoder_layer", "TransformerDecoderLayer")
    ):
        assert units[name]["available"], units[name]["reason"]
        params = {param["name"]: param for param in units[name]["params"]}
        assert params[layer]["kind"] == "unit"
        assert params[layer]["required"]
        assert params[layer]["suggestions"] == [f"nn.{cls}"]
        assert (params["norm"]["kind"], params["norm"]["default"]) == (
            "unit", None
        )


def test_unit_param_build_infer_export() -> None:
    """
    引数に部品を書いた定義を, 形の推定, 組み立て, Python の書き出しで同じ
    計算にできることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 確かめ, 形の推定 (層を重ねた数のパラメータ)
    spec = load_spec(yaml.safe_load(ENCODER))
    assert issues_of(spec) == []
    result = infer(spec)
    assert result["ok"], result
    assert result["outputs"]["y"]["shape"] == [2, 5, 16]
    model = build_model(spec, seed=0).eval()
    assert result["params"] == sum(
        param.numel() for param in model.parameters()
    )
    assert len(model.nodes.enc.layers) == 2

    # 書き出したコードは中の部品を先に作り, 重みのキーと計算が同じ
    code = export_python(spec)
    assert "_part0 = nn.TransformerEncoderLayer(" in code
    assert "encoder_layer=_part0" in code
    assert max(len(line) for line in code.splitlines()) <= 79
    namespace: dict = {}
    exec(compile(code, "<exported>", "exec"), namespace)
    exported = namespace["Model"]().eval()
    exported.load_state_dict(model.state_dict())
    x = torch.randn(2, 5, 16)
    assert torch.allclose(model(x), exported(x))


def test_unit_param_version() -> None:
    """
    引数に部品を書いた定義が版 3 で保存され, 版 2 までと書いた定義は
    読み込みで誤りになることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 版 3 での保存と, 版 2 と書いた定義の誤り (場所と版の案内)
    spec = load_spec(yaml.safe_load(ENCODER))
    assert yaml.safe_load(dump_spec(spec))["format_version"] == 3
    old = yaml.safe_load(ENCODER)
    old["format_version"] = 2
    with pytest.raises(ValueError, match="nodes.enc.params.encoder_layer"):
        load_spec(old)
    with pytest.raises(ValueError, match="format_version: 3"):
        load_spec(old)

    # 部分モデルの中でも版 3, 版 2 の項目も使う版 1 の定義は 3 を案内
    nested = yaml.safe_load(ENCODER)
    nested["subgraphs"] = {"block": {
        "inputs": {"x": {}},
        "nodes": {"enc": nested["nodes"]["enc"]},
        "outputs": {"y": "enc"}
    }}
    nested["nodes"] = {"block": {"unit": "sub.block", "inputs": "x"}}
    assert yaml.safe_load(
        dump_spec(load_spec(nested))
    )["format_version"] == 3
    nested["format_version"] = 1
    nested["nodes"]["block"]["train"] = False
    with pytest.raises(ValueError, match="版 2 の項目.*format_version: 3"):
        load_spec(nested)


def test_unit_param_errors(residual: Any) -> None:
    """
    引数に書いた部品の誤りを, 場所を付けて全て返すことを確かめる関数

    Parameters
    ----------
    residual: Any
        自作の部品 test.Residual を登録するフィクスチャ

    Returns
    ----------
    None
    """
    # 中の部品の引数の誤りは params.<引数>.params.<名前> の場所
    spec = load_spec(yaml.safe_load(ENCODER))
    layer = spec["nodes"]["enc"]["params"]["encoder_layer"]
    layer["params"]["nhead"] = "four"
    layer["params"]["heads"] = 4
    del layer["params"]["d_model"]
    assert sorted(issues_of(spec)) == [
        ("invalid_value", "params.encoder_layer.params.nhead"),
        ("missing_param", "params.encoder_layer.params.d_model"),
        ("unknown_param", "params.encoder_layer.params.heads")
    ]

    # 部品と引数の形でない値, 部分モデル, 登録されていない部品, 必須の不足
    cases = [
        (16, "invalid_value", "params.encoder_layer"),
        ({"unit": "nn.Linear", "weights": {}}, "invalid_value",
         "params.encoder_layer"),
        ({"params": {}}, "invalid_value", "params.encoder_layer"),
        ({"unit": "sub.block"}, "invalid_value",
         "params.encoder_layer.unit"),
        ({"unit": "nn.TransformerEncoderLayr"}, "unknown_unit",
         "params.encoder_layer.unit"),
        (None, "missing_param", "params.encoder_layer")
    ]
    for value, code, location in cases:
        spec = load_spec(yaml.safe_load(ENCODER))
        spec["nodes"]["enc"]["params"]["encoder_layer"] = value
        assert issues_of(spec) == [(code, location)], value

    # 学習済みの重みを読む部品と, 部品の引数でない引数への辞書
    spec = load_spec(yaml.safe_load(ENCODER))
    spec["nodes"]["enc"]["params"]["norm"] = {"unit": "test.Weighted"}
    register(UnitSpec(
        name="test.Weighted", category="test", source="custom",
        params=[ParamSpec("pretrained", "bool", True)],
        inputs=[PortSpec("x")], outputs=[PortSpec("out")],
        builder=lambda **values: nn.Identity(),
        offline={"pretrained": False}
    ))
    try:
        assert issues_of(spec) == [("invalid_value", "params.norm.unit")]
    finally:
        REGISTRY.pop("test.Weighted")
    spec = load_spec(yaml.safe_load(ENCODER))
    spec["nodes"]["enc"]["params"]["num_layers"] = {"unit": "nn.ReLU"}
    assert issues_of(spec) == [("invalid_value", "params.num_layers")]

    # 誤りのある中の部品のノードは, 形の推定で誤りになり先へ進まない
    spec = load_spec(yaml.safe_load(ENCODER))
    spec["nodes"]["enc"]["params"]["encoder_layer"]["unit"] = "nn.Linearr"
    result = infer(spec)
    assert result["nodes"]["enc"]["status"] == "error"


def test_unit_param_nested_custom(residual: Any) -> None:
    """
    自作の部品の部品の引数に, 部品の引数を持つ部品を入れ子で書けることを
    確かめる関数

    Parameters
    ----------
    residual: Any
        自作の部品 test.Residual を登録するフィクスチャ

    Returns
    ----------
    None
    """
    # 自作の部品の引数も unit になり, 入れ子の部品と演算を書ける
    unit = get_unit("test.Residual")
    assert unit.available
    assert [(param.name, param.kind) for param in unit.params] == [
        ("inner", "unit"), ("scale", "float")
    ]
    spec = load_spec({
        "format_version": 3,
        "inputs": {"x": {"shape": [3, 8]}},
        "nodes": {"res": {
            "unit": "test.Residual",
            "params": {"inner": {
                "unit": "test.Residual",
                "params": {
                    "inner": {"unit": "op.Scale", "params": {"factor": 2.0}},
                    "scale": 0.5
                }
            }},
            "inputs": "x"
        }},
        "outputs": {"y": "res"}
    })
    assert issues_of(spec) == []
    x = torch.ones(3, 8)
    model = build_model(spec)
    assert torch.allclose(model(x), x + (x + 0.5 * 2.0 * x))

    # 書き出しは中の部品を先に作り, 演算のクラスを書き写す
    code = export_python(spec)
    assert code.index("_part0 = Scale(") < code.index("_part1 = Residual(")
    assert "class Scale(nn.Module):" in code
    namespace: dict = {}
    exec(compile(code, "<exported>", "exec"), namespace)
    assert torch.allclose(namespace["Model"]()(x), model(x))

    # 外部の重みの部品の引数には部品を書けない
    spec["nodes"]["res"]["weights"] = {
        "unit": "test.Residual",
        "params": {"inner": {"unit": "nn.Identity"}}
    }
    assert issues_of(spec) == [
        ("invalid_weights", "weights.params.inner")
    ]
