import pytest
import torch
import yaml

from ml_modelcontrol import (
    build_model, describe_spec_units, dump_spec, embed_subgraph,
    export_python, infer, load_spec, register_unit, validate
)

# 部分モデルとして写す, 残差のブロックの定義 (保存した定義の想定)
BLOCK = """
format_version: 1
name: block
description: 残差のブロック
inputs:
  x: {shape: [4, 8]}
nodes:
  fc: {unit: nn.Linear, params: {in_features: 8, out_features: 8}, inputs: x}
  add: {unit: op.Add, inputs: {xs: [x, fc]}}
outputs:
  y: add
"""

# 部分モデルを 2 回使い, 出力が 2 つの部分モデルも使う本体
MAIN = """
format_version: 1
name: main
inputs:
  x: {shape: [4, 8]}
nodes:
  b1: {unit: sub.block, inputs: x}
  pair: {unit: sub.pair, inputs: b1}
  head:
    unit: nn.Linear
    params: {in_features: 8, out_features: 3}
    inputs: "pair:a"
outputs:
  logits: head
  skip: "pair:b"
subgraphs:
  pair:
    inputs:
      v: {}
    nodes:
      inner: {unit: sub.block, inputs: v}
      act: {unit: nn.ReLU, inputs: inner}
    outputs:
      a: act
      b: inner
"""


def make_main() -> dict:
    """
    ブロックを写した本体の定義を作る関数

    Parameters
    ----------
    None

    Returns
    ----------
    spec: dict
        部分モデル block と pair を持つ定義
    """
    spec = embed_subgraph(yaml.safe_load(MAIN), "block", yaml.safe_load(BLOCK))

    return spec


def test_build_and_infer():
    """
    部分モデルを使う定義を組み立て, 形を推定できることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 誤りも注意も無く, 中の重みは部分モデルのノードの下のキー
    spec = make_main()
    assert validate(spec) == []
    model = build_model(spec, seed=0)
    keys = list(model.state_dict())
    assert "nodes.b1.nodes.fc.weight" in keys
    assert "nodes.pair.nodes.inner.nodes.fc.weight" in keys
    result = model(torch.zeros(4, 8))
    assert result["logits"].shape == (4, 3)
    assert result["skip"].shape == (4, 8)

    # 同じ部分モデルの 2 回の使用は別の重み
    assert model.nodes["b1"].nodes["fc"].weight is not (
        model.nodes["pair"].nodes["inner"].nodes["fc"].weight
    )

    # 形の推定は部分モデルのノードの口ごとの形とパラメータ数
    inferred = infer(spec)
    assert inferred["ok"]
    assert inferred["nodes"]["pair"]["outputs"]["a"]["shape"] == [4, 8]
    assert inferred["nodes"]["b1"]["params"] == 72
    assert inferred["params"] == 72 * 2 + 27


def test_export_matches_model():
    """
    書き出したコードが部分モデルごとのクラスを持ち, 同じ計算をすることを
    確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    spec = make_main()
    code = export_python(spec)
    assert "class Block(nn.Module):" in code
    assert "class Pair(nn.Module):" in code
    assert "pair['a']" in code
    namespace: dict = {}
    exec(code, namespace)
    exported = namespace["Model"]()
    model = build_model(spec, seed=0)
    exported.load_state_dict(model.state_dict())
    x = torch.randn(4, 8)
    for key, value in model(x).items():
        assert torch.allclose(value, exported(x)[key])

    # 本体のクラスの名前と部分モデルのクラスの名前の重なりはエラー
    with pytest.raises(ValueError, match="重なります"):
        export_python(spec, class_name="Block")


def test_errors_inside_subgraph():
    """
    部分モデルの中の誤りが場所付きで返り, 使うノードだけが計算されない
    ことを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 中の部品の書き間違いは subgraphs.<名前> の場所, 使うノードは使えない
    spec = make_main()
    spec["subgraphs"]["pair"]["nodes"]["act"]["unit"] = "nn.Relu"
    issues = validate(spec)
    assert any(
        issue["location"] == "subgraphs.pair.nodes.act.unit"
        and issue["subgraph"] == "pair" and issue["node"] is None
        and "もしかして" in issue["message"] for issue in issues
    )
    assert any(
        issue["node"] == "pair" and issue["code"] == "unavailable_unit"
        for issue in issues
    )
    # 誤りの前のノードに, 出力に届かない注意を重ねない
    assert not any(issue["code"] == "unused_node" for issue in issues)

    # 形の推定は誤りの前のノードだけ
    inferred = infer(spec)
    assert not inferred["ok"]
    assert inferred["nodes"]["b1"]["status"] == "ok"
    assert inferred["nodes"]["pair"]["status"] == "error"
    assert inferred["nodes"]["head"]["status"] == "skipped"


def test_cycle_and_unknown():
    """
    部分モデルの循環, 無い部分モデル, 部分モデルに書けない項目を誤りに
    することを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    spec = make_main()
    spec["subgraphs"]["block"]["nodes"]["fc"] = {
        "unit": "sub.pair", "inputs": "x"
    }
    messages = [issue["message"] for issue in validate(spec)]
    assert any("互いに使い合っています" in message for message in messages)

    spec = make_main()
    spec["nodes"]["b1"]["unit"] = "sub.blok"
    issues = validate(spec)
    assert any(
        issue["code"] == "unknown_subgraph"
        and "sub.block" in issue["message"]
        for issue in issues
    )

    spec = make_main()
    spec["subgraphs"]["pair"]["subgraphs"] = {}
    assert any(
        issue["location"] == "subgraphs.pair.subgraphs"
        for issue in validate(spec)
    )


def test_embed_and_dump():
    """
    写す関数と保存の順, 使われない部分モデルの注意, GUI 向けの一覧を
    確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 写した定義は版を持たず, 元の辞書は変えない
    main = yaml.safe_load(MAIN)
    spec = embed_subgraph(main, "block", yaml.safe_load(BLOCK))
    assert "format_version" not in spec["subgraphs"]["block"]
    assert "block" not in main["subgraphs"]

    # 写す定義の部分モデルも一緒に写し, 中身の違う同じ名前はエラー
    outer = embed_subgraph(yaml.safe_load(BLOCK), "pair", {
        **spec["subgraphs"]["pair"]
    })
    nested = embed_subgraph(yaml.safe_load(MAIN), "wrap", {
        **outer, "subgraphs": {"block": spec["subgraphs"]["block"]}
    })
    assert set(nested["subgraphs"]) == {"pair", "block", "wrap"}
    changed = yaml.safe_load(BLOCK)
    changed["nodes"]["fc"]["params"]["bias"] = False
    with pytest.raises(ValueError, match="中身が違います"):
        embed_subgraph(spec, "block", changed)
    with pytest.raises(ValueError, match="名前"):
        embed_subgraph(spec, "1x", changed)

    # 置き換えると中身が変わり, 写す定義の中と同じ名前は誤り
    replaced = embed_subgraph(spec, "block", changed, replace=True)
    assert replaced["subgraphs"]["block"]["nodes"]["fc"]["params"]["bias"] is (
        False
    )
    with pytest.raises(ValueError, match="写す定義の中"):
        embed_subgraph(yaml.safe_load(BLOCK), "block", spec)

    # 保存して読み直しても同じ, subgraphs は outputs の後
    text = dump_spec(spec)
    assert load_spec(yaml.safe_load(text)) == load_spec(spec)
    assert text.index("subgraphs:") > text.index("outputs:")

    # 使われない部分モデルは注意 (使われない部分モデルだけが使うものも)
    spec["subgraphs"]["spare"] = spec["subgraphs"]["pair"]
    spec["subgraphs"]["pair"] = spec["subgraphs"]["block"]
    unused = {
        issue["location"] for issue in validate(spec)
        if issue["code"] == "unused_subgraph"
    }
    assert unused == {"subgraphs.spare"}
    spec["nodes"]["pair"]["unit"] = "sub.block"
    unused = {
        issue["location"] for issue in validate(spec)
        if issue["code"] == "unused_subgraph"
    }
    assert unused == {"subgraphs.spare", "subgraphs.pair"}

    # GUI 向けの一覧は口を持ち, 出力が 2 つの部分モデルは口ごと
    units = {unit["name"]: unit for unit in describe_spec_units(spec)}
    assert [port["name"] for port in units["sub.spare"]["inputs"]] == ["v"]
    assert [port["name"] for port in units["sub.spare"]["outputs"]] == [
        "a", "b"
    ]
    assert units["sub.block"]["description"] == "残差のブロック"


def test_register_rejects_sub_prefix():
    """
    自作の部品の名前に部分モデルの頭 sub. を使えないことを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    with pytest.raises(ValueError, match="sub"):
        register_unit(torch.nn.Identity, name="sub.Mine")


def test_awkward_names():
    """
    予約語や数字になる部分モデルの名前と, self という入力の名前でも組み立てと
    書き出しができることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 予約語や数字になるクラスの名前は頭を付けた名前
    for name, class_name in [("none", "SubgraphNone"), ("_1", "Subgraph1")]:
        spec = yaml.safe_load(f"""
inputs:
  x: {{shape: [2, 4]}}
nodes:
  n: {{unit: sub.{name}, inputs: x}}
outputs:
  o: n
subgraphs:
  {name}:
    inputs: {{x: {{}}}}
    nodes:
      r: {{unit: nn.ReLU, inputs: x}}
    outputs: {{y: r}}
""")
        assert validate(spec) == []
        code = export_python(spec)
        assert f"class {class_name}(nn.Module):" in code
        exec(code, {})

    # 部分モデルの入力の口は位置で渡すので, self という名前でも計算できる
    spec = yaml.safe_load("""
inputs:
  x: {shape: [2, 4]}
  z: {shape: [2, 4]}
nodes:
  n: {unit: sub.s, inputs: {self: x, b: z}}
outputs:
  o: n
subgraphs:
  s:
    inputs: {self: {}, b: {}}
    nodes:
      r: {unit: op.Add, inputs: {xs: [self, b]}}
    outputs: {y: r}
""")
    model = build_model(spec)
    assert model(torch.ones(2, 4), torch.zeros(2, 4)).sum().item() == 8
    assert infer(spec)["ok"]


@pytest.mark.skipif(
    not hasattr(Exception, "add_note"), reason="注記は Python 3.11 以上"
)
def test_error_inside_subgraph_names_inner_node():
    """
    部分モデルの中で計算の誤りが起きた場合に, 中のノードの名前を誤りの文に
    添えることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    spec = make_main()
    spec["subgraphs"]["block"]["nodes"]["fc"]["params"]["in_features"] = 5
    error = infer(spec)["nodes"]["b1"]["error"]
    assert error.startswith("RuntimeError")
    assert "ノード fc (nn.Linear)" in error
