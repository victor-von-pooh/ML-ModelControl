import math

import pytest
import torch
import yaml

from ml_modelcontrol import (
    build_model, describe_units, export_python, get_unit, infer, ops,
    validate
)

# VAE の符号化と再パラメータ化, 拡散の時刻の埋め込み, 対照学習の類似度を
# 演算の部品だけで組んだ定義
GENERATIVE = """
format_version: 1
name: generative_ops
inputs:
  x: {shape: [4, 6]}
  t: {shape: [4], dtype: int64}
nodes:
  enc:
    unit: nn.Linear
    params: {in_features: 6, out_features: 8}
    inputs: x
  mu: {unit: op.Chunk, params: {chunks: 2, index: 0, dim: 1}, inputs: enc}
  logvar: {unit: op.Chunk, params: {chunks: 2, index: -1}, inputs: enc}
  z:
    unit: op.Reparameterize
    inputs: {mu: mu, logvar: logvar}
  halved: {unit: op.Scale, params: {factor: 0.5}, inputs: logvar}
  std: {unit: op.Exp, inputs: halved}
  time:
    unit: op.SinusoidalEmbedding
    params: {embedding_dim: 5}
    inputs: t
  unit_z: {unit: op.L2Normalize, inputs: z}
  target: {unit: op.Detach, inputs: unit_z}
  target_t: {unit: op.Transpose, params: {dim0: 0, dim1: 1}, inputs: target}
  sim:
    unit: op.MatMul
    inputs: {input: unit_z, other: target_t}
outputs:
  z: z
  std: std
  time: time
  sim: sim
"""


def test_generative_ops_compute() -> None:
    """
    生成と表現の学習で使う演算の計算を, 手で計算した値と比べる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 指数, 定数倍, 勾配の切り離し
    x = torch.randn(3, 4, requires_grad=True)
    assert torch.allclose(ops.Exp()(x), torch.exp(x))
    assert torch.allclose(ops.Scale(0.5)(x), x * 0.5)
    detached = ops.Detach()(x)
    assert torch.equal(detached, x) and not detached.requires_grad

    # 等分した何番目か (負の番号は後ろから, 割り切れない場合は誤り)
    value = torch.arange(24.0).reshape(2, 12)
    first, second, third = value.chunk(3, dim=1)
    assert torch.equal(ops.Chunk(3, 0)(value), first)
    assert torch.equal(ops.Chunk(3, 1, dim=-1)(value), second)
    assert torch.equal(ops.Chunk(3, -1)(value), third)
    with pytest.raises(ValueError, match="割り切れません"):
        ops.Chunk(5, 0)(value)
    with pytest.raises(ValueError, match="index 3"):
        ops.Chunk(3, 3)(value)

    # L2 正規化 (ノルムが 1, 0 のベクトルは 0 のまま) と行列の積
    normed = ops.L2Normalize()(torch.tensor([[3.0, 4.0], [0.0, 0.0]]))
    assert torch.allclose(normed, torch.tensor([[0.6, 0.8], [0.0, 0.0]]))
    left, right = torch.randn(2, 3, 4), torch.randn(4, 5)
    assert torch.allclose(ops.MatMul()(left, right), left @ right)

    # 正弦波の埋め込み (前半が cos, 後半が sin, 時刻 0 は cos が 1)
    embedding = ops.SinusoidalEmbedding(4, max_period=100.0)(
        torch.tensor([0, 2])
    )
    angles = torch.tensor([2.0, 2.0 / 10.0])
    assert embedding.shape == (2, 4)
    assert torch.allclose(embedding[0], torch.tensor([1.0, 1.0, 0.0, 0.0]))
    assert torch.allclose(
        embedding[1], torch.cat([torch.cos(angles), torch.sin(angles)])
    )
    odd = ops.SinusoidalEmbedding(5)(torch.tensor([[1.0], [2.0]]))
    assert odd.shape == (2, 1, 5) and torch.all(odd[..., -1] == 0)


def test_reparameterize() -> None:
    """
    再パラメータ化が, 学習の形では乱数を引き勾配を流し, 評価の形では
    平均を返すことを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 学習の形: 同じ種なら mu + exp(logvar / 2) * eps
    mu = torch.zeros(2000, 3, requires_grad=True)
    logvar = torch.full((2000, 3), math.log(4.0), requires_grad=True)
    unit = ops.Reparameterize()
    torch.manual_seed(0)
    z = unit(mu, logvar)
    torch.manual_seed(0)
    eps = torch.randn(2000, 3)
    assert torch.allclose(z, mu + eps * 2.0)
    assert abs(z.std().item() - 2.0) < 0.1

    # 勾配は mu と logvar の両方に流れる
    z.sum().backward()
    assert mu.grad is not None and logvar.grad is not None

    # 評価の形は平均, sample_in_eval なら乱数
    assert torch.equal(unit.eval()(mu, logvar), mu)
    sampled = ops.Reparameterize(sample_in_eval=True).eval()(mu, logvar)
    assert not torch.equal(sampled, mu)


def test_generative_ops_registry() -> None:
    """
    演算の部品の登録 (口, 引数の説明と範囲, 組み合わせの確認) を確かめる
    関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 全ての部品が説明を持ち, 一覧にも出る
    names = [
        "op.Exp", "op.Scale", "op.Chunk", "op.L2Normalize", "op.MatMul",
        "op.Detach", "op.SinusoidalEmbedding", "op.Reparameterize"
    ]
    listed = {unit["name"]: unit for unit in describe_units()}
    for name in names:
        unit = get_unit(name)
        assert unit.category == "op" and unit.description
        assert [port.name for port in unit.outputs] == ["out"]
        assert all(param.description for param in unit.params)
        assert listed[name]["description"] == unit.description

    # 2 つの入力の口は名前と説明を持つ
    matmul = get_unit("op.MatMul").inputs
    assert [(port.name, port.description) for port in matmul] == [
        ("input", "左の行列"), ("other", "右の行列")
    ]
    reparam = get_unit("op.Reparameterize").inputs
    assert [port.name for port in reparam] == ["mu", "logvar"]
    assert not any(port.optional for port in [*matmul, *reparam])
    assert [port.name for port in get_unit("op.Exp").inputs] == ["input"]

    # 引数の種類と範囲
    chunk = {param.name: param for param in get_unit("op.Chunk").params}
    assert chunk["chunks"].kind == "int" and chunk["chunks"].minimum == 1
    sinusoidal = {
        param.name: param
        for param in get_unit("op.SinusoidalEmbedding").params
    }
    assert sinusoidal["embedding_dim"].required
    assert sinusoidal["max_period"].default == 10000.0


def test_generative_ops_graph() -> None:
    """
    演算の部品だけで組んだ定義の確認, 形の推定, 組み立て, 書き出しを
    確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 確認は誤りなし, 形の推定は出力ごとの形
    spec = yaml.safe_load(GENERATIVE)
    assert validate(spec) == []
    result = infer(spec)
    outputs = result["outputs"]
    assert outputs["z"]["shape"] == [4, 4]
    assert outputs["std"]["shape"] == [4, 4]
    assert outputs["time"]["shape"] == [4, 5]
    assert outputs["sim"]["shape"] == [4, 4]

    # 組み立てたモデルは手で計算した値と一致 (評価の形の z は平均)
    model = build_model(spec, seed=0).eval()
    x, t = torch.randn(4, 6), torch.arange(4)
    with torch.no_grad():
        out = model(x, t)
        hidden = model.nodes["enc"](x)
    mu, logvar = hidden[:, :4], hidden[:, 4:]
    unit_mu = mu / mu.norm(dim=-1, keepdim=True)
    assert torch.allclose(out["z"], mu)
    assert torch.allclose(out["std"], torch.exp(0.5 * logvar))
    assert torch.allclose(out["sim"], unit_mu @ unit_mu.T, atol=1e-6)

    # 書き出したコード (単独で動く) も同じ値
    code = export_python(spec)
    assert "class Reparameterize(nn.Module)" in code
    namespace: dict = {}
    exec(compile(code, "<exported>", "exec"), namespace)
    exported = namespace["Model"]()
    exported.load_state_dict(model.state_dict())
    with torch.no_grad():
        again = exported.eval()(x, t)
    for name in ("z", "std", "time", "sim"):
        assert torch.allclose(again[name], out[name])


def test_chunk_index_check() -> None:
    """
    op.Chunk の番号が分ける数の範囲の外の場合に, 確認が誤りを返すことを
    確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 範囲の外の番号は組み合わせの誤り, 範囲の中なら誤りなし
    spec = yaml.safe_load(GENERATIVE)
    spec["nodes"]["mu"]["params"]["index"] = 2
    issues = validate(spec)
    found = [
        (issue["code"], issue["node"], issue["location"]) for issue in issues
    ]
    assert found == [("invalid_combination", "mu", "params.index")]
    spec["nodes"]["mu"]["params"]["index"] = -2
    assert validate(spec) == []
