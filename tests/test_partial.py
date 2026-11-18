import pytest
import torch
from torch import nn

from ml_modelcontrol import build_model, load_example


# 呼ばれると誤りを出す部品 (計算しないノードの確認)
class Forbidden(nn.Module):
    def forward(self, *args: object, **kwargs: object) -> None:
        """
        呼ばれた場合に誤りを出す関数

        Parameters
        ----------
        *args: object
            位置で渡した値
        **kwargs: object
            名前を付けて渡した値

        Returns
        ----------
        None
        """
        raise RuntimeError("呼ばれないはずのノードが呼ばれました")


def test_vae_decoder_only() -> None:
    """
    VAE の例で, 潜在変数から復号器だけを呼んだ値が, 全体の順伝播の
    再構成と同じことを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 評価の形 (潜在変数は平均のまま) の全体の順伝播
    model = build_model(load_example("vae"), seed=0).eval()
    x = torch.randn(8, 16, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        full = model(x)

        # 平均を潜在変数として渡した復号器だけの計算
        part = model.compute({"z": full["mu"]}, "recon")
    assert list(part) == ["recon"]
    assert torch.allclose(part["recon"], full["recon"])


def test_vae_decoder_skips_encoder() -> None:
    """
    復号器だけを呼ぶ場合に, 符号化器のノードを計算しないことを確かめる
    関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 符号化器のノードを, 呼ばれると誤りを出す部品に差し替え
    model = build_model(load_example("vae"), seed=0)
    for name in ["enc1", "enc_act", "enc2", "mu", "logvar", "z"]:
        model.nodes[name] = Forbidden()

    # 潜在変数から再構成までは計算でき, 全体の順伝播は誤り
    with torch.no_grad():
        part = model.compute({"z": torch.randn(5, 8)}, ["recon"])
        with pytest.raises(RuntimeError, match="呼ばれない"):
            model(torch.randn(5, 16))
    assert part["recon"].shape == (5, 16)


def test_two_tower_item_only() -> None:
    """
    2 塔のモデルの例で, 品物の塔だけで計算した埋め込みが, 全体の
    順伝播の品物の埋め込みと同じことを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 全体の順伝播と, 品物だけを渡した計算 (バッチの大きさも別でよい)
    model = build_model(load_example("two_tower_retrieval"), seed=0)
    generator = torch.Generator().manual_seed(0)
    user = torch.randn(8, 12, generator=generator)
    item = torch.randn(8, 20, generator=generator)
    with torch.no_grad():
        full = model(user=user, item=item)
        part = model.compute({"item": item}, "item_embedding")
        catalog = model.compute(
            {"item": torch.randn(50, 20, generator=generator)},
            ["item_embedding"]
        )
    assert torch.allclose(part["item_embedding"], full["item_embedding"])
    assert catalog["item_embedding"].shape == (50, 16)


def test_compute_all_outputs() -> None:
    """
    出力を指定しない場合に, 全体の順伝播と同じ値を辞書で返すことを
    確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 出力が 1 つの例でも辞書で返すこと (評価の形. 例は Dropout を持つ)
    model = build_model(load_example("mlp"), seed=0).eval()
    x = torch.randn(4, 16, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        full = model(x)
        part = model.compute({"x": x})
    assert list(part) == list(model.outputs)
    assert torch.allclose(next(iter(part.values())), full)


def test_compute_missing_input() -> None:
    """
    要る入力を渡さない場合と, 名前の誤りの場合に, 日本語の分かる誤りを
    出すことを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 利用者の入力が要る点数を, 品物だけから求めた場合
    model = build_model(load_example("two_tower_retrieval"), seed=0)
    item = torch.randn(8, 20)
    with pytest.raises(ValueError, match="出力 score の計算に要る入力 user"):
        model.compute({"item": item}, "score")

    # 復号器の出力を, 潜在変数も入力も渡さずに求めた場合
    vae = build_model(load_example("vae"), seed=0)
    with pytest.raises(ValueError, match="途中のノードの値を渡す"):
        vae.compute({}, "recon")

    # 無い出力の名前と, 入力にもノードにも無い値の名前
    with pytest.raises(ValueError, match="出力 items はありません"):
        model.compute({"item": item}, "items")
    with pytest.raises(ValueError, match="入力にもノードにも itme はありません"):
        model.compute({"itme": item}, "item_embedding")


def test_compute_port_and_subgraph() -> None:
    """
    出力の口が 2 つ以上のノードには「ノード:口」で, 部分モデルのノードには
    名前で値を渡せることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # LSTM の最後の隠れ状態 (口 h_n) から分類の頭だけの計算
    model = build_model(load_example("lstm_classifier"), seed=0)
    ids = torch.randint(0, 100, (4, 12))
    with torch.no_grad():
        full = model(ids)
        output = model.nodes["lstm"](model.nodes["emb"](ids))
        part = model.compute({"lstm:h_n": output[1][0]}, "logits")
    assert torch.allclose(part["logits"], full)

    # 名前だけのキーは最初の口 (output) で, 分類に使う h_n の代わりにならない
    with pytest.raises(ValueError, match="入力 ids が渡されていません"):
        model.compute({"lstm": output[0]}, "logits")
    with pytest.raises(ValueError, match="出力の口 hn はありません"):
        model.compute({"lstm:hn": output[1][0]}, "logits")

    # 部分モデルを使うノード (block2) の出力から後ろだけの計算
    blocks = build_model(load_example("stacked_blocks"), seed=0)
    x = torch.randn(4, 32)
    with torch.no_grad():
        full = blocks(x)
        middle = blocks.nodes["block2"](
            blocks.nodes["block1"](blocks.nodes["stem"](x))
        )
        part = blocks.compute({"block2": middle}, "logits")
    assert torch.allclose(part["logits"], full)


def test_compute_mixed_values() -> None:
    """
    要らない値を無視することと, 一部の出力だけに入力が要る場合に, その
    出力だけを誤りに挙げることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 入力と潜在変数の両方を渡した場合 (再構成は潜在変数から)
    model = build_model(load_example("vae"), seed=0).eval()
    x = torch.randn(8, 16)
    z = torch.randn(8, 8)
    with torch.no_grad():
        both = model.compute({"x": x, "z": z}, ["recon", "mu"])
        decoded = model.compute({"z": z}, "recon")
    assert torch.allclose(both["recon"], decoded["recon"])
    assert torch.allclose(both["mu"], model(x)["mu"])

    # 潜在変数だけで平均も求めた場合は, 平均の出力だけが入力 x を要る
    with pytest.raises(ValueError) as error:
        model.compute({"z": z}, ["recon", "mu"])
    assert "出力 mu の計算に要る入力 x" in str(error.value)
    assert "出力 recon" not in str(error.value)

    # 値を辞書で渡さない場合
    with pytest.raises(TypeError, match="辞書で渡して"):
        model.compute(z, "recon")


def test_compute_input_output() -> None:
    """
    入力をそのまま出す出力も, 入力を渡せば求まり, 渡さなければ誤りに
    なることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 入力をそのまま出す出力を持つ定義
    spec = {
        "inputs": {"x": {"shape": [4, 8]}},
        "nodes": {
            "fc": {
                "unit": "nn.Linear",
                "params": {"in_features": 8, "out_features": 2},
                "inputs": "x"
            }
        },
        "outputs": {"y": "fc", "raw": "x"}
    }
    model = build_model(spec, seed=0)
    x = torch.randn(4, 8)
    part = model.compute({"x": x}, "raw")
    assert part["raw"] is x
    with pytest.raises(ValueError, match="出力 raw の計算に要る入力 x"):
        model.compute({"fc": torch.zeros(4, 2)}, ["y", "raw"])
