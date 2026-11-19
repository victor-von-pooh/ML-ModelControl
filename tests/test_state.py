import copy
import urllib.request
import warnings
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml
from torch import nn, optim

from ml_modelcontrol import (
    build_model, dump_spec, embed_subgraph, export_python, infer,
    list_examples, load_example, load_spec, validate
)
from ml_modelcontrol import pretrained, state
from ml_modelcontrol.state import remap_key


def mlp_spec(**extra: Any) -> dict:
    """
    2 層の全結合のモデルの定義 (版 2) を作る関数

    Parameters
    ----------
    **extra: Any
        ノードの名前ごとに足す項目 (fc1={"train": False} など)

    Returns
    ----------
    spec: dict
        モデルの定義
    """
    spec = {
        "format_version": 2,
        "inputs": {"x": {"shape": [4, 8]}},
        "nodes": {
            "fc1": {
                "unit": "nn.Linear",
                "params": {"in_features": 8, "out_features": 8},
                "inputs": "x"
            },
            "act": {"unit": "nn.ReLU", "inputs": "fc1"},
            "fc2": {
                "unit": "nn.Linear",
                "params": {"in_features": 8, "out_features": 3},
                "inputs": "act"
            }
        },
        "outputs": {"y": "fc2"}
    }
    for name, items in extra.items():
        spec["nodes"][name].update(items)

    return spec


def codes(spec: dict) -> list[str]:
    """
    確認の誤りと注意の識別子を返す関数

    Parameters
    ----------
    spec: dict
        モデルの定義

    Returns
    ----------
    found: list[str]
        識別子のリスト
    """
    found = [issue["code"] for issue in validate(spec)]

    return found


def test_freeze_only_trainable_get_gradients() -> None:
    """
    凍結したノードのパラメータに勾配が来ず, 最適化手法で変わらないことを
    確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 1 層目を凍結し, model.parameters() をそのまま最適化手法に渡して 1 歩
    model = build_model(mlp_spec(fc1={"train": False}), seed=0)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    x = torch.randn(4, 8, generator=torch.Generator().manual_seed(0))
    loss = model(x).pow(2).sum()
    loss.backward()
    optimizer.step()

    # 凍結した層は勾配が無く値も同じ, 学習する層は勾配があり値が変化
    assert model.nodes.fc1.weight.grad is None
    assert not model.nodes.fc1.weight.requires_grad
    assert model.nodes.fc2.weight.grad is not None
    after = model.state_dict()
    assert torch.equal(after["nodes.fc1.weight"], before["nodes.fc1.weight"])
    assert not torch.equal(
        after["nodes.fc2.weight"], before["nodes.fc2.weight"]
    )

    # 形の推定の学習するパラメータ数にも反映
    result = infer(mlp_spec(fc1={"train": False}))
    assert result["nodes"]["fc1"]["trainable"] == 0
    assert result["trainable"] == result["nodes"]["fc2"]["params"]


def test_freeze_subgraph_and_eval_mode() -> None:
    """
    部分モデルのノードの凍結が中の全てに及び, 評価の形のノードは学習の形に
    しても統計が変わらないことを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # BatchNorm を持つ部分モデルを凍結し, 評価の形に保つ
    spec = {
        "format_version": 2,
        "inputs": {"x": {"shape": [4, 8]}},
        "subgraphs": {
            "block": {
                "inputs": {"x": {}},
                "nodes": {
                    "fc": {
                        "unit": "nn.Linear",
                        "params": {"in_features": 8, "out_features": 8},
                        "inputs": "x"
                    },
                    "bn": {
                        "unit": "nn.BatchNorm1d",
                        "params": {"num_features": 8}, "inputs": "fc"
                    }
                },
                "outputs": {"y": "bn"}
            }
        },
        "nodes": {
            "base": {
                "unit": "sub.block", "inputs": "x", "train": False,
                "eval_mode": True
            },
            "head": {
                "unit": "nn.Linear",
                "params": {"in_features": 8, "out_features": 2},
                "inputs": "base"
            }
        },
        "outputs": {"y": "head"}
    }
    model = build_model(spec, seed=0)
    model.train()
    assert all(
        not parameter.requires_grad
        for parameter in model.nodes.base.parameters()
    )
    assert not model.nodes.base.training
    assert not model.nodes.base.nodes.bn.training
    assert model.nodes.head.training
    stats = model.nodes.base.nodes.bn.running_mean.clone()
    model(torch.randn(4, 8))
    assert torch.equal(model.nodes.base.nodes.bn.running_mean, stats)

    # 評価の形を書かなければ, 凍結しても統計は学習の形で更新
    loose = copy.deepcopy(spec)
    del loose["nodes"]["base"]["eval_mode"]
    model = build_model(loose, seed=0).train()
    assert model.nodes.base.nodes.bn.training
    stats = model.nodes.base.nodes.bn.running_mean.clone()
    model(torch.randn(4, 8))
    assert not torch.equal(model.nodes.base.nodes.bn.running_mean, stats)


def test_frozen_batchnorm_warning() -> None:
    """
    評価の形を書かずに凍結した BatchNorm に注意を出すことを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 凍結だけなら注意, eval_mode を書けば (偽でも) 注意なし
    spec = {
        "format_version": 2,
        "inputs": {"x": {"shape": [4, 8]}},
        "nodes": {
            "bn": {
                "unit": "nn.BatchNorm1d", "params": {"num_features": 8},
                "inputs": "x", "train": False
            }
        },
        "outputs": {"y": "bn"}
    }
    assert "frozen_norm_stats" in codes(spec)
    spec["nodes"]["bn"]["eval_mode"] = False
    assert "frozen_norm_stats" not in codes(spec)
    spec["nodes"]["bn"]["train"] = "no"
    assert "invalid_state" in codes(spec)


def test_zero_init_keeps_base_output() -> None:
    """
    足した枝の最後の層を零初期化すると, 始めの出力が元のモデルと同じに
    なることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 元の出力 base に, 零初期化した枝の出力を足す (ControlNet の形)
    spec = mlp_spec()
    spec["nodes"].update({
        "branch": {
            "unit": "nn.Linear",
            "params": {"in_features": 8, "out_features": 8},
            "inputs": "x"
        },
        "branch_out": {
            "unit": "nn.Linear",
            "params": {"in_features": 8, "out_features": 3},
            "inputs": "branch", "zero_init": True
        },
        "add": {"unit": "op.Add", "inputs": {"xs": ["fc2", "branch_out"]}}
    })
    spec["outputs"] = {"y": "add", "base": "fc2"}
    model = build_model(spec, seed=0)
    x = torch.randn(4, 8)
    out = model(x)
    assert torch.equal(out["y"], out["base"])
    assert torch.count_nonzero(model.nodes.branch_out.weight) == 0

    # 名前を選んだ零初期化と, 無い名前の誤り
    named = mlp_spec(fc2={"zero_init": ["bias"]})
    model = build_model(named, seed=0)
    assert torch.count_nonzero(model.nodes.fc2.bias) == 0
    assert torch.count_nonzero(model.nodes.fc2.weight) > 0
    with pytest.raises(ValueError, match="zero_init のパラメータ"):
        build_model(mlp_spec(fc2={"zero_init": ["gain"]}))
    assert "invalid_state" in codes(mlp_spec(fc2={"zero_init": "yes"}))


def encoder_spec(weights: dict) -> dict:
    """
    外部の重みを読む部分モデル (2 層) を持つ定義を作る関数

    Parameters
    ----------
    weights: dict
        部分モデルのノードの weights

    Returns
    ----------
    spec: dict
        モデルの定義
    """
    layer = {"unit": "nn.Linear", "params": {"in_features": 4,
                                              "out_features": 4}}
    spec = {
        "format_version": 2,
        "inputs": {"x": {"shape": [2, 4]}},
        "subgraphs": {
            "encoder": {
                "inputs": {"x": {}},
                "nodes": {
                    "block0": {**layer, "inputs": "x"},
                    "block1": {**layer, "inputs": "block0"}
                },
                "outputs": {"y": "block1"}
            }
        },
        "nodes": {
            "enc": {"unit": "sub.encoder", "inputs": "x",
                    "weights": weights},
            "head": {
                "unit": "nn.Linear",
                "params": {"in_features": 4, "out_features": 2},
                "inputs": "enc"
            }
        },
        "outputs": {"y": "head"}
    }

    return spec


def save_state(path: Path, tensors: dict) -> dict:
    """
    state_dict をファイルに保存する関数

    Parameters
    ----------
    path: Path
        保存先
    tensors: dict
        キーとテンソルの辞書

    Returns
    ----------
    saved: dict
        保存した辞書
    """
    torch.save(tensors, path)
    saved = tensors

    return saved


def test_partial_load_counts_and_prefix_remap(tmp_path: Path) -> None:
    """
    前置きの付け替えで一部を読み, 読めた / 形の違う / 新しい / 使わない
    重みの数を報告することを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # block0 は合い, block1 の重みは形が違い, block1 の bias は無く,
    # 頭の重みは使わない重み
    state = save_state(tmp_path / "base.pt", {
        "model.blocks.0.weight": torch.full((4, 4), 2.0),
        "model.blocks.0.bias": torch.full((4,), 3.0),
        "model.blocks.1.weight": torch.zeros(5, 4),
        "model.head.weight": torch.zeros(10, 4)
    })
    weights = {
        "path": "base.pt",
        "remap": [
            {"from": "model.blocks.0.", "to": "nodes.block0."},
            {"from": "model.blocks.1.", "to": "nodes.block1."}
        ],
        "strict": False
    }
    model = build_model(encoder_spec(weights), base_dir=tmp_path, seed=0)
    report = model.weights_report[0]
    assert report["node"] == "enc"
    assert (report["loaded"], report["mismatched"], report["new"],
            report["unused"]) == (2, 1, 1, 1)
    assert report["mismatched_keys"] == [
        ["nodes.block1.weight", [5, 4], [4, 4]]
    ]
    assert report["new_keys"] == ["nodes.block1.bias"]
    assert report["unused_keys"] == ["model.head.weight"]
    assert report["license"] is None
    assert torch.equal(
        model.nodes.enc.nodes.block0.weight,
        state["model.blocks.0.weight"]
    )

    # 既定 (strict) では形の違う重みと新しい重みで誤り
    del weights["strict"]
    with pytest.raises(ValueError, match="strict: false"):
        build_model(encoder_spec(weights), base_dir=tmp_path)

    # 1 つも読めない (付け替えの書き間違い) 場合の誤りと, 読まない設定
    weights["remap"] = [{"from": "model.block.", "to": "nodes.block"}]
    with pytest.raises(ValueError, match="読める重みがありません"):
        build_model(encoder_spec(weights), base_dir=tmp_path)
    model = build_model(encoder_spec(weights), load_weights=False)
    assert model.weights_report == []


def test_partial_load_regex_offset_and_state_key(tmp_path: Path) -> None:
    """
    正規表現の付け替え (番号をずらす) と, 入れ物の辞書からの取り出しを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 層 6 と 7 を block0 と block1 に (番号を -6), 入れ物は checkpoint.model
    layers = {
        f"layer.{index}.{kind}": torch.full(
            (4, 4) if kind == "weight" else (4,), float(index)
        )
        for index in range(8) for kind in ("weight", "bias")
    }
    torch.save({"checkpoint": {"model": layers}, "epoch": 3},
               tmp_path / "full.pt")
    weights = {
        "path": str(tmp_path / "full.pt"),
        "state_key": "checkpoint.model",
        "remap": [{
            "pattern": r"^layer\.([67])\.", "to": r"nodes.block\1.",
            "offset": -6
        }]
    }
    model = build_model(encoder_spec(weights))
    report = model.weights_report[0]
    assert (report["loaded"], report["mismatched"], report["new"],
            report["unused"]) == (4, 0, 0, 12)
    assert torch.all(model.nodes.enc.nodes.block0.weight == 6.0)
    assert torch.all(model.nodes.enc.nodes.block1.bias == 7.0)

    # 付け替えの規則 (最初に当たったもの, 当たらなければそのまま)
    rules = [
        {"from": "a.", "to": "b."},
        {"pattern": r"^x\.(\d+)\.", "to": r"y.\1.", "offset": 2}
    ]
    assert remap_key("a.w", rules) == "b.w"
    assert remap_key("x.3.w", rules) == "y.5.w"
    assert remap_key("z.w", rules) == "z.w"

    # 入れ物のキーの誤りと, 付け替えた後の重なり
    with pytest.raises(ValueError, match="state_key"):
        build_model(encoder_spec({**weights, "state_key": "missing"}))
    collide = {**weights, "remap": [{"pattern": r"^layer\.\d+\.",
                                     "to": "nodes.block0."}]}
    with pytest.raises(ValueError, match="同じキー"):
        build_model(encoder_spec(collide))


def test_weights_validation() -> None:
    """
    外部の重みの書き方の誤りを, ファイルを読まずに見つけることを確かめる
    関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 正しい書き方は誤りなし
    assert "invalid_weights" not in codes(encoder_spec({"path": "a.pt"}))

    # 出どころが無い, 2 つある, hub に file が無い, 知らない項目
    bad = [
        {},
        {"path": "a.pt", "unit": "nn.Linear"},
        {"hub": "org/name"},
        {"path": "a.pt", "strict": "yes"},
        {"path": "a.pt", "extra": 1},
        {"path": "a.pt", "remap": [{"from": "a"}]},
        {"path": "a.pt", "remap": [{"pattern": "(", "to": "x"}]},
        {"path": "a.pt", "remap": [{"pattern": "a", "to": "b",
                                    "offset": 1}]},
        {"unit": "nn.Missing"},
        {"hub": "../x", "file": "a.bin"},
        "a.pt"
    ]
    for weights in bad:
        assert "invalid_weights" in codes(encoder_spec(weights)), weights


def test_hub_source_reads_license(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Hugging Face の置き場から読む重みに, 配布元のライセンスを添えることを
    確かめる関数 (ネットワークは使わない)

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    monkeypatch: pytest.MonkeyPatch
        差し替えの道具

    Returns
    ----------
    None
    """
    # 取ってくる関数とモデルの説明を読む関数の差し替え
    save_state(tmp_path / "model.bin", {
        "encoder.0.weight": torch.ones(4, 4),
        "encoder.0.bias": torch.ones(4),
        "encoder.1.weight": torch.ones(4, 4),
        "encoder.1.bias": torch.ones(4)
    })
    monkeypatch.setattr(
        state, "hub_download", FakeDownload(tmp_path / "model.bin")
    )
    monkeypatch.setattr(
        pretrained, "fetch_hub_metadata", FakeMetadata()
    )
    monkeypatch.setattr(urllib.request, "urlopen", refuse_network)
    pretrained.clear_pretrained_cache()
    weights = {
        "hub": "org/base", "file": "model.bin",
        "remap": [{"pattern": r"^encoder\.(\d+)\.", "to": r"nodes.block\1."}]
    }
    model = build_model(encoder_spec(weights))
    report = model.weights_report[0]
    pretrained.clear_pretrained_cache()
    assert report["loaded"] == 4
    assert report["license"]["license"] == "cc-by-nc-4.0"
    assert report["license"]["status"] == "noncommercial"
    assert report["license"]["alert"]
    assert "Hugging Face" in report["source"]


# Hugging Face から取ってくる関数の代わり
class FakeDownload:
    def __init__(self, path: Path):
        # 返すファイルを保存
        self.path = path

    def __call__(self, hub_id: str, filename: str, revision: str) -> Path:
        """
        取ってきたファイルのパスを返す関数

        Parameters
        ----------
        hub_id: str
            置き場の名前
        filename: str
            ファイルの名前
        revision: str
            版

        Returns
        ----------
        path: Path
            ファイルのパス
        """
        path = self.path

        return path


# Hugging Face のモデルの説明を読む関数の代わり
class FakeMetadata:
    def __call__(self, hub_id: str, revision: str) -> dict:
        """
        非商用のライセンスのモデルの説明を返す関数

        Parameters
        ----------
        hub_id: str
            置き場の名前
        revision: str
            版

        Returns
        ----------
        metadata: dict
            モデルの説明
        """
        metadata = {
            "cardData": {"license": "cc-by-nc-4.0"}, "sha": "abc",
            "gated": False
        }

        return metadata


def refuse_network(*args: Any, **kwargs: Any) -> Any:
    """
    テストでネットワークを使った場合に失敗させる関数

    Parameters
    ----------
    *args: Any
        urlopen の引数
    **kwargs: Any
        urlopen のキーワード引数

    Returns
    ----------
    None
    """
    raise AssertionError("テストでネットワークを使った")


def test_unit_source() -> None:
    """
    登録簿の部品を出どころにして, その state_dict を読むことを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 同じ形の nn.Linear から読み (学習済みの重みは無いのでライセンスなし)
    spec = mlp_spec(fc1={"weights": {
        "unit": "nn.Linear",
        "params": {"in_features": 8, "out_features": 8}
    }})
    model = build_model(spec)
    report = model.weights_report[0]
    assert (report["loaded"], report["new"]) == (2, 0)
    assert report["license"] is None
    assert report["source"] == "部品 nn.Linear"


def tied_spec(out_features: int = 10) -> dict:
    """
    埋め込みと出力の重みを共有する定義を作る関数

    Parameters
    ----------
    out_features: int = 10
        出力の層の大きさ (語彙の数. 10 以外なら形が合わない)

    Returns
    ----------
    spec: dict
        モデルの定義
    """
    spec = {
        "format_version": 2,
        "inputs": {"ids": {"shape": [2, 3], "dtype": "int64"}},
        "nodes": {
            "embed": {
                "unit": "nn.Embedding",
                "params": {"num_embeddings": 10, "embedding_dim": 4},
                "inputs": "ids"
            },
            "head": {
                "unit": "nn.Linear",
                "params": {"in_features": 4, "out_features": out_features,
                           "bias": False},
                "inputs": "embed", "share": {"weight": "embed.weight"}
            }
        },
        "outputs": {"logits": "head", "hidden": "embed"}
    }

    return spec


def test_tied_parameters_gradient_from_both_uses() -> None:
    """
    パラメータごとの共有が 1 つのパラメータで, 勾配が両方の使い方から
    流れることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 同じパラメータの物で, パラメータ数は 1 回だけの数え方
    model = build_model(tied_spec(), seed=0)
    assert model.nodes.head.weight is model.nodes.embed.weight
    assert len(list(model.parameters())) == 1
    assert infer(tied_spec())["params"] == 40

    # 片方だけの勾配と, 両方の勾配の和
    ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    grads = []
    for key in ("hidden", "logits"):
        model.zero_grad()
        model(ids)[key].sum().backward()
        grads.append(model.nodes.embed.weight.grad.clone())
    model.zero_grad()
    out = model(ids)
    (out["hidden"].sum() + out["logits"].sum()).backward()
    assert grads[0].abs().sum() > 0 and grads[1].abs().sum() > 0
    assert torch.allclose(model.nodes.embed.weight.grad, grads[0] + grads[1])

    # 形の違う共有は組み立てと形の推定で誤り
    with pytest.raises(ValueError, match="形"):
        build_model(tied_spec(out_features=12))
    result = infer(tied_spec(out_features=12))
    assert result["nodes"]["head"]["status"] == "error"


def two_tower_spec() -> dict:
    """
    品物と利用者の塔を共有する 2 塔の定義を作る関数

    Parameters
    ----------
    None

    Returns
    ----------
    spec: dict
        モデルの定義
    """
    spec = {
        "format_version": 2,
        "inputs": {"user": {"shape": [2, 4]}, "item": {"shape": [2, 4]}},
        "subgraphs": {
            "tower": {
                "inputs": {"x": {}},
                "nodes": {
                    "fc": {
                        "unit": "nn.Linear",
                        "params": {"in_features": 4, "out_features": 4},
                        "inputs": "x"
                    }
                },
                "outputs": {"y": "fc"}
            }
        },
        "nodes": {
            "user_tower": {"unit": "sub.tower", "inputs": "user"},
            "item_tower": {
                "unit": "sub.tower", "inputs": "item", "share": "user_tower"
            }
        },
        "outputs": {"u": "user_tower", "i": "item_tower"}
    }

    return spec


def test_shared_tower_and_save_load(tmp_path: Path) -> None:
    """
    ノード全体の共有が同じ部品で, 定義と重みを保存して読み戻しても共有が
    残ることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 同じ部品で, 同じ入力には同じ出力
    model = build_model(two_tower_spec(), seed=0)
    assert model.nodes.item_tower is model.nodes.user_tower
    x = torch.randn(2, 4)
    out = model(x, x)
    assert torch.equal(out["u"], out["i"])

    # 学習で値を変え, 定義 (YAML) と state_dict を保存
    with torch.no_grad():
        model.nodes.user_tower.nodes.fc.weight.add_(1.0)
    dump_spec(two_tower_spec(), tmp_path / "model.yaml")
    torch.save(model.state_dict(), tmp_path / "weights.pt")
    text = (tmp_path / "model.yaml").read_text(encoding="utf-8")
    assert yaml.safe_load(text)["format_version"] == 2

    # 読み戻したモデルも共有し, 値は保存したもの
    loaded = build_model(tmp_path / "model.yaml", seed=1)
    loaded.load_state_dict(torch.load(tmp_path / "weights.pt"))
    assert loaded.nodes.item_tower is loaded.nodes.user_tower
    assert torch.equal(
        loaded.nodes.item_tower.nodes.fc.weight,
        model.nodes.user_tower.nodes.fc.weight
    )
    assert load_spec(tmp_path / "model.yaml") == load_spec(two_tower_spec())

    # 埋め込みの共有の, 保存と読み込みの後の保持
    tied = build_model(tied_spec(), seed=0)
    torch.save(tied.state_dict(), tmp_path / "tied.pt")
    dump_spec(tied_spec(), tmp_path / "tied.json")
    again = build_model(tmp_path / "tied.json", seed=1)
    again.load_state_dict(torch.load(tmp_path / "tied.pt"))
    assert again.nodes.head.weight is again.nodes.embed.weight
    assert torch.equal(again.nodes.embed.weight, tied.nodes.embed.weight)


def test_share_validation() -> None:
    """
    共有の書き方の誤り (元が無い, 自分, 連鎖, 部品の違い, 状態の項目) を
    確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 元のノードが無い, 自分自身, 書き方の誤り
    spec = two_tower_spec()
    spec["nodes"]["item_tower"]["share"] = "missing"
    assert "invalid_share" in codes(spec)
    spec["nodes"]["item_tower"]["share"] = "item_tower"
    assert "invalid_share" in codes(spec)
    spec["nodes"]["item_tower"]["share"] = {"weight": "user_tower"}
    assert "invalid_share" in codes(spec)

    # ノード全体の共有に状態の項目, 部品の違い
    spec = two_tower_spec()
    spec["nodes"]["item_tower"]["train"] = False
    assert "shared_setting" in codes(spec)
    spec = mlp_spec(fc2={"share": "fc1"})
    assert "share_mismatch" in codes(spec)

    # 共有の連鎖と, 共有する名前の零初期化
    spec = tied_spec()
    spec["nodes"]["extra"] = {
        "unit": "nn.Linear",
        "params": {"in_features": 4, "out_features": 10, "bias": False},
        "inputs": "embed", "share": {"weight": "head.weight"}
    }
    assert "invalid_share" in codes(spec)
    spec = tied_spec()
    spec["nodes"]["head"]["zero_init"] = ["weight"]
    assert "shared_setting" in codes(spec)


def test_versions() -> None:
    """
    版 1 の定義がそのまま読め, 版 2 の項目が無ければ版 1 で書き, 版 1 に
    版 2 の項目があるか新しい版なら止まることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 同梱の例 (版 1) の読み込みと組み立て, 版 1 のままの保存
    for example in list_examples():
        spec = load_example(example["name"])
        assert spec["format_version"] == 2
        assert yaml.safe_load(dump_spec(spec))["format_version"] == 1
    model = build_model(load_example("mlp"), seed=0)
    assert model.weights_report == []

    # 版 2 の項目を使う場合の版 2 での保存 (部分モデルの中だけでも)
    spec = two_tower_spec()
    del spec["nodes"]["item_tower"]["share"]
    spec["subgraphs"]["tower"]["nodes"]["fc"]["train"] = False
    assert yaml.safe_load(dump_spec(spec))["format_version"] == 2

    # 版 1 と書いた (か版を書かない) 定義に版 2 の項目があれば誤り
    old = mlp_spec(fc1={"train": False})
    old["format_version"] = 1
    with pytest.raises(ValueError, match="nodes.fc1.train"):
        load_spec(old)
    del old["format_version"]
    with pytest.raises(ValueError, match="format_version: 2"):
        load_spec(old)
    assert codes(old) == ["invalid_spec"]

    # ライブラリより新しい版のはっきりした停止
    newer = mlp_spec()
    newer["format_version"] = 3
    with pytest.raises(ValueError, match="format_version 3 の定義は"):
        load_spec(newer)


def test_export_state() -> None:
    """
    書き出すコードが, 凍結, 零初期化, 共有, 評価の形を同じにすることを
    確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 凍結, 零初期化, 評価の形, パラメータごとの共有を持つ定義の書き出し
    spec = tied_spec()
    spec["nodes"]["embed"]["train"] = False
    spec["nodes"]["norm"] = {
        "unit": "nn.LayerNorm", "params": {"normalized_shape": 4},
        "inputs": "embed", "zero_init": ["bias"], "eval_mode": True
    }
    spec["nodes"]["head"]["inputs"] = "norm"
    spec["nodes"]["head2"] = {
        "unit": "nn.Linear",
        "params": {"in_features": 4, "out_features": 10, "bias": False},
        "inputs": "norm", "share": "head2_base"
    }
    spec["nodes"]["head2_base"] = {
        "unit": "nn.Linear",
        "params": {"in_features": 4, "out_features": 10, "bias": False},
        "inputs": "norm"
    }
    spec["outputs"]["other"] = "head2"
    spec["outputs"]["other_base"] = "head2_base"
    code = export_python(spec)
    namespace: dict = {}
    exec(compile(code, "<export>", "exec"), namespace)
    exported = namespace["Model"]().train()
    built = build_model(spec)
    assert list(exported.state_dict()) == list(built.state_dict())
    assert exported.nodes["head"].weight is exported.nodes["embed"].weight
    assert exported.nodes["head2"] is exported.nodes["head2_base"]
    assert not exported.nodes["embed"].weight.requires_grad
    assert torch.count_nonzero(exported.nodes["norm"].bias) == 0
    assert not exported.nodes["norm"].training
    assert exported.nodes["head2"].training

    # 外部の重みはコードに書かず注意
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        export_python(mlp_spec(fc1={"weights": {"path": "a.pt"}}))
    assert any("外部の重み" in str(item.message) for item in caught)


def test_tied_checkpoint_and_embedded_paths(tmp_path: Path) -> None:
    """
    共有した重みを 1 回だけ持つファイルを strict で読めることと, 別の
    ディレクトリの定義を写しても外部の重みのパスが合うことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 埋め込みの重みだけのファイル (共有した出力の重みのキーは無い)
    save_state(tmp_path / "tied.pt", {"weight": torch.ones(10, 4)})
    spec = tied_spec()
    spec["subgraphs"] = {"lm": {
        "inputs": {"ids": {}}, "nodes": spec.pop("nodes"),
        "outputs": {"logits": "head"}
    }}
    spec["nodes"] = {"lm": {
        "unit": "sub.lm", "inputs": "ids",
        "weights": {"path": "tied.pt",
                    "remap": [{"from": "", "to": "nodes.embed."}]}
    }}
    spec["outputs"] = {"logits": "lm"}
    model = build_model(spec, base_dir=tmp_path)
    report = model.weights_report[0]
    assert (report["loaded"], report["new"]) == (1, 0)
    assert torch.all(model.nodes.lm.nodes.head.weight == 1.0)

    # 別のディレクトリの定義を写して保存しても, 重みのファイルを読める
    inner = tmp_path / "inner"
    inner.mkdir()
    (tmp_path / "tied.pt").rename(inner / "tied.pt")
    dump_spec(spec, inner / "lm.yaml")
    outer = {
        "format_version": 2, "inputs": {"ids": {"shape": [2, 3],
                                                "dtype": "int64"}},
        "nodes": {"m": {"unit": "sub.wrapped", "inputs": "ids"}},
        "outputs": {"logits": "m"}
    }
    outer = embed_subgraph(outer, "wrapped", inner / "lm.yaml")
    dump_spec(outer, tmp_path / "outer.yaml")
    model = build_model(tmp_path / "outer.yaml")
    assert model.weights_report[0]["node"] == "m.lm"


def test_share_source_skipped_in_infer() -> None:
    """
    形の推定で, 共有の元が計算できない場合に共有するノードも計算しない
    ことを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 誤りのあるノードの先の a を, b が共有
    spec = mlp_spec()
    spec["nodes"]["fc1"]["params"]["out_features"] = -1
    spec["nodes"]["b"] = {
        "unit": "nn.Linear",
        "params": {"in_features": 8, "out_features": 3},
        "inputs": "x", "share": "fc2"
    }
    spec["outputs"]["z"] = "b"
    result = infer(spec)
    assert result["nodes"]["fc2"]["status"] == "skipped"
    assert result["nodes"]["b"]["status"] == "skipped"

    # 引数を書かない (か違う) ノード全体の共有は誤り
    spec = mlp_spec()
    spec["nodes"]["b"] = {"unit": "nn.Linear", "inputs": "x",
                          "share": "fc1"}
    assert "missing_param" in codes(spec)
    spec["nodes"]["b"]["params"] = {"in_features": 8, "out_features": 4}
    assert "share_mismatch" in codes(spec)
