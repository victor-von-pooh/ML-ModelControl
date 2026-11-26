import pytest
import torch

from ml_modelcontrol import (
    TEMPLATE_ARGS, adapt_example, build_model, example_args, infer,
    load_example, validate
)
from ml_modelcontrol.infer import example_input


@pytest.mark.parametrize("name", sorted(TEMPLATE_ARGS))
def test_adapt_every_template(name: str) -> None:
    """
    雛形ごとに, 全ての書き換える引数を同梱の例と違う値にした定義が, 確認,
    形の推定, 組み立てと順伝播を通ることを確かめる関数

    Parameters
    ----------
    name: str
        雛形の名前

    Returns
    ----------
    None
    """
    # 今の値と違う値 (U-Net のチャンネル数などは小さく)
    current = example_args(name)
    assert set(current) == set(TEMPLATE_ARGS[name])
    values = {arg: value + 3 for arg, value in current.items()}
    spec = adapt_example(name, values)
    assert validate(spec) == []
    infer(spec)
    model = build_model(spec, seed=0)
    with torch.no_grad():
        model(**example_input(spec))
    assert example_args_of(spec, name) == values

    # 同梱の例は書き換わらない
    assert example_args(name) == current


def example_args_of(spec: dict, name: str) -> dict:
    """
    書き換えた定義から, 書き換える引数の値を読み直す関数

    Parameters
    ----------
    spec: dict
        書き換えた定義
    name: str
        雛形の名前

    Returns
    ----------
    values: dict
        引数の名前ごとの値 (全ての場所で同じ値であることも確かめる)
    """
    # 場所ごとの値を倍率で割り, 引数ごとに 1 つ
    from ml_modelcontrol.templates import read_place

    values = {}
    for arg, places in TEMPLATE_ARGS[name].items():
        found = {read_place(spec, place) // place[3] for place in places}
        assert len(found) == 1, (name, arg, found)
        values[arg] = found.pop()

    return values


def test_adapt_output_shapes() -> None:
    """
    書き換えた値が出力の形に出ることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 異常の検知の再構成, VAE の潜在, 領域のクラス, 点群の点の数, 需要の水準
    cases = [
        ("tabular_autoencoder", {"features": 10}, "recon", [8, 10]),
        ("vae", {"latent": 4}, "mu", [8, 4]),
        ("unet_segmentation", {"classes": 5, "channels": 1}, "logits",
         [4, 5, 32, 32]),
        ("point_cloud_predictor", {"points_out": 100}, "cloud",
         [4, 100, 3]),
        ("tabular_demand", {"levels": 5, "features": 7}, "quantiles",
         [8, 5]),
        ("two_tower_retrieval", {"users": 30, "items": 50},
         "item_embedding", [8, 16]),
    ]
    for name, values, output, shape in cases:
        result = infer(adapt_example(name, values))
        assert result["outputs"][output]["shape"] == shape, name


def test_adapt_errors() -> None:
    """
    知らない雛形, 知らない引数, 1 以上の整数でない値で止まることを確かめる
    関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 書き換えられない雛形は KeyError, 引数と値の誤りは ValueError
    with pytest.raises(KeyError, match="書き換えられる雛形ではありません"):
        adapt_example("cnn_classifier", {"features": 3})
    with pytest.raises(ValueError, match="書き換えられない引数"):
        adapt_example("vae", {"width": 3})
    for value in (0, -1, 2.0, True, "3"):
        with pytest.raises(ValueError, match="1 以上の整数"):
            adapt_example("vae", {"features": value})
    assert load_example("vae")["inputs"]["x"]["shape"] == [8, 16]
