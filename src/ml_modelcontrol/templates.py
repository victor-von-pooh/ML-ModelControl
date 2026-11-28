import copy
from numbers import Integral
from typing import Any

from ml_modelcontrol.spec import load_example

# 雛形ごとの, データに合わせて書き換える引数の名前と, 書き換える場所. 場所は
# (種類, 名前, 鍵, 倍率) で, 種類 input は入力の shape の鍵番目の軸, param は
# ノードの引数, shape はノードの引数 shape の鍵番目の要素 (値 × 倍率を書く)
TEMPLATE_ARGS: dict[str, dict[str, list[tuple[str, str, Any, int]]]] = {
    "vae": {
        "features": [("input", "x", 1, 1), ("param", "enc1", "in_features", 1),
                     ("param", "dec2", "out_features", 1)],
        "latent": [("param", "enc2", "out_features", 2),
                   ("param", "dec1", "in_features", 1)],
    },
    "gan_generator": {
        "noise": [("input", "z", 1, 1), ("param", "fc1", "in_features", 1)],
        "features": [("param", "fc2", "out_features", 1)],
    },
    "gan_discriminator": {
        "features": [("input", "x", 1, 1),
                     ("param", "fc1", "in_features", 1)],
    },
    "diffusion_denoiser": {
        "features": [("input", "x", 1, 1), ("param", "x_fc", "in_features", 1),
                     ("param", "head", "out_features", 1)],
    },
    "contrastive_encoder": {
        "features": [("input", "x", 1, 1),
                     ("param", "fc1", "in_features", 1)],
    },
    "two_tower_retrieval": {
        "users": [("input", "user", 1, 1),
                  ("param", "user_fc1", "in_features", 1)],
        "items": [("input", "item", 1, 1),
                  ("param", "item_fc1", "in_features", 1)],
    },
    "policy_value": {
        "observations": [("input", "obs", 1, 1),
                         ("param", "fc1", "in_features", 1)],
        "actions": [("param", "policy", "out_features", 1)],
    },
    "tabular_autoencoder": {
        "features": [("input", "x", 1, 1), ("param", "enc1", "in_features", 1),
                     ("param", "dec2", "out_features", 1)],
        "bottleneck": [("param", "enc2", "out_features", 1),
                       ("param", "dec1", "in_features", 1)],
    },
    "tabular_uplift": {
        "features": [("input", "x", 1, 1),
                     ("param", "treated1", "in_features", 1),
                     ("param", "control1", "in_features", 1)],
    },
    "tabular_uplift_s": {
        "features": [("input", "x", 1, 1),
                     ("param", "outcome1", "in_features", 1)],
    },
    "tabular_price": {
        "features": [("input", "x", 1, 1),
                     ("param", "units1", "in_features", 1)],
    },
    "tabular_demand": {
        "features": [("input", "x", 1, 1),
                     ("param", "demand1", "in_features", 1)],
        "levels": [("param", "demand2", "out_features", 1)],
    },
    "unet_segmentation": {
        "channels": [("input", "image", 1, 1),
                     ("param", "enc1a", "in_channels", 1)],
        "classes": [("param", "head", "out_channels", 1)],
    },
    "point_cloud_predictor": {
        "points_in": [("input", "observed", 1, 1)],
        "points_out": [("param", "decode2", "out_features", 3),
                       ("shape", "cloud", 0, 1)],
    },
}


def example_args(name: str) -> dict[str, int]:
    """
    雛形の, データに合わせて書き換える引数の名前と今の値を返す関数

    Parameters
    ----------
    name: str
        雛形の名前 (TEMPLATE_ARGS にある同梱の例)

    Returns
    ----------
    values: dict[str, int]
        引数の名前ごとの, 同梱の例の値

    Raises
    ----------
    KeyError
        書き換えられる雛形でない場合
    """
    # 最初の場所の値を倍率で割った値
    places = template_places(name)
    spec = load_example(name)
    values = {
        arg: read_place(spec, found[0]) // found[0][3]
        for arg, found in places.items()
    }

    return values


def adapt_example(name: str, values: dict[str, Any]) -> dict:
    """
    雛形の, データに合わせて書き換える引数を渡した値にした定義を返す関数

    関わるノードの引数と入力の shape を全てそろえて書き換えるので, 使う側は
    ノードの名前を知らなくてよい

    Parameters
    ----------
    name: str
        雛形の名前 (TEMPLATE_ARGS にある同梱の例)
    values: dict[str, Any]
        引数の名前ごとの値 (1 以上の整数. NumPy の整数も可. 書かない引数は
        同梱の例のまま)

    Returns
    ----------
    spec: dict
        書き換えた定義 (同梱の例の写し)

    Raises
    ----------
    KeyError
        書き換えられる雛形でない場合
    ValueError
        知らない引数の名前と, 1 以上の整数でない値の場合
    """
    # 名前と値の確かめ (真偽値と小数は断る)
    places = template_places(name)
    unknown = sorted(set(values) - set(places))
    if unknown:
        raise ValueError(
            f"{name} に書き換えられない引数です: {', '.join(unknown)} "
            f"(書き換えられるのは {', '.join(places)})"
        )
    checked = {}
    for arg, value in values.items():
        if isinstance(value, bool) or not isinstance(value, Integral) or (
            value < 1
        ):
            raise ValueError(f"{arg} は 1 以上の整数にしてください")
        checked[arg] = int(value)

    # 写しの定義の, 引数ごとの全ての場所の書き換え
    spec = copy.deepcopy(load_example(name))
    for arg, value in checked.items():
        for place in places[arg]:
            write_place(spec, place, value * place[3])

    return spec


def template_places(name: str) -> dict[str, list[tuple[str, str, Any, int]]]:
    """
    雛形の, 引数ごとの書き換える場所を返す関数

    Parameters
    ----------
    name: str
        雛形の名前

    Returns
    ----------
    places: dict[str, list[tuple[str, str, Any, int]]]
        引数の名前ごとの場所

    Raises
    ----------
    KeyError
        書き換えられる雛形でない場合
    """
    # 表にある雛形だけ
    if name not in TEMPLATE_ARGS:
        raise KeyError(
            f"{name} は書き換えられる雛形ではありません (雛形: "
            f"{', '.join(TEMPLATE_ARGS)})"
        )
    places = TEMPLATE_ARGS[name]

    return places


def read_place(spec: dict, place: tuple[str, str, Any, int]) -> int:
    """
    定義の, 書き換える場所の今の値を返す関数

    Parameters
    ----------
    spec: dict
        定義
    place: tuple[str, str, Any, int]
        (種類, 名前, 鍵, 倍率)

    Returns
    ----------
    value: int
        今の値
    """
    # 入力の shape の軸, ノードの引数, ノードの引数 shape の要素
    kind, owner, key, _ = place
    if kind == "input":
        value = int(spec["inputs"][owner]["shape"][key])
    elif kind == "param":
        value = int(spec["nodes"][owner]["params"][key])
    else:
        value = int(spec["nodes"][owner]["params"]["shape"][key])

    return value


def write_place(
    spec: dict, place: tuple[str, str, Any, int], value: int
) -> None:
    """
    定義の, 書き換える場所に値を書く関数

    Parameters
    ----------
    spec: dict
        定義 (書き換える)
    place: tuple[str, str, Any, int]
        (種類, 名前, 鍵, 倍率)
    value: int
        書く値 (倍率をかけた後)

    Returns
    ----------
    None
    """
    # 入力の shape の軸, ノードの引数, ノードの引数 shape の要素
    kind, owner, key, _ = place
    if kind == "input":
        spec["inputs"][owner]["shape"][key] = value
    elif kind == "param":
        spec["nodes"][owner]["params"][key] = value
    else:
        spec["nodes"][owner]["params"]["shape"][key] = value
