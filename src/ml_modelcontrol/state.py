import contextvars
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from torch import nn

from ml_modelcontrol.libraries import import_quietly
from ml_modelcontrol.pretrained import hub_weights_info, pretrained_info
from ml_modelcontrol.registry import get_unit

# 報告に並べるキーの数の上限 (使わない重みは数千になることがあるため)
REPORT_KEYS = 50

# 新しい重みに数えないキーの終わり (古いファイルには無い BatchNorm の数)
SKIP_NEW_SUFFIXES = ("num_batches_tracked",)


@dataclass
class LoadContext:
    # 相対のパスの元のディレクトリと, ライセンスを読むのにネットワークを
    # 使うか
    base_dir: Path
    fetch: bool = True


# 組み立てで外部の重みを読む設定 (build_model の中だけで設定. None なら
# 読まない. 形の推定は読まない)
LOAD_CONTEXT: contextvars.ContextVar[Optional[LoadContext]] = (
    contextvars.ContextVar("LOAD_CONTEXT", default=None)
)


def to_value(value: Any) -> Any:
    """
    定義の引数の値 (JSON の値) を, 部品に渡す値に直す関数

    Parameters
    ----------
    value: Any
        定義に書いた値

    Returns
    ----------
    arg: Any
        リストはタプルに, それ以外はそのまま
    """
    if isinstance(value, list):
        arg = tuple(to_value(item) for item in value)
    else:
        arg = value

    return arg


def remap_key(key: str, remap: list[dict]) -> str:
    """
    ファイルのキーを, 付け替えのリストの最初に当たったもので付け替える
    関数

    Parameters
    ----------
    key: str
        ファイルの state_dict のキー
    remap: list[dict]
        {from, to} (前置きの置き換え) か {pattern, to, offset} (正規表現)
        のリスト

    Returns
    ----------
    renamed: str
        付け替えたキー (どれにも当たらなければそのまま)

    Raises
    ----------
    ValueError
        offset で付け替えた番号が負になる場合と, 番号でない場合
    """
    # 上から見て最初に当たった 1 つだけを使用
    for rule in remap:
        if "from" in rule:
            if key.startswith(rule["from"]):
                renamed = rule["to"] + key[len(rule["from"]):]
                return renamed
            continue
        match = re.search(rule["pattern"], key)
        if match is None:
            continue
        to = rule["to"]
        if "offset" in rule:
            to = shifted(to, match.group(1), int(rule["offset"]), key)
        renamed = key[:match.start()] + match.expand(to) + key[match.end():]
        return renamed
    renamed = key

    return renamed


def shifted(to: str, number: str, offset: int, key: str) -> str:
    """
    置き換えの文字の \\1 を, 番号に offset を足した数に置き換える関数

    Parameters
    ----------
    to: str
        置き換えの文字
    number: str
        正規表現のグループに当たった文字
    offset: int
        足す数
    key: str
        ファイルのキー (誤りの文に添える)

    Returns
    ----------
    replaced: str
        番号を置き換えた文字

    Raises
    ----------
    ValueError
        番号でない場合と, 負になる場合
    """
    # 数字だけのグループを数として足し, 参照の書き方の 2 通りを置換
    if not number.isdigit():
        raise ValueError(
            f"キー {key} の番号 {number!r} は数字ではないので offset を"
            "足せません"
        )
    value = int(number) + offset
    if value < 0:
        raise ValueError(
            f"キー {key} の番号 {number} に offset {offset} を足すと負に"
            "なります"
        )
    replaced = to.replace("\\g<1>", str(value)).replace("\\1", str(value))

    return replaced


def pick_state(loaded: Any, state_key: Optional[str]) -> dict:
    """
    読み込んだ中身から, state_dict (キーとテンソルの辞書) を取り出す関数

    Parameters
    ----------
    loaded: Any
        ファイルから読んだ値
    state_key: Optional[str]
        入れ物の辞書から取り出すキー (. で区切って入れ子も), None なら
        そのまま

    Returns
    ----------
    state: dict
        キーとテンソルの辞書 (テンソルでない値は除く)

    Raises
    ----------
    ValueError
        キーが無い場合と, 辞書でない場合
    """
    # 入れ物の辞書を順にたどり, テンソルの値だけを残す
    value = loaded
    for part in (state_key.split(".") if state_key else []):
        if not isinstance(value, dict) or part not in value:
            keys = list(value)[:10] if isinstance(value, dict) else []
            raise ValueError(
                f"読んだ中身に state_key の {part} がありません "
                f"(あるキー: {', '.join(map(str, keys))})"
            )
        value = value[part]
    if not isinstance(value, dict):
        raise ValueError(
            "読んだ中身が state_dict (キーとテンソルの辞書) ではありません. "
            "入れ物の辞書なら state_key を書いてください"
        )
    state = {
        str(key): tensor for key, tensor in value.items()
        if isinstance(tensor, torch.Tensor)
    }
    if not state and any(isinstance(item, dict) for item in value.values()):
        raise ValueError(
            "読んだ中身にテンソルがありません. 入れ物の辞書なら state_key "
            f"を書いてください (あるキー: {', '.join(map(str, value))})"
        )

    return state


def read_file(path: Path) -> Any:
    """
    重みのファイルを読む関数

    Parameters
    ----------
    path: Path
        .safetensors か torch.save したファイル

    Returns
    ----------
    loaded: Any
        読んだ値 (torch.load は weights_only=True で読む)
    """
    # safetensors は専用の読み方, それ以外は任意のコードを動かさない読み方
    if not path.is_file():
        raise ValueError(f"重みのファイル {path} がありません")
    if path.suffix == ".safetensors":
        module, reason = import_quietly("safetensors.torch")
        if module is None:
            raise ValueError(f".safetensors を読めません: {reason}")
        loaded = module.load_file(str(path), device="cpu")
        return loaded
    loaded = torch.load(path, map_location="cpu", weights_only=True)

    return loaded


def hub_download(hub_id: str, filename: str, revision: str) -> Path:
    """
    Hugging Face の置き場からファイルを取ってくる関数

    Parameters
    ----------
    hub_id: str
        置き場の名前
    filename: str
        置き場の中のファイルの名前
    revision: str
        版

    Returns
    ----------
    path: Path
        取ってきた (手元に覚えた) ファイルのパス

    Raises
    ----------
    ValueError
        huggingface_hub が無い場合
    """
    # 取ってくるのは huggingface_hub に任せ, トークンも同じ環境変数を使用
    hub, reason = import_quietly("huggingface_hub")
    if hub is None:
        raise ValueError(f"Hugging Face から重みを取れません: {reason}")
    path = Path(hub.hf_hub_download(
        repo_id=hub_id, filename=filename, revision=revision
    ))

    return path


def read_source(
    weights: dict, context: LoadContext
) -> tuple[dict, str, Optional[dict]]:
    """
    外部の重みの出どころから state_dict を読む関数

    Parameters
    ----------
    weights: dict
        ノードの weights の定義
    context: LoadContext
        相対のパスの元のディレクトリとネットワークの設定

    Returns
    ----------
    state: dict
        キーとテンソルの辞書 (付け替える前)
    source: str
        出どころの説明
    license: Optional[dict]
        ライセンスの情報 (読めない出どころは None)
    """
    # ファイル (相対のパスは元のディレクトリから)
    license_info: Optional[dict] = None
    if "path" in weights:
        path = Path(weights["path"]).expanduser()
        if not path.is_absolute():
            path = context.base_dir / path
        loaded = read_file(path)
        source = f"ファイル {weights['path']}"

    # Hugging Face の置き場のファイル (ライセンスは配布元から)
    elif "hub" in weights:
        revision = str(weights.get("revision") or "main")
        loaded = read_file(
            hub_download(weights["hub"], weights["file"], revision)
        )
        source = (
            f"Hugging Face {weights['hub']} の {weights['file']} "
            f"(版 {revision})"
        )
        license_info = hub_weights_info(
            weights["hub"], revision, context.fetch
        )

    # 学習済みの重みを読む部品 (ライセンスは pretrained_info)
    else:
        params = dict(weights.get("params") or {})
        unit = get_unit(weights["unit"])
        if unit.builder is None:
            raise ValueError(f"部品 {unit.name} は使えません: {unit.reason}")
        module = unit.builder(
            **{key: to_value(value) for key, value in params.items()}
        )
        loaded = module.state_dict()
        source = f"部品 {unit.name}"
        license_info = pretrained_info(unit.name, params, context.fetch)
    state = pick_state(loaded, weights.get("state_key"))

    return state, source, license_info


def remap_state(state: dict, remap: list[dict]) -> dict:
    """
    state_dict のキーを付け替える関数

    Parameters
    ----------
    state: dict
        キーとテンソルの辞書
    remap: list[dict]
        付け替えのリスト

    Returns
    ----------
    renamed: dict
        付け替えたキーとテンソルの辞書

    Raises
    ----------
    ValueError
        付け替えた後に同じキーになる 2 つのキーがある場合
    """
    # キーごとに付け替え, 重なりを確認
    renamed: dict = {}
    origins: dict[str, str] = {}
    for key, tensor in state.items():
        new_key = remap_key(key, remap)
        if new_key in renamed:
            raise ValueError(
                f"付け替えると {origins[new_key]} と {key} が同じキー "
                f"{new_key} になります"
            )
        renamed[new_key] = tensor
        origins[new_key] = key

    return renamed


def load_node_weights(
    name: str, module: nn.Module, weights: dict, context: LoadContext
) -> dict:
    """
    外部の重みを 1 つのノードに読み込み, 報告を返す関数

    Parameters
    ----------
    name: str
        ノードの名前
    module: nn.Module
        ノードの部品
    weights: dict
        ノードの weights の定義
    context: LoadContext
        相対のパスの元のディレクトリとネットワークの設定

    Returns
    ----------
    report: dict
        node, source, strict, loaded, mismatched, new, unused (数),
        mismatched_keys, new_keys, unused_keys, license

    Raises
    ----------
    ValueError
        1 つも読めない場合と, strict で形の違う重みか新しい重みがある場合
    """
    # 出どころから読んで付け替え, ノードのキーと比較
    state, source, license_info = read_source(weights, context)
    state = remap_state(state, list(weights.get("remap") or []))
    target = module.state_dict()
    matched = {
        key: tensor for key, tensor in state.items()
        if key in target and tensor.shape == target[key].shape
    }
    mismatched = [
        [key, list(tensor.shape), list(target[key].shape)]
        for key, tensor in state.items()
        if key in target and tensor.shape != target[key].shape
    ]
    new = [
        key for key in target
        if key not in state and not key.endswith(SKIP_NEW_SUFFIXES)
    ]
    unused = [key for key in state if key not in target]
    strict = bool(weights.get("strict", True))
    report = {
        "node": name, "source": source, "strict": strict,
        "loaded": len(matched), "mismatched": len(mismatched),
        "new": len(new), "unused": len(unused),
        "mismatched_keys": mismatched[:REPORT_KEYS],
        "new_keys": new[:REPORT_KEYS], "unused_keys": unused[:REPORT_KEYS],
        "license": license_info
    }

    # 1 つも読めない場合と, strict で読めないものがある場合は誤り
    if not matched:
        raise ValueError(
            f"{source} から読める重みがありません (ファイルのキーの例: "
            f"{', '.join(list(state)[:3])}. ノードのキーの例: "
            f"{', '.join(list(target)[:3])}). remap でキーを付け替えて"
            "ください"
        )
    if strict and (mismatched or new):
        raise ValueError(
            f"{source} の重みが合いません (形の違う重み {len(mismatched)}, "
            f"ファイルに無い重み {len(new)}. 例: "
            f"{', '.join([item[0] for item in mismatched[:2]] + new[:2])}). "
            "読めるものだけを読む場合は strict: false にしてください"
        )
    module.load_state_dict(matched, strict=False)

    return report


def zero_parameters(module: nn.Module, names: Any) -> None:
    """
    部品のパラメータを 0 にする関数

    Parameters
    ----------
    module: nn.Module
        ノードの部品
    names: Any
        True なら全て, リストならその名前のパラメータ

    Returns
    ----------
    None

    Raises
    ----------
    ValueError
        部品に無いパラメータの名前の場合
    """
    # 名前の確認と, 勾配を記録しない書き換え
    parameters = dict(module.named_parameters())
    if names is True:
        targets = list(parameters.values())
    elif names is False:
        targets = []
    else:
        missing = [name for name in names if name not in parameters]
        if missing:
            raise ValueError(
                f"zero_init のパラメータ {', '.join(missing)} はありません "
                f"(パラメータ: {', '.join(list(parameters)[:10])})"
            )
        targets = [parameters[name] for name in names]
    with torch.no_grad():
        for parameter in targets:
            parameter.zero_()


def tie_parameter(
    module: nn.Module, name: str, source: nn.Module, source_name: str
) -> None:
    """
    部品のパラメータを, 別の部品のパラメータと同じものにする関数

    Parameters
    ----------
    module: nn.Module
        共有するノードの部品
    name: str
        共有するノードのパラメータの名前
    source: nn.Module
        共有の元のノードの部品
    source_name: str
        共有の元のノードのパラメータの名前

    Returns
    ----------
    None

    Raises
    ----------
    ValueError
        パラメータが無い場合と, 形が違う場合
    """
    # 両方のパラメータがあり, 形が同じことの確認
    own = dict(module.named_parameters(remove_duplicate=False))
    other = dict(source.named_parameters(remove_duplicate=False))
    if name not in own:
        raise ValueError(
            f"共有するパラメータ {name} はありません (パラメータ: "
            f"{', '.join(list(own)[:10])})"
        )
    if source_name not in other:
        raise ValueError(
            f"共有の元のパラメータ {source_name} はありません (パラメータ: "
            f"{', '.join(list(other)[:10])})"
        )
    if own[name].shape != other[source_name].shape:
        raise ValueError(
            f"共有するパラメータ {name} の形 {list(own[name].shape)} と, "
            f"共有の元の {source_name} の形 "
            f"{list(other[source_name].shape)} が違います"
        )

    # 持ち主の部品の属性を, 元のパラメータに置換
    owner_name, _, leaf = name.rpartition(".")
    owner = module.get_submodule(owner_name)
    setattr(owner, leaf, other[source_name])


def share_targets(share: Any) -> list[tuple[str, str, str]]:
    """
    パラメータごとの共有の辞書を, 名前と元のノードと元の名前の組にする
    関数

    Parameters
    ----------
    share: Any
        {パラメータの名前: <ノード>.<パラメータの名前>} の辞書

    Returns
    ----------
    targets: list[tuple[str, str, str]]
        このノードのパラメータの名前, 元のノード, 元のパラメータの名前
    """
    targets = []
    for name, ref in share.items():
        node, _, source_name = str(ref).partition(".")
        targets.append((str(name), node, source_name))

    return targets
