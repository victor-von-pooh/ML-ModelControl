import contextvars
import os
import pickle
import re
import warnings
from dataclasses import dataclass, field
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


# 読んでよいディレクトリ (定義のディレクトリのほか. パスの区切りで並べる)
# と, 重みのファイルの大きさの上限 (バイト) の環境変数
DIRS_ENV = "ML_MODELCONTROL_WEIGHTS_DIRS"
MAX_BYTES_ENV = "ML_MODELCONTROL_WEIGHTS_MAX_BYTES"

# 重みのファイルの大きさの既定の上限 (4 GiB)
DEFAULT_MAX_BYTES = 4 * 1024 ** 3


@dataclass
class LoadContext:
    # 相対のパスの元のディレクトリ (読んでよいディレクトリでもある),
    # ライセンスの読み込みでのネットワークの使用の有無, 他に読んでよい
    # ディレクトリ, 重みのファイルの大きさの上限
    base_dir: Path
    fetch: bool = True
    allowed_dirs: list[Path] = field(default_factory=list)
    max_bytes: int = DEFAULT_MAX_BYTES


def make_context(
    base_dir: Path, fetch: bool, extra_dirs: Optional[list] = None
) -> LoadContext:
    """
    外部の重みを読む設定を, 環境変数と引数から作る関数

    Parameters
    ----------
    base_dir: Path
        相対のパスの元のディレクトリ
    fetch: bool
        ライセンスの読み込みでネットワークを使うか
    extra_dirs: Optional[list] = None
        他に読んでよいディレクトリ

    Returns
    ----------
    context: LoadContext
        読み込みの設定

    Raises
    ----------
    ValueError
        大きさの上限の環境変数が整数でない場合
    """
    # 環境変数のディレクトリ (空の項目は除く) と大きさの上限
    dirs = [
        Path(item).expanduser()
        for item in os.environ.get(DIRS_ENV, "").split(os.pathsep) if item
    ]
    dirs += [Path(item).expanduser() for item in (extra_dirs or [])]
    text = os.environ.get(MAX_BYTES_ENV, "")
    if text and not text.isdigit():
        raise ValueError(f"{MAX_BYTES_ENV} はバイト数の整数にしてください")
    max_bytes = int(text) if text else DEFAULT_MAX_BYTES
    context = LoadContext(base_dir, fetch, dirs, max_bytes)

    return context


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


def shifted_number(number: Optional[str], offset: int) -> Optional[str]:
    """
    番号に offset を足した文字を返す関数 (0 で詰めた幅は保つ)

    Parameters
    ----------
    number: Optional[str]
        正規表現のグループに当たった文字 (当たらなければ None)
    offset: int
        足す数

    Returns
    ----------
    text: Optional[str]
        足した数の文字, 番号でない場合と負になる場合は None (その付け替えは
        当たらないものとする)
    """
    # 数字だけのグループを数として足し, 07 のような 0 の詰め方は幅を保持
    if number is None or not number.isdigit():
        return None
    value = int(number) + offset
    if value < 0:
        return None
    text = str(value)
    if len(number) > 1 and number.startswith("0"):
        text = text.zfill(len(number))

    return text


def apply_rule(key: str, rule: dict) -> Optional[str]:
    """
    1 つの付け替えをファイルのキーに当てる関数

    前置きも正規表現もキーの先頭から当てる

    Parameters
    ----------
    key: str
        ファイルの state_dict のキー
    rule: dict
        {from, to} (前置きの置き換え) か {pattern, to, offset} (正規表現)

    Returns
    ----------
    renamed: Optional[str]
        付け替えたキー, 当たらない場合は None (offset で番号が負になる
        場合も当たらないものとする)
    """
    # 前置きは先頭が同じキー, 正規表現は先頭から当たるキー
    if "from" in rule:
        if not key.startswith(rule["from"]):
            return None
        renamed = rule["to"] + key[len(rule["from"]):]
        return renamed
    match = re.match(rule["pattern"], key)
    if match is None:
        return None
    to = rule["to"]
    if "offset" in rule:
        number = shifted_number(match.group(1), int(rule["offset"]))
        if number is None:
            return None
        to = to.replace("\\g<1>", number).replace("\\1", number)
    renamed = match.expand(to) + key[match.end():]

    return renamed


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
    """
    # 上から見て最初に当たった 1 つだけを使用
    for rule in remap:
        renamed = apply_rule(key, rule)
        if renamed is not None:
            return renamed
    renamed = key

    return renamed


# 入れ物の辞書の値の種類を誤りの文に出す上限
SHOWN_KEYS = 10


def pick_state(loaded: Any, state_key: Any) -> dict:
    """
    読み込んだ中身から, state_dict (キーとテンソルの辞書) を取り出す関数

    Parameters
    ----------
    loaded: Any
        ファイルから読んだ値
    state_key: Any
        入れ物の辞書から取り出すキー (. で区切った文字か, . を含むキーの
        ためのリスト), None ならそのまま

    Returns
    ----------
    state: dict
        キーとテンソルの辞書

    Raises
    ----------
    ValueError
        キーが無い場合, 辞書でない場合, 中に辞書がある場合 (入れ物のまま
        で, state_key の書き忘れ)
    """
    # 入れ物の辞書を順にたどり (文字は . で区切る)
    parts = state_key.split(".") if isinstance(state_key, str) else list(
        state_key or []
    )
    value = loaded
    for part in parts:
        if not isinstance(value, dict) or part not in value:
            keys = list(value)[:SHOWN_KEYS] if isinstance(value, dict) else []
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

    # 中に辞書があれば, 黙って除かずに誤り (入れ物のままのため)
    nested = [
        str(key) for key, item in value.items() if isinstance(item, dict)
    ]
    if nested:
        raise ValueError(
            f"読んだ中身に辞書 ({', '.join(nested[:SHOWN_KEYS])}) があります."
            " 入れ物の辞書なら state_key に取り出すキーを書いてください"
        )
    state = {
        str(key): tensor for key, tensor in value.items()
        if isinstance(tensor, torch.Tensor)
    }

    return state


# 外部の重みのファイル (キーごとの形と, 選んだキーだけのテンソルの読み込み)
class WeightsFile:
    def __init__(
        self, shapes: dict[str, list[int]], tensors: Optional[dict] = None,
        path: Optional[Path] = None
    ):
        # キーごとの形と, 読み込み済みのテンソル (safetensors は読む場所)
        self.shapes = shapes
        self.tensors = tensors
        self.path = path

    def load(self, keys: list[str]) -> dict:
        """
        選んだキーのテンソルを返す関数

        safetensors は選んだキーだけをファイルから読む

        Parameters
        ----------
        keys: list[str]
            ファイルのキー

        Returns
        ----------
        tensors: dict
            キーとテンソルの辞書
        """
        # 読み込み済みならその一部, safetensors は選んだキーだけを読み込み
        if self.tensors is not None:
            tensors = {key: self.tensors[key] for key in keys}
            return tensors
        module, reason = import_quietly("safetensors")
        if module is None:
            raise ValueError(f".safetensors を読めません: {reason}")
        tensors = {}
        with module.safe_open(str(self.path), framework="pt",
                              device="cpu") as handle:
            for key in keys:
                tensors[key] = handle.get_tensor(key)

        return tensors


def from_tensors(tensors: dict) -> WeightsFile:
    """
    キーとテンソルの辞書を WeightsFile にする関数

    Parameters
    ----------
    tensors: dict
        キーとテンソルの辞書

    Returns
    ----------
    weights_file: WeightsFile
        読み込み済みのファイル
    """
    shapes = {key: list(tensor.shape) for key, tensor in tensors.items()}
    weights_file = WeightsFile(shapes, tensors)

    return weights_file


def check_size(path: Path, written: str, context: LoadContext) -> None:
    """
    重みのファイルがあり, 大きさが上限までかを確かめる関数

    Parameters
    ----------
    path: Path
        ファイルのパス
    written: str
        誤りの文に出す名前 (定義に書いたパスか置き場のファイルの名前)
    context: LoadContext
        大きさの上限

    Returns
    ----------
    None

    Raises
    ----------
    ValueError
        ファイルが無い場合と, 上限を超える場合
    """
    # 読む前に stat で大きさを確認 (誤りの文に手元の絶対のパスは出さない)
    if not path.is_file():
        raise ValueError(f"重みのファイル {written} がありません")
    size = path.stat().st_size
    if size > context.max_bytes:
        raise ValueError(
            f"重みのファイル {written} の大きさ ({size} バイト) が上限 "
            f"({context.max_bytes} バイト) を超えます (環境変数 "
            f"{MAX_BYTES_ENV} で変えられます)"
        )


def torch_load(path: Path, written: str) -> Any:
    """
    torch.save したファイルを, 任意のコードを動かさない読み方で読む関数

    Parameters
    ----------
    path: Path
        ファイルのパス
    written: str
        誤りの文に出す名前

    Returns
    ----------
    loaded: Any
        読んだ値

    Raises
    ----------
    ValueError
        テンソル以外の値 (任意のコードを動かせる値) を含む場合
    """
    # 大きなファイルはメモリに写さず (mmap. 古い形のファイルは写して) 読み,
    # weights_only で読めないものは日本語の誤り (weights_only を外す案内は
    # しない)
    try:
        try:
            loaded = torch.load(
                path, map_location="cpu", weights_only=True, mmap=True
            )
        except RuntimeError as error:
            if "mmap" not in str(error):
                raise
            loaded = torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as error:
        raise ValueError(
            f"重みのファイル {written} はテンソル以外の値 (読むとコードを"
            "動かせる値) を含むので読みません. 信頼できる出どころなら, 元の"
            "環境で state_dict だけを torch.save し直すか, safetensors に"
            "してください"
        ) from error

    return loaded


def read_file(
    path: Path, written: str, state_key: Any, context: LoadContext
) -> WeightsFile:
    """
    重みのファイルを読む関数

    Parameters
    ----------
    path: Path
        .safetensors か torch.save したファイル
    written: str
        誤りの文に出す名前
    state_key: Any
        入れ物の辞書から取り出すキー (safetensors では使えない)
    context: LoadContext
        大きさの上限

    Returns
    ----------
    weights_file: WeightsFile
        キーごとの形と, テンソルを読むもの

    Raises
    ----------
    ValueError
        ファイルが無い場合, 大きすぎる場合, 読めない場合
    """
    # safetensors はキーと形だけを先に読み (テンソルは読めるものだけを後で),
    # それ以外は任意のコードを動かさない読み方
    check_size(path, written, context)
    if path.suffix != ".safetensors":
        loaded = torch_load(path, written)
        weights_file = from_tensors(pick_state(loaded, state_key))
        return weights_file
    if state_key:
        raise ValueError(".safetensors には state_key を書けません")
    module, reason = import_quietly("safetensors")
    if module is None:
        raise ValueError(f".safetensors を読めません: {reason}")
    with module.safe_open(str(path), framework="pt", device="cpu") as handle:
        shapes = {
            key: list(handle.get_slice(key).get_shape())
            for key in handle.keys()
        }
    weights_file = WeightsFile(shapes, path=path)

    return weights_file


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


def allowed_path(written: str, context: LoadContext) -> Path:
    """
    定義に書いたパスを, 読んでよいディレクトリの中のパスにする関数

    Parameters
    ----------
    written: str
        定義に書いたパス (相対のパスは元のディレクトリから)
    context: LoadContext
        元のディレクトリと, 読んでよいディレクトリ

    Returns
    ----------
    path: Path
        リンクをたどった後のパス

    Raises
    ----------
    ValueError
        読んでよいディレクトリの外の場合 (リンクで外に出るものも)
    """
    # リンクをたどった先で中にあるかを確認
    path = Path(written).expanduser()
    if not path.is_absolute():
        path = context.base_dir / path
    path = path.resolve()
    roots = [root.resolve() for root in [context.base_dir,
                                         *context.allowed_dirs]]
    if not any(path.is_relative_to(root) for root in roots):
        raise ValueError(
            f"重みのファイル {written} は, 読んでよいディレクトリ (定義の"
            "ディレクトリと, 環境変数 "
            f"{DIRS_ENV} に書いたもの) の外にあるので読みません"
        )

    return path


def read_source(
    weights: dict, context: LoadContext
) -> tuple[WeightsFile, str, Optional[dict]]:
    """
    外部の重みの出どころから state_dict を読む関数

    Parameters
    ----------
    weights: dict
        ノードの weights の定義
    context: LoadContext
        相対のパスの元のディレクトリ, 読んでよいディレクトリ, 大きさの
        上限, ネットワークの設定

    Returns
    ----------
    weights_file: WeightsFile
        キーごとの形とテンソルを読むもの (付け替える前)
    source: str
        出どころの説明
    license_info: Optional[dict]
        ライセンスの情報 (読めない出どころは None)

    Raises
    ----------
    ValueError
        ファイルが無い場合, 読んでよいディレクトリの外の場合, 読めない
        場合, 部品が使えない場合
    """
    # ファイル (相対のパスは元のディレクトリから. 読んでよい所だけ)
    license_info: Optional[dict] = None
    state_key = weights.get("state_key")
    if "path" in weights:
        written = str(weights["path"])
        weights_file = read_file(
            allowed_path(written, context), written, state_key, context
        )
        source = f"ファイル {written}"

    # Hugging Face の置き場のファイル (ライセンスは配布元から)
    elif "hub" in weights:
        revision = str(weights.get("revision") or "main")
        written = f"{weights['hub']} の {weights['file']}"
        weights_file = read_file(
            hub_download(weights["hub"], weights["file"], revision),
            written, state_key, context
        )
        source = f"Hugging Face {written} (版 {revision})"
        license_info = hub_weights_info(
            weights["hub"], revision, context.fetch, str(weights["file"])
        )

    # 学習済みの重みを読む部品 (ライセンスは pretrained_info)
    else:
        params = dict(weights.get("params") or {})
        unit = get_unit(weights["unit"])
        if unit.builder is None:
            raise ValueError(f"部品 {unit.name} は使えません: {unit.reason}")
        # 外の乱数の状態を変えない作成 (種を決めた初期値が読み込みに依らない)
        with torch.random.fork_rng(devices=[]):
            module = unit.builder(
                **{key: to_value(value) for key, value in params.items()}
            )
        weights_file = from_tensors(pick_state(module.state_dict(), None))
        source = f"部品 {unit.name}"
        license_info = pretrained_info(unit.name, params, context.fetch)

    return weights_file, source, license_info


def remap_keys(keys: list[str], remap: list[dict]) -> dict[str, str]:
    """
    ファイルのキーを付け替え, 付け替えたキーから元のキーへの辞書を返す関数

    Parameters
    ----------
    keys: list[str]
        ファイルのキー
    remap: list[dict]
        付け替えのリスト

    Returns
    ----------
    origins: dict[str, str]
        付け替えたキーごとの元のキー

    Raises
    ----------
    ValueError
        付け替えた後に同じキーになる 2 つのキーがある場合
    """
    # キーごとに付け替え, 重なりを確認
    origins: dict[str, str] = {}
    for key in keys:
        new_key = remap_key(key, remap)
        if new_key in origins:
            raise ValueError(
                f"付け替えると {origins[new_key]} と {key} が同じキー "
                f"{new_key} になります"
            )
        origins[new_key] = key

    return origins


def load_node_weights(
    name: str, module: nn.Module, weights: dict, context: LoadContext,
    shared: Optional[set[str]] = None
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
        相対のパスの元のディレクトリ, 読んでよいディレクトリ, 大きさの
        上限, ネットワークの設定
    shared: Optional[set[str]] = None
        パラメータごとの共有で, 共有の元から受け取るパラメータの名前 (読む
        対象から除く)

    Returns
    ----------
    report: dict
        node, source, strict, loaded, mismatched, new, unused (数),
        mismatched_keys, new_keys, unused_keys, license, warnings

    Raises
    ----------
    ValueError
        1 つも読めない場合と, strict で形の違う重みか新しい重みがある場合
    """
    # 出どころから読んで付け替え, ノードのキー (共有で受け取るものを除く)
    # と比較
    weights_file, source, license_info = read_source(weights, context)
    origins = remap_keys(
        list(weights_file.shapes), list(weights.get("remap") or [])
    )
    shapes = {key: weights_file.shapes[origin]
              for key, origin in origins.items()}
    shared = shared or set()
    target = {
        key: list(tensor.shape) for key, tensor in module.state_dict().items()
        if key not in shared
    }
    matched = [
        key for key, shape in shapes.items()
        if key in target and shape == target[key]
    ]
    mismatched = [
        [key, shape, target[key]] for key, shape in shapes.items()
        if key in target and shape != target[key]
    ]
    # 共有したパラメータ (同じ物の別のキー) は, どれかが読めれば新しい
    # 重みに数えない (共有した重みはファイルに 1 回だけ保存されることが多い)
    aliases: dict[int, list[str]] = {}
    for key, tensor in module.state_dict(keep_vars=True).items():
        aliases.setdefault(id(tensor), []).append(key)
    covered = {
        key for keys in aliases.values()
        if any(item in matched for item in keys) for key in keys
    }
    new = [
        key for key in target
        if key not in shapes and key not in covered
        and not key.endswith(SKIP_NEW_SUFFIXES)
    ]
    unused = [key for key in shapes if key not in target]
    warnings_list = [
        f"共有するパラメータ {key} はファイルから読まず, 共有の元のノードの"
        "値を使います (使わない重みに数えた)"
        for key in shapes if key in shared
    ]
    strict = bool(weights.get("strict", True))
    report = {
        "node": name, "source": source, "strict": strict,
        "loaded": len(matched), "mismatched": len(mismatched),
        "new": len(new), "unused": len(unused),
        "mismatched_keys": mismatched[:REPORT_KEYS],
        "new_keys": new[:REPORT_KEYS], "unused_keys": unused[:REPORT_KEYS],
        "license": license_info, "warnings": warnings_list
    }

    # 1 つも読めない場合と, strict で読めないものがある場合は誤り
    if not matched:
        raise ValueError(
            f"{source} から読める重みがありません (ファイルのキーの例: "
            f"{', '.join(list(shapes)[:3])}. ノードのキーの例: "
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

    # 読めるキーのテンソルだけを読み込み
    tensors = weights_file.load([origins[key] for key in matched])
    module.load_state_dict(
        {key: tensors[origins[key]] for key in matched}, strict=False
    )
    for message in warnings_list:
        warnings.warn(f"ノード {name}: {message}", stacklevel=2)

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
    # 値を最初の . で元のノードとパラメータの名前に分割
    targets = []
    for name, ref in share.items():
        node, _, source_name = str(ref).partition(".")
        targets.append((str(name), node, source_name))

    return targets
