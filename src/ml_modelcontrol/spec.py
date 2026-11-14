import copy
import json
from pathlib import Path
from typing import Any, Optional, Union

import yaml

# このライブラリが読める定義の版
FORMAT_VERSION = 1

# 定義の一番上, 入力, ノードで書ける項目
GRAPH_KEYS = {
    "format_version", "name", "description", "inputs", "nodes", "outputs",
    "ui"
}
INPUT_KEYS = {"shape", "dtype", "fill", "description", "ui"}
NODE_KEYS = {
    "unit", "params", "forward_params", "inputs", "description", "ui"
}


def load_spec(source: Union[str, Path, dict]) -> dict:
    """
    モデルの定義を読み込む関数

    Parameters
    ----------
    source: Union[str, Path, dict]
        YAML か JSON のファイルのパス, または読み込み済みの辞書

    Returns
    ----------
    spec: dict
        モデルの定義 (版を今の版に直したもの)
    """
    # 辞書はそのまま, ファイルは拡張子で JSON と YAML を読み分け
    if isinstance(source, dict):
        spec = copy.deepcopy(source)
    else:
        path = Path(source)
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".json":
            spec = json.loads(text)
        else:
            spec = yaml.safe_load(text)
    if not isinstance(spec, dict):
        raise ValueError("モデルの定義は辞書 (YAML のマップ) にしてください")
    spec = migrate(spec)

    return spec


def migrate(spec: dict) -> dict:
    """
    古い版の定義を今の版に直す関数

    Parameters
    ----------
    spec: dict
        モデルの定義

    Returns
    ----------
    spec: dict
        今の版の定義

    Raises
    ----------
    ValueError
        ライブラリより新しい版の場合
    """
    # 版の無い定義は今の版として扱い, 新しい版はエラー
    version = spec.get("format_version", FORMAT_VERSION)
    if not isinstance(version, int) or isinstance(version, bool):
        raise ValueError(f"format_version は整数にしてください: {version!r}")
    if version > FORMAT_VERSION:
        raise ValueError(
            f"format_version {version} の定義は, このライブラリ (版 "
            f"{FORMAT_VERSION} まで) では読めません"
        )
    spec["format_version"] = FORMAT_VERSION

    return spec


def dump_spec(spec: dict, path: Optional[Union[str, Path]] = None) -> str:
    """
    モデルの定義を YAML か JSON の文字にし, パスがあれば保存する関数

    一番上の項目は決まった順に並べ, ノードは書いた順のまま残す

    Parameters
    ----------
    spec: dict
        モデルの定義
    path: Optional[Union[str, Path]] = None
        保存先 (.json なら JSON, それ以外は YAML), None なら保存しない

    Returns
    ----------
    text: str
        保存した文字
    """
    # 一番上の項目の順をそろえ, 知らない項目は後ろに残す
    order = [
        "format_version", "name", "description", "inputs", "nodes",
        "outputs", "ui"
    ]
    ordered = {key: spec[key] for key in order if key in spec}
    ordered.update({
        key: value for key, value in spec.items() if key not in ordered
    })

    # 拡張子に合わせて書き出し
    is_json = path is not None and Path(path).suffix == ".json"
    if is_json:
        text = json.dumps(ordered, ensure_ascii=False, indent=2) + "\n"
    else:
        text = yaml.safe_dump(
            ordered, allow_unicode=True, sort_keys=False,
            default_flow_style=None
        )
    if path is not None:
        Path(path).write_text(text, encoding="utf-8")

    return text


def parse_ref(ref: str) -> tuple[str, Optional[str]]:
    """
    つなぎ先の文字を, 名前と口の名前に分ける関数

    Parameters
    ----------
    ref: str
        「名前」か「名前:口の名前」

    Returns
    ----------
    name: str
        入力かノードの名前
    port: Optional[str]
        出力の口の名前, 書いていない場合は None
    """
    # : で 1 回だけ分割
    name, _, port = ref.partition(":")
    port_name = port if port else None

    return name, port_name


def node_inputs(node: dict) -> Any:
    """
    ノードの inputs を返す関数 (書いていない場合は空の辞書)

    Parameters
    ----------
    node: dict
        ノードの定義

    Returns
    ----------
    inputs: Any
        口の名前とつなぎ先の辞書, 省略形の場合は文字
    """
    inputs = node.get("inputs", {})
    if inputs is None:
        inputs = {}

    return inputs
