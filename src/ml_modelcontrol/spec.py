import copy
import json
import re
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


# YAML 1.2 の書き方の数 (1e-5 など) も数として読む読み込み方
class SpecLoader(yaml.SafeLoader):
    pass


SpecLoader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        r"^[-+]?(?:[0-9][0-9_]*\.[0-9_]*(?:[eE][-+]?[0-9]+)?"
        r"|[0-9][0-9_]*[eE][-+]?[0-9]+|\.[0-9_]+(?:[eE][-+]?[0-9]+)?"
        r"|\.(?:inf|Inf|INF)|\.(?:nan|NaN|NAN))$"
    ),
    list("-+0123456789.")
)


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
        # パスの代わりに YAML の文字を渡した場合の案内
        if isinstance(source, str) and ("\n" in source or ": " in source):
            raise ValueError(
                "ファイルのパスではなく YAML の文字のようです. "
                "yaml.safe_load で辞書にしてから渡してください"
            )
        path = Path(source)
        text = path.read_text(encoding="utf-8")
        try:
            if path.suffix == ".json":
                spec = json.loads(text)
            else:
                spec = yaml.load(text, Loader=SpecLoader)
        except (yaml.YAMLError, json.JSONDecodeError) as error:
            raise ValueError(f"{path} を読めません: {error}") from error
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
    if version < 1:
        raise ValueError(f"format_version は 1 以上にしてください: {version}")
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
    # 一番上の項目の順の整列 (知らない項目は後ろ)
    order = [
        "format_version", "name", "description", "inputs", "nodes",
        "outputs", "ui"
    ]
    ordered = {key: spec[key] for key in order if key in spec}
    ordered.update({
        key: value for key, value in spec.items() if key not in ordered
    })

    # ノードの項目の順もそろえ, 拡張子に合わせて書き出し
    node_order = [
        "unit", "description", "params", "forward_params", "inputs", "ui"
    ]
    if isinstance(ordered.get("nodes"), dict):
        ordered["nodes"] = {
            name: (
                {
                    **{key: node[key] for key in node_order if key in node},
                    **{
                        key: value for key, value in node.items()
                        if key not in node_order
                    }
                } if isinstance(node, dict) else node
            )
            for name, node in ordered["nodes"].items()
        }
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
    ノードの inputs を, 書いていない場合は空の辞書にして返す関数

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


# 同梱した例のモデルの定義を置く場所
EXAMPLES_DIR = Path(__file__).parent / "examples"


def list_examples() -> list[dict]:
    """
    同梱した例のモデルの定義の一覧を返す関数

    Parameters
    ----------
    None

    Returns
    ----------
    examples: list[dict]
        名前 (ファイル名から拡張子を除いたもの) と説明
    """
    # 例のファイルごとに名前と説明を取り出し
    examples = []
    for path in sorted(EXAMPLES_DIR.glob("*.yaml")):
        spec = load_spec(path)
        examples.append({
            "name": path.stem, "description": spec.get("description", "")
        })

    return examples


def load_example(name: str) -> dict:
    """
    同梱した例のモデルの定義を読み込む関数

    Parameters
    ----------
    name: str
        例の名前 (list_examples の name)

    Returns
    ----------
    spec: dict
        モデルの定義

    Raises
    ----------
    KeyError
        例が無い場合
    """
    # 一覧にある名前だけを受け付け (パスとして解釈しない)
    names = {path.stem: path for path in EXAMPLES_DIR.glob("*.yaml")}
    if name not in names:
        raise KeyError(f"例 {name} はありません")
    spec = load_spec(names[name])

    return spec
