import inspect
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional

from torch import nn

# 引数として扱わない __init__ の引数 (置き場所と型は組み立てた後に決める)
SKIPPED_PARAMS = {"self", "device", "dtype"}

# 入力の口として扱う forward の引数の既定値 (None は省略できるテンソル)
# 真偽値や数の既定値を持つ引数 (need_weights など) は, 口ではなく呼び出しの設定
PORT_DEFAULTS = (None,)


@dataclass
class ParamSpec:
    # 引数の名前, 種類 (int, float, bool, str, list, any), 既定値, 必須か
    name: str
    kind: str = "any"
    default: Any = None
    required: bool = False

    # 説明, 範囲 (端を含む), 選択肢
    description: str = ""
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    choices: Optional[list] = None


@dataclass
class PortSpec:
    # 口の名前, 説明, 省略できるか, 可変の数のつなぎ先を受け取るか
    name: str
    description: str = ""
    optional: bool = False
    variadic: bool = False

    # 出力の口の取り出し方 (番号か属性の名前の並び, 空なら戻り値そのもの)
    path: list = field(default_factory=list)


@dataclass
class UnitSpec:
    # 部品の名前 (nn.Linear など), 分類, 出どころ, 説明
    name: str
    category: str
    source: str
    description: str = ""

    # 引数と, 入力と出力の口, 呼び出すときの設定の値
    params: list[ParamSpec] = field(default_factory=list)
    inputs: list[PortSpec] = field(default_factory=list)
    outputs: list[PortSpec] = field(default_factory=list)
    forward_params: list[ParamSpec] = field(default_factory=list)

    # 引数の一覧に無い引数も受け取るか (**kwargs を持つクラス)
    extra_params: bool = False

    # 使えるかどうかと, 使えない場合の理由
    available: bool = True
    reason: str = ""

    # 引数の値から nn.Module を作る関数 (一覧の表示には含めない)
    builder: Optional[Callable[..., nn.Module]] = None

    # 書き出す Python のコードで, 作る式に使う名前 (nn.Linear など)
    code_name: str = ""
    code_import: str = ""


# 名前から部品を引く登録簿
REGISTRY: dict[str, UnitSpec] = {}


def register(spec: UnitSpec) -> UnitSpec:
    """
    部品を登録簿に加える関数

    Parameters
    ----------
    spec: UnitSpec
        登録する部品

    Returns
    ----------
    spec: UnitSpec
        登録した部品
    """
    # 同じ名前は後から登録したもので上書き
    REGISTRY[spec.name] = spec

    return spec


def get_unit(name: str) -> UnitSpec:
    """
    名前から部品を取り出す関数

    Parameters
    ----------
    name: str
        部品の名前 (nn.Linear など)

    Returns
    ----------
    spec: UnitSpec
        部品

    Raises
    ----------
    KeyError
        登録されていない名前の場合
    """
    # 組み込みの部品を読み込んでから引く
    ensure_loaded()
    if name not in REGISTRY:
        raise KeyError(f"部品 {name} は登録されていません")
    spec = REGISTRY[name]

    return spec


def describe_units() -> list[dict]:
    """
    登録した部品の一覧を, JSON にできる辞書のリストで返す関数

    GUI の部品の一覧と, 引数のフォームに使う

    Parameters
    ----------
    None

    Returns
    ----------
    units: list[dict]
        名前の順の部品ごとの辞書 (作る関数は除く)
    """
    # 組み込みの部品を読み込み, 作る関数を除いて辞書に変換
    ensure_loaded()
    units = []
    for name in sorted(REGISTRY):
        data = asdict(REGISTRY[name])
        for key in ("builder", "code_name", "code_import"):
            data.pop(key)
        units.append(data)

    return units


def ensure_loaded() -> None:
    """
    組み込みの部品 (torch.nn, 演算, 外部のライブラリ) を登録簿に読み込む
    関数

    最初に呼ばれたときだけ読み込む

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 循環 import を避けるため, ここで読み込み
    from ml_modelcontrol import units

    units.load_builtin_units()


def to_plain(value: Any) -> Any:
    """
    既定値を JSON にできる値に直す関数

    Parameters
    ----------
    value: Any
        既定値

    Returns
    ----------
    plain: Any
        数, 文字, 真偽値, None, リストはそのまま (タプルはリスト),
        それ以外 (関数など) は表示用の文字
    """
    # 無限大と非数は JSON にできないので文字に変換
    if isinstance(value, float) and not math.isfinite(value):
        plain = str(value)
    elif value is None or isinstance(value, (bool, int, float, str)):
        plain = value
    elif isinstance(value, (list, tuple)):
        plain = [to_plain(item) for item in value]
    elif callable(value):
        plain = getattr(value, "__name__", repr(value))
    else:
        plain = repr(value)

    return plain


def kind_of(annotation: Any, default: Any) -> str:
    """
    型ヒントと既定値から, 引数の種類を決める関数

    Parameters
    ----------
    annotation: Any
        型ヒント (無い場合は inspect.Parameter.empty)
    default: Any
        既定値 (無い場合は inspect.Parameter.empty)

    Returns
    ----------
    kind: str
        int, float, bool, str, list, any のどれか
    """
    # 既定値の型を優先し, 無ければ型ヒントの文字から判定
    if isinstance(default, bool):
        kind = "bool"
    elif isinstance(default, int):
        kind = "int"
    elif isinstance(default, float):
        kind = "float"
    elif isinstance(default, str):
        kind = "str"
    else:
        text = "" if annotation is inspect.Parameter.empty else str(
            getattr(annotation, "__name__", annotation)
        )
        kind = "any"
        for name in ("bool", "int", "float", "str"):
            if text == name:
                kind = name
        if text.startswith(("tuple", "list", "Union[int, tuple")):
            kind = "list" if text.startswith(("tuple", "list")) else "any"

    return kind


def params_from_init(cls: type) -> tuple[list[ParamSpec], bool]:
    """
    クラスの __init__ のシグネチャから引数の一覧を取り出す関数

    Parameters
    ----------
    cls: type
        nn.Module のクラス

    Returns
    ----------
    params: list[ParamSpec]
        引数 (self, device, dtype と, _ で始まるものを除く)
    extra: bool
        **kwargs を持つかどうか
    """
    # シグネチャの引数を順に変換
    params = []
    extra = False
    signature = inspect.signature(cls.__init__)
    for name, parameter in signature.parameters.items():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            extra = True
            continue
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            continue
        if name in SKIPPED_PARAMS or name.startswith("_"):
            continue
        required = parameter.default is inspect.Parameter.empty
        default = None if required else parameter.default
        params.append(ParamSpec(
            name=name, kind=kind_of(parameter.annotation, parameter.default),
            default=to_plain(default), required=required
        ))

    return params, extra


def ports_from_forward(cls: type) -> tuple[list[PortSpec], list[ParamSpec]]:
    """
    クラスの forward のシグネチャから入力の口と設定の値を取り出す関数

    既定値の無い引数は必須の口, 既定値が None の引数は省略できる口,
    *args は可変の数の口 (xs) にする. 真偽値や数の既定値を持つ引数は,
    口ではなく呼び出すときの設定の値にする

    Parameters
    ----------
    cls: type
        nn.Module のクラス

    Returns
    ----------
    ports: list[PortSpec]
        入力の口
    forward_params: list[ParamSpec]
        呼び出すときの設定の値
    """
    # シグネチャの引数を順に変換
    ports = []
    forward_params = []
    signature = inspect.signature(cls.forward)
    for name, parameter in signature.parameters.items():
        if name == "self" or parameter.kind is inspect.Parameter.VAR_KEYWORD:
            continue
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            ports.append(PortSpec(name="xs", variadic=True))
            continue
        if parameter.default is inspect.Parameter.empty:
            ports.append(PortSpec(name=name))
        elif parameter.default in PORT_DEFAULTS:
            ports.append(PortSpec(name=name, optional=True))
        else:
            forward_params.append(ParamSpec(
                name=name,
                kind=kind_of(parameter.annotation, parameter.default),
                default=to_plain(parameter.default)
            ))

    return ports, forward_params


def apply_notes(spec: UnitSpec, notes: dict) -> None:
    """
    手で書いた説明と範囲を部品に重ねる関数

    Parameters
    ----------
    spec: UnitSpec
        部品, その場で書き換える
    notes: dict
        description (部品の説明) と, params (引数の名前ごとの
        description, minimum, maximum, choices, kind)

    Returns
    ----------
    None
    """
    # 部品の説明と, 引数ごとの説明と範囲の上書き
    spec.description = notes.get("description", spec.description)
    by_name = {param.name: param for param in spec.params}
    for name, values in notes.get("params", {}).items():
        if name not in by_name:
            raise ValueError(f"{spec.name} に引数 {name} はありません")
        for key, value in values.items():
            setattr(by_name[name], key, value)
