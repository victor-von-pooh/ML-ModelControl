import inspect
import math
from dataclasses import asdict, dataclass, field
from types import UnionType
from typing import (
    Any, Callable, Literal, Optional, Union, get_args, get_origin,
    get_type_hints
)

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

    # 値の候補 (選択肢と違い, 候補以外も書ける. モデルの名前など)
    suggestions: Optional[list] = None

    # 種類が任意 (any) の引数が受け取る値の種類 (int, float, bool, str,
    # list, null の並び. 型ヒントから分からない場合は None で確かめない)
    types: Optional[list] = None


@dataclass
class PortSpec:
    # 口の名前, 説明, 省略できるか, 可変の数のつなぎ先を受け取るか
    name: str
    description: str = ""
    optional: bool = False
    variadic: bool = False

    # 出力の口の取り出し方 (番号か属性の名前の並び, 空なら戻り値そのもの)
    path: list = field(default_factory=list)

    # 位置で渡す必要がある入力の口か (*args より前の口と, 位置だけの引数)
    positional: bool = False


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

    # 形の推定で置き換える引数の値 (学習済みの重みを読まない値)
    offline: dict = field(default_factory=dict)

    # 引数の値から nn.Module を作る関数 (一覧の表示には含めない)
    builder: Optional[Callable[..., nn.Module]] = None

    # 書き出す Python のコードで, 作る式に使う名前 (nn.Linear など)
    code_name: str = ""
    code_import: str = ""
    code_args: list = field(default_factory=list)


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
    # 組み込みの部品を読み込んでから検索
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
        for key in (
            "builder", "code_name", "code_import", "code_args", "offline"
        ):
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

    型ヒントがある場合はそれを優先し (int と組の両方を受け取る kernel_size
    などは any), 無い場合は既定値の型から決める

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
    # 型ヒントが単純な型ならその種類, 選択肢なら str, それ以外は any
    simple = {bool: "bool", int: "int", float: "float", str: "str"}
    names = {"bool": "bool", "int": "int", "float": "float", "str": "str"}
    if annotation is not inspect.Parameter.empty:
        if isinstance(annotation, str):
            kind = names.get(annotation.strip(), "any")
        elif annotation in simple:
            kind = simple[annotation]
        elif get_origin(annotation) is Literal and all(
            isinstance(value, str) for value in get_args(annotation)
        ):
            kind = "str"
        else:
            kind = "any"
        return kind

    # 型ヒントが無い場合は既定値の型
    if isinstance(default, bool):
        kind = "bool"
    elif isinstance(default, int):
        kind = "int"
    elif isinstance(default, float):
        kind = "float"
    elif isinstance(default, str):
        kind = "str"
    else:
        kind = "any"

    return kind


def types_of(annotation: Any) -> Optional[list]:
    """
    型ヒントから, 定義の値 (JSON) として受け取れる種類を取り出す関数

    int | tuple[int, int] は int と list, Optional[float] は float と null の
    ように, Union を分けて調べる. テンソルや関数など値で書けない型を含む
    場合と, 型ヒントが無いか読めない場合は確かめないので None

    Parameters
    ----------
    annotation: Any
        型ヒント

    Returns
    ----------
    types: Optional[list]
        int, float, bool, str, list, null の並び, 分からない場合は None
    """
    # 型ヒントが無い場合と, 文字のまま読めない場合は確かめない
    if annotation is inspect.Parameter.empty or isinstance(annotation, str):
        return None
    options = (
        list(get_args(annotation)) if is_union(annotation) else [annotation]
    )
    simple = {
        bool: "bool", int: "int", float: "float", str: "str",
        type(None): "null"
    }
    found: list[str] = []
    for option in options:
        origin = get_origin(option) or option
        # 選択肢 (Literal) は, 選択肢の値の種類
        if get_origin(option) is Literal:
            for value in get_args(option):
                kind = simple.get(type(value))
                if kind is None:
                    return None
                if kind not in found:
                    found.append(kind)
            continue
        if option in simple:
            name = simple[option]
        elif origin in (list, tuple) or (
            inspect.isclass(origin) and origin.__name__ == "Sequence"
        ):
            name = "list"
        else:
            return None
        if name not in found:
            found.append(name)

    return found


def choices_of(annotation: Any) -> Optional[list]:
    """
    型ヒントが選択肢 (Literal) の場合に, その値を返す関数

    Parameters
    ----------
    annotation: Any
        型ヒント

    Returns
    ----------
    choices: Optional[list]
        選択肢, 選択肢でない場合は None
    """
    choices = None
    if get_origin(annotation) is Literal:
        choices = list(get_args(annotation))

    return choices


def needs_module(cls: type) -> bool:
    """
    __init__ の引数に別の部品 (nn.Module や関数) が必要なクラスかを返す関数

    Parameters
    ----------
    cls: type
        nn.Module のクラス

    Returns
    ----------
    needed: bool
        必須の引数の型ヒントが nn.Module か Callable なら True
    """
    # 既定値の無い引数の型ヒントを確認
    # 文字の型ヒント (from __future__ import annotations) の型への変換
    needed = False
    try:
        hints = get_type_hints(cls.__init__)
    except Exception:
        hints = {}
    for name, parameter in inspect.signature(
        cls.__init__
    ).parameters.items():
        if parameter.default is not inspect.Parameter.empty:
            continue
        annotation = hints.get(name, parameter.annotation)
        if isinstance(annotation, str):
            annotation = getattr(nn, annotation, annotation)
        if needs_object(annotation):
            needed = True

    return needed


def needs_object(annotation: Any) -> bool:
    """
    型ヒントが, 定義の値 (JSON) では渡せない部品かテンソルだけかを返す関数

    Parameters
    ----------
    annotation: Any
        型ヒント

    Returns
    ----------
    needed: bool
        nn.Module, 関数, テンソルだけを受け取る型ヒントなら True.
        テンソルか数の組 (Tensor | tuple[float, float] など) のように,
        値でも渡せる型ヒントは False
    """
    # 値でも渡せる選択肢を含む Union は対象外
    words = ("Module", "Callable", "MessagePassing", "Tensor")
    options = [
        option for option in get_args(annotation)
        if option is not type(None)
    ] if is_union(annotation) else []
    if options:
        needed = all(needs_object(option) for option in options)
        return needed

    # 型に直せなかった文字の型ヒントは, 値の型を含めば渡せるものと判定
    text = str(annotation)
    values = ("int", "float", "tuple", "list", "str", "bool")
    if isinstance(annotation, str) and "Tensor" in text and not any(
        word in text for word in ("Module", "Callable")
    ):
        needed = not any(word in text for word in values)
        return needed
    needed = (
        inspect.isclass(annotation) and issubclass(annotation, nn.Module)
    ) or any(word in text for word in words)

    return needed


def is_union(annotation: Any) -> bool:
    """
    型ヒントが Union (Optional と X | Y を含む) かを返す関数

    Parameters
    ----------
    annotation: Any
        型ヒント

    Returns
    ----------
    union: bool
        Union なら True
    """
    origin = get_origin(annotation)
    union = origin is Union or origin is UnionType

    return union


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
    # nn.Module の __init__ をそのまま使うクラスは引数なし
    params: list[ParamSpec] = []
    extra = False
    if cls.__init__ is nn.Module.__init__:
        return params, extra

    # シグネチャの引数を順に変換 (文字の型ヒントは型に直して使用)
    signature = inspect.signature(cls.__init__)
    try:
        hints = get_type_hints(cls.__init__)
    except Exception:
        hints = {}
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
        annotation = parameter.annotation
        if isinstance(annotation, str):
            annotation = hints.get(name, annotation)
        kind = kind_of(annotation, parameter.default)
        params.append(ParamSpec(
            name=name, kind=kind, default=to_plain(default),
            required=required, choices=choices_of(annotation),
            types=types_of(annotation) if kind == "any" else None
        ))

    return params, extra


def ports_from_forward(cls: type) -> tuple[list[PortSpec], list[ParamSpec]]:
    """
    クラスの forward のシグネチャから入力の口と設定の値を取り出す関数

    既定値の無い引数は必須の口, 既定値が None の引数は省略できる口,
    *args は可変の数の口 (xs) にする. 真偽値や数の既定値を持つ引数と,
    既定値が None でもテンソル以外の型ヒントを持つ引数は, 口ではなく
    呼び出すときの設定の値にする

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
    # *args より前の口と位置だけの引数は位置で, それ以外はキーワードで受け渡し
    ports = []
    forward_params = []
    parameters = inspect.signature(cls.forward).parameters
    has_varargs = any(
        parameter.kind is inspect.Parameter.VAR_POSITIONAL
        for parameter in parameters.values()
    )
    before_varargs = has_varargs
    for name, parameter in parameters.items():
        if name == "self" or parameter.kind is inspect.Parameter.VAR_KEYWORD:
            continue
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            ports.append(PortSpec(name="xs", variadic=True))
            before_varargs = False
            continue
        positional = before_varargs or (
            parameter.kind is inspect.Parameter.POSITIONAL_ONLY
        )
        # 既定値が None でも, 値の型ヒント (bool, int, list[int] など) を
        # 持ちテンソルを含まない引数は設定の値
        text = str(parameter.annotation)
        flag = (
            parameter.annotation is not inspect.Parameter.empty
            and "Tensor" not in text
            and any(
                word in text
                for word in ("bool", "int", "float", "str", "list", "tuple",
                             "Tuple", "List")
            )
        )
        if parameter.default is inspect.Parameter.empty:
            ports.append(PortSpec(name=name, positional=positional))
        elif parameter.default in PORT_DEFAULTS and not flag:
            ports.append(PortSpec(
                name=name, optional=True, positional=positional
            ))
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
