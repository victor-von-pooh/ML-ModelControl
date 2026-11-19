import importlib
import inspect
import re
from types import ModuleType
from typing import Any

from ml_modelcontrol.units import register_unit

# モジュールの名前 (識別子を . でつないだもの. ファイルのパスは不可)
MODULE_NAME = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*"
)

# 読み込んだモジュールの名前ごとの, 登録した部品の名前 (読み込んだ順)
LOADED_MODULES: dict[str, list[str]] = {}

# UNITS の辞書の要素に書ける register_unit の引数の名前
UNIT_KEYS = {
    "cls", "name", "description", "category", "outputs", "notes",
    "overwrite"
}


def register_units_from(module_names: list[str]) -> dict[str, list[str]]:
    """
    起動の設定に書いたモジュールを読み込み, 宣言した部品を登録する関数

    モジュールは UNITS (クラスか register_unit の引数の辞書の並び) か
    register_units(register) の関数で部品を宣言する. 読み込んだモジュール
    は任意のコードとして動くので, 信頼できるものだけを渡す

    Parameters
    ----------
    module_names: list[str]
        モジュールの名前の並び (my_lab.units など)

    Returns
    ----------
    loaded: dict[str, list[str]]
        モジュールの名前ごとの, 登録した部品の名前 (渡した順)

    Raises
    ----------
    ValueError
        名前の形が違う, 読み込めない, 部品を宣言していない, 登録できない
        モジュールがある場合 (そこで止める)
    """
    # 1 つの文字を渡した場合の取り違えの防止
    if isinstance(module_names, str):
        raise TypeError(
            "module_names はモジュールの名前のリストにしてください "
            f"(例: [{module_names!r}])"
        )

    # モジュールごとに読み込んで登録 (読み込み済みは前の結果)
    loaded = {}
    for module_name in module_names:
        loaded[module_name] = load_module_units(module_name)

    return loaded


def load_module_units(module_name: str) -> list[str]:
    """
    1 つのモジュールを読み込み, 宣言した部品を登録する関数

    Parameters
    ----------
    module_name: str
        モジュールの名前

    Returns
    ----------
    names: list[str]
        登録した部品の名前 (宣言した順)
    """
    # 読み込み済みのモジュールは, 読み込みも登録もし直さない
    if module_name in LOADED_MODULES:
        return list(LOADED_MODULES[module_name])
    module = import_listed_module(module_name)

    # UNITS, register_units(register) の順に登録
    units = getattr(module, "UNITS", None)
    hook = getattr(module, "register_units", None)
    if units is None and hook is None:
        raise ValueError(
            f"モジュール {module_name} は部品を宣言していません. UNITS "
            "(クラスか register_unit の引数の辞書の並び) か, "
            "register_units(register) の関数を書いてください"
        )
    names = []
    if units is not None:
        names += register_declared(module_name, units)
    if hook is not None:
        names += call_register_hook(module_name, hook)
    LOADED_MODULES[module_name] = list(names)

    return names


def import_listed_module(module_name: str) -> ModuleType:
    """
    起動の設定に書いたモジュールを名前で読み込む関数

    Parameters
    ----------
    module_name: str
        モジュールの名前

    Returns
    ----------
    module: ModuleType
        読み込んだモジュール

    Raises
    ----------
    ValueError
        名前の形が違う場合と, 読み込めない場合
    """
    # ファイルのパスや空の名前を受け付けない
    if not isinstance(module_name, str) or not MODULE_NAME.fullmatch(
        module_name
    ):
        raise ValueError(
            f"モジュールの名前 {module_name!r} は, 識別子を . でつないだ名前に"
            "してください (例: my_lab.units. ファイルのパスは書けません)"
        )

    # 読み込みの誤りは, モジュールの名前と元の誤りを添えて返却
    try:
        module = importlib.import_module(module_name)
    except Exception as error:
        raise ValueError(
            f"自作の部品のモジュール {module_name} を読み込めません: "
            f"{type(error).__name__}: {error}. Python の import の探す場所 "
            "(入れたパッケージか PYTHONPATH) にあるかを確かめてください"
        ) from error

    return module


def register_declared(module_name: str, units: Any) -> list[str]:
    """
    モジュールの UNITS の部品を登録する関数

    Parameters
    ----------
    module_name: str
        モジュールの名前
    units: Any
        UNITS の値 (クラスか register_unit の引数の辞書の並び)

    Returns
    ----------
    names: list[str]
        登録した部品の名前
    """
    # 並びでない値の拒否
    if not isinstance(units, (list, tuple)):
        raise ValueError(
            f"モジュール {module_name} の UNITS はリストにしてください "
            f"(今は {type(units).__name__})"
        )

    # 要素ごとに, クラスはそのまま, 辞書は引数として登録
    names = []
    for index, item in enumerate(units):
        if inspect.isclass(item):
            arguments: dict = {"cls": item}
        elif isinstance(item, dict):
            arguments = dict(item)
        else:
            raise ValueError(
                f"モジュール {module_name} の UNITS[{index}] は, nn.Module の"
                "クラスか register_unit の引数の辞書にしてください "
                f"(今は {type(item).__name__})"
            )
        unknown = sorted(set(arguments) - UNIT_KEYS)
        if "cls" not in arguments or unknown:
            raise ValueError(
                f"モジュール {module_name} の UNITS[{index}] の辞書は, cls と "
                f"register_unit の引数 ({', '.join(sorted(UNIT_KEYS))}) "
                f"だけにしてください (知らない項目: {unknown})"
            )
        cls = arguments.pop("cls")
        spec = register_listed(module_name, cls, **arguments)
        names.append(spec.name)

    return names


def call_register_hook(module_name: str, hook: Any) -> list[str]:
    """
    モジュールの register_units(register) を呼び, 登録した部品を集める関数

    Parameters
    ----------
    module_name: str
        モジュールの名前
    hook: Any
        register_units の値

    Returns
    ----------
    names: list[str]
        登録した部品の名前
    """
    # 呼べない値の拒否
    if not callable(hook):
        raise ValueError(
            f"モジュール {module_name} の register_units は関数にしてください"
        )

    # 渡す関数は, 登録した名前を記録しながら register_unit を呼ぶもの
    # (register_unit と同じ引数. モジュールの名前を付ける)
    recorder = Recorder(module_name)
    hook(recorder.record)
    names = list(recorder.names)

    return names


class Recorder:
    def __init__(self, module_name: str):
        # モジュールの名前と登録した部品の名前を保存
        self.module_name = module_name
        self.names: list[str] = []

    def record(self, cls: type, **arguments: Any) -> Any:
        """
        部品を登録し, 名前を記録する関数

        Parameters
        ----------
        cls: type
            nn.Module のクラス
        **arguments: Any
            register_unit の残りの引数 (module は書けない)

        Returns
        ----------
        spec: UnitSpec
            登録した部品
        """
        spec = register_listed(self.module_name, cls, **arguments)
        self.names.append(spec.name)

        return spec


def register_listed(module_name: str, cls: Any, **arguments: Any) -> Any:
    """
    モジュールの名前を付けて部品を登録する関数

    Parameters
    ----------
    module_name: str
        部品を宣言したモジュールの名前
    cls: Any
        nn.Module のクラス
    **arguments: Any
        register_unit の残りの引数

    Returns
    ----------
    spec: UnitSpec
        登録した部品

    Raises
    ----------
    ValueError
        登録できない場合 (モジュールの名前を添える)
    """
    # モジュールの名前は自分で付けるので, 書いた場合は拒否
    if "module" in arguments:
        raise ValueError(
            f"モジュール {module_name} の部品に module は書けません "
            "(宣言したモジュールの名前を付けます)"
        )

    # 登録の誤りはモジュールの名前を添えて返却
    try:
        spec = register_unit(cls, module=module_name, **arguments)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"モジュール {module_name} の部品を登録できません: {error}"
        ) from error

    return spec
