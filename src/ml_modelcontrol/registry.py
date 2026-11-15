import inspect
import math
import re
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

    # 引数の組み合わせの誤り (場所と説明の組) を返す関数 (部品ごとの確認.
    # 一覧の表示には含めない)
    checker: Optional[Callable[[dict], list[tuple[str, str]]]] = None

    # 書き出す Python のコードで, 作る式に使う名前 (nn.Linear など)
    code_name: str = ""
    code_import: str = ""
    code_args: list = field(default_factory=list)


# 名前から部品を引く登録簿
REGISTRY: dict[str, UnitSpec] = {}


# 手で説明を書いていない引数の, 名前ごとの説明 (多くの層とライブラリで
# 同じ意味の引数)
PARAM_DESCRIPTIONS = {
    "in_channels": "入力のチャンネル数",
    "out_channels": "出力のチャンネル数",
    "in_features": "入力の特徴量の数",
    "out_features": "出力の特徴量の数",
    "in1_features": "1 つ目の入力の特徴量の数",
    "in2_features": "2 つ目の入力の特徴量の数",
    "kernel_size": "カーネルの大きさ (整数か次元ごとのリスト)",
    "stride": "ずらす幅 (整数か次元ごとのリスト)",
    "padding": "端に足す幅 (整数か次元ごとのリスト, 畳み込みは same も可)",
    "output_padding": "出力の片側に足す幅 (転置畳み込みの大きさの調整)",
    "dilation": "カーネルの要素の間隔",
    "groups": "チャンネルを分けるグループの数 (入力と出力のチャンネル数を割り"
              "切る)",
    "bias": "バイアスを足すか",
    "padding_mode": "端を埋める値の作り方",
    "inplace": "入力を書き換えて計算するか (メモリを減らす)",
    "eps": "割り算を安定させる小さな数",
    "momentum": "平均と分散の移動平均の割合 (null なら累積平均)",
    "affine": "学習する倍率とずれを持つか",
    "track_running_stats": "推論に使う平均と分散を記録するか",
    "num_features": "特徴量 (チャンネル) の数",
    "ceil_mode": "出力の大きさの計算で切り上げるか",
    "return_indices": "最大値の位置も返すか",
    "count_include_pad": "平均に端に足した 0 も含めるか",
    "divisor_override": "平均の割る数 (null なら要素の数)",
    "output_size": "出力の大きさ (整数か次元ごとのリスト)",
    "output_ratio": "出力の大きさの入力に対する割合",
    "p": "確率 (ドロップアウトでは 0 にする確率)",
    "dim": "計算する次元",
    "size": "出力の大きさ",
    "scale_factor": "大きさの倍率",
    "mode": "計算の方法",
    "norm_type": "ノルムの p",
    "alpha": "係数 alpha",
    "beta": "係数 beta",
    "value": "埋める値か置き換える値",
    "dropout": "ドロップアウトの確率",
    "batch_first": "入力と出力を (バッチ, 系列, 特徴量) の順にするか",
    "input_size": "入力の特徴量の数",
    "hidden_size": "隠れ状態の大きさ",
    "dim_feedforward": "全結合の層の中間の大きさ",
    "activation": "活性化関数",
    "layer_norm_eps": "層正規化の eps",
    "norm_first": "注意と全結合の前に正規化するか (Pre-LN)",
    "d_model": "埋め込みの大きさ",
    "nhead": "注意のヘッドの数 (d_model を割り切る)",
    "num_layers": "重ねる層の数",
    "padding_idx": "埋め込みを 0 のままにする番号 (詰め物の単語など)",
    "max_norm": "埋め込みのノルムの上限",
    "scale_grad_by_freq": "勾配を単語の出現の回数で割るか",
    "sparse": "勾配を疎なテンソルにするか",
    "num_embeddings": "埋め込みの数 (語彙の大きさ)",
    "embedding_dim": "埋め込みの大きさ",
    "start_dim": "平らにする最初の次元",
    "end_dim": "平らにする最後の次元",
    "num_groups": "チャンネルを分けるグループの数",
    "num_channels": "チャンネル数",
    "elementwise_affine": "要素ごとの学習する倍率とずれを持つか",
    "negative_slope": "負の側の傾き",
    "min_val": "下限", "max_val": "上限",
    "min_value": "下限", "max_value": "上限",
    "threshold": "しきい値",
    "lambd": "しきい値 lambda",
    "k": "係数 k",
    "approximate": "近似の方法 (tanh で近似するか)",
    "add_bias_kv": "キーと値にバイアスの系列を足すか",
    "add_zero_attn": "キーと値に 0 の系列を足すか",
    "kdim": "キーの特徴量の数 (null なら embed_dim)",
    "vdim": "値の特徴量の数 (null なら embed_dim)",
    "num_heads": "注意のヘッドの数",
    "embed_dim": "埋め込みの大きさ",
    "num_parameters": "学習する傾きの数 (1 か チャンネル数)",
    "init": "最初の値",
    "upscale_factor": "解像度を上げる倍率",
    "downscale_factor": "解像度を下げる倍率",
    "unflattened_size": "分けた後の形",
    "normalized_shape": "正規化する最後の次元の形",
    "nonlinearity": "活性化関数",
    "proj_size": "出力の射影の大きさ (0 なら射影しない)",
    "bidirectional": "双方向にするか",
    "lower": "傾きの下限", "upper": "傾きの上限",
    "beta_softplus": "係数 beta",
    "align_corners": "角の画素をそろえて補間するか",
    "recompute_scale_factor": "倍率から大きさを計算し直すか",
    "act_layer": "活性化関数の層 (名前で選ぶ. ReLU, GELU など)",
    "act_kwargs": "活性化関数の層に渡す引数",
    "apply_act": "活性化関数を通すか",
    "norm_layer": "正規化の層 (名前で選ぶ. BatchNorm2d, LayerNorm など)",
    "gate_layer": "ゲートの活性化関数 (名前で選ぶ. sigmoid など)",
    "spatial_dims": "空間の次元の数 (1 は系列, 2 は画像, 3 は立体)",
    "pretrained": "学習済みの重みを読むか (形の推定では読まない)",
    "progress": "重みの取得の進み具合を表示するか",
    "hidden_features": "中間の特徴量の数",
    "drop": "ドロップアウトの割合",
    "attn_drop": "注意の重みのドロップアウトの割合",
    "proj_drop": "出力の射影のドロップアウトの割合",
    "rd_ratio": "チャンネルを減らす割合 (Squeeze-and-Excitation など)",
    "rd_channels": "減らした後のチャンネル数 (None なら rd_ratio から)",
    "rd_divisor": "減らした後のチャンネル数を割り切れる数にする約数",
    "dim_out": "出力の次元の大きさ",
    "growth_rate": "層ごとに増やすチャンネル数 (DenseNet)",
    "dropout_prob": "ドロップアウトの割合"
}

# 名前ごとの説明があっても, ライブラリによって意味が違うので docstring の
# 説明を優先する引数の名前 (MONAI の input_size は空間の大きさ, timm の
# InplaceAbn の act_layer は選べる種類が少ないなど)
AMBIGUOUS_NAMES = {"input_size", "act_layer"}


def describe_params(spec: UnitSpec, cls: Optional[type] = None) -> None:
    """
    説明の無い引数に名前ごとか docstring の説明を付ける関数

    手で書いた説明, 名前ごとの説明 (日本語), docstring の説明 (ライブラリの
    言葉のまま) の順に使う

    Parameters
    ----------
    spec: UnitSpec
        部品, その場で書き換える
    cls: Optional[type] = None
        部品のクラス (docstring を読む), None の場合は名前ごとの説明だけ

    Returns
    ----------
    None
    """
    # ライブラリごとに意味の違う名前は, docstring の説明を優先
    documented = docstring_params(cls) if cls is not None else {}
    for param in spec.params:
        if param.description:
            continue
        if param.name in AMBIGUOUS_NAMES and param.name in documented:
            param.description = documented[param.name]
            continue
        param.description = PARAM_DESCRIPTIONS.get(
            param.name, documented.get(param.name, "")
        )


# docstring の引数の節の見出し (Google 形式) と, NumPy 形式の見出しの下線
DOC_SECTION = re.compile(
    r"^(\s*)(Args|Arguments|Parameters|Params|Keyword Args)"
    r"\s*:\s*$"
)
DOC_UNDERLINE = re.compile(r"^\s*-{3,}\s*$")

# Google 形式の 1 つの引数の行 (name (型): 説明) と, NumPy 形式の行
# (name : 型), Sphinx 形式の行 (:param 型 name: 説明)
DOC_GOOGLE = re.compile(
    r"^(\s*)\**([A-Za-z_][A-Za-z0-9_]*)\s*"
    r"(\((?:[^()]|\([^()]*\))*\))?\s*:\s*(.*)$"
)
DOC_NUMPY = re.compile(
    r"^(\s*)\**([A-Za-z_][A-Za-z0-9_, *]*?)\s*:\s*(.*)$"
)
DOC_SPHINX = re.compile(
    r"^\s*:param\s+(?:[^:]*\s)?([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*)$"
)

# reST の役割の書き方 (:obj:`True` など. 中の文字だけを残す)
DOC_ROLE = re.compile(r":[A-Za-z]+:`~?([^`]*)`")

# docstring から取り出す説明の文字数の上限
DOC_MAX_LENGTH = 300


def docstring_params(cls: type) -> dict[str, str]:
    """
    クラスの docstring から引数ごとの説明を取り出す関数

    クラスと __init__ の docstring の, Google 形式 (Args:), NumPy 形式
    (Parameters と下線), Sphinx 形式 (:param name:) を読む. 読めない
    書き方の引数は含めない

    Parameters
    ----------
    cls: type
        nn.Module のクラス

    Returns
    ----------
    descriptions: dict[str, str]
        引数の名前ごとの説明 (空白をまとめ, 長いものは切り詰めたもの)
    """
    # クラスと __init__ の docstring を順に読み, 先に見つけた説明を使用
    descriptions: dict[str, str] = {}
    for doc in (cls.__doc__, getattr(cls.__init__, "__doc__", None)):
        if not isinstance(doc, str) or (
            cls.__init__ is nn.Module.__init__
            and doc is nn.Module.__init__.__doc__
        ):
            continue
        for name, text in parse_doc_params(inspect.cleandoc(doc)).items():
            descriptions.setdefault(name, text)

    return descriptions


def underlined(lines: list[str], index: int) -> bool:
    """
    行の次の行が NumPy 形式の見出しの下線かを返す関数

    Parameters
    ----------
    lines: list[str]
        docstring の行
    index: int
        行の番号

    Returns
    ----------
    found: bool
        次の行が下線 (--- の並び) なら True
    """
    found = index + 1 < len(lines) and bool(
        DOC_UNDERLINE.match(lines[index + 1])
    )

    return found


def parse_doc_params(doc: str) -> dict[str, str]:
    """
    1 つの docstring から, 引数ごとの説明を取り出す関数

    Parameters
    ----------
    doc: str
        字下げを整えた docstring

    Returns
    ----------
    descriptions: dict[str, str]
        引数の名前ごとの説明
    """
    lines = doc.splitlines()
    found: dict[str, list[str]] = {}

    # Sphinx 形式は行ごと (続く字下げした行は説明の続き)
    current: Optional[list[str]] = None
    for line in lines:
        match = DOC_SPHINX.match(line)
        if match:
            current = found.setdefault(match.group(1), [])
            current.append(match.group(2))
        elif current is not None and line.startswith((" ", "\t")) and (
            line.strip()
        ):
            current.append(line.strip())
        else:
            current = None

    # Google 形式と NumPy 形式は, 引数の節の中だけ
    index = 0
    while index < len(lines):
        google = DOC_SECTION.match(lines[index])
        numpy_style = lines[index].strip() in (
            "Parameters", "Args", "Arguments"
        ) and underlined(lines, index)
        if not google and not numpy_style:
            index += 1
            continue
        base = len(lines[index]) - len(lines[index].lstrip())
        index += 2 if numpy_style else 1
        entry_indent: Optional[int] = None
        current = None
        while index < len(lines):
            line = lines[index]
            if not line.strip():
                index += 1
                continue
            indent = len(line) - len(line.lstrip())
            # 節の終わり (見出しと同じ字下げの, 引数でない行や次の見出し)
            if indent < base or DOC_SECTION.match(line) or underlined(
                lines, index
            ):
                break
            if google and indent <= base:
                break
            if entry_indent is None or indent <= entry_indent:
                match = (
                    DOC_GOOGLE.match(line) if google else DOC_NUMPY.match(line)
                )
                # 引数の字下げで読めない行 (書き方の崩れた引数や地の文) は,
                # 前の引数の説明に混ぜずに読み飛ばし
                if match is None:
                    if indent <= base:
                        break
                    current = None
                    index += 1
                    continue
                entry_indent = indent
                names = [
                    name.strip().lstrip("*")
                    for name in match.group(2).split(",")
                ]
                current = []
                for name in names:
                    if name.isidentifier():
                        found.setdefault(name, current)
                # Google 形式は同じ行に説明, NumPy 形式の同じ行は型
                if google and match.group(4):
                    current.append(match.group(4).strip())
            elif current is not None:
                current.append(line.strip())
            index += 1

    # 空白をまとめ, 長い説明は切り詰め
    descriptions = {}
    for name, parts in found.items():
        text = " ".join(" ".join(parts).split())
        text = DOC_ROLE.sub(r"\1", text).replace("``", "")
        if not text:
            continue
        if len(text) > DOC_MAX_LENGTH:
            text = text[:DOC_MAX_LENGTH - 1].rstrip() + "…"
        descriptions[name] = text

    return descriptions


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
            "builder", "checker", "code_name", "code_import", "code_args",
            "offline"
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
