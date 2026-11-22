import importlib
import inspect
import keyword
import re
import warnings
from pathlib import Path
from typing import Any, Optional, Union

from ml_modelcontrol.model import SpecError, to_arg
from ml_modelcontrol.spec import SUBGRAPH_PREFIX, load_spec, parse_ref
from ml_modelcontrol.registry import UnitSpec, get_unit
from ml_modelcontrol.validate import (
    MAX_UNIT_DEPTH, analyze, class_name_of, inner_units
)

# 書き出すコードで変数の名前に使わない名前 (読み込むモジュールなど)
RESERVED_VARS = {"self", "torch", "nn", "ops", "result", "_pick"}


def var_name(name: str) -> str:
    """
    入力かノードの名前から, 書き出すコードの変数の名前を作る関数

    Parameters
    ----------
    name: str
        入力かノードの名前 (識別子の形は確認済み)

    Returns
    ----------
    var: str
        変数の名前 (重なる名前は check_names でエラーにするので同じ名前)
    """
    var = name

    return var


def path_code(base: str, path: list, keyed: bool) -> str:
    """
    出力の口の取り出し方を, 値を取り出す式にする関数

    Parameters
    ----------
    base: str
        部品の戻り値の変数
    path: list
        番号か名前の並び
    keyed: bool
        名前を常に辞書のキーとして取り出せる値か (Hugging Face の出力.
        無いキーは None)

    Returns
    ----------
    code: str
        lstm[1][0], bert.get('pooler_output'),
        _pick(_pick(stats, 'max'), 'indices') のような式. 型の分からない値の
        名前は, 組み立てたモデルと同じく辞書ならキー, それ以外は属性で
        取り出す _pick を使う
    """
    code = base
    for step in path:
        if isinstance(step, int):
            code += f"[{step!r}]"
        elif keyed:
            code += f".get({step!r})"
        else:
            code = f"_pick({code}, {step!r})"

    return code


# 型の分からない値から名前で取り出す関数 (書き出すコードに必要な場合だけ置く)
PICK_CODE = [
    "def _pick(value, name):",
    "    # 辞書はキー (無いキーで同じ名前の属性があれば属性), それ以外は属性",
    "    if isinstance(value, dict) and name in value:",
    "        return value[name]",
    "    if isinstance(value, dict) and not hasattr(value, name):",
    "        return value[name]",
    "    return getattr(value, name)",
    "", ""
]


# このライブラリの部品のクラスを読み込む行と, そのクラスのモジュール
# (単独で動くコードではクラスの定義を書き写して置き換える)
OWN_IMPORTS = {
    "from ml_modelcontrol import ops": "ml_modelcontrol.ops",
    "from ml_modelcontrol.external import HuggingFaceModel": (
        "ml_modelcontrol.external"
    )
}


def export_python(
    spec: Union[str, Path, dict], class_name: str = "Model",
    path: Optional[Union[str, Path]] = None, standalone: bool = True
) -> str:
    """
    モデルの定義から, 同じ計算をする nn.Module のクラスの Python のコードを
    書き出す関数

    部品は self.nodes (nn.ModuleDict) に持つので, state_dict のキーは
    build_model で組み立てたモデルと同じになる

    Parameters
    ----------
    spec: Union[str, Path, dict]
        モデルの定義, またはその YAML か JSON のファイルのパス
    class_name: str = "Model"
        クラスの名前
    path: Optional[Union[str, Path]] = None
        保存先, None の場合は保存しない
    standalone: bool = True
        このライブラリを入れなくても動くコードにするか. 演算 (op.Add など)
        と Hugging Face の部品のクラスの定義をコードに書き写す. False の
        場合は ml_modelcontrol から読み込む

    Returns
    ----------
    code: str
        Python のコード

    Raises
    ----------
    SpecError
        定義に誤りがある場合
    """
    # 定義を確かめ, 誤りがあれば書き出さない
    spec = load_spec(spec)
    analysis = analyze(spec)
    if not analysis.ok:
        raise SpecError([vars(issue) for issue in analysis.issues])
    if not class_name.isidentifier() or keyword.iskeyword(class_name):
        raise ValueError(f"クラスの名前 {class_name!r} は識別子にしてください")

    # 書き出すクラス (部分モデルは中のものから先に, 本体は最後)
    used = used_units(spec, analysis)
    graphs = [
        (class_name_of(name), sub, sub_analysis)
        for name, (sub, sub_analysis) in analysis.subgraphs.items()
        if f"{SUBGRAPH_PREFIX}{name}" in used
    ]
    graphs.append((class_name, spec, analysis))
    names = [name for name, _, _ in graphs]
    for name in names:
        if names.count(name) > 1:
            raise ValueError(
                f"クラスの名前 {name} が部分モデルのクラスの名前と重なります. "
                "クラスか部分モデルの名前を変えてください"
            )

    # 読み込むモジュールと, 名前の重なりの確認
    # (同じ名前の別のクラスは, クラスをまたいでも確認)
    imports = ["from torch import nn"]
    imported: dict[str, str] = {"nn": "torch.nn"}
    for _, graph, graph_analysis in graphs:
        for name in graph_analysis.order:
            for unit in node_units(graph, graph_analysis, name):
                if unit.code_import and unit.code_import not in imports:
                    imports.append(unit.code_import)
        check_names(graph, graph_analysis, names, imported)

    # モデルの名前は文字の値として書き, コードとして解釈されないようにする
    title = (
        f"{spec.get('name', class_name)} のモデル (ML-ModelControl の定義から"
        "書き出したコード)"
    )
    # 単独で動くコードでは, このライブラリの部品のクラスを書き写し
    copied: list[str] = []
    renames: dict[str, str] = {}
    if standalone:
        imports, copied, renames = copy_own_classes(graphs, imports, names)
    lines = [repr(title), *merge_imports(imports)]
    for block in copied:
        lines += ["", "", *block.splitlines()]
    for name, graph, graph_analysis in graphs:
        lines += [
            "", "", *class_code(graph, graph_analysis, name, renames)
        ]
    if any("_pick(" in line for line in lines):
        start = lines.index(f"class {graphs[0][0]}(nn.Module):")
        lines[start:start] = PICK_CODE
    code = "\n".join(lines) + "\n"
    if path is not None:
        Path(path).write_text(code, encoding="utf-8")

    return code


def node_units(spec: dict, analysis: Any, name: str) -> list[UnitSpec]:
    """
    ノードの部品と, 引数に書いた部品を並べる関数

    Parameters
    ----------
    spec: dict
        モデルの定義
    analysis: Any
        定義の解析の結果
    name: str
        ノードの名前

    Returns
    ----------
    units: list[UnitSpec]
        ノードの部品と, 引数に書いた部品 (中の部品も)
    """
    params = spec["nodes"][name].get("params")
    units = [
        analysis.units[name], *[unit for _, unit in inner_units(params)]
    ]

    return units


def used_units(spec: dict, analysis: Any) -> set[str]:
    """
    本体と, 使われる部分モデルが使う部品の名前を集める関数

    Parameters
    ----------
    spec: dict
        モデルの定義
    analysis: Any
        定義の解析の結果

    Returns
    ----------
    names: set[str]
        部品の名前 (sub.block など)
    """
    # 本体の部品から, 部分モデルの中の部品へたどる
    stack = [analysis.units[name].name for name in analysis.order]
    names: set[str] = set()
    while stack:
        unit_name = stack.pop()
        if unit_name in names:
            continue
        names.add(unit_name)
        sub = analysis.subgraphs.get(unit_name[len(SUBGRAPH_PREFIX):])
        if unit_name.startswith(SUBGRAPH_PREFIX) and sub is not None:
            stack += [sub[1].units[name].name for name in sub[1].order]

    return names


def merge_imports(imports: list[str]) -> list[str]:
    """
    同じモジュールから読み込む from の行を 1 行にまとめる関数

    Parameters
    ----------
    imports: list[str]
        読み込む行 (空の行は区切り)

    Returns
    ----------
    merged: list[str]
        まとめた行 (最初に現れた位置に, 名前を現れた順に並べたもの)
    """
    # モジュールごとの名前を集め, 最初の行の位置にまとめて出力
    names: dict[str, list[str]] = {}
    merged: list[str] = []
    for line in imports:
        if not line.startswith("from ") or " import " not in line:
            merged.append(line)
            continue
        module, _, items = line[len("from "):].partition(" import ")
        if module not in names:
            names[module] = []
            merged.append(f"from {module} import ")
        for item in items.split(","):
            if item.strip() and item.strip() not in names[module]:
                names[module].append(item.strip())
    merged = [
        line + ", ".join(names[line[len("from "):-len(" import ")]])
        if line.startswith("from ") and line.endswith(" import ") else line
        for line in merged
    ]

    return merged


def copy_own_classes(
    graphs: list, imports: list[str], class_names: list[str]
) -> tuple[list[str], list[str], dict[str, str]]:
    """
    このライブラリの部品の読み込みをクラスの定義の書き写しに置き換える関数

    書き写すクラスが使う名前 (torch, Any, Optional, importlib) の import を
    クラスの中身から決めて加える. クラスの中身を読めない環境 (ソースの無い
    インストール) では, 書き写さずに読み込む行を残して注意を出す

    Parameters
    ----------
    graphs: list
        書き出すクラスの名前, 定義, 解析の組
    imports: list[str]
        読み込む行
    class_names: list[str]
        書き出すクラスの名前 (書き写すクラスとの重なりの確認)

    Returns
    ----------
    imports: list[str]
        このライブラリの読み込みを除き, 書き写したクラスが使う行を加えた
        読み込む行 (標準ライブラリ, torch, その他の順)
    copied: list[str]
        書き写したクラスの定義 (使う順)
    renames: dict[str, str]
        部品を作る式の名前の置き換え (ops.Add を Add に)

    Raises
    ----------
    ValueError
        書き写すクラスの名前が, 書き出すクラスや読み込む名前と重なる場合
    """
    # 他の読み込む行が持ち込む名前 (from x import A の A, import x の x)
    kept = [line for line in imports if line not in OWN_IMPORTS]
    bound = {
        name.strip().split(" as ")[-1]
        for line in kept
        for name in line.split(" import ", 1)[-1].split(",")
    }

    # 使う部品のクラスの書き写しと, 書き写したクラスが使う名前の収集
    copied: list[str] = []
    renames: dict[str, str] = {}
    used: set[str] = set()
    fallback: list[str] = []
    units = [
        unit
        for _, graph, graph_analysis in graphs
        for name in graph_analysis.order
        for unit in node_units(graph, graph_analysis, name)
    ]
    for unit in units:
        if unit.code_import not in OWN_IMPORTS:
            continue
        if unit.code_name in renames or unit.code_import in fallback:
            continue
        short = unit.code_name.split(".")[-1]
        module = importlib.import_module(OWN_IMPORTS[unit.code_import])
        try:
            source = inspect.getsource(getattr(module, short))
        except (OSError, TypeError):
            warnings.warn(
                f"{unit.code_name} のクラスの中身を読めないので, 書き出す"
                "コードは ml_modelcontrol を読み込みます", stacklevel=3
            )
            fallback.append(unit.code_import)
            continue
        if short in class_names or short in bound:
            raise ValueError(
                f"部品のクラス {short} の名前が, 書き出すクラスか読み込む"
                "クラスの名前と重なります. 名前を変えるか, "
                "standalone=False で書き出してください"
            )
        copied.append(source)
        renames[unit.code_name] = short
        used |= {
            word for word in ("importlib", "torch", "Any", "Optional")
            if re.search(rf"\b{word}\b", source)
        }

    # 書き写したクラスが使う名前と, 書き出すクラスの名前の重なりの確認
    overlap = used & set(class_names)
    if overlap:
        raise ValueError(
            f"クラスの名前 {sorted(overlap)} は, 書き写した部品のクラスが使う"
            "名前と重なるので使えません"
        )

    # 標準ライブラリ, torch, その他の順の読み込む行
    standard = ["import importlib"] if "importlib" in used else []
    typing_names = [name for name in ("Any", "Optional") if name in used]
    if typing_names:
        standard.append(f"from typing import {', '.join(typing_names)}")
    first = ["import torch"] if "torch" in used else []
    # (標準ライブラリの後は 1 行空ける)
    imports = [
        *standard, *([""] if standard else []), *first,
        *[line for line in imports if line in fallback], *kept
    ]

    return imports, copied, renames


def class_code(
    spec: dict, analysis: Any, class_name: str,
    renames: Optional[dict[str, str]] = None
) -> list[str]:
    """
    1 つの定義 (本体か部分モデル) の nn.Module のクラスの行を作る関数

    Parameters
    ----------
    spec: dict
        モデルの定義
    analysis: Any
        定義の解析の結果
    class_name: str
        クラスの名前
    renames: Optional[dict[str, str]] = None
        部品を作る式の名前の置き換え (書き写したクラス)

    Returns
    ----------
    lines: list[str]
        クラスの定義の行
    """
    # 引数に渡す部品を作る行 (中の部品から), ノードの部品を作る行と,
    # forward の行
    lines = [
        f"class {class_name}(nn.Module):",
        "    def __init__(self):",
        "        # 親クラスのコンストラクタを呼び出す",
        f"        super({class_name}, self).__init__()", ""
    ]
    parts: list[str] = []
    args = {
        name: build_args(spec, analysis, name, parts, renames or {})
        for name in analysis.order
    }
    if parts:
        lines += ["        # 引数に渡す部品の作成", *parts, ""]
    lines += [
        "        # ノードの部品の作成",
        "        self.nodes = nn.ModuleDict({"
    ]
    for name in analysis.order:
        code = analysis.units[name].code_name
        lines += wrap_call(
            f"            {name!r}: {(renames or {}).get(code, code)}(",
            args[name], "),", "            "
        )
    lines.append("        })")
    lines += state_code(spec, analysis)
    lines += forward_code(spec, analysis)
    lines += train_code(spec, analysis, class_name)

    return lines


def state_code(spec: dict, analysis: Any) -> list[str]:
    """
    パラメータの状態 (版 2 の項目) を設定する __init__ の行を作る関数

    外部の重みはコードに書かず, コメントと注意にする

    Parameters
    ----------
    spec: dict
        モデルの定義
    analysis: Any
        定義の解析の結果

    Returns
    ----------
    lines: list[str]
        零初期化, 凍結, 共有, 評価の形の行 (無ければ空)
    """
    # 外部の重み (コメントと注意), 零初期化, 凍結の行 (組み立てと同じ順)
    nodes = spec["nodes"]
    body: list[str] = []
    for name in analysis.order:
        node = nodes[name]
        module = f"self.nodes[{name!r}]"
        if "weights" in node:
            body.append(
                f"        # ノード {name} の外部の重みは読み込みの対象外"
            )
            body.append(
                "        # (build_model で組み立てたモデルの state_dict を "
                "load_state_dict で読み込み)"
            )
            warnings.warn(
                f"ノード {name} の外部の重み (weights) は書き出すコードでは"
                "読みません. ML-ModelControl の build_model で組み立てた"
                "モデルの state_dict を保存し, 書き出したクラスの "
                "load_state_dict で読み込んでください",
                stacklevel=4
            )
        zero = node.get("zero_init")
        if zero is True:
            body += [
                f"        for parameter in {module}.parameters():",
                "            nn.init.zeros_(parameter)"
            ]
        elif isinstance(zero, list):
            body += [
                f"        nn.init.zeros_({module}.get_parameter({item!r}))"
                for item in zero
            ]
        if node.get("train") is False:
            body += [
                f"        for parameter in {module}.parameters():",
                "            parameter.requires_grad_(False)"
            ]

    # 共有の行 (ノード全体は部品, パラメータごとは属性の置き換え)
    for name in analysis.order:
        share = nodes[name].get("share")
        if isinstance(share, str):
            body.append(
                f"        self.nodes[{name!r}] = self.nodes[{share!r}]"
            )
        elif isinstance(share, dict):
            for own, ref in share.items():
                source, _, source_name = str(ref).partition(".")
                owner, _, leaf = str(own).rpartition(".")
                body += [
                    "        setattr(",
                    f"            self.nodes[{name!r}].get_submodule("
                    f"{owner!r}), {leaf!r},",
                    f"            self.nodes[{source!r}].get_parameter("
                    f"{source_name!r})",
                    "        )"
                ]

    # 評価の形に保つノード
    eval_nodes = [
        name for name in analysis.order
        if nodes[name].get("eval_mode") is True
    ]
    if eval_nodes:
        body += [
            f"        self.eval_nodes = {eval_nodes!r}",
            "        self.train(self.training)"
        ]
    lines = ["", "        # パラメータの状態の設定", *body] if body else []

    return lines


def train_code(spec: dict, analysis: Any, class_name: str) -> list[str]:
    """
    評価の形に保つノードがある場合の train の行を作る関数

    Parameters
    ----------
    spec: dict
        モデルの定義
    analysis: Any
        定義の解析の結果
    class_name: str
        クラスの名前

    Returns
    ----------
    lines: list[str]
        train の定義の行 (評価の形に保つノードが無ければ空)
    """
    # eval_mode のノードがある場合だけ, 全体の切り替えの後に戻す
    if not any(
        spec["nodes"][name].get("eval_mode") is True
        for name in analysis.order
    ):
        lines: list[str] = []
        return lines
    lines = [
        "", "    def train(self, mode=True):",
        "        # 全体の切り替えと, 評価の形に保つノードの戻し",
        f"        super({class_name}, self).train(mode)",
        "        for name in self.eval_nodes:",
        "            self.nodes[name].train(False)",
        "        model = self", "",
        "        return model"
    ]

    return lines


def check_names(
    spec: dict, analysis: Any, class_names: list[str],
    imported: Optional[dict[str, str]] = None
) -> None:
    """
    書き出すコードで名前が重ならないことを確かめる関数

    入力とノードの名前, クラスの名前が, 読み込むモジュールやクラスの名前と
    重なる場合と, 同じ名前の別のクラスを読み込む場合はエラーにする

    Parameters
    ----------
    spec: dict
        モデルの定義
    analysis: Any
        定義の解析の結果
    class_names: list[str]
        書き出す全てのクラスの名前 (部分モデルと本体)
    imported: Optional[dict[str, str]] = None
        読み込む名前ごとの import の行 (書き出す全てのクラスで共有し,
        その場で書き加える). None の場合はこの定義だけで確認

    Returns
    ----------
    None
    """
    # 読み込む名前 (nn, ops, 部品のクラス) と, 予約した変数の名前
    if imported is None:
        imported = {"nn": "torch.nn"}
    units = [
        unit for name in analysis.order
        for unit in node_units(spec, analysis, name)
    ]
    for unit in units:
        code_name = unit.code_name.split(".")[0]
        if unit.source == "subgraph":
            continue
        if "<locals>" in unit.code_import or "__main__" in unit.code_import:
            raise ValueError(
                f"部品 {unit.name} は関数の中か実行したスクリプトで定義した"
                "クラスなので, 読み込むコードを書き出せません"
            )
        if imported.get(code_name, unit.code_import) != unit.code_import and (
            code_name not in ("nn", "ops")
        ):
            raise ValueError(
                f"同じ名前 {code_name} の別のクラスを読み込むので書き出せません"
            )
        imported.setdefault(code_name, unit.code_import)
        if code_name in {"self", "result", "_pick"}:
            raise ValueError(
                f"部品 {unit.name} のクラスの名前 {code_name} は書き出すコード"
                "で使う名前と重なるので書き出せません"
            )
    reserved = RESERVED_VARS | set(imported)
    for name in [*spec["inputs"], *analysis.order]:
        if name in reserved:
            raise ValueError(
                f"名前 {name} は書き出すコードで使う名前と重なるので, 変えて"
                "ください"
            )
    for class_name in class_names:
        if class_name in reserved or class_name in spec["inputs"] or (
            class_name in analysis.order
        ):
            raise ValueError(
                f"クラスの名前 {class_name} は他の名前と重なるので使えません"
            )


def build_args(
    spec: dict, analysis: Any, name: str, parts: list[str],
    renames: dict[str, str]
) -> list[str]:
    """
    1 つのノードの部品を作る式の引数を作る関数

    Parameters
    ----------
    spec: dict
        モデルの定義
    analysis: Any
        定義の解析の結果
    name: str
        ノードの名前
    parts: list[str]
        引数に渡す部品を作る行 (その場で書き加える)
    renames: dict[str, str]
        部品を作る式の名前の置き換え (書き写したクラス)

    Returns
    ----------
    args: list[str]
        in_features=16 のような引数の式の並び
    """
    # 位置の引数 (Hugging Face のモデルの種類など) と, 名前を付けた引数
    unit = analysis.units[name]
    params = spec["nodes"][name].get("params") or {}
    args = [repr(value) for value in unit.code_args]
    args += [
        f"{key}={part_code(value, parts, renames)}"
        if isinstance(value, dict) else f"{key}={to_arg(value)!r}"
        for key, value in params.items()
    ]

    return args


def part_code(
    value: dict, parts: list[str], renames: dict[str, str], depth: int = 1
) -> str:
    """
    引数に書いた部品を作る行を書き加え, その変数の名前を返す関数

    Parameters
    ----------
    value: dict
        部品の引数 (unit と params)
    parts: list[str]
        部品を作る行 (その場で書き加える. 中の部品を先に)
    renames: dict[str, str]
        部品を作る式の名前の置き換え (書き写したクラス)
    depth: int = 1
        入れ子の深さ (ノードの引数に書いた部品は 1)

    Returns
    ----------
    var: str
        作った部品の変数の名前 (_part0 のような, 他の名前と重ならない名前)
    """
    # 中の部品を先に作り, この部品を作る行を足す (深さは確かめで上限の内)
    if depth > MAX_UNIT_DEPTH:
        raise ValueError(
            f"引数の部品の入れ子が深すぎます ({MAX_UNIT_DEPTH} 段まで)"
        )
    unit = get_unit(str(value["unit"]))
    args = [repr(item) for item in unit.code_args]
    args += [
        f"{key}={part_code(item, parts, renames, depth + 1)}"
        if isinstance(item, dict) else f"{key}={to_arg(item)!r}"
        for key, item in (value.get("params") or {}).items()
    ]
    var = f"_part{sum(line.startswith('        _part') for line in parts)}"
    code = renames.get(unit.code_name, unit.code_name)
    parts += wrap_call(f"        {var} = {code}(", args, ")", "        ")

    return var


def wrap_call(
    head: str, args: list[str], tail: str, indent: str, width: int = 79
) -> list[str]:
    """
    関数の呼び出しの行を, 幅に収まるように折り返す関数

    Parameters
    ----------
    head: str
        引数の前までの文字 (開きかっこまで)
    args: list[str]
        引数の式の並び
    tail: str
        引数の後の文字 (閉じかっこから)
    indent: str
        head の行の字下げ
    width: int = 79
        1 行の幅の上限

    Returns
    ----------
    lines: list[str]
        1 行に収まればその 1 行, 収まらなければ引数を次の行から詰めて並べ,
        最後に閉じかっこの行を置いた行の並び. head と 1 つの引数が幅より
        長い場合 (とても長いノードの名前など) は, その行だけ幅を超える
    """
    # 1 行に収まる場合と, 引数が無い場合はそのまま
    line = head + ", ".join(args) + tail
    if len(line) <= width or not args:
        return [line]

    # 引数を 1 段深い字下げで, 幅に収まるだけ詰めて並べる
    inner = indent + "    "
    lines = [head]
    current = ""
    for arg in args:
        piece = f"{arg},"
        candidate = f"{current} {piece}" if current else inner + piece
        if current and len(candidate) > width:
            lines.append(current)
            candidate = inner + piece
        current = candidate
    lines += [current, indent + tail]

    return lines


def forward_code(spec: dict, analysis: Any) -> list[str]:
    """
    forward の行を作る関数

    Parameters
    ----------
    spec: dict
        モデルの定義
    analysis: Any
        定義の解析の結果

    Returns
    ----------
    lines: list[str]
        forward の定義の行
    """
    # 入力を引数に, ノードを計算の順に呼び出す行
    inputs = [var_name(str(name)) for name in spec["inputs"]]
    lines = [
        "", f"    def forward(self, {', '.join(inputs)}):",
        "        # ノードを計算の順に計算"
    ]
    for name in analysis.order:
        lines += wrap_call(
            f"        {var_name(name)} = self.nodes[{name!r}](",
            call_args(spec, analysis, name), ")", "        "
        )

    # 出力 (1 つならその値, 2 つ以上なら辞書)
    values = {
        str(output): ref_code(analysis, ref)
        for output, ref in spec["outputs"].items()
    }
    lines.append("")
    if len(values) == 1:
        lines.append(f"        result = {next(iter(values.values()))}")
    else:
        lines.append("        result = {")
        for output, value in values.items():
            lines.append(f"            {output!r}: {value},")
        lines.append("        }")
    lines += ["", "        return result"]

    return lines


def ref_code(analysis: Any, ref: str) -> str:
    """
    つなぎ先を, 変数と添字の式にする関数

    Parameters
    ----------
    analysis: Any
        定義の解析の結果
    ref: str
        「名前」か「名前:口の名前」

    Returns
    ----------
    code: str
        lstm[1][0] のような式
    """
    # 入力はそのまま, ノードは出力の口の取り出し方の添字を付加
    name, port = parse_ref(ref)
    code = var_name(name)
    if name in analysis.units:
        outputs = analysis.units[name].outputs
        output = next(
            (item for item in outputs if item.name == port), outputs[0]
        )
        if analysis.units[name].source == "subgraph":
            code += "".join(f"[{step!r}]" for step in output.path)
        else:
            code = path_code(
                code, output.path,
                analysis.units[name].source == "transformers"
            )

    return code


def call_args(spec: dict, analysis: Any, name: str) -> list[str]:
    """
    ノードを呼び出す引数の式を作る関数 (組み立てたモデルと同じ渡し方)

    Parameters
    ----------
    spec: dict
        モデルの定義
    analysis: Any
        定義の解析の結果
    name: str
        ノードの名前

    Returns
    ----------
    args: list[str]
        x, edge_index=edge_index のような引数の式の並び
    """
    # 位置で渡す口, 可変の数の口, キーワードの口と設定の値の順
    connections = analysis.connections[name]
    args = []
    kwargs = []
    for port in analysis.units[name].inputs:
        refs = [
            ref_code(analysis, target if port_name is None else (
                f"{target}:{port_name}"
            ))
            for target, port_name in connections.get(port.name, [])
        ]
        if port.variadic:
            args += refs
        elif port.positional:
            args.append(refs[0] if refs else "None")
        elif refs:
            kwargs.append(f"{port.name}={refs[0]}")
    forward_params = spec["nodes"][name].get("forward_params") or {}
    kwargs += [
        f"{key}={to_arg(value)!r}" for key, value in forward_params.items()
    ]
    args += kwargs

    return args
