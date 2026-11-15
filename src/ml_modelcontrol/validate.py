import contextvars
import difflib
import keyword
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Union

from torch import nn

from ml_modelcontrol.registry import (
    REGISTRY, ParamSpec, PortSpec, UnitSpec, get_unit
)
from ml_modelcontrol.spec import (
    GRAPH_KEYS, INPUT_KEYS, NODE_KEYS, SUBGRAPH_PREFIX, load_spec,
    node_inputs, parse_ref
)

# 入力, ノード, 出力の名前の形
NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# ノードに使えない名前 (組み立てたモデルの部品の入れ物の属性と重なるもの)
RESERVED_NAMES = set(dir(nn.ModuleDict()))

# 書き出す Python のコードで使う名前 (ノードと入力に付けると書き出せない)
EXPORT_NAMES = {"self", "torch", "nn", "ops", "result", "_pick"}

# 入力の例に使える型と, 値の作り方
DTYPES = {
    "float32", "float64", "float16", "bfloat16", "int64", "int32", "bool"
}
FILLS = {"zeros", "ones", "arange"}


@dataclass
class Issue:
    # 誤りか注意か, 決まった英字の識別子, 説明, ノードの名前 (定義全体と
    # 部分モデルの中なら None), 場所 (params.out_features など)
    severity: str
    code: str
    message: str
    node: Optional[str] = None
    location: Optional[str] = None

    # 部分モデルの中の誤りの場合の部分モデルの名前 (node は None で,
    # location は subgraphs.<名前>.nodes.<ノード>.<場所> のような全体の場所)
    subgraph: Optional[str] = None


@dataclass
class Analysis:
    # 誤りと注意, 計算の順, ノードごとの口のつなぎ先 (名前と口の名前の組)
    issues: list[Issue] = field(default_factory=list)
    order: list[str] = field(default_factory=list)
    connections: dict[str, dict[str, list[tuple[str, Optional[str]]]]] = (
        field(default_factory=dict)
    )
    units: dict[str, UnitSpec] = field(default_factory=dict)

    # 誤りの無い部分モデルの, 名前ごとの定義と解析 (入れ子のものも含む)
    subgraphs: dict[str, tuple[dict, "Analysis"]] = field(
        default_factory=dict
    )

    @property
    def ok(self) -> bool:
        """
        誤りが無いかを返す関数

        Parameters
        ----------
        None

        Returns
        ----------
        ok: bool
            誤り (severity が error のもの) が無ければ True
        """
        ok = not any(issue.severity == "error" for issue in self.issues)

        return ok


def validate(spec: Union[str, Path, dict]) -> list[dict]:
    """
    モデルの定義を確かめ, 全ての誤りと注意を返す関数

    ネットワークは使わない. 最初の誤りで止めず, ノードごとの結果を返す

    Parameters
    ----------
    spec: Union[str, Path, dict]
        モデルの定義, またはその YAML か JSON のファイルのパス

    Returns
    ----------
    issues: list[dict]
        severity, code, message, node, location を持つ辞書のリスト
    """
    # 読み込めない定義 (形の誤り, 新しい版) は 1 件の誤り
    try:
        loaded = load_spec(spec)
    except (ValueError, OSError) as error:
        issues = [asdict(Issue("error", "invalid_spec", str(error)))]
        return issues
    issues = [asdict(issue) for issue in analyze(loaded).issues]

    return issues


# 確かめている定義の部分モデルの部品を引くもの (analyze の中だけで設定)
LOCAL_UNITS: contextvars.ContextVar[Optional["Subgraphs"]] = (
    contextvars.ContextVar("LOCAL_UNITS", default=None)
)


def analyze(spec: dict) -> Analysis:
    """
    モデルの定義を確かめ, 組み立てに使う計算の順とつなぎ先を求める関数

    Parameters
    ----------
    spec: dict
        モデルの定義

    Returns
    ----------
    analysis: Analysis
        誤りと注意, 計算の順, つなぎ先
    """
    # 部分モデルを先に全て確かめ (使われないものも), 本体を確認
    issues: list[Issue] = []
    subgraphs = Subgraphs(spec.get("subgraphs"), issues)
    token = LOCAL_UNITS.set(subgraphs)
    try:
        subgraphs.check_all()
        analysis = analyze_graph(spec, issues, top=True)
        check_unused_subgraphs(spec, analysis, subgraphs)
    finally:
        LOCAL_UNITS.reset(token)
    analysis.subgraphs = subgraphs.done

    return analysis


def analyze_graph(spec: dict, issues: list[Issue], top: bool) -> Analysis:
    """
    1 つの定義 (本体か部分モデル) を確かめる関数

    Parameters
    ----------
    spec: dict
        モデルの定義
    issues: list[Issue]
        誤りと注意を加えるリスト (本体の解析の issues になる)
    top: bool
        本体か (部分モデルは format_version と subgraphs を書けず, 入力の
        shape が無くても注意しない)

    Returns
    ----------
    analysis: Analysis
        誤りと注意, 計算の順, つなぎ先
    """
    # 一番上の項目, 入力, ノード, 出力の順に確認
    analysis = Analysis(issues=issues)
    keys = GRAPH_KEYS if top else GRAPH_KEYS - {
        "format_version", "subgraphs"
    }
    for key in spec:
        if key not in keys:
            issues.append(Issue(
                "error", "unknown_key", f"知らない項目 {key} があります"
                + ("" if top else " (部分モデルの中には書けません)"),
                location=key
            ))
    input_names = check_inputs(spec.get("inputs"), issues, need_shape=top)
    nodes = spec.get("nodes")
    if isinstance(nodes, dict) and not nodes or nodes is None:
        issues.append(Issue(
            "error", "no_nodes", "nodes に 1 つ以上のノードを書いてください",
            location="nodes"
        ))
        nodes = {}
    elif not isinstance(nodes, dict):
        issues.append(Issue(
            "error", "invalid_nodes",
            "nodes はノードの名前と定義の辞書にしてください (リストではなく)",
            location="nodes"
        ))
        nodes = {}
    for name, node in nodes.items():
        check_node(str(name), node, input_names, nodes, analysis)
    check_outputs(spec.get("outputs"), input_names, nodes, analysis)

    # 計算の順 (循環があればそのノードは誤り) と, 使われないものの確認
    analysis.order = topological_order(nodes, analysis)
    check_unused(spec, input_names, nodes, analysis)

    return analysis


def check_name(
    name: Any, kind: str, issues: list[Issue], node: bool,
    location: Optional[str] = None
) -> bool:
    """
    入力, ノード, 出力の名前が規則に合うかを確かめる関数

    Parameters
    ----------
    name: Any
        名前
    kind: str
        誤りの説明に使う種類 (入力, ノード, 出力)
    issues: list[Issue]
        誤りを加えるリスト
    node: bool
        ノードの名前か (組み立てたモデルの部品の入れ物の属性と同じ名前を禁止)
    location: Optional[str] = None
        誤りの場所 (inputs.x など)

    Returns
    ----------
    valid: bool
        規則に合う場合は True
    """
    # 識別子の形, 予約語, 組み立てたモデルの属性との重なりの確認
    valid = (
        isinstance(name, str) and bool(NAME_PATTERN.match(name))
        and not keyword.iskeyword(name)
    )
    if not valid:
        issues.append(Issue(
            "error", "invalid_name",
            f"{kind}の名前 {name!r} は, 英字か _ で始まり英数字と _ だけからなる"
            "名前 (予約語を除く) にしてください",
            node=str(name) if node else None, location=location
        ))
    elif node and name in RESERVED_NAMES:
        valid = False
        issues.append(Issue(
            "error", "reserved_name",
            f"ノードの名前 {name} は組み立てたモデルの属性と重なるので使えません",
            node=name
        ))
    elif name in EXPORT_NAMES:
        issues.append(Issue(
            "warning", "export_name",
            f"{kind}の名前 {name} は書き出す Python のコードで使う名前と重なる"
            "ので, コードを書き出せません (組み立てと形の推定はできる)",
            node=str(name) if node else None, location=location
        ))

    return valid


def check_inputs(
    inputs: Any, issues: list[Issue], need_shape: bool = True
) -> list[str]:
    """
    モデルの入力の定義を確かめる関数

    Parameters
    ----------
    inputs: Any
        定義の inputs
    issues: list[Issue]
        誤りと注意を加えるリスト
    need_shape: bool = True
        shape が無い場合に注意するか (部分モデルでは形は前のノードから
        決まるので注意しない)

    Returns
    ----------
    names: list[str]
        入力の名前 (書いた順)
    """
    # 1 つ以上の入力の辞書
    if not isinstance(inputs, dict) or not inputs:
        issues.append(Issue(
            "error", "no_inputs", "inputs に 1 つ以上の入力を書いてください",
            location="inputs"
        ))
        return []

    # 入力ごとに名前, 項目, 形, 型, 値の作り方を確認
    names = []
    for name, value in inputs.items():
        check_name(
            name, "入力", issues, node=False, location=f"inputs.{name}"
        )
        names.append(str(name))
        value = value if value is not None else {}
        if not isinstance(value, dict):
            issues.append(Issue(
                "error", "invalid_input",
                f"入力 {name} は shape などを持つ辞書にしてください",
                location=f"inputs.{name}"
            ))
            continue
        for key in value:
            if key not in INPUT_KEYS:
                issues.append(Issue(
                    "error", "unknown_key",
                    f"入力 {name} に知らない項目 {key} があります",
                    location=f"inputs.{name}.{key}"
                ))
        shape = value.get("shape")
        if shape is None and need_shape:
            issues.append(Issue(
                "warning", "no_shape",
                f"入力 {name} に shape が無いので, 形を推定できません",
                location=f"inputs.{name}.shape"
            ))
        elif shape is not None and (not isinstance(shape, list) or not all(
            isinstance(size, int) and not isinstance(size, bool) and size >= 0
            for size in shape
        )):
            issues.append(Issue(
                "error", "invalid_shape",
                f"入力 {name} の shape は 0 以上の整数のリストにしてください: "
                f"{shape!r}",
                location=f"inputs.{name}.shape"
            ))
        dtype = value.get("dtype", "float32")
        if not isinstance(dtype, str) or dtype not in DTYPES:
            issues.append(Issue(
                "error", "invalid_dtype",
                f"入力 {name} の dtype は {sorted(DTYPES)} のどれかにしてください",
                location=f"inputs.{name}.dtype"
            ))
        fill = value.get("fill", "zeros")
        if not isinstance(fill, str) or fill not in FILLS:
            issues.append(Issue(
                "error", "invalid_fill",
                f"入力 {name} の fill は {sorted(FILLS)} のどれかにしてください",
                location=f"inputs.{name}.fill"
            ))

    return names


def is_json_value(value: Any) -> bool:
    """
    値が JSON の値 (数, 文字, 真偽値, None と, それらのリスト) かを返す関数

    Parameters
    ----------
    value: Any
        値

    Returns
    ----------
    plain: bool
        JSON の値なら True
    """
    # リストは中の値も確認, 無限大と非数は JSON に書けないので不可
    if isinstance(value, list):
        plain = all(is_json_value(item) for item in value)
    elif isinstance(value, float):
        plain = math.isfinite(value)
    else:
        plain = value is None or isinstance(value, (bool, int, str))

    return plain


# 値の種類の名前 (誤りの文に使う)
TYPE_NAMES = {
    "int": "整数", "float": "数", "bool": "真偽値", "str": "文字",
    "list": "リスト", "null": "null"
}


def matches_types(value: Any, types: list) -> bool:
    """
    値が, 受け取れる種類のどれかに合うかを返す関数

    Parameters
    ----------
    value: Any
        値 (None は呼び出し側で除く)
    types: list
        int, float, bool, str, list, null の並び

    Returns
    ----------
    matched: bool
        どれかに合えば True (整数は数としても可)
    """
    is_bool = isinstance(value, bool)
    checks = {
        "int": isinstance(value, int) and not is_bool,
        "float": isinstance(value, (int, float)) and not is_bool,
        "bool": is_bool,
        "str": isinstance(value, str),
        "list": isinstance(value, list),
        "null": value is None
    }
    matched = any(checks.get(item, True) for item in types)

    return matched


def check_value(param: ParamSpec, value: Any) -> Optional[str]:
    """
    引数の値が種類と範囲に合うかを確かめる関数

    Parameters
    ----------
    param: ParamSpec
        引数の定義
    value: Any
        値

    Returns
    ----------
    message: Optional[str]
        合わない場合の説明, 合う場合は None
    """
    # 種類 (None は省略の意味で常に可) と, 範囲と選択肢の確認
    message = None
    if value is None:
        return message
    is_number = isinstance(value, (int, float)) and not isinstance(value, bool)
    if param.kind == "int" and (not isinstance(value, int) or isinstance(
        value, bool
    )):
        message = f"{param.name} は整数にしてください (今は {value!r})"
    elif param.kind == "float" and not is_number:
        message = f"{param.name} は数にしてください (今は {value!r})"
    elif param.kind == "bool" and not isinstance(value, bool):
        message = f"{param.name} は true か false にしてください (今は {value!r})"
    elif param.kind == "str" and not isinstance(value, str):
        message = f"{param.name} は文字にしてください (今は {value!r})"
    elif param.kind == "list" and not isinstance(value, list):
        message = f"{param.name} はリストにしてください (今は {value!r})"
    elif is_number and param.minimum is not None and value < param.minimum:
        message = f"{param.name} は {param.minimum} 以上 (今は {value})"
    elif is_number and param.maximum is not None and value > param.maximum:
        message = f"{param.name} は {param.maximum} 以下 (今は {value})"
    elif param.kind == "any" and param.types and not matches_types(
        value, param.types
    ):
        names = " か ".join(TYPE_NAMES[item] for item in param.types)
        message = f"{param.name} は {names}にしてください (今は {value!r})"
    elif param.choices is not None and value not in param.choices:
        message = f"{param.name} は {param.choices} のどれか (今は {value!r})"

    return message


def check_params(
    name: str, unit: UnitSpec, values: Any, field_name: str,
    specs: list[ParamSpec], issues: list[Issue]
) -> None:
    """
    ノードの params か forward_params を確かめる関数

    Parameters
    ----------
    name: str
        ノードの名前
    unit: UnitSpec
        ノードの部品
    values: Any
        書いた値の辞書
    field_name: str
        params か forward_params
    specs: list[ParamSpec]
        部品の引数の定義
    issues: list[Issue]
        誤りを加えるリスト

    Returns
    ----------
    None
    """
    # 辞書であることの確認
    if values is None:
        values = {}
    if not isinstance(values, dict):
        issues.append(Issue(
            "error", "invalid_params", f"{field_name} は辞書にしてください",
            node=name, location=field_name
        ))
        return

    # 書いた値ごとに, 名前, JSON の値, 種類と範囲を確認
    by_name = {param.name: param for param in specs}
    extra = unit.extra_params and field_name == "params"
    for key, value in values.items():
        location = f"{field_name}.{key}"
        if key not in by_name and not extra:
            issues.append(Issue(
                "error", "unknown_param",
                f"{unit.name} に引数 {key} はありません"
                f"{suggest(str(key), list(by_name))} (引数: "
                f"{', '.join(list(by_name)[:12]) or 'なし'}"
                f"{', …' if len(by_name) > 12 else ''})", node=name,
                location=location
            ))
            continue
        # 一覧に無い引数は, Python のキーワード引数として書ける名前だけ
        # (書き出すコードの呼び出しで使うため)
        if key not in by_name and (
            not isinstance(key, str) or not key.isidentifier()
            or keyword.iskeyword(key)
        ):
            issues.append(Issue(
                "error", "invalid_param_name",
                f"引数の名前 {key!r} は, 英字か _ で始まる英数字と _ で, "
                "Python の予約語 (class など) 以外にしてください",
                node=name, location=location
            ))
            continue
        if not is_json_value(value):
            issues.append(Issue(
                "error", "invalid_value",
                f"{key} は数, 文字, 真偽値, null とそのリストで書いてください",
                node=name, location=location
            ))
            continue
        message = check_value(by_name[key], value) if key in by_name else None
        if message is not None:
            issues.append(Issue(
                "error", "invalid_value", message, node=name,
                location=location
            ))

    # 必須の引数の不足 (null も不足)
    for param in specs:
        if param.required and values.get(param.name) is None:
            issues.append(Issue(
                "error", "missing_param",
                f"{unit.name} の必須の引数 {param.name} がありません",
                node=name, location=f"{field_name}.{param.name}"
            ))


def check_node(
    name: str, node: Any, input_names: list[str], nodes: dict,
    analysis: Analysis
) -> None:
    """
    1 つのノードの名前, 部品, 引数, 入力のつなぎ先を確かめる関数

    Parameters
    ----------
    name: str
        ノードの名前
    node: Any
        ノードの定義
    input_names: list[str]
        モデルの入力の名前
    nodes: dict
        全てのノードの定義
    analysis: Analysis
        誤りとつなぎ先を加える解析の結果

    Returns
    ----------
    None
    """
    # 名前, 入力との重なり, ノードの形の確認
    issues = analysis.issues
    check_name(name, "ノード", issues, node=True)
    if name in input_names:
        issues.append(Issue(
            "error", "duplicate_name",
            f"ノード {name} は入力と同じ名前です", node=name
        ))
    if not isinstance(node, dict):
        issues.append(Issue(
            "error", "invalid_node", f"ノード {name} は辞書にしてください",
            node=name
        ))
        return
    for key in node:
        if key not in NODE_KEYS:
            issues.append(Issue(
                "error", "unknown_key", f"ノードに知らない項目 {key} があります",
                node=name, location=key
            ))

    # 部品が登録簿にあり, 使えることの確認
    if "unit" not in node:
        issues.append(Issue(
            "error", "missing_unit", f"ノード {name} に unit がありません",
            node=name, location="unit"
        ))
        return
    unit_name = str(node.get("unit"))
    if unit_name.startswith(SUBGRAPH_PREFIX) and not has_subgraph(unit_name):
        issues.append(Issue(
            "error", "unknown_subgraph",
            f"部分モデル {unit_name[len(SUBGRAPH_PREFIX):]} は定義の "
            f"subgraphs にありません{suggest(unit_name, unit_names())}",
            node=name, location="unit"
        ))
        return
    try:
        unit = lookup_unit(unit_name)
    except KeyError:
        issues.append(Issue(
            "error", "unknown_unit",
            f"部品 {node.get('unit')!r} は登録されていません"
            f"{suggest(str(node.get('unit')), unit_names())}", node=name,
            location="unit"
        ))
        return
    if not unit.available:
        issues.append(Issue(
            "error", "unavailable_unit",
            f"部品 {unit.name} は使えません: {unit.reason}", node=name,
            location="unit"
        ))
        return
    analysis.units[name] = unit

    # Hugging Face のモデルの名前は, ネットワークを使わないので確かめない
    if unit.source == "transformers":
        issues.append(Issue(
            "warning", "unchecked_model",
            "Hugging Face のモデルの名前と設定は, 組み立てと形の推定のときに"
            "読み込むまで確かめられません", node=name,
            location="params.model_name"
        ))

    # 引数, 呼び出すときの設定の値, 入力のつなぎ先の確認
    check_params(
        name, unit, node.get("params"), "params", unit.params, issues
    )
    check_params(
        name, unit, node.get("forward_params"), "forward_params",
        unit.forward_params, issues
    )
    analysis.connections[name] = check_connections(
        name, unit, node_inputs(node), input_names, nodes, issues
    )


def check_connections(
    name: str, unit: UnitSpec, inputs: Any, input_names: list[str],
    nodes: dict, issues: list[Issue]
) -> dict[str, list[tuple[str, Optional[str]]]]:
    """
    ノードの入力の口とつなぎ先を確かめる関数

    Parameters
    ----------
    name: str
        ノードの名前
    unit: UnitSpec
        ノードの部品
    inputs: Any
        定義の inputs (口の名前とつなぎ先の辞書, または省略形の文字)
    input_names: list[str]
        モデルの入力の名前
    nodes: dict
        全てのノードの定義
    issues: list[Issue]
        誤りを加えるリスト

    Returns
    ----------
    connections: dict[str, list[tuple[str, Optional[str]]]]
        口の名前ごとのつなぎ先 (名前と出力の口の名前の組のリスト)
    """
    # 省略形は, 必須の口がちょうど 1 つの部品だけで, その口につなぐ
    ports = {port.name: port for port in unit.inputs}
    required = [
        port for port in unit.inputs if not port.optional and not port.variadic
    ]
    # 必須の口が無く, 可変の数の口が 1 つの部品 (op.Add など) は, その口に
    # 1 つだけつなぐ
    variadic = [port for port in unit.inputs if port.variadic]
    if isinstance(inputs, str) and not required and len(variadic) == 1:
        inputs = {variadic[0].name: [inputs]}
    if isinstance(inputs, str):
        if len(required) != 1:
            issues.append(Issue(
                "error", "ambiguous_inputs",
                f"{unit.name} は必須の口が {len(required)} 個なので, inputs は"
                "口の名前とつなぎ先の辞書にしてください", node=name,
                location="inputs"
            ))
            return {}
        inputs = {required[0].name: inputs}
    if not isinstance(inputs, dict):
        issues.append(Issue(
            "error", "invalid_inputs",
            "inputs は口の名前とつなぎ先の辞書にしてください (必須の口が 1 つの"
            "部品は, つなぎ先の文字 1 つでも書ける)", node=name,
            location="inputs"
        ))
        return {}

    # 口ごとに, 口があること, つなぎ先の形 (可変の口はリスト) と存在を確認
    connections: dict[str, list[tuple[str, Optional[str]]]] = {}
    for port_name, value in inputs.items():
        location = f"inputs.{port_name}"
        port = ports.get(port_name)
        if port is None:
            issues.append(Issue(
                "error", "unknown_port",
                f"{unit.name} に入力の口 {port_name} はありません "
                f"(口: {list(ports)})", node=name, location=location
            ))
            continue
        refs = value if port.variadic else [value]
        if port.variadic and not isinstance(value, list):
            issues.append(Issue(
                "error", "invalid_inputs",
                f"口 {port_name} は複数のつなぎ先を受け取るので, リストで書いて"
                "ください", node=name, location=location
            ))
            continue
        if port.variadic and not value:
            issues.append(Issue(
                "error", "missing_input",
                f"口 {port_name} に 1 つ以上のつなぎ先を書いてください",
                node=name, location=location
            ))
            continue
        parsed = []
        for ref in refs:
            target = check_ref(ref, input_names, nodes, issues, name, location)
            if target is not None:
                parsed.append(target)
        if len(parsed) == len(refs):
            connections[port_name] = parsed

    # 必須の口につなぎ先が無い場合は誤り
    for port in unit.inputs:
        if not port.optional and port.name not in inputs:
            issues.append(Issue(
                "error", "missing_input",
                f"{unit.name} の口 {port.name} につなぎ先がありません",
                node=name, location=f"inputs.{port.name}"
            ))

    return connections


def check_ref(
    ref: Any, input_names: list[str], nodes: dict, issues: list[Issue],
    node: Optional[str], location: str
) -> Optional[tuple[str, Optional[str]]]:
    """
    つなぎ先 (入力の名前, ノードの名前, ノードの名前:口の名前) を確かめる関数

    Parameters
    ----------
    ref: Any
        つなぎ先の文字
    input_names: list[str]
        モデルの入力の名前
    nodes: dict
        全てのノードの定義
    issues: list[Issue]
        誤りを加えるリスト
    node: Optional[str]
        誤りを付けるノードの名前 (出力の場合は None)
    location: str
        誤りの場所

    Returns
    ----------
    target: Optional[tuple[str, Optional[str]]]
        名前と出力の口の名前の組, 誤りの場合は None
    """
    # 文字であることと, 名前が入力かノードにあることの確認
    if not isinstance(ref, str):
        issues.append(Issue(
            "error", "invalid_ref", f"つなぎ先 {ref!r} は文字で書いてください",
            node=node, location=location
        ))
        return None
    name, port = parse_ref(ref)
    if name in input_names:
        if port is not None:
            issues.append(Issue(
                "error", "invalid_ref",
                f"入力 {name} には口の名前を付けられません", node=node,
                location=location
            ))
            return None
        target = (name, None)
        return target
    if name not in nodes:
        issues.append(Issue(
            "error", "unknown_ref",
            f"つなぎ先 {name} は入力にもノードにもありません"
            f"{suggest(name, [*input_names, *map(str, nodes)])}", node=node,
            location=location
        ))
        return None

    # ノードの出力の口の確認 (出力が複数の部品は口の名前が必須)
    target_node = nodes[name]
    try:
        unit = lookup_unit(str(target_node.get("unit"))) if isinstance(
            target_node, dict
        ) else None
    except KeyError:
        unit = None
    if unit is not None:
        outputs = [output.name for output in unit.outputs]
        if port is None and len(outputs) > 1:
            issues.append(Issue(
                "error", "ambiguous_ref",
                f"{name} ({unit.name}) は出力が複数あるので, 「{name}:口の名前」"
                f"で選んでください (口: {outputs})", node=node,
                location=location
            ))
            return None
        if port is not None and port not in outputs:
            issues.append(Issue(
                "error", "unknown_port",
                f"{name} ({unit.name}) に出力の口 {port} はありません "
                f"(口: {outputs})", node=node, location=location
            ))
            return None
    target = (name, port)

    return target


def check_outputs(
    outputs: Any, input_names: list[str], nodes: dict, analysis: Analysis
) -> None:
    """
    モデルの出力の定義を確かめる関数

    Parameters
    ----------
    outputs: Any
        定義の outputs
    input_names: list[str]
        モデルの入力の名前
    nodes: dict
        全てのノードの定義
    analysis: Analysis
        誤りとつなぎ先を加える解析の結果

    Returns
    ----------
    None
    """
    # 1 つ以上の出力の辞書で, 名前とつなぎ先を確認
    issues = analysis.issues
    if not isinstance(outputs, dict) or not outputs:
        issues.append(Issue(
            "error", "no_outputs", "outputs に 1 つ以上の出力を書いてください",
            location="outputs"
        ))
        return
    for name, ref in outputs.items():
        check_name(
            name, "出力", issues, node=False, location=f"outputs.{name}"
        )
        check_ref(ref, input_names, nodes, issues, None, f"outputs.{name}")


def dependencies(analysis: Analysis, name: str) -> list[str]:
    """
    ノードが入力に使うノードの名前を返す関数

    Parameters
    ----------
    analysis: Analysis
        解析の結果
    name: str
        ノードの名前

    Returns
    ----------
    names: list[str]
        つなぎ先のノードの名前 (重なりなし, モデルの入力は除く)
    """
    names = []
    for targets in analysis.connections.get(name, {}).values():
        for target, _ in targets:
            if target in analysis.connections and target not in names:
                names.append(target)

    return names


def topological_order(nodes: dict, analysis: Analysis) -> list[str]:
    """
    ノードの計算の順を求め, 循環を誤りにする関数

    同じ順位のノードは定義に書いた順に並べる

    Parameters
    ----------
    nodes: dict
        全てのノードの定義
    analysis: Analysis
        解析の結果, 循環の誤りを加える

    Returns
    ----------
    order: list[str]
        計算の順のノードの名前 (循環に入るノードは含まない)
    """
    # 前のノードを全て計算し終えたノードを, 書いた順に繰り返し選択
    names = [str(name) for name in nodes if str(name) in analysis.connections]
    order: list[str] = []
    done: set[str] = set()
    changed = True
    while changed:
        changed = False
        for name in names:
            if name in done:
                continue
            if all(dep in done for dep in dependencies(analysis, name)):
                order.append(name)
                done.add(name)
                changed = True
                break

    # 残ったノードのうち, 自分に戻ってくるものは循環 (誤り), それ以外は
    # 循環の先 (計算できないので注意)
    left = [name for name in names if name not in done]
    for name in left:
        if in_cycle(analysis, name, set(left)):
            analysis.issues.append(Issue(
                "error", "cycle", f"ノード {name} のつなぎ方が循環しています",
                node=name, location="inputs"
            ))
        else:
            analysis.issues.append(Issue(
                "warning", "after_cycle",
                f"ノード {name} は循環の先にあるので計算できません",
                node=name, location="inputs"
            ))

    return order


def in_cycle(analysis: Analysis, name: str, left: set[str]) -> bool:
    """
    ノードが, 前のノードをたどって自分に戻ってくる (循環に入る) かを返す関数

    Parameters
    ----------
    analysis: Analysis
        解析の結果 (つなぎ先)
    name: str
        ノードの名前
    left: set[str]
        計算の順に入らなかったノードの名前 (この中だけをたどる)

    Returns
    ----------
    cyclic: bool
        自分に戻ってくれば True
    """
    # 前のノードを深さ優先でたどり, 自分に着くかを確認
    stack = [dep for dep in dependencies(analysis, name) if dep in left]
    seen: set[str] = set()
    cyclic = False
    while stack:
        current = stack.pop()
        if current == name:
            cyclic = True
            break
        if current in seen:
            continue
        seen.add(current)
        stack.extend(
            dep for dep in dependencies(analysis, current) if dep in left
        )

    return cyclic


def suggest(name: str, candidates: list[str]) -> str:
    """
    書き間違いの近い候補を, 誤りの文に付ける文字にする関数

    Parameters
    ----------
    name: str
        書いた名前
    candidates: list[str]
        正しい名前の候補

    Returns
    ----------
    hint: str
        「 (もしかして: nn.Linear)」のような文字, 近い候補が無ければ空
    """
    # 大文字と小文字を区別せずに比べ, 元の綴りで返す
    lowered = {candidate.lower(): candidate for candidate in candidates}
    close = difflib.get_close_matches(
        name.lower(), list(lowered), n=3, cutoff=0.7
    )
    hint = (
        f" (もしかして: {', '.join(lowered[item] for item in close)})"
        if close else ""
    )

    return hint


def check_unused(
    spec: dict, input_names: list[str], nodes: dict, analysis: Analysis
) -> None:
    """
    出力に届かないノードと, 使われない入力を注意にする関数

    Parameters
    ----------
    spec: dict
        モデルの定義
    input_names: list[str]
        モデルの入力の名前
    nodes: dict
        全てのノードの定義
    analysis: Analysis
        注意を加える解析の結果

    Returns
    ----------
    None
    """
    # 出力から逆にたどって, 使われる名前の収集
    outputs = spec.get("outputs")
    if not isinstance(outputs, dict):
        outputs = {}
    # 誤りのあるノード (つなぎ先を読めないことがある) からもたどり, その
    # 前のノードに出力に届かない注意を重ねない
    stack = [
        parse_ref(ref)[0] for ref in outputs.values() if isinstance(ref, str)
    ] + [
        str(issue.node) for issue in analysis.issues
        if issue.severity == "error" and issue.node is not None
    ]
    used: set[str] = set()
    while stack:
        name = stack.pop()
        if name in used:
            continue
        used.add(name)
        if name in analysis.connections:
            for targets in analysis.connections[name].values():
                stack.extend(target for target, _ in targets)
        elif isinstance(nodes.get(name), dict):
            # 部品を確かめられなかったノードは, 書いたつなぎ先の文字からの
            # 追跡
            stack.extend(raw_refs(node_inputs(nodes[name])))

    # 届かないノードと入力を注意として追加
    for name in nodes:
        if str(name) not in used:
            analysis.issues.append(Issue(
                "warning", "unused_node",
                f"ノード {name} は出力に届きません", node=str(name)
            ))
    for name in input_names:
        if name not in used:
            analysis.issues.append(Issue(
                "warning", "unused_input", f"入力 {name} は使われていません",
                location=f"inputs.{name}"
            ))


def raw_refs(inputs: Any) -> list[str]:
    """
    ノードの inputs に書いたつなぎ先の名前を, 口を確かめずに集める関数

    Parameters
    ----------
    inputs: Any
        定義の inputs (辞書か省略形の文字)

    Returns
    ----------
    names: list[str]
        入力かノードの名前
    """
    values = [inputs] if isinstance(inputs, str) else (
        list(inputs.values()) if isinstance(inputs, dict) else []
    )
    refs = [
        item for value in values
        for item in (value if isinstance(value, list) else [value])
        if isinstance(item, str)
    ]
    names = [parse_ref(ref)[0] for ref in refs]

    return names


def has_subgraph(name: str) -> bool:
    """
    sub. で始まる部品の名前の部分モデルが, 確かめている定義にあるかを返す
    関数

    Parameters
    ----------
    name: str
        部品の名前 (sub.block など)

    Returns
    ----------
    found: bool
        定義の subgraphs にある場合は True
    """
    subgraphs = LOCAL_UNITS.get()
    found = subgraphs is not None and (
        name[len(SUBGRAPH_PREFIX):] in subgraphs.specs
    )

    return found


def lookup_unit(name: str) -> UnitSpec:
    """
    名前から部品を取り出す関数 (確かめている定義の部分モデルも含む)

    Parameters
    ----------
    name: str
        部品の名前 (nn.Linear, sub.block など)

    Returns
    ----------
    unit: UnitSpec
        部品

    Raises
    ----------
    KeyError
        登録されていない名前と, 定義に無い部分モデルの場合
    """
    # sub. で始まる名前は定義の部分モデル, それ以外は登録簿
    subgraphs = LOCAL_UNITS.get()
    if name.startswith(SUBGRAPH_PREFIX):
        if subgraphs is None:
            raise KeyError(f"部分モデル {name} はありません")
        unit = subgraphs.unit(name[len(SUBGRAPH_PREFIX):])
        return unit
    unit = get_unit(name)

    return unit


def unit_names() -> list[str]:
    """
    書き間違いの候補に使う, 部品の名前の一覧を返す関数

    Parameters
    ----------
    None

    Returns
    ----------
    names: list[str]
        登録簿の部品と, 確かめている定義の部分モデルの名前
    """
    subgraphs = LOCAL_UNITS.get()
    names = list(REGISTRY)
    if subgraphs is not None:
        names += [SUBGRAPH_PREFIX + name for name in subgraphs.names]

    return names


def class_name_of(name: str) -> str:
    """
    部分モデルの名前から, 書き出すコードのクラスの名前を作る関数

    Parameters
    ----------
    name: str
        部分モデルの名前 (res_block など)

    Returns
    ----------
    class_name: str
        ResBlock のような名前
    """
    parts = [part for part in name.split("_") if part]
    class_name = "".join(
        part[:1].upper() + part[1:] for part in parts
    ) or "Subgraph"

    return class_name


# 定義の部分モデルを, 必要になったときに確かめて部品にするもの
class Subgraphs:
    def __init__(self, subgraphs: Any, issues: list[Issue]):
        # 部分モデルの定義と, 本体の誤りと注意を加えるリストを保存
        self.issues = issues
        self.specs: dict[str, Any] = {}
        if subgraphs is not None and not isinstance(subgraphs, dict):
            issues.append(Issue(
                "error", "invalid_subgraphs",
                "subgraphs は部分モデルの名前と定義の辞書にしてください",
                location="subgraphs"
            ))
        elif isinstance(subgraphs, dict):
            self.specs = {str(key): value for key, value in subgraphs.items()}
        self.names = list(self.specs)

        # 作った部品, 確かめた定義と解析, 確かめている途中の名前 (循環の検出)
        self.units: dict[str, UnitSpec] = {}
        self.done: dict[str, tuple[dict, Analysis]] = {}
        self.stack: list[str] = []

    def check_all(self) -> None:
        """
        全ての部分モデルを確かめる関数 (使われないものの誤りも返すため)

        Parameters
        ----------
        None

        Returns
        ----------
        None
        """
        # 名前の順に部分モデルを確認
        for name in self.names:
            try:
                self.unit(name)
            except KeyError:
                pass

    def unit(self, name: str) -> UnitSpec:
        """
        部分モデルを確かめ, 部品にして返す関数

        Parameters
        ----------
        name: str
            部分モデルの名前 (sub. を除いたもの)

        Returns
        ----------
        unit: UnitSpec
            部品 (誤りのある部分モデルは使えない部品)

        Raises
        ----------
        KeyError
            定義に無い部分モデルの場合
        """
        # 確かめ済みのものと, 確かめている途中のもの (循環) の扱い
        if name in self.units:
            return self.units[name]
        if name not in self.specs:
            raise KeyError(f"部分モデル {name} はありません")
        unit_name = SUBGRAPH_PREFIX + name
        if name in self.stack:
            cycle = " → ".join([*self.stack[self.stack.index(name):], name])
            unit = UnitSpec(
                name=unit_name, category="sub", source="subgraph",
                available=False,
                reason=f"部分モデルが互いに使い合っています ({cycle})"
            )
            return unit

        # 部分モデルの定義を確かめ, 誤りと注意を場所を付けて本体に追加
        spec = self.specs[name]
        location = f"subgraphs.{name}"
        inner: list[Issue] = []
        self.stack.append(name)
        try:
            check_name(name, "部分モデル", inner, node=False)
            if not isinstance(spec, dict):
                inner.append(Issue(
                    "error", "invalid_subgraph",
                    "部分モデルは定義の辞書にしてください"
                ))
                analysis = Analysis(issues=inner)
            else:
                analysis = analyze_graph(spec, inner, top=False)
        finally:
            self.stack.pop()
        for issue in inner:
            where = ".".join(
                part for part in (
                    location,
                    f"nodes.{issue.node}" if issue.node else None,
                    issue.location
                ) if part
            )
            self.issues.append(Issue(
                issue.severity, issue.code,
                f"部分モデル {name}: {issue.message}", location=where,
                subgraph=name
            ))

        # 入力は全て必須の口, 出力は出力の口 (2 つ以上なら辞書のキー)
        errors = [issue for issue in inner if issue.severity == "error"]
        inputs = spec.get("inputs") if isinstance(spec, dict) else None
        outputs = spec.get("outputs") if isinstance(spec, dict) else None
        input_ports = [
            PortSpec(name=str(key), description=str(
                (value or {}).get("description", "")
                if isinstance(value, dict) or value is None else ""
            ))
            for key, value in (
                inputs.items() if isinstance(inputs, dict) else []
            )
        ]
        output_names = [
            str(key) for key in (outputs if isinstance(outputs, dict) else {})
        ]
        output_ports = [
            PortSpec(
                name=key, path=[] if len(output_names) == 1 else [key]
            )
            for key in output_names
        ]
        unit = UnitSpec(
            name=unit_name, category="sub", source="subgraph",
            description=str(
                spec.get("description", "") if isinstance(spec, dict) else ""
            ),
            inputs=input_ports, outputs=output_ports,
            available=not errors,
            reason=(
                f"部分モデル {name} の中に誤りがあります. 先に部分モデルの"
                "中の誤りを直してください" if errors else ""
            ),
            code_name=class_name_of(name)
        )
        if not errors:
            self.done[name] = (spec, analysis)
            unit.builder = subgraph_builder(spec, analysis)
        self.units[name] = unit

        return unit


def subgraph_builder(
    spec: dict, analysis: Analysis
) -> Callable[[], nn.Module]:
    """
    部分モデルの GraphModel を作る関数を返す関数

    Parameters
    ----------
    spec: dict
        部分モデルの定義
    analysis: Analysis
        部分モデルの解析

    Returns
    ----------
    builder: Callable[[], nn.Module]
        引数なしで部分モデルを作る関数
    """
    def build() -> nn.Module:
        """
        部分モデルの GraphModel を作る関数

        Parameters
        ----------
        None

        Returns
        ----------
        model: nn.Module
            部分モデル
        """
        # 循環 import を避けるため, ここで読み込み
        from ml_modelcontrol.model import GraphModel

        model = GraphModel(spec, analysis)

        return model

    return build


def units_of(graph: Any) -> list[str]:
    """
    定義 (本体か部分モデル) のノードの部品の名前を返す関数

    Parameters
    ----------
    graph: Any
        定義 (辞書でない場合は空として扱う)

    Returns
    ----------
    names: list[str]
        ノードの部品の名前
    """
    nodes = graph.get("nodes") if isinstance(graph, dict) else None
    names = [
        str(node.get("unit"))
        for node in (nodes if isinstance(nodes, dict) else {}).values()
        if isinstance(node, dict)
    ]

    return names


def check_unused_subgraphs(
    spec: dict, analysis: Analysis, subgraphs: Subgraphs
) -> None:
    """
    本体からも他の部分モデルからも使われない部分モデルを注意にする関数

    Parameters
    ----------
    spec: dict
        モデルの定義
    analysis: Analysis
        注意を加える本体の解析
    subgraphs: Subgraphs
        定義の部分モデル

    Returns
    ----------
    None
    """
    # 本体のノードから, 使う部分モデルの中のノードへたどる
    stack = units_of(spec)
    used: set[str] = set()
    while stack:
        unit_name = stack.pop()
        if unit_name in used:
            continue
        used.add(unit_name)
        if unit_name.startswith(SUBGRAPH_PREFIX):
            stack += units_of(
                subgraphs.specs.get(unit_name[len(SUBGRAPH_PREFIX):])
            )
    for name in subgraphs.names:
        if SUBGRAPH_PREFIX + name not in used:
            analysis.issues.append(Issue(
                "warning", "unused_subgraph",
                f"部分モデル {name} はどのノードにも使われていません",
                location=f"subgraphs.{name}"
            ))


def describe_spec_units(spec: Union[str, Path, dict]) -> list[dict]:
    """
    定義の部分モデルの部品の一覧を, describe_units と同じ形で返す関数

    GUI の部品の一覧と, ノードの口の表示に使う

    Parameters
    ----------
    spec: Union[str, Path, dict]
        モデルの定義, またはその YAML か JSON のファイルのパス

    Returns
    ----------
    units: list[dict]
        部分モデルの名前の順の部品ごとの辞書 (誤りのあるものは使えない部品)
    """
    # 部分モデルを確かめて部品にし, 作る関数を除いて辞書に変換
    loaded = load_spec(spec)
    subgraphs = Subgraphs(loaded.get("subgraphs"), [])
    token = LOCAL_UNITS.set(subgraphs)
    try:
        subgraphs.check_all()
    finally:
        LOCAL_UNITS.reset(token)
    units = []
    for name in subgraphs.names:
        if name not in subgraphs.units:
            continue
        data = asdict(subgraphs.units[name])
        for key in (
            "builder", "code_name", "code_import", "code_args", "offline"
        ):
            data.pop(key)
        units.append(data)

    return units
