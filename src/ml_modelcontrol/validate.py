import keyword
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

from torch import nn

from ml_modelcontrol.registry import ParamSpec, UnitSpec, get_unit
from ml_modelcontrol.spec import (
    GRAPH_KEYS, INPUT_KEYS, NODE_KEYS, load_spec, node_inputs, parse_ref
)

# 入力, ノード, 出力の名前の形
NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# ノードに使えない名前 (組み立てたモデルの部品の入れ物の属性と重なるもの)
RESERVED_NAMES = set(dir(nn.ModuleDict()))

# 入力の例に使える型と, 値の作り方
DTYPES = {
    "float32", "float64", "float16", "bfloat16", "int64", "int32", "bool"
}
FILLS = {"zeros", "ones", "arange"}


@dataclass
class Issue:
    # 誤りか注意か, 決まった英字の識別子, ノードの名前 (定義全体なら None),
    # 場所 (params.out_features など), 説明
    severity: str
    code: str
    message: str
    node: Optional[str] = None
    location: Optional[str] = None


@dataclass
class Analysis:
    # 誤りと注意, 計算の順, ノードごとの口のつなぎ先 (名前と口の名前の組)
    issues: list[Issue] = field(default_factory=list)
    order: list[str] = field(default_factory=list)
    connections: dict[str, dict[str, list[tuple[str, Optional[str]]]]] = (
        field(default_factory=dict)
    )
    units: dict[str, UnitSpec] = field(default_factory=dict)

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
    # 一番上の項目, 入力, ノード, 出力の順に確認
    analysis = Analysis()
    issues = analysis.issues
    for key in spec:
        if key not in GRAPH_KEYS:
            issues.append(Issue(
                "error", "unknown_key", f"知らない項目 {key} があります",
                location=key
            ))
    input_names = check_inputs(spec.get("inputs"), issues)
    nodes = spec.get("nodes")
    if not isinstance(nodes, dict) or not nodes:
        issues.append(Issue(
            "error", "no_nodes", "nodes に 1 つ以上のノードを書いてください",
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

    return valid


def check_inputs(inputs: Any, issues: list[Issue]) -> list[str]:
    """
    モデルの入力の定義を確かめる関数

    Parameters
    ----------
    inputs: Any
        定義の inputs
    issues: list[Issue]
        誤りと注意を加えるリスト

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
        if shape is None:
            issues.append(Issue(
                "warning", "no_shape",
                f"入力 {name} に shape が無いので, 形を推定できません",
                location=f"inputs.{name}.shape"
            ))
        elif not isinstance(shape, list) or not all(
            isinstance(size, int) and not isinstance(size, bool) and size >= 0
            for size in shape
        ):
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
                f"{unit.name} に引数 {key} はありません", node=name,
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
    try:
        unit = get_unit(str(node.get("unit")))
    except KeyError:
        issues.append(Issue(
            "error", "unknown_unit",
            f"部品 {node.get('unit')!r} は登録されていません", node=name,
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
            "inputs は口の名前とつなぎ先の辞書にしてください", node=name,
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
            f"つなぎ先 {name} は入力にもノードにもありません", node=node,
            location=location
        ))
        return None

    # ノードの出力の口の確認 (出力が複数の部品は口の名前が必須)
    target_node = nodes[name]
    try:
        unit = get_unit(str(target_node.get("unit"))) if isinstance(
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

    # 残ったノードは循環 (かその先) のノード
    for name in names:
        if name not in done:
            analysis.issues.append(Issue(
                "error", "cycle",
                f"ノード {name} のつなぎ方が循環しています (または循環の先です)",
                node=name, location="inputs"
            ))

    return order


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
    stack = [
        parse_ref(ref)[0] for ref in outputs.values() if isinstance(ref, str)
    ]
    used: set[str] = set()
    while stack:
        name = stack.pop()
        if name in used:
            continue
        used.add(name)
        for targets in analysis.connections.get(name, {}).values():
            stack.extend(target for target, _ in targets)

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
