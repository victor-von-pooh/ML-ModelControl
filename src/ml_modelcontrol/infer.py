from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional, Union

import torch
from torch import nn

from ml_modelcontrol.model import GraphModel, NodeBuildError, follow_path
from ml_modelcontrol.registry import get_unit
from ml_modelcontrol.spec import load_spec
from ml_modelcontrol.validate import Analysis, Issue, analyze

# 形の推定で計算する前に確かめる, パラメータ数と入力の例の要素数の上限の既定値
MAX_PARAMS = 200_000_000
MAX_INPUT_ELEMENTS = 10_000_000


def describe(value: Any) -> dict:
    """
    出力の値の形と型を, JSON にできる辞書で表す関数

    Parameters
    ----------
    value: Any
        部品の出力の値

    Returns
    ----------
    info: dict
        kind (tensor, tuple, dict, none, other) と, テンソルなら shape と
        dtype, タプルと辞書なら中の値の items
    """
    # テンソル, タプルとリスト, 辞書 (Hugging Face の出力も), None の順に判定
    if isinstance(value, torch.Tensor):
        info: dict = {
            "kind": "tensor", "shape": list(value.shape),
            "dtype": str(value.dtype).removeprefix("torch.")
        }
    elif isinstance(value, (tuple, list)):
        info = {"kind": "tuple", "items": [describe(item) for item in value]}
    elif isinstance(value, Mapping):
        info = {
            "kind": "dict",
            "items": {str(key): describe(item) for key, item in value.items()}
        }
    elif value is None:
        info = {"kind": "none"}
    else:
        info = {"kind": "other", "type": type(value).__name__}

    return info


def example_input(spec: dict) -> dict[str, torch.Tensor]:
    """
    入力の例 (shape, dtype, fill) から, 形の推定に使う値を作る関数

    Parameters
    ----------
    spec: dict
        モデルの定義

    Returns
    ----------
    values: dict[str, torch.Tensor]
        入力の名前ごとの値
    """
    # 入力ごとに, 決めた型の 0, 1, 通し番号のテンソルを作成
    values = {}
    for name, value in spec["inputs"].items():
        value = value or {}
        shape = list(value.get("shape") or [])
        dtype = getattr(torch, value.get("dtype", "float32"))
        fill = value.get("fill", "zeros")
        if fill == "ones":
            tensor = torch.ones(shape, dtype=dtype)
        elif fill == "arange":
            count = 1
            for size in shape:
                count *= size
            tensor = torch.arange(count).reshape(shape).to(dtype)
        else:
            tensor = torch.zeros(shape, dtype=dtype)
        values[str(name)] = tensor

    return values


def count_params(
    module: nn.Module, seen: set[int]
) -> tuple[int, int]:
    """
    部品のパラメータ数を, 他の部品と共有した重みを除いて数える関数

    Parameters
    ----------
    module: nn.Module
        部品
    seen: set[int]
        数え済みのパラメータの id, その場で加える

    Returns
    ----------
    total: int
        パラメータの要素数
    trainable: int
        そのうち学習するものの要素数
    """
    # まだ数えていないパラメータだけを合計
    total = 0
    trainable = 0
    for parameter in module.parameters():
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        total += parameter.numel()
        if parameter.requires_grad:
            trainable += parameter.numel()

    return total, trainable


def offline_values(unit_name: str) -> dict:
    """
    部品の, 学習済みの重みを読まない引数の値を返す関数

    Parameters
    ----------
    unit_name: str
        部品の名前

    Returns
    ----------
    values: dict
        引数の名前ごとの値, 登録されていない部品の場合は空
    """
    try:
        values = dict(get_unit(unit_name).offline)
    except KeyError:
        values = {}

    return values


def describe_error(error: Exception) -> str:
    """
    ノードの計算の誤りの文を作る関数

    部分モデルのノードの誤りには, 中のどのノードで起きたかを添える
    (組み立てたモデルが誤りに付けた注記のうち, 最後の 1 つ (外のノード
    自身のもの) を除いたもの. 注記は Python 3.11 以上)

    Parameters
    ----------
    error: Exception
        ノードの計算で起きた誤り

    Returns
    ----------
    message: str
        「型の名前: 誤りの文」と, 中のノードの場所
    """
    message = f"{type(error).__name__}: {error}"
    notes = [
        str(note) for note in getattr(error, "__notes__", [])[:-1]
    ]
    if notes:
        message += f" ({' / '.join(notes)})"

    return message


def incoming_text(model: Any, name: str, values: dict[str, Any]) -> str:
    """
    計算に失敗したノードに入った値の形と型を, 誤りの文に添える文にする関数

    形や型の合わない誤りを, 前のノードを見に行かずに直せるようにする

    Parameters
    ----------
    model: Any
        組み立てたモデル (GraphModel)
    name: str
        ノードの名前
    values: dict[str, Any]
        入力と, 前に計算したノードの戻り値

    Returns
    ----------
    text: str
        「 (入った値: x [4, 16] float32, ...)」, 分からない場合は空
    """
    # 口ごとのつなぎ先の値の形と型 (テンソルでない値は種類だけ)
    parts = []
    for port, targets in model.connections.get(name, {}).items():
        for target in targets:
            try:
                info = describe(model.resolve(values, target))
            except Exception:
                continue
            if info.get("kind") == "tensor":
                parts.append(
                    f"{port} {info.get('shape')} {info.get('dtype')}"
                )
            else:
                parts.append(f"{port} {info.get('kind')}")
    text = f" (入った値: {', '.join(parts)})" if parts else ""

    return text


# 入る値の次元と比べる引数 (引数の名前と, 比べる次元の位置と説明)
SHAPE_PARAMS = {
    "in_features": (-1, "最後の次元"),
    "normalized_shape": (-1, "最後の次元"),
    "in_channels": (1, "チャネル (2 番目の次元)"),
    "num_features": (1, "チャネル (2 番目の次元)"),
    "num_channels": (1, "チャネル (2 番目の次元)"),
}

# 番号 (整数) を受け取る部品と, 小数を受け取る部品の分類
INDEX_UNITS = {"nn.Embedding", "nn.EmbeddingBag"}


def shape_hint(
    model: Any, spec: dict, name: str, values: dict[str, Any]
) -> str:
    """
    計算に失敗したノードの, よくある形と型の食い違いの直し方を返す関数

    最初の口に入った値の次元と引数 (in_features, in_channels など) の
    食い違いと, 番号を受け取る部品に小数が入った場合などを日本語で示す

    Parameters
    ----------
    model: Any
        組み立てたモデル (GraphModel)
    spec: dict
        モデルの定義
    name: str
        ノードの名前
    values: dict[str, Any]
        入力と, 前に計算したノードの戻り値

    Returns
    ----------
    hint: str
        「 → 直し方」の文, 分からない場合は空
    """
    # 最初の口に入ったテンソルの形と型
    targets = next(iter(model.connections.get(name, {}).values()), [])
    if not targets:
        return ""
    try:
        info = describe(model.resolve(values, targets[0]))
    except Exception:
        return ""
    if info.get("kind") != "tensor":
        return ""
    shape = info.get("shape") or []
    dtype = str(info.get("dtype"))
    node = spec["nodes"][name]
    params = node.get("params") or {}
    unit = model.unit_names.get(name, "")

    # 番号と小数の食い違い
    hints = []
    if unit in INDEX_UNITS and not dtype.startswith("int"):
        hints.append(
            f"{unit} には番号 (int64) を入れてください (入った値は {dtype})"
        )
    elif dtype.startswith("int") and any(
        key in params for key in SHAPE_PARAMS
    ):
        hints.append(
            f"{dtype} の値が入っています. 小数 (float32) の値を入れてください"
        )

    # 次元と引数の食い違い
    for key, (axis, label) in SHAPE_PARAMS.items():
        value = params.get(key)
        if isinstance(value, list) and len(value) == 1:
            value = value[0]
        if not isinstance(value, int) or isinstance(value, bool):
            continue
        if len(shape) <= max(axis, -axis - 1):
            hints.append(
                f"入った値は {len(shape)} 次元で, {key} と比べる{label}が"
                "ありません"
            )
        elif shape[axis] != value:
            hints.append(
                f"入る値の{label} {shape[axis]} と {key} {value} が合いません "
                f"({key} を {shape[axis]} にするか, 前のノードを直してください)"
            )
    hint = f" → {'. '.join(hints)}" if hints else ""

    return hint


def infer(
    spec: Union[str, Path, dict], max_params: int = MAX_PARAMS,
    max_input_elements: int = MAX_INPUT_ELEMENTS
) -> dict:
    """
    入力の例をモデルに通し, ノードごとの出力の形とパラメータ数を求める関数

    cpu で eval() と勾配なしで, ノードを 1 つずつ計算する. 失敗したノードは
    誤りの文を, その先のノードは計算できないことを返す

    Parameters
    ----------
    spec: Union[str, Path, dict]
        モデルの定義, またはその YAML か JSON のファイルのパス
    max_params: int = MAX_PARAMS
        計算する前に確かめるパラメータ数の上限. 超える場合は計算しない
    max_input_elements: int = MAX_INPUT_ELEMENTS
        入力の例の要素数の上限. 超える場合は計算しない (大きすぎる形で
        メモリを使い切らないため)

    Returns
    ----------
    result: dict
        ok (全て計算できたか), issues (定義の誤りと注意), nodes (ノードごとの
        status, outputs, params, trainable, error), outputs (モデルの出力の
        形), params, trainable (全体), estimated_only (上限を超えて計算
        しなかったか)
    """
    # 定義の誤りがある場合と, 形の無い入力がある場合は計算なし
    # 学習済みの重みを読む部品は読まない値で作成 (形は同じで, 大きな重みの
    # 取得と読み込みを避けるため)
    # (部分モデルの中のノードも)
    spec = load_spec(spec)
    # 引数の組み合わせの誤り (学習済みの重みとクラスの数など) は, 重みを
    # 読まない値に置き換える前の定義で確かめる
    combinations = [
        issue for issue in analyze(spec).issues
        if issue.code == "invalid_combination"
    ]
    subgraphs = spec.get("subgraphs")
    graphs = [spec, *(
        subgraphs.values() if isinstance(subgraphs, dict) else []
    )]
    for graph in graphs:
        nodes = graph.get("nodes") if isinstance(graph, dict) else None
        for node in (nodes if isinstance(nodes, dict) else {}).values():
            if not isinstance(node, dict):
                continue
            offline = offline_values(str(node.get("unit", "")))
            if offline and isinstance(node.get("params", {}) or {}, dict):
                node["params"] = {**(node.get("params") or {}), **offline}
    analysis = analyze(spec)
    analysis.issues.extend(combinations)
    result: dict = {
        "ok": False, "issues": [vars(issue) for issue in analysis.issues],
        "nodes": {}, "outputs": {}, "params": None, "trainable": None,
        "estimated_only": False
    }
    no_shape = [
        name for name, value in spec["inputs"].items()
        if not isinstance(value, dict) or value.get("shape") is None
    ]
    # 組み立ての途中でも形を確かめられるように, 誤りのあるノードとその先
    # だけを除いて計算 (全体の ok は False のまま). 出力が無いことと出力の
    # つなぎ先の誤り以外の, ノードに付かない誤り (入力の形の書き方など) が
    # ある場合は計算しない. 部分モデルの中の誤りは, それを使うノードの誤り
    # (使えない部品) になるので止めない
    blocking = [
        issue for issue in analysis.issues
        if issue.severity == "error" and issue.node is None
        and issue.code != "no_outputs"
        and not str(issue.location or "").startswith(
            ("outputs", "subgraphs")
        )
    ]
    if blocking or no_shape:
        return result
    # 誤りのあるノード (誤りの文をつなぐ) と, 誤りのある出力
    bad: dict[str, str] = {}
    bad_outputs: dict[str, str] = {}
    for issue in analysis.issues:
        if issue.severity != "error":
            continue
        location = str(issue.location or "")
        if issue.node is not None:
            node = str(issue.node)
            bad[node] = f"{bad[node]} / {issue.message}" if node in bad else (
                issue.message
            )
        elif location.startswith("outputs."):
            bad_outputs[location.split(".", 2)[1]] = issue.message
    for name, value in spec["inputs"].items():
        count = 1
        for size in value["shape"]:
            count *= size
        if count > max_input_elements:
            result["issues"].append(vars(Issue(
                "warning", "input_too_large",
                f"入力 {name} の例の要素数 ({count}) が上限 "
                f"({max_input_elements}) を超えるので, 形を推定しません",
                location=f"inputs.{name}.shape"
            )))
            return result

    # 先にパラメータ数を数え, 上限を超える場合は計算なし
    # 数えられない場合 (meta デバイスで作れない部品) は注意を加えて計算
    # 重みを作らずに (meta デバイスで) 組み立てられる部分のパラメータ数を
    # 数え, それだけで上限を超える場合は計算なし. meta デバイスで作れない
    # 部品 (作るときに値を読むものなど) は数えずに注意を加える
    counted = dict(bad)
    try:
        with torch.device("meta"):
            meta_model = build_usable(spec, analysis, counted)[0]
        estimate: Optional[int] = count_params(meta_model, set())[0]
    except Exception:
        estimate = None
    if estimate is not None and estimate > max_params:
        result["params"] = estimate
        result["estimated_only"] = True
        return result
    if estimate is None or set(counted) - set(bad):
        result["issues"].append(vars(Issue(
            "warning", "no_estimate",
            "パラメータ数を前もって数えられない部品があるので, 上限を確かめ"
            "ずに計算します"
        )))

    # cpu で組み立て, 入力の例をノードごとに計算
    # 部品を作れないノード (引数の組み合わせの誤りなど) は, そのノードの
    # エラーにして除き, 組み立て直し
    with torch.device("cpu"), torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model, usable, skipped = build_usable(spec, analysis, bad)
        model.eval()
        run_nodes(model, spec, result, bad_outputs)

    # 誤りのあるノードとその先のノードの結果
    for name, message in bad.items():
        if name in spec["nodes"]:
            result["nodes"][name] = {
                "status": "error", "outputs": {}, "params": 0,
                "trainable": 0, "error": message
            }
    for name, cause in skipped.items():
        result["nodes"][name] = {
            "status": "skipped", "outputs": {}, "params": 0, "trainable": 0,
            "error": f"前のノード {cause} の誤りで計算できません"
        }
    for name, ref in spec["outputs"].items():
        target = str(ref).partition(":")[0]
        if str(name) not in bad_outputs and (
            target in bad or target in skipped
        ):
            result["outputs"].pop(str(name), None)

    # ノードは定義に書いた順に並べ, 誤りがあれば全体は ok ではない
    result["nodes"] = {
        str(name): result["nodes"][str(name)] for name in spec["nodes"]
        if str(name) in result["nodes"]
    }
    if not analysis.ok or bad or skipped:
        result["ok"] = False

    # 読み込むまで確かめられない注意 (Hugging Face のモデルの名前) は,
    # 読み込んで計算できたノードでは除く
    computed = {
        name for name, entry in result["nodes"].items()
        if entry["status"] == "ok"
    }
    result["issues"] = [
        issue for issue in result["issues"]
        if not (issue["code"] == "unchecked_model"
                and issue["node"] in computed)
    ]

    return result


def build_usable(
    spec: dict, analysis: Analysis, bad: dict[str, str]
) -> tuple[GraphModel, Analysis, dict[str, str]]:
    """
    計算できる部分のモデルを組み立てる関数

    部品を作れないノードは, そのノードの誤りとして bad に加えて除き,
    組み立て直す

    Parameters
    ----------
    spec: dict
        モデルの定義
    analysis: Analysis
        定義の解析の結果
    bad: dict[str, str]
        誤りのあるノードの名前ごとの誤りの文, その場で書き加える

    Returns
    ----------
    model: GraphModel
        計算できる部分のモデル
    usable: Analysis
        計算できる部分の解析
    skipped: dict[str, str]
        除いた先のノードの名前ごとの, 計算できない前のノードの名前
    """
    # 1 回ごとに計算の順が 1 つ以上減るので, 必ず終わる
    usable, skipped = usable_analysis(analysis, bad)
    while True:
        try:
            model = GraphModel(spec, usable)
            break
        except NodeBuildError as error:
            bad[error.node] = str(error)
            usable, skipped = usable_analysis(analysis, bad)

    return model, usable, skipped


def usable_analysis(
    analysis: Analysis, bad: dict[str, str]
) -> tuple[Analysis, dict[str, str]]:
    """
    誤りのあるノードとその先のノードを除いた, 計算できる部分の解析を作る
    関数

    Parameters
    ----------
    analysis: Analysis
        定義の解析の結果
    bad: dict[str, str]
        誤りのあるノードの名前ごとの誤りの文

    Returns
    ----------
    usable: Analysis
        計算の順から除いた解析
    skipped: dict[str, str]
        除いた先のノードの名前ごとの, 計算できない前のノードの名前
    """
    # 計算の順に, つなぎ先に除いたノードがあれば除外
    skipped: dict[str, str] = {}
    order = []
    for name in analysis.order:
        if name in bad or name not in analysis.units:
            continue
        # つなぎ先と共有の元 (作られないと共有できない) の確認
        causes = [
            target
            for targets in analysis.connections.get(name, {}).values()
            for target, _ in targets
            if target in bad or target in skipped
        ] + [
            source for source in analysis.shares.get(name, [])
            if source in bad or source in skipped
        ]
        if causes:
            skipped[name] = causes[0]
            continue
        order.append(name)
    usable = replace(analysis, order=order)

    return usable, skipped


def run_nodes(
    model: GraphModel, spec: dict, result: dict,
    bad_outputs: Optional[dict[str, str]] = None
) -> None:
    """
    組み立てたモデルのノードを 1 つずつ計算し, 結果を書き込む関数

    Parameters
    ----------
    model: GraphModel
        組み立てたモデル
    spec: dict
        モデルの定義
    result: dict
        infer の結果, その場で書き込む
    bad_outputs: Optional[dict[str, str]] = None
        確認で誤りのある出力の名前ごとの誤りの文 (計算せずにエラー)

    Returns
    ----------
    None
    """
    # 入力の例を作り, 計算の順にノードを計算
    values: dict[str, Any] = example_input(spec)
    failed: dict[str, str] = {}
    seen: set[int] = set()
    total = 0
    trainable = 0
    with torch.no_grad():
        for name in model.order:
            params, learn = count_params(model.nodes[name], seen)
            total += params
            trainable += learn
            entry: dict = {
                "status": "ok", "outputs": {}, "params": params,
                "trainable": learn, "error": None
            }
            result["nodes"][name] = entry

            # 前のノードが失敗した場合は計算なし
            upstream = [
                target for targets in model.connections[name].values()
                for target, _ in targets if target in failed
            ]
            if upstream:
                entry["status"] = "skipped"
                entry["error"] = (
                    f"前のノード {upstream[0]} の誤りで計算できません"
                )
                failed[name] = entry["error"]
                continue

            # 計算して出力の口ごとの形を記録, 失敗したらエラーを記録
            try:
                values[name] = model.run_node(name, values)
                for port, path in model.paths[name].items():
                    entry["outputs"][port] = describe(
                        follow_path(
                            values[name], path, name in model.lenient
                        )
                    )
            except Exception as error:
                entry["status"] = "error"
                entry["error"] = describe_error(error) + incoming_text(
                    model, name, values
                ) + shape_hint(model, spec, name, values)
                failed[name] = entry["error"]
    result["params"] = total
    result["trainable"] = trainable

    # モデルの出力の形 (取り出せない場合はエラー)
    output_errors = False
    for name, ref in spec["outputs"].items():
        if str(name) in (bad_outputs or {}):
            output_errors = True
            result["outputs"][str(name)] = {
                "kind": "error", "error": (bad_outputs or {})[str(name)]
            }
            continue
        ref = str(ref)
        target = ref.partition(":")[0]
        if target in failed:
            output_errors = True
            continue
        try:
            result["outputs"][str(name)] = describe(model.resolve(
                values, (target, ref.partition(":")[2] or None)
            ))
        except Exception as error:
            output_errors = True
            result["outputs"][str(name)] = {
                "kind": "error", "error": f"{type(error).__name__}: {error}"
            }
    result["ok"] = not failed and not output_errors
