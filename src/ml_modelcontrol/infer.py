from collections.abc import Mapping
from typing import Any, Optional, Union

import torch
from torch import nn

from ml_modelcontrol.model import GraphModel, follow_path
from ml_modelcontrol.spec import load_spec
from ml_modelcontrol.validate import analyze

# 形の推定で計算する前に確かめる, パラメータ数の上限の既定値
MAX_PARAMS = 200_000_000


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


def estimate_params(spec: dict, analysis: Any) -> Optional[int]:
    """
    重みの値を作らずに (meta デバイスで), モデルのパラメータ数を数える関数

    Parameters
    ----------
    spec: dict
        モデルの定義
    analysis: Any
        定義の解析の結果

    Returns
    ----------
    total: Optional[int]
        パラメータ数, meta デバイスで作れない部品がある場合は None
    """
    # 作れない部品がある場合は数えずに返す
    try:
        with torch.device("meta"):
            model = GraphModel(spec, analysis)
    except Exception:
        return None
    total, _ = count_params(model, set())

    return total


def infer(
    spec: Union[str, dict], max_params: int = MAX_PARAMS
) -> dict:
    """
    入力の例をモデルに通し, ノードごとの出力の形とパラメータ数を求める関数

    cpu で eval() と勾配なしで, ノードを 1 つずつ計算する. 失敗したノードは
    エラーを, その先のノードは計算できないことを返す

    Parameters
    ----------
    spec: Union[str, dict]
        モデルの定義, またはその YAML か JSON のファイルのパス
    max_params: int = MAX_PARAMS
        計算する前に確かめるパラメータ数の上限. 超える場合は計算しない

    Returns
    ----------
    result: dict
        ok (全て計算できたか), issues (定義の誤りと注意), nodes (ノードごとの
        status, outputs, params, trainable, error), outputs (モデルの出力の
        形), params, trainable (全体), estimated_only (上限を超えて計算
        しなかったか)
    """
    # 定義の誤りがある場合は計算しない
    spec = load_spec(spec)
    analysis = analyze(spec)
    result: dict = {
        "ok": False, "issues": [vars(issue) for issue in analysis.issues],
        "nodes": {}, "outputs": {}, "params": None, "trainable": None,
        "estimated_only": False
    }
    if not analysis.ok:
        return result

    # 先にパラメータ数を数え, 上限を超える場合は計算しない
    estimate = estimate_params(spec, analysis)
    if estimate is not None and estimate > max_params:
        result["params"] = estimate
        result["estimated_only"] = True
        return result

    # cpu で組み立て, 入力の例をノードごとに計算
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = GraphModel(spec, analysis)
    model.eval()
    run_nodes(model, spec, result)

    return result


def run_nodes(model: GraphModel, spec: dict, result: dict) -> None:
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

            # 前のノードが失敗した場合は計算しない
            upstream = [
                target for targets in model.connections[name].values()
                for target, _ in targets if target in failed
            ]
            if upstream:
                entry["status"] = "skipped"
                entry["error"] = f"前のノード {upstream[0]} が計算できません"
                failed[name] = entry["error"]
                continue

            # 計算して出力の口ごとの形を記録, 失敗したらエラーを記録
            try:
                values[name] = model.run_node(name, values)
                for port, path in model.paths[name].items():
                    entry["outputs"][port] = describe(
                        follow_path(values[name], path)
                    )
            except Exception as error:
                entry["status"] = "error"
                entry["error"] = f"{type(error).__name__}: {error}"
                failed[name] = entry["error"]
    result["params"] = total
    result["trainable"] = trainable

    # モデルの出力の形 (計算できた場合)
    for name, ref in spec["outputs"].items():
        target = ref.partition(":")[0]
        if target in failed:
            continue
        try:
            result["outputs"][str(name)] = describe(model.resolve(
                values, (target, ref.partition(":")[2] or None)
            ))
        except Exception as error:
            failed[str(name)] = f"{type(error).__name__}: {error}"
    result["ok"] = not failed
