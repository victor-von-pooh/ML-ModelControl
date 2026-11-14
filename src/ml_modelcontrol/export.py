import keyword
from pathlib import Path
from typing import Any, Optional, Union

from ml_modelcontrol.model import SpecError, to_arg
from ml_modelcontrol.spec import load_spec, parse_ref
from ml_modelcontrol.validate import analyze

# 書き出すコードで変数の名前に使わない名前 (読み込むモジュールなど)
RESERVED_VARS = {"self", "torch", "nn", "ops", "result"}


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
        変数の名前 (読み込むモジュールなどと重なる場合は末尾に _)
    """
    var = f"{name}_" if name in RESERVED_VARS or keyword.iskeyword(name) else (
        name
    )

    return var


def path_code(path: list) -> str:
    """
    出力の口の取り出し方を, 添字の式にする関数

    Parameters
    ----------
    path: list
        番号か名前の並び

    Returns
    ----------
    code: str
        [0][1] や ["last_hidden_state"] のような式
    """
    code = "".join(f"[{step!r}]" for step in path)

    return code


def export_python(
    spec: Union[str, Path, dict], class_name: str = "Model",
    path: Optional[Union[str, Path]] = None
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

    # 読み込むモジュール, 部品の作成, 計算の行を組み立て
    imports = ["import torch", "from torch import nn"]
    for name in analysis.order:
        line = analysis.units[name].code_import
        if line and line not in imports:
            imports.append(line)
    lines = [
        f'"""{spec.get("name", class_name)} のモデル (ML-ModelControl の定義から'
        '書き出したコード)"""',
        *imports, "", "",
        f"class {class_name}(nn.Module):",
        "    def __init__(self):",
        "        # 親クラスのコンストラクタを呼び出す",
        f"        super({class_name}, self).__init__()", "",
        "        # ノードの部品の作成",
        "        self.nodes = nn.ModuleDict({"
    ]
    for name in analysis.order:
        code = build_code(spec, analysis, name)
        lines.append(f"            {name!r}: {code},")
    lines.append("        })")
    lines += forward_code(spec, analysis)
    code = "\n".join(lines) + "\n"
    if path is not None:
        Path(path).write_text(code, encoding="utf-8")

    return code


def build_code(spec: dict, analysis: Any, name: str) -> str:
    """
    1 つのノードの部品を作る式を作る関数

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
    code: str
        nn.Linear(in_features=16, out_features=3) のような式
    """
    # 位置の引数 (Hugging Face のモデルの種類など) と, 名前を付けた引数
    unit = analysis.units[name]
    params = spec["nodes"][name].get("params") or {}
    args = [repr(value) for value in unit.code_args]
    args += [f"{key}={to_arg(value)!r}" for key, value in params.items()]
    code = f"{unit.code_name}({', '.join(args)})"

    return code


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
        lines.append(
            f"        {var_name(name)} = self.nodes[{name!r}]"
            f"({call_args(spec, analysis, name)})"
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
        code += path_code(output.path)

    return code


def call_args(spec: dict, analysis: Any, name: str) -> str:
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
    code: str
        x, edge_index=edge_index のような引数の式
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
    code = ", ".join(args + kwargs)

    return code
