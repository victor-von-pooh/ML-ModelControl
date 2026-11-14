import keyword
from pathlib import Path
from typing import Any, Optional, Union

from ml_modelcontrol.model import SpecError, to_arg
from ml_modelcontrol.spec import load_spec, parse_ref
from ml_modelcontrol.validate import analyze

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
    "    # 辞書はキー (無いキーは None), それ以外は属性",
    "    if isinstance(value, dict):",
    "        return value.get(name)",
    "    return getattr(value, name)",
    "", ""
]


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

    # 読み込むモジュールと, 名前の重なりの確認
    imports = ["from torch import nn"]
    for name in analysis.order:
        line = analysis.units[name].code_import
        if line and line not in imports:
            imports.append(line)
    check_names(spec, analysis, class_name)

    # モデルの名前は文字の値として書き, コードとして解釈されないようにする
    title = (
        f"{spec.get('name', class_name)} のモデル (ML-ModelControl の定義から"
        "書き出したコード)"
    )
    lines = [
        repr(title),
        *imports, "", "",
        f"class {class_name}(nn.Module):",
        "    def __init__(self):",
        "        # 親クラスのコンストラクタを呼び出す",
        f"        super({class_name}, self).__init__()", "",
        "        # ノードの部品の作成",
        "        self.nodes = nn.ModuleDict({"
    ]
    for name in analysis.order:
        lines += wrap_call(
            f"            {name!r}: {analysis.units[name].code_name}(",
            build_args(spec, analysis, name), "),", "            "
        )
    lines.append("        })")
    forward = forward_code(spec, analysis)
    if any("_pick(" in line for line in forward):
        start = lines.index(f"class {class_name}(nn.Module):")
        lines[start:start] = PICK_CODE
    lines += forward
    code = "\n".join(lines) + "\n"
    if path is not None:
        Path(path).write_text(code, encoding="utf-8")

    return code


def check_names(spec: dict, analysis: Any, class_name: str) -> None:
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
    class_name: str
        書き出すクラスの名前

    Returns
    ----------
    None
    """
    # 読み込む名前 (nn, ops, 部品のクラス) と, 予約した変数の名前
    imported: dict[str, str] = {"nn": "torch.nn"}
    for name in analysis.order:
        unit = analysis.units[name]
        code_name = unit.code_name.split(".")[0]
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
    if class_name in reserved or class_name in spec["inputs"] or (
        class_name in analysis.order
    ):
        raise ValueError(
            f"クラスの名前 {class_name} は他の名前と重なるので使えません"
        )


def build_args(spec: dict, analysis: Any, name: str) -> list[str]:
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

    Returns
    ----------
    args: list[str]
        in_features=16 のような引数の式の並び
    """
    # 位置の引数 (Hugging Face のモデルの種類など) と, 名前を付けた引数
    unit = analysis.units[name]
    params = spec["nodes"][name].get("params") or {}
    args = [repr(value) for value in unit.code_args]
    args += [f"{key}={to_arg(value)!r}" for key, value in params.items()]

    return args


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
