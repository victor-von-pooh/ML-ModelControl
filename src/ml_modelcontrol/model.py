from pathlib import Path
from typing import Any, Optional, Union

import torch
from torch import nn

from ml_modelcontrol.spec import load_spec, parse_ref
from ml_modelcontrol.validate import Analysis, analyze


class SpecError(ValueError):
    def __init__(self, issues: list[dict]):
        # 誤りの一覧の説明の文への変換
        errors = [issue for issue in issues if issue["severity"] == "error"]
        lines = [
            f"{issue['node'] or '定義'}: {issue['message']}" for issue in errors
        ]

        # 親クラスのコンストラクタを呼び出す
        super(SpecError, self).__init__(
            "モデルの定義に誤りがあります:\n" + "\n".join(lines)
        )

        # 誤りと注意の一覧を保存
        self.issues = issues


class NodeBuildError(ValueError):
    def __init__(self, node: str, error: Exception):
        # 親クラスのコンストラクタを呼び出す
        super(NodeBuildError, self).__init__(
            f"ノード {node} の部品を作れません: {type(error).__name__}: {error}"
        )

        # 作れなかったノードの名前と元のエラーを保存
        self.node = node
        self.error = error


def to_arg(value: Any) -> Any:
    """
    定義の引数の値 (JSON の値) を, 部品に渡す値に直す関数

    Parameters
    ----------
    value: Any
        定義に書いた値

    Returns
    ----------
    arg: Any
        リストはタプルに (kernel_size などの組), それ以外はそのまま
    """
    if isinstance(value, list):
        arg = tuple(to_arg(item) for item in value)
    else:
        arg = value

    return arg


def follow_path(value: Any, path: list) -> Any:
    """
    出力の値から, 口の取り出し方 (番号か名前の並び) で値を取り出す関数

    Parameters
    ----------
    value: Any
        部品の戻り値
    path: list
        番号 (タプルの要素) か名前 (属性かキー) の並び

    Returns
    ----------
    item: Any
        取り出した値
    """
    # 番号は要素, 名前は辞書ならキー (無いキーは None. Hugging Face の出力は
    # モデルによって pooler_output などを持たない), それ以外は属性
    item = value
    for step in path:
        if isinstance(step, int):
            item = item[step]
        elif isinstance(item, dict):
            item = item.get(step)
        else:
            item = getattr(item, step)

    return item


# 定義の部品を計算の順につないだモデル
class GraphModel(nn.Module):
    def __init__(self, spec: dict, analysis: Analysis):
        # 親クラスのコンストラクタを呼び出す
        super(GraphModel, self).__init__()

        # ノードの部品を作成 (名前は定義のノードの名前)
        self.nodes = nn.ModuleDict()
        for name in analysis.order:
            node = spec["nodes"][name]
            params = {
                key: to_arg(value)
                for key, value in (node.get("params") or {}).items()
            }
            try:
                self.nodes[name] = analysis.units[name].builder(**params)
            except Exception as error:
                raise NodeBuildError(name, error) from error

        # 計算の順, 入力と出力の名前, つなぎ先, 口の順, 設定の値を保存
        self.order = list(analysis.order)
        self.input_names = [str(name) for name in spec["inputs"]]
        self.outputs = {
            str(name): ref for name, ref in spec["outputs"].items()
        }
        self.connections = analysis.connections
        self.ports = {
            name: list(analysis.units[name].inputs) for name in self.order
        }
        self.paths = {
            name: {
                port.name: list(port.path)
                for port in analysis.units[name].outputs
            }
            for name in self.order
        }
        self.forward_params = {
            name: {
                key: to_arg(value) for key, value in (
                    spec["nodes"][name].get("forward_params") or {}
                ).items()
            }
            for name in self.order
        }

    def bind_inputs(self, args: tuple, kwargs: dict) -> dict[str, Any]:
        """
        forward の引数を, 入力の名前ごとの値に対応させる関数

        Parameters
        ----------
        args: tuple
            定義の順の入力の値
        kwargs: dict
            名前を付けた入力の値

        Returns
        ----------
        values: dict[str, Any]
            入力の名前ごとの値
        """
        # 位置で渡した値の対応付けと, 名前で渡した値の追加
        if len(args) > len(self.input_names):
            raise TypeError(
                f"入力は {len(self.input_names)} 個ですが, {len(args)} 個の値が"
                "渡されました"
            )
        values = dict(zip(self.input_names, args))
        for key, value in kwargs.items():
            if key not in self.input_names:
                raise TypeError(f"入力 {key} はありません")
            if key in values:
                raise TypeError(f"入力 {key} が 2 回渡されました")
            values[key] = value
        missing = [name for name in self.input_names if name not in values]
        if missing:
            raise TypeError(f"入力 {missing} が渡されていません")

        return values

    def resolve(self, values: dict[str, Any], target: tuple) -> Any:
        """
        つなぎ先 (名前と出力の口の名前) の値を取り出す関数

        Parameters
        ----------
        values: dict[str, Any]
            入力とノードの戻り値
        target: tuple
            名前と出力の口の名前 (口の名前が無い場合は None)

        Returns
        ----------
        value: Any
            つなぎ先の値
        """
        # 入力はそのまま, ノードは口の取り出し方で取り出し
        name, port = target
        value = values[name]
        if name in self.paths:
            paths = self.paths[name]
            port_name = port if port is not None else next(iter(paths))
            value = follow_path(value, paths[port_name])

        return value

    def run_node(self, name: str, values: dict[str, Any]) -> Any:
        """
        1 つのノードを計算する関数

        *args より前の口と位置だけの口は位置で, 可変の数の口は位置で並べ,
        それ以外の口と呼び出すときの設定の値はキーワードで渡す

        Parameters
        ----------
        name: str
            ノードの名前
        values: dict[str, Any]
            入力と, 前に計算したノードの戻り値

        Returns
        ----------
        output: Any
            部品の戻り値
        """
        # 口ごとのつなぎ先の値を集めて呼び出し
        # 位置で渡す口 (*args より前) は省略しても None を置き, 可変の数の口は
        # 位置で並べ, それ以外はキーワードで受け渡し
        connections = self.connections[name]
        args: list[Any] = []
        kwargs: dict[str, Any] = dict(self.forward_params[name])
        for port in self.ports[name]:
            items = [
                self.resolve(values, target)
                for target in connections.get(port.name, [])
            ]
            if port.variadic:
                args.extend(items)
            elif port.positional:
                args.append(items[0] if items else None)
            elif items:
                kwargs[port.name] = items[0]
        output = self.nodes[name](*args, **kwargs)

        return output

    def forward(self, *args: Any, **kwargs: Any) -> Union[Any, dict[str, Any]]:
        """
        入力から全てのノードを計算の順に計算し, 出力を返す関数

        Parameters
        ----------
        *args: Any
            定義の順の入力の値
        **kwargs: Any
            名前を付けた入力の値

        Returns
        ----------
        result: Union[Any, dict[str, Any]]
            出力が 1 つならその値, 2 つ以上なら出力の名前ごとの値の辞書
        """
        # 入力を対応させ, ノードを順に計算
        values = self.bind_inputs(args, kwargs)
        for name in self.order:
            values[name] = self.run_node(name, values)

        # 出力の取り出し
        result = {
            name: self.resolve(values, parse_ref(ref))
            for name, ref in self.outputs.items()
        }
        if len(result) == 1:
            result = next(iter(result.values()))

        return result


def build_model(
    spec: Union[str, Path, dict], seed: Optional[int] = None
) -> GraphModel:
    """
    モデルの定義から nn.Module を組み立てる関数

    Parameters
    ----------
    spec: Union[str, Path, dict]
        モデルの定義, またはその YAML か JSON のファイルのパス
    seed: Optional[int] = None
        重みの初期値の乱数の種, None の場合は今の乱数の状態のまま

    Returns
    ----------
    model: GraphModel
        組み立てたモデル

    Raises
    ----------
    SpecError
        定義に誤りがある場合
    """
    # 定義を読み込んで確かめ, 誤りがあれば組み立てずにエラー
    spec = load_spec(spec)
    analysis = analyze(spec)
    if not analysis.ok:
        raise SpecError([vars(issue) for issue in analysis.issues])

    # 種を決めた場合は, 外の乱数の状態を変えずに初期値を決めて組み立て
    if seed is None:
        model = GraphModel(spec, analysis)
        return model
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = GraphModel(spec, analysis)

    return model
