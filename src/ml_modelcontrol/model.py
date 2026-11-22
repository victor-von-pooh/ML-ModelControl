from pathlib import Path
from typing import Any, Optional, Union

import torch
from torch import nn

from ml_modelcontrol.registry import get_unit
from ml_modelcontrol.spec import load_spec, parse_ref
from ml_modelcontrol.state import (
    LOAD_CONTEXT, load_node_weights, make_context, share_targets,
    tie_parameter, zero_parameters
)
from ml_modelcontrol.validate import MAX_UNIT_DEPTH, Analysis, analyze


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


def to_arg(value: Any, depth: int = 0) -> Any:
    """
    定義の引数の値 (JSON の値) を, 部品に渡す値に直す関数

    Parameters
    ----------
    value: Any
        定義に書いた値
    depth: int = 0
        引数の部品の入れ子の深さ (確かめていない値で限りなくたどらない
        ように, 上限を超えたら止める)

    Returns
    ----------
    arg: Any
        リストはタプルに (kernel_size などの組), 部品の引数 (unit と
        params の辞書) は作った部品に, それ以外はそのまま
    """
    # 確かめ済みの定義で辞書の値は, 部品の引数だけ
    if depth > MAX_UNIT_DEPTH:
        raise ValueError(
            f"引数の部品の入れ子が深すぎます ({MAX_UNIT_DEPTH} 段まで)"
        )
    if isinstance(value, list):
        arg = tuple(to_arg(item, depth) for item in value)
    elif isinstance(value, dict):
        params = {
            key: to_arg(item, depth + 1)
            for key, item in (value.get("params") or {}).items()
        }
        arg = get_unit(str(value["unit"])).builder(**params)
    else:
        arg = value

    return arg


def follow_path(value: Any, path: list, lenient: bool = False) -> Any:
    """
    出力の値から, 口の取り出し方 (番号か名前の並び) で値を取り出す関数

    Parameters
    ----------
    value: Any
        部品の戻り値
    path: list
        番号 (タプルの要素) か名前 (属性かキー) の並び
    lenient: bool = False
        辞書に無いキーを None にするか (Hugging Face の出力はモデルによって
        pooler_output などを持たないため). False なら KeyError

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
            # 辞書に無いキーでも同じ名前の属性があれば属性 (Hugging Face の
            # ModelOutput は値が None の項目をキーに持たない)
            if step in item:
                item = item[step]
            elif hasattr(item, step):
                item = getattr(item, step)
            elif lenient:
                item = None
            else:
                raise KeyError(f"戻り値の辞書にキー {step!r} がありません")
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
        # 無いキーを None にするノード (Hugging Face の出力)
        self.lenient = {
            name for name in self.order
            if analysis.units[name].source == "transformers"
        }
        # 誤りの文に添える部品の名前
        self.unit_names = {
            name: analysis.units[name].name for name in self.order
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

        # パラメータの状態 (版 2. 外部の重み, 零初期化, 凍結, 共有, 評価の
        # 形) の適用
        self.eval_nodes: list[str] = []
        self.weights_report: list[dict] = []
        self.apply_state(spec["nodes"])

    def apply_state(self, nodes: dict) -> None:
        """
        ノードのパラメータの状態 (版 2 の項目) を適用する関数

        外部の重み, 零初期化, 凍結, 共有, 評価の形の順に行う

        Parameters
        ----------
        nodes: dict
            定義のノード

        Returns
        ----------
        None

        Raises
        ----------
        NodeBuildError
            外部の重みを読めない場合, パラメータの名前が無い場合, 共有の
            形が違う場合
        """
        # ノードごとの外部の重み (build_model の中だけ), 零初期化, 凍結
        context = LOAD_CONTEXT.get()
        for name in self.order:
            node = nodes[name]
            module = self.nodes[name]
            try:
                if context is not None and "weights" in node:
                    share = node.get("share")
                    self.weights_report.append(load_node_weights(
                        name, module, node["weights"], context,
                        set(share) if isinstance(share, dict) else set()
                    ))
                if node.get("zero_init"):
                    zero_parameters(module, node["zero_init"])
                if node.get("train") is False:
                    for parameter in module.parameters():
                        parameter.requires_grad_(False)
            except Exception as error:
                raise NodeBuildError(name, error) from error

            # 部分モデルの中で読んだ重みの報告 (ノードの名前を . でつなぐ)
            if isinstance(module, GraphModel) and "share" not in node:
                self.weights_report += [
                    {**item, "node": f"{name}.{item['node']}"}
                    for item in module.weights_report
                ]

        # 共有 (ノード全体は同じ部品, パラメータごとは同じパラメータ)
        for name in self.order:
            share = nodes[name].get("share")
            if share is None:
                continue
            try:
                self.share_node(name, share)
            except Exception as error:
                raise NodeBuildError(name, error) from error

        # 評価の形に保つノード
        self.eval_nodes = [
            name for name in self.order
            if nodes[name].get("eval_mode") is True
        ]
        self.train(self.training)

    def share_node(self, name: str, share: Any) -> None:
        """
        ノードの部品かパラメータを, 共有の元のノードのものにする関数

        Parameters
        ----------
        name: str
            共有するノードの名前
        share: Any
            元のノードの名前 (ノード全体) か, パラメータの名前と
            <ノード>.<パラメータの名前> の辞書

        Returns
        ----------
        None

        Raises
        ----------
        ValueError
            元のノードを作れなかった場合 (形の推定で誤りのあるノード) と,
            パラメータが無いか形が違う場合
        """
        # ノード全体は部品の入れ替え, パラメータごとは属性の入れ替え
        if isinstance(share, str):
            if share not in self.nodes:
                raise ValueError(
                    f"共有の元のノード {share} は作られていません (その"
                    "ノードの誤りを先に直してください)"
                )
            self.nodes[name] = self.nodes[share]
            return
        for own, source, source_name in share_targets(share):
            if source not in self.nodes:
                raise ValueError(
                    f"共有の元のノード {source} は作られていません (その"
                    "ノードの誤りを先に直してください)"
                )
            tie_parameter(
                self.nodes[name], own, self.nodes[source], source_name
            )

    def train(self, mode: bool = True) -> "GraphModel":
        """
        学習の形か評価の形にする関数

        eval_mode のノードは, 学習の形にしても評価の形のままにする

        Parameters
        ----------
        mode: bool = True
            学習の形にするか

        Returns
        ----------
        model: GraphModel
            このモデル
        """
        # 全体の切り替えと, 評価の形に保つノードの戻し
        super(GraphModel, self).train(mode)
        for name in self.eval_nodes:
            self.nodes[name].train(False)
        model = self

        return model

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
            入力とノードの戻り値 (compute では, 渡した口の値も
            「ノード:口」のキーで持つ)
        target: tuple
            名前と出力の口の名前 (口の名前が無い場合は None)

        Returns
        ----------
        value: Any
            つなぎ先の値
        """
        # 入力はそのまま, ノードは口の取り出し方で取り出し
        # (compute で口の値を渡した場合はその値)
        name, port = target
        paths = self.paths.get(name)
        port_name = None
        if paths is not None:
            port_name = port if port is not None else next(iter(paths))
        if paths is None:
            value = values[name]
        elif f"{name}:{port_name}" in values:
            value = values[f"{name}:{port_name}"]
        else:
            # 取り出せない場合は, どのノードのどの口かを添える
            try:
                value = follow_path(
                    values[name], paths[port_name], name in self.lenient
                )
            except (KeyError, IndexError, AttributeError) as error:
                raise ValueError(
                    f"ノード {name} ({self.unit_names.get(name, '?')}) の口 "
                    f"{port_name} を取り出せません: {error}"
                ) from error

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
        # 部品の中の誤りへの, 起きたノードの名前の付加
        try:
            output = self.nodes[name](*args, **kwargs)
        except Exception as error:
            if hasattr(error, "add_note"):
                error.add_note(
                    f"ノード {name} ({self.unit_names.get(name, '?')}) で起きた"
                    "誤り"
                )
            raise

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

    def port_key(self, ref: str) -> str:
        """
        compute に渡した値の名前を, 入力の名前か「ノード:口」に直す関数

        Parameters
        ----------
        ref: str
            「入力の名前」「ノードの名前」(最初の出力の口) か
            「ノードの名前:口の名前」

        Returns
        ----------
        key: str
            入力の名前か「ノードの名前:口の名前」

        Raises
        ----------
        ValueError
            入力にもノードにも無い名前と, ノードに無い口の場合
        """
        # 入力はそのまま, ノードは口の名前を補ったもの
        name, port = parse_ref(str(ref))
        if port is None and name in self.input_names:
            return name
        if name not in self.paths:
            raise ValueError(
                f"入力にもノードにも {ref} はありません"
            )
        ports = list(self.paths[name])
        port_name = port if port is not None else ports[0]
        if port_name not in ports:
            raise ValueError(
                f"ノード {name} に出力の口 {port_name} はありません "
                f"(口は {', '.join(ports)})"
            )
        key = f"{name}:{port_name}"

        return key

    def needed_nodes(self, outputs: list[str], given: set[str]) -> list[str]:
        """
        求める出力に要るノードを, 計算の順に返す関数

        出力のつなぎ先から前へたどり, 値を渡した入力と口の所で止める

        Parameters
        ----------
        outputs: list[str]
            求める出力の名前 (定義の出力にあるもの)
        given: set[str]
            値を渡した入力の名前と「ノード:口」(port_key で直したもの)

        Returns
        ----------
        needed: list[str]
            計算するノードの名前 (計算の順)

        Raises
        ----------
        ValueError
            要る入力の値が渡されていない場合
        """
        # 出力ごとに前へたどり, 要るノードと渡されていない入力を集約
        needed_set: set[str] = set()
        missing: dict[str, list[str]] = {}
        for output in outputs:
            visited: set[str] = set()
            stack = [parse_ref(self.outputs[output])]
            while stack:
                name, port = stack.pop()
                if name in self.input_names:
                    if name not in given and name not in visited:
                        missing.setdefault(output, []).append(name)
                    visited.add(name)
                    continue
                port_name = port if port is not None else next(
                    iter(self.paths[name])
                )
                if f"{name}:{port_name}" in given or name in visited:
                    continue
                visited.add(name)
                needed_set.add(name)
                for targets in self.connections[name].values():
                    stack.extend(targets)

        # 渡されていない入力の, 出力ごとの案内
        if missing:
            lines = [
                f"出力 {output} の計算に要る入力 "
                f"{', '.join(sorted(names))} が渡されていません"
                for output, names in missing.items()
            ]
            raise ValueError(
                "\n".join(lines) + "\n(途中のノードの値を渡す場合は, その"
                "ノードの名前か「ノード:口」をキーにし, 要らない出力は "
                "outputs で除いてください)"
            )

        # 要るノードの計算の順への並べ替え
        needed = [name for name in self.order if name in needed_set]

        return needed

    def compute(
        self, values: dict[str, Any],
        outputs: Optional[Union[str, list[str]]] = None
    ) -> dict[str, Any]:
        """
        渡した値から, 求める出力に要るノードだけを計算する関数

        値は入力だけでなく途中のノードの出力の口にも渡せ (定義のつなぎ先と
        同じく「ノード」か「ノード:口」), その値だけから求まるノードは
        計算しない. 重みはこのモデルのものをそのまま使う (復号器だけを
        呼ぶ, 品物の塔だけを呼ぶなど)

        Parameters
        ----------
        values: dict[str, Any]
            入力の名前, ノードの名前 (最初の出力の口) か「ノード:口」ごとの
            値
        outputs: Optional[Union[str, list[str]]] = None
            求める出力の名前 (1 つなら文字, 複数ならリスト), None なら
            全ての出力

        Returns
        ----------
        result: dict[str, Any]
            求めた出力の名前ごとの値 (出力が 1 つでも辞書)

        Raises
        ----------
        TypeError
            values が辞書でない場合
        ValueError
            出力か値の名前が無い場合と, 要る入力の値が渡されていない場合
        """
        # 求める出力と, 渡した値の名前の確認
        if not isinstance(values, dict):
            raise TypeError(
                "values は名前ごとの値の辞書で渡してください "
                "(例: {\"z\": z})"
            )
        if outputs is None:
            names = list(self.outputs)
        elif isinstance(outputs, str):
            names = [outputs]
        else:
            names = list(dict.fromkeys(outputs))
        unknown = [name for name in names if name not in self.outputs]
        if unknown:
            raise ValueError(
                f"出力 {', '.join(map(str, unknown))} はありません (出力は "
                f"{', '.join(self.outputs)})"
            )
        given = {self.port_key(key): value for key, value in values.items()}

        # 要るノードだけを計算の順に計算
        computed = dict(given)
        for name in self.needed_nodes(names, set(given)):
            computed[name] = self.run_node(name, computed)

        # 求めた出力の取り出し
        result = {
            name: self.resolve(computed, parse_ref(self.outputs[name]))
            for name in names
        }

        return result


def build_model(
    spec: Union[str, Path, dict], seed: Optional[int] = None,
    load_weights: bool = True, base_dir: Optional[Union[str, Path]] = None,
    fetch: bool = True, weights_dirs: Optional[list] = None
) -> GraphModel:
    """
    モデルの定義から nn.Module を組み立てる関数

    外部の重み (ノードの weights) を読んだ場合の報告は, 組み立てたモデルの
    weights_report に入る

    Parameters
    ----------
    spec: Union[str, Path, dict]
        モデルの定義, またはその YAML か JSON のファイルのパス
    seed: Optional[int] = None
        重みの初期値の乱数の種, None の場合は今の乱数の状態のまま
    load_weights: bool = True
        外部の重みを読むか (学習した重みを後から load_state_dict で読む
        場合は False)
    base_dir: Optional[Union[str, Path]] = None
        外部の重みの相対のパスの元のディレクトリ, None の場合は定義の
        ファイルのディレクトリ (辞書で渡した場合は今のディレクトリ)
    fetch: bool = True
        外部の重みのライセンスを読むのにネットワークを使うか
    weights_dirs: Optional[list] = None
        外部の重みを読んでよいディレクトリ (base_dir と, 環境変数
        ML_MODELCONTROL_WEIGHTS_DIRS のものに加える)

    Returns
    ----------
    model: GraphModel
        組み立てたモデル

    Raises
    ----------
    SpecError
        定義に誤りがある場合
    NodeBuildError
        部品を作れない場合, 外部の重みを読めない場合, パラメータの名前が
        無い場合, 共有の形が違う場合
    """
    # 定義を読み込んで確かめ, 誤りがあれば組み立てずにエラー
    source = spec
    spec = load_spec(spec)
    analysis = analyze(spec)
    if not analysis.ok:
        raise SpecError([vars(issue) for issue in analysis.issues])

    # 外部の重みを読む設定 (相対のパスの元は定義のファイルのディレクトリ)
    context = None
    if load_weights:
        if base_dir is None and isinstance(source, (str, Path)):
            base_dir = Path(source).parent
        context = make_context(Path(base_dir or "."), fetch, weights_dirs)

    # 種を決めた場合は, 外の乱数の状態を変えずに初期値を決めて組み立て
    token = LOAD_CONTEXT.set(context)
    try:
        if seed is None:
            model = GraphModel(spec, analysis)
            return model
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model = GraphModel(spec, analysis)
    finally:
        LOAD_CONTEXT.reset(token)

    return model
