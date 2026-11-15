import importlib.util
import sys

import pytest
import torch

from ml_modelcontrol import infer, list_examples, load_example, validate
from ml_modelcontrol.infer import example_input


@pytest.mark.parametrize(
    "name", [example["name"] for example in list_examples()]
)
def test_example(name: str) -> None:
    """
    同梱した例のモデルの定義が, 確かめて誤りが無く, 形を推定できることを
    確かめる関数

    Parameters
    ----------
    name: str
        例の名前

    Returns
    ----------
    None
    """
    # 使うライブラリが無い例は飛ばし, Hugging Face の例は確認だけ (形の推定
    # ではモデルの設定を取得するのでネットワークが要る)
    spec = load_example(name)
    units = {node["unit"] for node in spec["nodes"].values()}
    if any(unit.startswith("pyg.") for unit in units) and (
        importlib.util.find_spec("torch_geometric") is None
    ):
        pytest.skip("PyTorch Geometric が入っていません")
    if any(unit.startswith("hf.") for unit in units):
        if importlib.util.find_spec("transformers") is None:
            pytest.skip("transformers が入っていません")
        errors = [
            issue for issue in validate(spec) if issue["severity"] == "error"
        ]
        assert errors == []
        return
    assert validate(spec) == []
    result = infer(spec)
    assert result["ok"], result
    assert spec["description"]


def test_example_names() -> None:
    """
    例の一覧と, 一覧に無い名前の読み込みを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 一覧に名前と説明, 無い名前やパスは読めない
    names = {example["name"] for example in list_examples()}
    assert {"mlp", "cnn_classifier", "gcn_graph_classifier"} <= names
    with pytest.raises(KeyError):
        load_example("../pyproject")


def test_standalone_export(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    書き出したコードが, このライブラリを読み込めなくても同じ計算をすること
    を確かめる関数 (演算のクラスを書き写すこと)

    Parameters
    ----------
    monkeypatch: pytest.MonkeyPatch
        読み込めないライブラリを作るための差し替え

    Returns
    ----------
    None
    """
    from ml_modelcontrol import build_model, export_python

    for name in [
        "residual_mlp", "lstm_classifier", "transformer_encoder",
        "stacked_blocks"
    ]:
        spec = load_example(name)
        code = export_python(spec)
        assert "from ml_modelcontrol" not in code
        assert "import ml_modelcontrol" not in code
        model = build_model(spec, seed=0).eval()

        # このライブラリを読み込めない状態で書き出したコードを実行
        namespace: dict = {}
        with monkeypatch.context() as patch:
            patch.setitem(sys.modules, "ml_modelcontrol", None)
            exec(code, namespace)
        exported = namespace["Model"]().eval()
        exported.load_state_dict(model.state_dict())
        inputs = example_input(spec)
        with torch.no_grad():
            expected = model(**inputs)
            actual = exported(**inputs)
        assert torch.allclose(expected, actual)

    # standalone=False は今までどおり読み込む
    assert "from ml_modelcontrol import ops" in export_python(
        load_example("residual_mlp"), standalone=False
    )

    # 書き写したクラスが使う名前だけを読み込む (使わない Any を読み込まない)
    code = export_python(load_example("mlp"))
    assert "import torch\n" not in code and "from typing" not in code
    code = export_python(load_example("residual_mlp"))
    assert "from typing" not in code


def test_standalone_export_conflicts(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    書き写すクラスの名前の重なりと, 中身を読めない場合の扱いを確かめる関数

    Parameters
    ----------
    monkeypatch: pytest.MonkeyPatch
        中身を読めない状態を作るための差し替え

    Returns
    ----------
    None
    """
    import inspect

    from torch import nn

    from ml_modelcontrol import export_python, register_unit

    # 自作の部品のクラスと書き写す演算のクラスの名前が同じ場合はエラー
    class Add(nn.Module):
        def forward(self, x):
            return x

    Add.__module__ = "mypkg"
    register_unit(Add, name="test_export.Add")
    spec = {
        "inputs": {"x": {"shape": [2, 3]}},
        "nodes": {
            "d": {"unit": "test_export.Add", "inputs": "x"},
            "s": {"unit": "op.Add", "inputs": {"xs": ["x", "d"]}}
        },
        "outputs": {"y": "s"}
    }
    with pytest.raises(ValueError, match="Add"):
        export_python(spec)

    # 書き写したクラスが使う名前はクラスの名前に使えない
    with pytest.raises(ValueError, match="torch"):
        export_python(load_example("residual_mlp"), class_name="torch")

    # 中身を読めない環境では, 注意を出して読み込む行を残す
    def broken(value: object) -> str:
        raise OSError("no source")

    monkeypatch.setattr(inspect, "getsource", broken)
    with pytest.warns(UserWarning, match="読めない"):
        code = export_python(load_example("residual_mlp"))
    assert "from ml_modelcontrol import ops" in code



def test_merge_imports() -> None:
    """
    同じモジュールから読み込む行を 1 行にまとめることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    from ml_modelcontrol.export import merge_imports

    assert merge_imports([
        "import torch", "", "from torch import nn",
        "from torch_geometric.nn import GCNConv",
        "from torch_geometric.nn import MeanAggregation",
        "from torch_geometric.nn import GCNConv"
    ]) == [
        "import torch", "", "from torch import nn",
        "from torch_geometric.nn import GCNConv, MeanAggregation"
    ]
