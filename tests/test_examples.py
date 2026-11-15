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


def test_standalone_export(monkeypatch: pytest.MonkeyPatch):
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
    from ml_modelcontrol import build_model, export_python, load_example

    for name in ["residual_mlp", "lstm_classifier", "transformer_encoder"]:
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
