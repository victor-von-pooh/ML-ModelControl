import importlib.util
import sys

import pytest
import torch
from torch import nn

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
    assert {
        "mlp", "cnn_classifier", "gcn_graph_classifier",
        "unet_segmentation", "point_cloud_predictor"
    } <= names
    with pytest.raises(KeyError):
        load_example("../pyproject")


# 受け持ちごとの雛形の例 (生成, 表現の学習, 推薦, 強化学習, 異常の検知)
ROLE_TEMPLATES = [
    "vae", "gan_generator", "gan_discriminator", "diffusion_denoiser",
    "contrastive_encoder", "two_tower_retrieval", "policy_value",
    "tabular_autoencoder"
]

# 雛形の出力の名前ごとの形 (入力の例のバッチは 8)
ROLE_SHAPES = {
    "vae": {"recon": [8, 16], "mu": [8, 8], "logvar": [8, 8]},
    "gan_generator": {"fake": [8, 24]},
    "gan_discriminator": {"logit": [8, 1]},
    "diffusion_denoiser": {"noise": [8, 16]},
    "contrastive_encoder": {"embedding": [8, 16], "similarity": [8, 8]},
    "two_tower_retrieval": {
        "score": [8, 8], "user_embedding": [8, 16],
        "item_embedding": [8, 16]
    },
    "policy_value": {"logits": [8, 4], "value": [8, 1]},
    "tabular_autoencoder": {"recon": [8, 16]}
}


def random_input(spec: dict) -> dict[str, torch.Tensor]:
    """
    入力の例の小数の値を, 決まった種の乱数に置き換えた値を作る関数

    0 だけの入力では全結合の層が偏りしか返さず, 書き出したコードとの
    食い違いを見逃すため

    Parameters
    ----------
    spec: dict
        モデルの定義

    Returns
    ----------
    values: dict[str, torch.Tensor]
        入力の名前ごとの値 (整数と真偽値の入力は例のまま)
    """
    # 小数の入力だけを乱数に置き換え
    generator = torch.Generator().manual_seed(0)
    values = example_input(spec)
    for name, value in values.items():
        if value.is_floating_point():
            values[name] = torch.randn(value.shape, generator=generator)

    return values


def assert_same_output(expected: object, actual: object) -> None:
    """
    組み立てたモデルと書き出したコードの出力が同じことを確かめる関数

    Parameters
    ----------
    expected: object
        組み立てたモデルの出力 (テンソルか, 出力の名前をキーにした辞書)
    actual: object
        書き出したコードのモデルの出力

    Returns
    ----------
    None
    """
    # 出力が 2 つ以上の場合は名前ごとに比較
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert expected.keys() == actual.keys()
        for key in expected:
            assert torch.allclose(expected[key], actual[key]), key
    else:
        assert torch.allclose(expected, actual)


@pytest.mark.parametrize(
    "name", [example["name"] for example in list_examples()]
)
def test_example_build_and_export(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    同梱した例が組み立てられ, 書き出したコードがこのライブラリを読み込め
    なくても同じ計算をすることを確かめる関数 (演算のクラスを書き写すこと)

    Parameters
    ----------
    name: str
        例の名前
    monkeypatch: pytest.MonkeyPatch
        読み込めないライブラリを作るための差し替え

    Returns
    ----------
    None
    """
    from ml_modelcontrol import build_model, export_python

    # 使うライブラリが無い例と, Hugging Face の例 (組み立てにネットワークが
    # 要る) は飛ばし
    spec = load_example(name)
    units = {node["unit"] for node in spec["nodes"].values()}
    if any(unit.startswith("pyg.") for unit in units) and (
        importlib.util.find_spec("torch_geometric") is None
    ):
        pytest.skip("PyTorch Geometric が入っていません")
    if any(unit.startswith("hf.") for unit in units):
        pytest.skip("Hugging Face の例は組み立てにネットワークを使います")

    # 書き出したコードがこのライブラリを読み込まないことと, 組み立て
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
    inputs = random_input(spec)
    with torch.no_grad():
        expected = model(**inputs)
        actual = exported(**inputs)
    assert_same_output(expected, actual)


def test_role_templates() -> None:
    """
    受け持ちごとの雛形の例が一覧にあり, 説明に学習の step で行うことを
    書いていることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 一覧の名前と説明
    descriptions = {
        example["name"]: example["description"]
        for example in list_examples()
    }
    assert set(ROLE_SHAPES) == set(ROLE_TEMPLATES)
    for name in ROLE_TEMPLATES:
        assert name in descriptions
        assert "学習の step" in descriptions[name], name

    # 推定した出力の名前と形 (損失に使う値と, 類似度の表の形)
    for name, shapes in ROLE_SHAPES.items():
        result = infer(load_example(name))
        found = {
            key: output["shape"] for key, output in result["outputs"].items()
        }
        assert found == shapes, name


def test_vae_sampling(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    VAE の例が, 学習の形では乱数を引き, 評価の形では平均を使うことと,
    書き出したコードも同じ種で同じ値を引くことを確かめる関数

    Parameters
    ----------
    monkeypatch: pytest.MonkeyPatch
        読み込めないライブラリを作るための差し替え

    Returns
    ----------
    None
    """
    from ml_modelcontrol import build_model, export_python

    # 組み立てたモデルと, このライブラリを読み込めない状態で書き出したコード
    spec = load_example("vae")
    model = build_model(spec, seed=0)
    code = export_python(spec)
    namespace: dict = {}
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "ml_modelcontrol", None)
        exec(code, namespace)
    exported = namespace["Model"]()
    exported.load_state_dict(model.state_dict())
    inputs = random_input(spec)

    # 学習の形の同じ種での一致と, 評価の形 (平均のまま) との違い
    with torch.no_grad():
        torch.manual_seed(1)
        sampled = model(**inputs)
        torch.manual_seed(1)
        assert_same_output(sampled, exported(**inputs))
        mean = model.eval()(**inputs)
    assert not torch.allclose(sampled["recon"], mean["recon"])
    assert torch.allclose(sampled["mu"], mean["mu"])


def test_standalone_export() -> None:
    """
    書き出すコードの読み込みの行を確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    from ml_modelcontrol import export_python

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


def test_segmentation_example() -> None:
    """
    U-Net の例が, 問題の設定の骨組みの仮のモデルと同じく (バッチ,
    チャンネル, 高さ, 幅) の画像から同じ高さと幅のクラスのスコアを出す
    ことを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    from ml_modelcontrol import build_model

    # 例の形と, 例と違う 4 の倍数の高さと幅のどちらでも同じ大きさで返す
    spec = load_example("unet_segmentation")
    result = infer(spec)
    assert result["outputs"]["logits"]["shape"] == [4, 2, 32, 32]
    model = build_model(spec, seed=0).eval()
    with torch.no_grad():
        logits = model(torch.rand(2, 3, 48, 64))
    assert logits.shape == (2, 2, 48, 64)

    # 骨組みと同じく, 画素ごとの交差エントロピーで学べる (勾配が入力の
    # 最初の層まで届く)
    model.train()
    labels = torch.randint(0, 2, (2, 16, 16))
    loss = torch.nn.CrossEntropyLoss()(
        model(torch.rand(2, 3, 16, 16)), labels
    )
    loss.backward()
    assert model.nodes["enc1a"].weight.grad is not None


def test_point_cloud_example() -> None:
    """
    点群の例が, 問題の設定の骨組みの仮のモデルと同じく (バッチ, 入力の
    点の数, 3) から (バッチ, 出す点の数, 3) を出し, 入力の点の数と並びに
    依らないことを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    from ml_modelcontrol import build_model

    # 例の形と, 例と違う入力の点の数
    spec = load_example("point_cloud_predictor")
    result = infer(spec)
    assert result["outputs"]["cloud"]["shape"] == [4, 256, 3]
    model = build_model(spec, seed=0).eval()
    points = torch.randn(2, 1000, 3)
    with torch.no_grad():
        cloud = model(points)
        shuffled = model(points[:, torch.randperm(1000)])
    assert cloud.shape == (2, 256, 3)
    assert torch.allclose(cloud, shuffled, atol=1e-6)

    # 骨組みと同じく Chamfer 距離で学べる (勾配が最初の層まで届く)
    model.train()
    target = torch.randn(2, 300, 3)
    pairs = torch.cdist(model(points), target) ** 2
    loss = (pairs.min(dim=2).values.mean() + pairs.min(dim=1).values.mean())
    loss.backward()
    assert model.nodes["encode1"].weight.grad is not None


def test_tabular_autoencoder_example() -> None:
    """
    表の行の自己符号化器の例が, 入力と同じ幅の再構成を出し, 再構成の
    誤差から行ごとの異常の度合いを出せることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    from ml_modelcontrol import build_model

    # 例の形と, 例と違う行の数で入力と同じ形の再構成
    spec = load_example("tabular_autoencoder")
    result = infer(spec)
    assert result["outputs"]["recon"]["shape"] == [8, 16]
    model = build_model(spec, seed=0).eval()
    rows = torch.randn(5, 16)
    with torch.no_grad():
        recon = model(rows)
    assert recon.shape == rows.shape

    # 行ごとの異常の度合い (特徴の次元で平均した二乗誤差)
    scores = ((recon - rows) ** 2).mean(dim=1)
    assert scores.shape == (5,)
    assert bool((scores >= 0).all())

    # 再構成の二乗誤差で学べる (勾配が最初の層まで届く)
    model.train()
    loss = nn.MSELoss()(model(rows), rows)
    loss.backward()
    assert model.nodes["enc1"].weight.grad is not None
