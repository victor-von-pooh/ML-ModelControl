import importlib.util
import re
import sys
from pathlib import Path

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


# 受け持ちごとの雛形の例 (生成, 表現の学習, 推薦, 強化学習, 異常の検知,
# 効果の見積もり, 値段を変える判断, 時間をまたぐ量の判断)
ROLE_TEMPLATES = [
    "vae", "gan_generator", "gan_discriminator", "diffusion_denoiser",
    "contrastive_encoder", "two_tower_retrieval", "policy_value",
    "tabular_autoencoder", "tabular_uplift", "tabular_price",
    "tabular_demand"
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
    "tabular_autoencoder": {"recon": [8, 16]},
    "tabular_uplift": {"treated": [8, 1], "control": [8, 1]},
    "tabular_price": {"units": [8, 1]},
    "tabular_demand": {"quantiles": [8, 9]}
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

    # 骨組みと同じ画素ごとの交差エントロピーでの学習 (入力の最初の層まで
    # 届く勾配)
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

    # 骨組みと同じ Chamfer 距離での学習 (最初の層まで届く勾配)
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
    assert bool(torch.isfinite(scores).all())

    # 再構成の二乗誤差での学習 (最初の層まで届く勾配)
    model.train()
    loss = nn.MSELoss()(model(rows), rows)
    loss.backward()
    assert model.nodes["enc1"].weight.grad is not None


def test_tabular_uplift_example() -> None:
    """
    表の行の効果の例が, 処置したときとしないときの 2 つの予測を (バッチ, 1)
    で出し, 処置の値で片方だけに付けた損失がその枝だけに届くことを確かめる
    関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    from ml_modelcontrol import build_model

    # 例の形と, 例と違う行の数で 2 つの出力
    spec = load_example("tabular_uplift")
    result = infer(spec)
    assert result["outputs"]["treated"]["shape"] == [8, 1]
    assert result["outputs"]["control"]["shape"] == [8, 1]
    model = build_model(spec, seed=0).eval()
    rows = torch.randn(6, 16)
    with torch.no_grad():
        outputs = model(rows)
    assert outputs["treated"].shape == (6, 1)
    assert outputs["control"].shape == (6, 1)

    # 行ごとの効果 (0 と 1 の結果は確率の差)
    effect = (
        torch.sigmoid(outputs["treated"]) - torch.sigmoid(outputs["control"])
    ).squeeze(1)
    assert effect.shape == (6,)
    assert bool((effect.abs() <= 1).all())

    # 処置の群の行だけの損失による勾配の範囲
    model.train()
    treatment = torch.tensor([1, 0, 1, 0, 1, 0])
    labels = torch.tensor([1.0, 0.0, 0.0, 1.0, 1.0, 0.0])
    criterion = nn.BCEWithLogitsLoss()
    treated = treatment == 1
    loss = criterion(
        model(rows)["treated"].squeeze(1)[treated], labels[treated]
    )
    loss.backward()
    assert model.nodes["treated1"].weight.grad is not None
    assert model.nodes["control1"].weight.grad is None

    # 両方の群の損失の和での学習 (両方の枝の最初の層まで届く勾配)
    model.zero_grad(set_to_none=True)
    outputs = model(rows)
    loss = criterion(
        outputs["treated"].squeeze(1)[treated], labels[treated]
    ) + criterion(outputs["control"].squeeze(1)[~treated], labels[~treated])
    loss.backward()
    assert model.nodes["treated1"].weight.grad is not None
    assert model.nodes["control1"].weight.grad is not None


def test_tabular_price_example() -> None:
    """
    値段を入力に入れる売れ行きの例が, 入力の幅を変えても売れた数の予測を
    (バッチ, 1) の負でない値で出し, 候補の値段ごとに予測し直せることと,
    値段の列まで勾配が届くことを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    from ml_modelcontrol import build_model

    # 例の形と, 例と違う行の数での売れた数の予測
    spec = load_example("tabular_price")
    result = infer(spec)
    assert result["outputs"]["units"]["shape"] == [8, 1]
    model = build_model(spec, seed=0).eval()
    rows = torch.randn(6, 17)
    with torch.no_grad():
        units = model(rows)
    assert units.shape == (6, 1)
    assert bool((units >= 0).all())

    # 入力の幅を書き換えた定義 (特徴の 24 列 + 値段の 1 列)
    wide = load_example("tabular_price")
    wide["inputs"]["x"]["shape"] = [8, 25]
    wide["nodes"]["units1"]["params"]["in_features"] = 25
    assert validate(wide) == []
    assert infer(wide)["outputs"]["units"]["shape"] == [8, 1]
    with torch.no_grad():
        units = build_model(wide, seed=0)(torch.randn(3, 25))
    assert units.shape == (3, 1)

    # 候補の値段ごとの予測 (元の値段の候補は 0 より大きく, 値段の列には
    # 標準化をかけた値を入れる. 最後の候補は記録の範囲の外でも負でない値)
    prices = torch.tensor([80.0, 100.0, 120.0, 3000.0])
    predictions = []
    for price in prices:
        varied = rows.clone()
        varied[:, -1] = (price - 100.0) / 20.0
        with torch.no_grad():
            predictions.append(model(varied).squeeze(1))
    table = torch.stack(predictions, dim=1)
    assert table.shape == (6, 4)
    assert bool((table >= 0).all())

    # 値段の列だけを変えたときの予測の違い (選び方の式は学習のスクリプトの
    # 側で行う)
    assert not torch.allclose(table[:, 0], table[:, 3])

    # 売れた数の二乗誤差での学習 (最初の層と値段の列まで届く勾配)
    model.train()
    inputs = rows.clone().requires_grad_(True)
    target = torch.tensor([[0.0], [1.0], [3.0], [2.0], [0.0], [5.0]])
    loss = nn.MSELoss()(model(inputs), target)
    loss.backward()
    assert model.nodes["units1"].weight.grad is not None
    assert inputs.grad is not None
    assert bool(inputs.grad[:, -1].abs().sum() > 0)


def test_tabular_demand_example() -> None:
    """
    需要の分位点の例が, 入力の幅と水準の数を変えても分位点を (バッチ,
    水準の数) で出し, 値が負にならず水準の順に下がらないことと, 分位点の
    損失で学べることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    from ml_modelcontrol import build_model

    # 例の形と, 例と違う行の数での分位点 (大きな値の入力も含める)
    spec = load_example("tabular_demand")
    result = infer(spec)
    assert result["outputs"]["quantiles"]["shape"] == [8, 9]
    model = build_model(spec, seed=0).eval()
    rows = torch.cat([torch.randn(5, 16), 50.0 * torch.randn(1, 16)])
    with torch.no_grad():
        quantiles = model(rows)
    assert quantiles.shape == (6, 9)
    assert bool((quantiles >= 0).all())
    assert bool((quantiles[:, 1:] >= quantiles[:, :-1]).all())

    # 入力の幅と水準の数を書き換えた定義 (学習に使う 24 列, 水準の 3 つ)
    wide = load_example("tabular_demand")
    wide["inputs"]["x"]["shape"] = [8, 24]
    wide["nodes"]["demand1"]["params"]["in_features"] = 24
    wide["nodes"]["demand2"]["params"]["out_features"] = 3
    assert validate(wide) == []
    assert infer(wide)["outputs"]["quantiles"]["shape"] == [8, 3]
    with torch.no_grad():
        narrow = build_model(wide, seed=0)(torch.randn(4, 24))
    assert narrow.shape == (4, 3)
    assert bool((narrow[:, 1:] >= narrow[:, :-1]).all())

    # 分位点の損失での学習 (水準ごとの pinball loss を水準と行で平均)
    model.train()
    levels = torch.linspace(0.1, 0.9, 9)
    target = torch.tensor([[0.0], [3.0], [7.0], [12.0], [1.0], [40.0]])
    error = target - model(rows)
    loss = torch.maximum(levels * error, (levels - 1.0) * error).mean()
    loss.backward()
    assert model.nodes["demand1"].weight.grad is not None
    assert bool(model.nodes["demand2"].weight.grad.abs().sum() > 0)


def contract_rows() -> dict[str, tuple[list[str], list[str]]]:
    """
    docs/DESIGN.md の「受け持ちごとの雛形の入力と出力の約束」の表を読む関数

    Parameters
    ----------
    None

    Returns
    ----------
    rows: dict[str, tuple[list[str], list[str]]]
        雛形の名前ごとの, 入力の名前と出力の名前 (書いた順)
    """
    # 見出しの後の表の行 (雛形, 受け持ち, 入力, 出力, ...)
    path = Path(__file__).resolve().parents[1] / "docs" / "DESIGN.md"
    text = path.read_text(encoding="utf-8")
    section = text.split("### 受け持ちごとの雛形の入力と出力の約束", 1)[1]
    rows = {}
    for line in section.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 4 or not cells[0].startswith("`"):
            continue
        # 6 つの列で, 入力と出力の名前は全て形付き
        assert len(cells) == 6, line
        for cell in cells[2:4]:
            assert len(re.findall(r"`\w+`", cell)) == len(
                re.findall(r"`(\w+)` \(", cell)
            ), line
        rows[cells[0].strip("`")] = (
            re.findall(r"`(\w+)` \(", cells[2]),
            re.findall(r"`(\w+)` \(", cells[3]),
        )

    return rows


def test_role_contract_table() -> None:
    """
    DESIGN.md の約束の表が, 受け持ちごとの雛形と領域と点群の例の入力と
    出力の名前と合うことを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 表の雛形は受け持ちごとの雛形と, 領域と点群の例
    rows = contract_rows()
    assert set(rows) == {*ROLE_TEMPLATES, "unet_segmentation",
                         "point_cloud_predictor"}
    for name, (inputs, outputs) in rows.items():
        spec = load_example(name)
        assert inputs == list(spec["inputs"]), name
        assert outputs == list(infer(spec)["outputs"]), name
        if name in ROLE_SHAPES:
            assert outputs == list(ROLE_SHAPES[name]), name
