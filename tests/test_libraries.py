import warnings
from pathlib import Path
from typing import Optional, Union

import pytest
import torch
from torch import nn

from ml_modelcontrol import (
    build_model, describe_units, export_python, get_unit, infer, validate
)
from ml_modelcontrol.libraries import has_module, import_quietly
from ml_modelcontrol.registry import (
    ParamSpec, PortSpec, UnitSpec, needs_module, needs_object, register
)
from test_core import make_spec

# ライブラリごとの部品の例 (部品, 引数, 入力の口, 入力の形, 出力の形)
CASES = [
    ("torchvision", "tv.SqueezeExcitation",
     {"input_channels": 8, "squeeze_channels": 2}, "input", [2, 8, 4, 4],
     [2, 8, 4, 4]),
    ("torchvision", "tv.Model", {"name": "resnet18", "num_classes": 5},
     "x", [2, 3, 32, 32], [2, 5]),
    ("timm", "timm.Mlp", {"in_features": 8, "hidden_features": 16}, "x",
     [2, 5, 8], [2, 5, 8]),
    ("timm", "timm.Model", {"model_name": "resnet18", "num_classes": 3},
     "x", [2, 3, 32, 32], [2, 3]),
    ("torchaudio", "audio.MelSpectrogram",
     {"sample_rate": 16000, "n_mels": 32}, "waveform", [2, 1600],
     [2, 32, 9]),
    ("einops", "einops.Rearrange", {"pattern": "b c h w -> b (h w) c"},
     "input", [2, 3, 4, 4], [2, 16, 3]),
    ("kornia", "kornia.Normalize",
     {"mean": [0.5, 0.5, 0.5], "std": [0.2, 0.2, 0.2]}, "input",
     [2, 3, 8, 8], [2, 3, 8, 8]),
    ("monai", "monai.UNet",
     {"spatial_dims": 2, "in_channels": 1, "out_channels": 2,
      "channels": [4, 8], "strides": [2]}, "x", [1, 1, 16, 16],
     [1, 2, 16, 16]),
    ("segmentation_models_pytorch", "smp.Unet",
     {"encoder_name": "resnet18", "encoder_depth": 3,
      "decoder_channels": [16, 8, 4], "classes": 3}, "x", [1, 3, 32, 32],
     [1, 3, 32, 32])
]


def one_node_spec(
    unit: str, params: dict, port: str, shape: list
) -> dict:
    """
    1 つの部品だけのモデルの定義を作る関数

    Parameters
    ----------
    unit: str
        部品の名前
    params: dict
        部品の引数
    port: str
        入力をつなぐ口の名前
    shape: list
        入力の例の形

    Returns
    ----------
    spec: dict
        モデルの定義
    """
    spec = make_spec(f"""
inputs:
  x: {{shape: {shape}}}
nodes:
  n:
    unit: {unit}
    inputs: {{{port}: x}}
outputs:
  y: n
""")
    spec["nodes"]["n"]["params"] = params

    return spec


@pytest.mark.parametrize(
    ("package", "unit", "params", "port", "shape", "expected"), CASES,
    ids=[case[1] for case in CASES]
)
def test_library_units(
    package: str, unit: str, params: dict, port: str, shape: list,
    expected: list
) -> None:
    """
    外部のライブラリの部品で, 形を推定し, 書き出したコードも組み立てた
    モデルと同じ出力になることを確かめる関数

    Parameters
    ----------
    package: str
        ライブラリの名前
    unit: str
        部品の名前
    params: dict
        部品の引数
    port: str
        入力をつなぐ口の名前
    shape: list
        入力の例の形
    expected: list
        出力の形

    Returns
    ----------
    None
    """
    # ライブラリが無い場合は使えない部品として, 入れ方の案内付きで出る
    if not has_module(package):
        spec = get_unit(unit)
        assert not spec.available
        assert "pip install" in spec.reason
        pytest.skip(f"{package} が入っていません")

    # 形の推定
    spec = one_node_spec(unit, params, port, shape)
    assert [
        issue for issue in validate(spec) if issue["severity"] == "error"
    ] == []
    result = infer(spec)
    assert result["ok"], result
    assert result["outputs"]["y"]["shape"] == expected

    # 書き出したコードも同じ state_dict のキーと出力
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        built = build_model(spec, seed=0).eval()
        namespace: dict = {}
        exec(compile(export_python(spec), "<exported>", "exec"), namespace)
        exported = namespace["Model"]().eval()
    assert list(exported.state_dict()) == list(built.state_dict())
    exported.load_state_dict(built.state_dict())
    x = torch.randn(*shape)
    # 出力が 1 つのモデルはテンソルをそのまま返す
    with torch.no_grad():
        assert torch.allclose(built(x), exported(x))


def test_library_listing() -> None:
    """
    名前で作るモデルの候補と, 書き出しの別名と, 重みを読まない値を確かめる
    関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 名前で作るモデルは, ライブラリが入っていれば名前の候補を持つ
    units = {unit["name"]: unit for unit in describe_units()}
    for name, key, package in (
        ("tv.Model", "name", "torchvision"),
        ("timm.Model", "model_name", "timm")
    ):
        params = {param["name"]: param for param in units[name]["params"]}
        if has_module(package):
            assert "resnet18" in params[key]["suggestions"]
        assert "offline" not in units[name]
    assert get_unit("tv.Model").offline == {"weights": None}
    assert get_unit("hf.AutoModel").offline == {"pretrained": False}

    # 違うライブラリのクラスは別名のモジュールから読み込む
    if has_module("torchvision"):
        unit = get_unit("tv.DeformConv2d")
        assert unit.code_name == "tv_ops.DeformConv2d"
        assert unit.code_import == "from torchvision import ops as tv_ops"
    if has_module("segmentation_models_pytorch"):
        assert get_unit("smp.Unet").offline == {"encoder_weights": None}

    # 読み込めないモジュールは理由を返す
    module, reason = import_quietly("no_such_library_for_test.layers")
    assert module is None
    assert "no_such_library_for_test" in reason


def test_offline_values_in_infer() -> None:
    """
    形の推定では, 学習済みの重みを読む引数を読まない値に置き換え, 組み立て
    では書いた値のまま使うことを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 重みを読む指定で作ると失敗する部品
    class Pretrained(nn.Module):
        def __init__(self, pretrained: bool = True):
            # 親クラスのコンストラクタを呼び出す
            super(Pretrained, self).__init__()
            if pretrained:
                raise RuntimeError("重みを読もうとしました")
            self.fc = nn.Linear(4, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            全結合の層に通す関数

            Parameters
            ----------
            x: torch.Tensor
                入力

            Returns
            ----------
            y: torch.Tensor
                出力
            """
            y = self.fc(x)

            return y

    register(UnitSpec(
        name="test.Pretrained", category="test", source="custom",
        params=[ParamSpec("pretrained", "bool", True)], inputs=[PortSpec("x")], outputs=[PortSpec("out")],
        builder=lambda **values: Pretrained(**values),
        offline={"pretrained": False}
    ))

    # 書いていなくても, 書いていても, 推定では読まない
    for params in ({}, {"pretrained": True}):
        spec = one_node_spec("test.Pretrained", params, "x", [3, 4])
        result = infer(spec)
        assert result["ok"], result
        assert result["outputs"]["y"]["shape"] == [3, 2]
    with pytest.raises(Exception, match="重みを読もうとしました"):
        build_model(one_node_spec("test.Pretrained", {}, "x", [3, 4]))


def test_needs_object() -> None:
    """
    テンソルか値のどちらでも渡せる引数は, 定義から作れるものとすることを
    確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 値の選択肢を含む Union は渡せる, 部品とテンソルだけは渡せない
    assert not needs_object(Union[torch.Tensor, tuple[float, float]])
    assert not needs_object(Union[float, torch.Tensor])
    assert needs_object(torch.Tensor)
    assert needs_object(nn.Module)
    assert needs_object(Optional[torch.Tensor])
    assert not needs_object("tuple[float, float] | torch.Tensor")
    assert needs_object("torch.Tensor")

    class Block(nn.Module):
        def __init__(
            self, scale: "Union[float, torch.Tensor]", inner: nn.Module
        ):
            # 親クラスのコンストラクタを呼び出す
            super(Block, self).__init__()

    assert needs_module(Block)


def test_huggingface_image_units(tmp_path: Path) -> None:
    """
    手元に保存した小さな ViT を画像の分類の部品として読み込み, 形を推定
    できることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 小さな ViT を保存し, ネットワークを使わずに読み込む定義
    transformers = pytest.importorskip("transformers")
    config = transformers.ViTConfig(
        image_size=16, patch_size=8, hidden_size=8, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=16
    )
    transformers.ViTModel(config).save_pretrained(tmp_path)
    spec = make_spec(f"""
inputs:
  image: {{shape: [2, 3, 16, 16]}}
nodes:
  vit:
    unit: hf.AutoModelForImageClassification
    params: {{model_name: "{tmp_path}", local_files_only: true,
             num_labels: 4}}
    inputs: {{pixel_values: image}}
  body:
    unit: hf.AutoVisionModel
    params: {{model_name: "{tmp_path}", local_files_only: true}}
    inputs: {{pixel_values: image}}
outputs:
  logits: "vit:logits"
  hidden: "body:last_hidden_state"
""")
    result = infer(spec)
    assert result["ok"], result
    assert result["outputs"]["logits"]["shape"] == [2, 4]
    assert result["outputs"]["hidden"]["shape"] == [2, 5, 8]
