import builtins
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch

from ml_modelcontrol import (
    SpecError, build_model, describe_units, get_unit, register_units_from,
    validate
)
from ml_modelcontrol.custom import LOADED_MODULES
from ml_modelcontrol.registry import REGISTRY

# 起動の設定から読み込む見本のモジュールの置き場
MODULES_DIR = Path(__file__).parent / "custom_modules"

# 見本のモジュールの名前
SAMPLE_MODULES = [
    "lab_units", "lab_hook", "lab_empty", "lab_clash", "lab_partial",
    "lab_hook_fail", "lab_overwrite", "never_import"
]


@pytest.fixture
def lab_path(monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """
    見本のモジュールを読み込める場所に加え, 後で登録を元に戻す関数

    Parameters
    ----------
    monkeypatch: pytest.MonkeyPatch
        sys.path を一時的に変える道具

    Returns
    ----------
    path: Iterator[Path]
        見本のモジュールの置き場
    """
    monkeypatch.syspath_prepend(str(MODULES_DIR))
    yield MODULES_DIR

    # 他のテストに残さないように, 登録した部品と読み込んだ記録を消去
    for name in [
        name for name, unit in REGISTRY.items() if unit.module
    ]:
        REGISTRY.pop(name)
    LOADED_MODULES.clear()
    for module_name in SAMPLE_MODULES:
        sys.modules.pop(module_name, None)


def test_register_units_from_module(lab_path: Path) -> None:
    """
    UNITS と register_units の宣言から部品を登録し, 使えることを確かめる
    関数

    Parameters
    ----------
    lab_path: Path
        見本のモジュールの置き場

    Returns
    ----------
    None
    """
    # モジュールごとの部品の名前 (宣言した順)
    loaded = register_units_from(["lab_units", "lab_hook"])
    assert loaded == {
        "lab_units": ["lab.Scale", "lab.SplitPair"],
        "lab_hook": ["hook.Doubler"]
    }

    # 部品の一覧に宣言したモジュールの名前, 組み込みの部品は空
    units = {unit["name"]: unit for unit in describe_units()}
    assert units["lab.Scale"]["module"] == "lab_units"
    assert units["lab.Scale"]["source"] == "custom"
    assert units["hook.Doubler"]["module"] == "lab_hook"
    assert units["nn.Linear"]["module"] == ""
    params = {param["name"]: param for param in units["lab.Scale"]["params"]}
    assert params["factor"]["description"] == "倍率"
    outputs = [port["name"] for port in units["lab.SplitPair"]["outputs"]]
    assert outputs == ["first", "second"]

    # 定義から組み立てて計算
    spec = {
        "format_version": 1,
        "inputs": {"x": {"shape": [2, 4]}},
        "nodes": {
            "scale": {
                "unit": "lab.Scale", "params": {"factor": 3.0},
                "inputs": "x"
            },
            "split": {"unit": "lab.SplitPair", "inputs": "scale"},
            "twice": {"unit": "hook.Doubler", "inputs": "split:second"}
        },
        "outputs": {"y": "twice"}
    }
    assert [i for i in validate(spec) if i["severity"] == "error"] == []
    model = build_model(spec)
    result = model(torch.ones(2, 4))
    assert torch.equal(result, torch.full((2, 2), 6.0))

    # 同じモジュールの 2 回目は登録し直さずに前の結果
    again = register_units_from(["lab_units"])
    assert again == {"lab_units": ["lab.Scale", "lab.SplitPair"]}


def test_register_units_from_errors(lab_path: Path) -> None:
    """
    起動の設定の誤りが, モジュールの名前を添えたエラーになることを確かめる
    関数

    Parameters
    ----------
    lab_path: Path
        見本のモジュールの置き場

    Returns
    ----------
    None
    """
    # 名前の形の誤り (ファイルのパスは不可)
    with pytest.raises(ValueError, match="識別子を . でつないだ名前"):
        register_units_from(["custom_modules/lab_units.py"])
    with pytest.raises(TypeError, match="リストにしてください"):
        register_units_from("lab_units")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="リストにしてください"):
        register_units_from({"lab_units": 1})  # type: ignore[arg-type]

    # 読み込めないモジュール
    with pytest.raises(ValueError, match="no_such_lab_module を読み込めません"):
        register_units_from(["no_such_lab_module"])

    # 部品を宣言していないモジュール
    with pytest.raises(ValueError, match="lab_empty は部品を宣言していません"):
        register_units_from(["lab_empty"])

    # 別のモジュールと同じ名前の部品は上書きしない
    register_units_from(["lab_units"])
    with pytest.raises(ValueError, match="lab_clash の部品を登録できません"):
        register_units_from(["lab_clash"])
    assert get_unit("lab.Scale").module == "lab_units"
    with pytest.raises(ValueError, match="overwrite は書けません"):
        register_units_from(["lab_overwrite"])
    assert get_unit("lab.Scale").module == "lab_units"


def test_failed_module_is_removed(lab_path: Path) -> None:
    """
    途中で止まったモジュールの部品を外し, 同じ誤りで読み直せることを
    確かめる関数

    Parameters
    ----------
    lab_path: Path
        見本のモジュールの置き場

    Returns
    ----------
    None
    """
    # UNITS の 2 つ目で止まると, 1 つ目も外し, 2 回目も同じ誤り
    for _ in range(2):
        with pytest.raises(ValueError, match=r"lab_partial の UNITS\[1\]"):
            register_units_from(["lab_partial"])
        assert "partial.Scale" not in REGISTRY

    # register_units の誤りはモジュールの名前を添え, 登録した部品を外す
    with pytest.raises(
        ValueError, match="lab_hook_fail の register_units.*部品の準備に失敗"
    ):
        register_units_from(["lab_hook_fail"])
    assert "hookfail.Scale" not in REGISTRY


def test_unknown_unit_is_not_imported(lab_path: Path) -> None:
    """
    登録されていない部品の名前が, 案内付きの誤りになり, 何も読み込まない
    ことを確かめる関数

    Parameters
    ----------
    lab_path: Path
        見本のモジュールの置き場 (読み込める場所にあっても読み込まない)

    Returns
    ----------
    None
    """
    # 読み込める場所にあるモジュールの名前を部品の名前に書いた定義
    spec = {
        "format_version": 1,
        "inputs": {"x": {"shape": [2, 4]}},
        "nodes": {"block": {"unit": "never_import.Block", "inputs": "x"}},
        "outputs": {"y": "block"}
    }
    issues = validate(spec)
    unknown = [i for i in issues if i["code"] == "unknown_unit"]
    assert len(unknown) == 1
    message = unknown[0]["message"]
    assert "この部品 ('never_import.Block') は登録されていません" in message
    assert "自作の部品なら起動の設定に足してください" in message
    assert "register_units_from" in message
    with pytest.raises(SpecError):
        build_model(spec)
    with pytest.raises(KeyError, match="起動の設定に足してください"):
        get_unit("never_import.Block")

    # モジュールは読み込まれていない
    assert "never_import" not in sys.modules
    assert not hasattr(builtins, "ML_MODELCONTROL_NEVER_IMPORTED")
