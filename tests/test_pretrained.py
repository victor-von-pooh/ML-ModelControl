import io
import json
import urllib.error
import urllib.request
from pathlib import Path
from collections.abc import Iterator
from typing import Any

import pytest

from ml_modelcontrol import (
    classify_license, clear_pretrained_cache, describe_units,
    pretrained_info
)
from ml_modelcontrol import pretrained

# テストで返す Hugging Face のモデルの情報 (置き場の名前ごと)
HUB = {
    "org/apache": {
        "sha": "a" * 40, "gated": False,
        "cardData": {"license": "apache-2.0"},
        "tags": ["license:apache-2.0"]
    },
    "org/noncommercial": {
        "sha": "b" * 40, "gated": False,
        "cardData": {"license": "cc-by-nc-4.0"}
    },
    "org/gated": {
        "sha": "c" * 40, "gated": "manual",
        "cardData": {"license": "llama3.1"}
    },
    "org/tag-only": {"sha": "d" * 40, "tags": ["license:mit"]},
    "org/list": {"sha": "e" * 40, "cardData": {"license": ["bsd-3-clause"]}},
    "org/other": {"sha": "f" * 40, "cardData": {"license": "other"}},
    "org/openrail": {"sha": "1" * 40, "cardData": {"license": "openrail"}},
    "org/empty": {"sha": "2" * 40, "cardData": {}},
    "org/mixed": {
        "sha": "3" * 40,
        "cardData": {"license": ["apache-2.0", "cc-by-nc-4.0"]}
    },
    "org/bad-tags": {"sha": "4" * 40, "tags": 3}
}


class FakeHub:
    def __init__(self):
        # 呼ばれた置き場の名前と版を保存
        self.calls: list[tuple[str, str]] = []

    def __call__(self, hub_id: str, revision: str) -> dict:
        """
        置き場の名前に決めた情報を返し, 無い名前は読めない誤りにする関数

        Parameters
        ----------
        hub_id: str
            置き場の名前
        revision: str
            版

        Returns
        ----------
        metadata: dict
            モデルの情報
        """
        self.calls.append((hub_id, revision))
        if hub_id not in HUB:
            raise pretrained.HubError("ネットワークの誤り")
        metadata = HUB[hub_id]

        return metadata


class RecordingOpen:
    def __init__(
        self, requests: list, body: bytes = b"{}", error: int = -1
    ):
        # 記録先, 返す中身, 誤り (HTTP の番号, 0 はネットワーク) を保存
        self.requests = requests
        self.body = body
        self.error = error

    def __call__(self, request: Any, timeout: float = 0.0) -> Any:
        """
        要求を記録し, 決めた中身か誤りを返す関数

        Parameters
        ----------
        request: Any
            urllib の要求
        timeout: float = 0.0
            待つ秒数

        Returns
        ----------
        response: Any
            中身を読める値
        """
        self.requests.append(request)
        if self.error == 0:
            raise urllib.error.URLError("つながらない")
        if self.error > 0:
            raise urllib.error.HTTPError(
                request.full_url, self.error, "誤り", None, None
            )
        response = io.BytesIO(self.body)

        return response


# 差し替える前の Hugging Face を読む関数
ORIGINAL_FETCH = pretrained.fetch_hub_metadata


def refuse_network(*args: Any, **kwargs: Any) -> Any:
    """
    テストでネットワークを使った場合に失敗させる関数

    Parameters
    ----------
    *args: Any
        urlopen の引数
    **kwargs: Any
        urlopen のキーワード引数

    Returns
    ----------
    None
    """
    raise AssertionError("テストでネットワークを使った")


@pytest.fixture(autouse=True)
def fake_hub(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeHub]:
    """
    Hugging Face を読む関数を差し替え, ネットワークを使わなくする関数

    Parameters
    ----------
    monkeypatch: pytest.MonkeyPatch
        差し替えの道具

    Returns
    ----------
    hub: Iterator[FakeHub]
        差し替えた関数 (呼ばれた記録を持つ)
    """
    # 覚えた結果と環境変数を消し, 読む関数を差し替え
    clear_pretrained_cache()
    for name in (
        "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_ENDPOINT", "HF_HUB_OFFLINE"
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(urllib.request, "urlopen", refuse_network)
    hub = FakeHub()
    monkeypatch.setattr(pretrained, "fetch_hub_metadata", hub)

    yield hub

    clear_pretrained_cache()


def hf_info(model_name: str, **params: Any) -> dict:
    """
    Hugging Face の部品のライセンスの情報を求める関数

    Parameters
    ----------
    model_name: str
        モデルの名前
    **params: Any
        他の引数の値

    Returns
    ----------
    info: dict
        ライセンスの情報
    """
    info = pretrained_info(
        "hf.AutoModel", {"model_name": model_name, **params}
    )
    assert info is not None

    return info


def test_classify_license() -> None:
    """
    ライセンスの識別子の分け方を確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 許諾の広いもの, 非商用, 表に無いもの (独自, 制限つき, 空)
    for license_id in ("apache-2.0", "MIT", "bsd-3-clause", "cc-by-4.0",
                       "cc0-1.0"):
        assert classify_license(license_id) == "commercial"
    for license_id in ("cc-by-nc-4.0", "cc-by-nc-sa-4.0", "cc-by-nc-1.0"):
        assert classify_license(license_id) == "noncommercial"
    for license_id in (
        "other", "llama3.1", "openrail", "creativeml-openrail-m", "gpl-3.0",
        "cc-by-nd-4.0", "", None
    ):
        assert classify_license(license_id) == "unknown"


def test_huggingface_commercial(fake_hub: FakeHub) -> None:
    """
    商用で使えるライセンスは警告せず, 版と再配布を返すことを確かめる関数

    Parameters
    ----------
    fake_hub: FakeHub
        差し替えた Hugging Face を読む関数

    Returns
    ----------
    None
    """
    info = hf_info("org/apache", revision="v1")
    assert info["status"] == "commercial"
    assert info["license"] == "apache-2.0"
    assert info["alert"] is False
    assert info["redistributable"] is True
    assert info["pinned_revision"] == "a" * 40
    assert info["revision"] == "v1"
    assert info["hub_id"] == "org/apache"
    assert info["url"] == "https://huggingface.co/org/apache"
    assert "Hugging Face のモデルの説明から読んだ" in info["source"]
    assert info["unit"] == "hf.AutoModel"
    assert fake_hub.calls == [("org/apache", "v1")]
    json.dumps(info)


def test_huggingface_alerts() -> None:
    """
    非商用, 申請制, 独自の条件, 空のライセンスを警告することを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 非商用は商用で使えないと警告し, 写さない
    info = hf_info("org/noncommercial")
    assert info["status"] == "noncommercial"
    assert info["alert"] is True
    assert info["redistributable"] is False
    assert "商用で使えない" in info["message"]

    # 申請制と独自の条件, 制限つき, 空は分からないもの
    gated = hf_info("org/gated")
    assert gated["status"] == "unknown" and gated["gated"] is True
    assert "申請制" in gated["message"]
    for name in ("org/other", "org/openrail", "org/empty"):
        info = hf_info(name)
        assert info["status"] == "unknown"
        assert info["alert"] is True
        assert info["redistributable"] is False
    assert hf_info("org/empty")["license"] is None


def test_license_from_tags_and_lists() -> None:
    """
    モデルの説明に無いライセンスをタグから, リストを先頭から読むことを
    確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    assert hf_info("org/tag-only")["license"] == "mit"
    assert hf_info("org/list")["license"] == "bsd-3-clause"

    # 複数のライセンスは最も厳しいもの
    mixed = hf_info("org/mixed")
    assert mixed["licenses"] == ["apache-2.0", "cc-by-nc-4.0"]
    assert mixed["status"] == "noncommercial"
    assert hf_info("org/bad-tags")["status"] == "unknown"


def test_invalid_inputs(fake_hub: FakeHub) -> None:
    """
    置き場の名前の形でない名前は読まず, 辞書でない引数は誤りにすることを
    確かめる関数

    Parameters
    ----------
    fake_hub: FakeHub
        差し替えた Hugging Face を読む関数

    Returns
    ----------
    None
    """
    for name in ("../../api/whoami-v2", "a/b/c", "org/..", "a b"):
        info = hf_info(name)
        assert info["status"] == "unknown" and info["url"] is None
    assert fake_hub.calls == []
    assert pretrained_info(
        "hf.AutoModel", {"model_name": "org/apache", "pretrained": ""}
    ) is None
    with pytest.raises(TypeError):
        pretrained_info("hf.AutoModel", ["org/apache"])  # type: ignore


def test_unreadable_and_cache(
    fake_hub: FakeHub, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    読めない場合は分からないものとし, 結果を覚えることを確かめる関数

    Parameters
    ----------
    fake_hub: FakeHub
        差し替えた Hugging Face を読む関数
    monkeypatch: pytest.MonkeyPatch
        差し替えの道具

    Returns
    ----------
    None
    """
    # 読めない場合は理由を添えて警告
    info = hf_info("org/missing")
    assert info["status"] == "unknown" and info["alert"] is True
    assert info["source"].startswith("Hugging Face から読めなかった")
    assert info["pinned_revision"] is None

    # 読めた結果も読めなかった結果も, 期限の内は読み直さない
    hf_info("org/apache")
    hf_info("org/apache")
    hf_info("org/missing")
    assert fake_hub.calls.count(("org/apache", "main")) == 1
    assert fake_hub.calls.count(("org/missing", "main")) == 1

    # トークンを設定したら読み直す
    monkeypatch.setenv("HF_TOKEN", "token")
    hf_info("org/missing")
    assert fake_hub.calls.count(("org/missing", "main")) == 2
    monkeypatch.delenv("HF_TOKEN")

    # 読めなかった結果は期限が過ぎたら読み直し, 消したら全て読み直す
    monkeypatch.setattr(pretrained, "FAILURE_CACHE_SECONDS", -1)
    hf_info("org/missing")
    assert fake_hub.calls.count(("org/missing", "main")) == 3
    clear_pretrained_cache()
    hf_info("org/apache")
    assert fake_hub.calls.count(("org/apache", "main")) == 2


def test_offline(fake_hub: FakeHub, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    ネットワークを使わない設定では読まず, 覚えた結果は使うことを確かめる
    関数

    Parameters
    ----------
    fake_hub: FakeHub
        差し替えた Hugging Face を読む関数
    monkeypatch: pytest.MonkeyPatch
        差し替えの道具

    Returns
    ----------
    None
    """
    # 読まない設定では分からないもの
    info = pretrained_info(
        "hf.AutoModel", {"model_name": "org/apache"}, fetch=False
    )
    assert info is not None and info["status"] == "unknown"
    assert "ネットワークを使わない設定" in info["source"]
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert hf_info("org/apache")["status"] == "unknown"
    assert fake_hub.calls == []

    # 覚えた結果があればネットワークを使わずに使う
    monkeypatch.delenv("HF_HUB_OFFLINE")
    hf_info("org/apache")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert hf_info("org/apache")["status"] == "commercial"
    assert len(fake_hub.calls) == 1


def test_no_weights_and_local(
    fake_hub: FakeHub, tmp_path: Path
) -> None:
    """
    重みを読まない場合は None, 手元のディレクトリは読まずに分からない
    ものとすることを確かめる関数

    Parameters
    ----------
    fake_hub: FakeHub
        差し替えた Hugging Face を読む関数
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    assert pretrained_info(
        "hf.AutoModel", {"model_name": "org/apache", "pretrained": False}
    ) is None
    assert pretrained_info("nn.Linear", {"in_features": 2}) is None
    info = hf_info(str(tmp_path))
    assert info["status"] == "unknown"
    assert "手元のファイル" in info["source"]
    assert fake_hub.calls == []
    with pytest.raises(KeyError):
        pretrained_info("nn.NoSuchUnit", {})


def test_weights_params_in_listing() -> None:
    """
    部品の一覧に重みを読む引数が入ることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    units = {unit["name"]: unit for unit in describe_units()}
    assert units["hf.AutoModel"]["weights_params"] == ["pretrained"]
    assert units["nn.Linear"]["weights_params"] == []


def test_fetch_hub_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Hugging Face の API を読む関数の URL, トークン, 誤りを確かめる関数

    Parameters
    ----------
    monkeypatch: pytest.MonkeyPatch
        差し替えの道具

    Returns
    ----------
    None
    """
    # 要求を記録する urlopen で, 差し替える前の関数を呼び出し
    requests: list = []
    monkeypatch.setenv("HF_TOKEN", "secret-token")
    monkeypatch.setenv("HF_ENDPOINT", "https://mirror.example/")
    monkeypatch.setattr(
        urllib.request, "urlopen", RecordingOpen(requests, b'{"sha": "x"}')
    )
    metadata = ORIGINAL_FETCH("org/model", "refs/pr/1")
    assert metadata == {"sha": "x"}
    request = requests[0]
    assert request.full_url == (
        "https://mirror.example/api/models/org/model/revision/refs%2Fpr%2F1"
    )
    assert request.unredirected_hdrs["Authorization"] == (
        "Bearer secret-token"
    )

    # 無いモデルとネットワークの誤りは読めない理由
    monkeypatch.setattr(
        urllib.request, "urlopen", RecordingOpen(requests, error=404)
    )
    with pytest.raises(pretrained.HubError, match="HF_TOKEN"):
        ORIGINAL_FETCH("org/model", "main")
    monkeypatch.setattr(
        urllib.request, "urlopen", RecordingOpen(requests, error=0)
    )
    with pytest.raises(pretrained.HubError, match="ネットワーク"):
        ORIGINAL_FETCH("org/model", "main")


def test_token_not_in_info(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    トークンが情報に入らないことを確かめる関数

    Parameters
    ----------
    monkeypatch: pytest.MonkeyPatch
        差し替えの道具

    Returns
    ----------
    None
    """
    monkeypatch.setenv("HF_TOKEN", "secret-token")
    info = hf_info("org/missing")
    assert "secret-token" not in json.dumps(info, ensure_ascii=False)


def test_torchvision_unknown(fake_hub: FakeHub) -> None:
    """
    torchvision の重みは読まずに分からないものとすることを確かめる関数

    Parameters
    ----------
    fake_hub: FakeHub
        差し替えた Hugging Face を読む関数

    Returns
    ----------
    None
    """
    pytest.importorskip("torchvision")
    info = pretrained_info(
        "tv.Model", {"name": "resnet18", "weights": "DEFAULT"}
    )
    assert info is not None and info["status"] == "unknown"
    assert info["source"].startswith("読めなかった")
    assert pretrained_info("tv.Model", {"name": "resnet18"}) is None
    assert fake_hub.calls == []


def test_timm(fake_hub: FakeHub, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    timm の重みを設定の置き場から読み, 読めなければ設定のライセンスを
    使うことを確かめる関数

    Parameters
    ----------
    fake_hub: FakeHub
        差し替えた Hugging Face を読む関数
    monkeypatch: pytest.MonkeyPatch
        差し替えの道具

    Returns
    ----------
    None
    """
    pytest.importorskip("timm")
    assert pretrained_info("timm.Model", {"model_name": "resnet18"}) is None

    # 設定の置き場が読めない場合は設定のライセンスを見せるが, 配布元で
    # 確かめていないので分からないもの
    info = pretrained_info(
        "timm.Model", {"model_name": "resnet18", "pretrained": True}
    )
    assert info is not None
    hub_id = info["hub_id"]
    assert hub_id and hub_id.startswith("timm/")
    assert fake_hub.calls == [(hub_id, "main")]
    assert info["source"].startswith("timm のモデルの設定から読んだ")
    assert info["license"] and info["status"] == "unknown"
    assert info["alert"] is True and info["pinned_revision"] is None

    # 置き場が読めた場合は Hugging Face のライセンスと版
    clear_pretrained_cache()
    monkeypatch.setitem(HUB, hub_id, HUB["org/noncommercial"])
    info = pretrained_info(
        "timm.Model", {"model_name": "resnet18", "pretrained": True}
    )
    assert info is not None and info["status"] == "noncommercial"
    assert info["pinned_revision"] == "b" * 40

    # 知らないモデルの名前は分からないもの
    info = pretrained_info(
        "timm.Model", {"model_name": "no_such_model", "pretrained": True}
    )
    assert info is not None and info["status"] == "unknown"


def test_smp(fake_hub: FakeHub) -> None:
    """
    segmentation_models_pytorch の符号化器の重みを, ライブラリの置き場と
    版から読むことを確かめる関数

    Parameters
    ----------
    fake_hub: FakeHub
        差し替えた Hugging Face を読む関数

    Returns
    ----------
    None
    """
    smp = pytest.importorskip("segmentation_models_pytorch")
    setting = smp.encoders.encoders["resnet34"]["pretrained_settings"][
        "imagenet"
    ]

    # 既定の重み (imagenet) は書かなくても読む
    info = pretrained_info("smp.Unet", {})
    assert info is not None and info["library"] == "smp"
    assert info["hub_id"] == setting["repo_id"]
    assert fake_hub.calls == [(setting["repo_id"], setting["revision"])]
    assert pretrained_info("smp.Unet", {"encoder_weights": None}) is None
