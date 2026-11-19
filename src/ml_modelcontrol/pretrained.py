import http.client
import json
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Optional

from ml_modelcontrol.libraries import import_quietly
from ml_modelcontrol.registry import UnitSpec, get_unit

# ライセンスの識別子 (Hugging Face の license の書き方. 小文字) ごとの分類.
# commercial は許諾の広い (商用で使えて再配布もできる. 表示の義務は残る)
# もの, noncommercial は商用で使えないもの. 表に無いもの (other, llama
# などの独自の条件, openrail などの利用の制限つき, GPL, cc-by-nd, cc-by-sa
# などの条件の確かめが要るもの) は全て unknown で, 確かめを促す警告
LICENSE_TABLE = {
    "apache-2.0": "commercial",
    "mit": "commercial",
    "bsd": "commercial",
    "bsd-2-clause": "commercial",
    "bsd-3-clause": "commercial",
    "bsd-3-clause-clear": "commercial",
    "isc": "commercial",
    "zlib": "commercial",
    "unlicense": "commercial",
    "cc0-1.0": "commercial",
    "cc-by-2.0": "commercial",
    "cc-by-2.5": "commercial",
    "cc-by-3.0": "commercial",
    "cc-by-4.0": "commercial",
    "cc-by-nc-2.0": "noncommercial",
    "cc-by-nc-3.0": "noncommercial",
    "cc-by-nc-4.0": "noncommercial",
    "cc-by-nc-nd-3.0": "noncommercial",
    "cc-by-nc-nd-4.0": "noncommercial",
    "cc-by-nc-sa-2.0": "noncommercial",
    "cc-by-nc-sa-3.0": "noncommercial",
    "cc-by-nc-sa-4.0": "noncommercial"
}

# 表に無い識別子のうち, 頭で商用で使えないと分かるもの (版の違い)
NONCOMMERCIAL_PREFIXES = ("cc-by-nc",)

# 分類の厳しさの順 (複数のライセンスを持つ重みは最も厳しいもの)
STATUS_ORDER = ("commercial", "unknown", "noncommercial")

# Hugging Face の置き場の名前の形 (組織の名前/モデルの名前. .. は除く)
HUB_ID_PATTERN = re.compile(r"[A-Za-z0-9_.-]+(/[A-Za-z0-9_.-]+)?")

# 分類ごとの画面の言葉
STATUS_TEXT = {
    "commercial": "商用で使える",
    "noncommercial": "商用で使えない",
    "unknown": "商用で使えるか分からない"
}

# 覚えた Hugging Face の結果を読み直さない秒数 (読めた結果は 1 日,
# 読めなかった結果はつながった後に読み直せるように 10 分)
CACHE_SECONDS = 24 * 60 * 60
FAILURE_CACHE_SECONDS = 10 * 60

# 覚える結果の数の上限 (画面から打ちかけの名前が来ても増え続けないように,
# 超えたら古いものから消す)
CACHE_LIMIT = 512

# Hugging Face の API を待つ秒数
FETCH_TIMEOUT = 10.0

# 置き場の名前, 版, トークンの有無ごとの Hugging Face の結果
# (読んだ時刻, 結果)
CACHE: dict[tuple[str, str, bool], tuple[float, dict]] = {}
CACHE_LOCK = threading.Lock()


class HubError(Exception):
    # Hugging Face の情報を読めなかった理由 (画面に出す日本語)
    pass


def classify_license(license_id: Optional[str]) -> str:
    """
    ライセンスの識別子を, 商用で使えるかで分ける関数

    Parameters
    ----------
    license_id: Optional[str]
        ライセンスの識別子 (apache-2.0 など), 読めなかった場合は None

    Returns
    ----------
    status: str
        commercial, noncommercial, unknown のどれか
    """
    # 表と頭で分け, それ以外は分からないもの
    key = (license_id or "").strip().lower()
    status = LICENSE_TABLE.get(key, "unknown")
    if status == "unknown" and key.startswith(NONCOMMERCIAL_PREFIXES):
        status = "noncommercial"

    return status


def classify_licenses(license_ids: list[str]) -> str:
    """
    複数のライセンスの識別子を, 最も厳しい分類にまとめる関数

    Parameters
    ----------
    license_ids: list[str]
        ライセンスの識別子のリスト (空なら分からないもの)

    Returns
    ----------
    status: str
        commercial, noncommercial, unknown のどれか
    """
    # 1 つも無ければ分からないもの, あれば厳しさの順で最大
    statuses = [classify_license(item) for item in license_ids]
    status = max(statuses or ["unknown"], key=STATUS_ORDER.index)

    return status


def valid_hub_id(hub_id: str) -> bool:
    """
    Hugging Face の置き場の名前として読める形かを返す関数

    Parameters
    ----------
    hub_id: str
        置き場の名前

    Returns
    ----------
    valid: bool
        組織の名前/モデルの名前 (か名前だけ) の形で, . と .. でない場合は
        True (API の別の道を読まないため)
    """
    parts = hub_id.split("/")
    valid = bool(HUB_ID_PATTERN.fullmatch(hub_id)) and all(
        part not in (".", "..") for part in parts
    )

    return valid


def hub_token() -> Optional[str]:
    """
    Hugging Face のトークンを環境変数から読む関数

    Parameters
    ----------
    None

    Returns
    ----------
    token: Optional[str]
        HF_TOKEN (無ければ HUGGING_FACE_HUB_TOKEN) の値, 無ければ None
    """
    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None
    )

    return token


def hub_endpoint() -> str:
    """
    Hugging Face の置き場の URL を返す関数

    Parameters
    ----------
    None

    Returns
    ----------
    endpoint: str
        HF_ENDPOINT の値 (無ければ https://huggingface.co), 末尾の / なし
    """
    endpoint = (
        os.environ.get("HF_ENDPOINT") or "https://huggingface.co"
    ).rstrip("/")

    return endpoint


def hub_offline() -> bool:
    """
    ネットワークを使わない設定 (HF_HUB_OFFLINE) かを返す関数

    Parameters
    ----------
    None

    Returns
    ----------
    offline: bool
        HF_HUB_OFFLINE が 1, true, yes, on のどれかなら True
    """
    value = os.environ.get("HF_HUB_OFFLINE", "").strip().lower()
    offline = value in ("1", "true", "yes", "on")

    return offline


def fetch_hub_metadata(hub_id: str, revision: str) -> dict:
    """
    Hugging Face の API からモデルの情報を読む関数

    テストではこの関数を差し替え, ネットワークを使わない

    Parameters
    ----------
    hub_id: str
        置き場の名前 (google-bert/bert-base-uncased など)
    revision: str
        版 (ブランチかコミット)

    Returns
    ----------
    metadata: dict
        API の JSON (cardData, tags, sha, gated などを持つ)
    """
    # 名前と版を URL の形にし, トークンがあれば添えて読み込み
    url = (
        f"{hub_endpoint()}/api/models/{urllib.parse.quote(hub_id)}"
        f"/revision/{urllib.parse.quote(revision, safe='')}"
    )
    request = urllib.request.Request(url)
    token = hub_token()
    if token:
        # 転送先 (別のホストのこともある) には渡さない
        request.add_unredirected_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(
            request, timeout=FETCH_TIMEOUT
        ) as response:
            metadata = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        if error.code in (401, 403, 404):
            raise HubError(
                "Hugging Face にモデルが無いか, 非公開 (トークンが要る場合は"
                "環境変数 HF_TOKEN に設定)"
            ) from error
        raise HubError(
            f"Hugging Face の応答の誤り (HTTP {error.code})"
        ) from error
    except (
        urllib.error.URLError, http.client.HTTPException, TimeoutError,
        OSError
    ) as error:
        raise HubError("ネットワークの誤り") from error
    except ValueError as error:
        raise HubError("Hugging Face の応答を読めない") from error
    if not isinstance(metadata, dict):
        raise HubError("Hugging Face の応答を読めない")

    return metadata


def licenses_from_metadata(metadata: dict) -> list[str]:
    """
    Hugging Face のモデルの情報からライセンスの識別子を取り出す関数

    Parameters
    ----------
    metadata: dict
        API の JSON

    Returns
    ----------
    license_ids: list[str]
        モデルの説明の license の全て (無ければ license: のタグの全て),
        無ければ空
    """
    # モデルの説明を先に, 無ければタグ (どちらもリストの全て)
    card = metadata.get("cardData")
    value = card.get("license") if isinstance(card, dict) else None
    values = value if isinstance(value, list) else [value]
    license_ids = [item for item in values if isinstance(item, str) and item]
    tags = metadata.get("tags")
    if not license_ids and isinstance(tags, list):
        license_ids = [
            tag.split(":", 1)[1] for tag in tags
            if isinstance(tag, str) and tag.startswith("license:")
            and tag.split(":", 1)[1]
        ]

    return license_ids


def read_hub(hub_id: str, revision: str, fetch: bool) -> dict:
    """
    Hugging Face のライセンス, コミット, 申請制かを, 覚えた結果か API から
    読む関数

    Parameters
    ----------
    hub_id: str
        置き場の名前
    revision: str
        版
    fetch: bool
        覚えた結果が無い場合にネットワークを使うか

    Returns
    ----------
    result: dict
        ok (読めたか), licenses, sha, gated, error (読めなかった理由)
    """
    # 期限の内の覚えた結果があれば使用 (トークンを設定したら読み直す)
    key = (hub_id, revision, hub_token() is not None)
    now = time.monotonic()
    with CACHE_LOCK:
        cached = CACHE.get(key)
    if cached is not None:
        limit = CACHE_SECONDS if cached[1]["ok"] else FAILURE_CACHE_SECONDS
        if now - cached[0] < limit:
            result = dict(cached[1])
            return result
    if not fetch or hub_offline():
        result = {
            "ok": False, "licenses": [], "sha": None, "gated": False,
            "error": "ネットワークを使わない設定"
        }
        return result

    # API から読み, 結果を覚える
    try:
        metadata = fetch_hub_metadata(hub_id, revision)
        sha = metadata.get("sha")
        result = {
            "ok": True, "licenses": licenses_from_metadata(metadata),
            "sha": sha if isinstance(sha, str) else None,
            "gated": bool(metadata.get("gated")), "error": ""
        }
    except HubError as error:
        result = {
            "ok": False, "licenses": [], "sha": None, "gated": False,
            "error": str(error)
        }
    with CACHE_LOCK:
        CACHE.pop(key, None)
        CACHE[key] = (now, dict(result))
        while len(CACHE) > CACHE_LIMIT:
            CACHE.pop(next(iter(CACHE)))

    return result


def clear_pretrained_cache() -> None:
    """
    覚えた Hugging Face の結果を消す関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    with CACHE_LOCK:
        CACHE.clear()


def loads_weights(spec: UnitSpec, params: dict) -> bool:
    """
    引数の値で, 部品が学習済みの重みを読むかを返す関数

    Parameters
    ----------
    spec: UnitSpec
        部品
    params: dict
        ノードの引数の値 (書いていない引数は既定値)

    Returns
    ----------
    loads: bool
        重みを読む引数のどれかが, 読まない値 (offline) と違い空でも偽でも
        ない場合は True
    """
    # 既定値に書いた値を重ねて, 読まない値と比べる
    values = {param.name: param.default for param in spec.params}
    values.update(params)
    loads = False
    for name, offline in spec.offline.items():
        value = values.get(name)
        if value != offline and value not in (None, False, ""):
            loads = True

    return loads


def make_info(
    library: str, model: str, hub_id: Optional[str] = None,
    revision: Optional[str] = None
) -> dict:
    """
    ライセンスの情報の辞書の雛形を作る関数

    Parameters
    ----------
    library: str
        重みの出どころのライブラリ (huggingface, timm など)
    model: str
        書いたモデルの名前
    hub_id: Optional[str] = None
        Hugging Face の置き場の名前
    revision: Optional[str] = None
        求めた版

    Returns
    ----------
    info: dict
        分からないものとして埋めた情報
    """
    info = {
        "library": library, "model": model, "hub_id": hub_id,
        "revision": revision, "pinned_revision": None, "license": None,
        "licenses": [], "confirmed": False, "status": "unknown",
        "redistributable": False, "gated": False,
        "alert": True, "message": "", "source": "",
        "url": f"{hub_endpoint()}/{hub_id}" if hub_id else None
    }

    return info


def fill_from_hub(info: dict, fetch: bool) -> bool:
    """
    Hugging Face の情報でライセンス, 版, 申請制かを埋める関数

    Parameters
    ----------
    info: dict
        hub_id と revision を持つ情報 (書き換える)
    fetch: bool
        ネットワークを使うか

    Returns
    ----------
    ok: bool
        読めた場合は True, 読めなかった場合は False (source に理由)
    """
    # 置き場の名前の形でないものと, . か .. の版は読まない (API の別の道を
    # 読まないため)
    revision = info["revision"] or "main"
    if not valid_hub_id(info["hub_id"]) or revision in (".", ".."):
        info["source"] = (
            f"読めなかった ({info['hub_id']} は Hugging Face の置き場の名前の"
            "形でない)"
        )
        info["url"] = None
        return False

    # 読めた場合はライセンスと版, 読めなかった場合は理由を記録
    result = read_hub(info["hub_id"], revision, fetch)
    ok = bool(result["ok"])
    if ok:
        info["licenses"] = list(result["licenses"])
        info["license"] = ", ".join(info["licenses"]) or None
        info["confirmed"] = True
        info["pinned_revision"] = result["sha"]
        info["gated"] = result["gated"]
        info["source"] = (
            f"Hugging Face のモデルの説明から読んだ ({info['hub_id']})"
        )
    else:
        info["source"] = (
            f"Hugging Face から読めなかった ({info['hub_id']}: "
            f"{result['error']})"
        )

    return ok


def timm_config(model_name: str) -> Any:
    """
    timm のモデルの学習済みの重みの設定を返す関数

    Parameters
    ----------
    model_name: str
        timm のモデルの名前 (resnet50.a1_in1k のように重みの名前も可)

    Returns
    ----------
    config: Any
        timm の PretrainedCfg (hf_hub_id, license を持つ), 無ければ None
    """
    # timm が無い場合と, 知らない名前の場合は None
    timm, _ = import_quietly("timm.models")
    config = None
    if timm is not None:
        try:
            config = timm.get_pretrained_cfg(model_name)
        except Exception:
            config = None

    return config


def timm_info(model_name: str, fetch: bool) -> dict:
    """
    timm のモデルの重みのライセンスの情報を作る関数

    Parameters
    ----------
    model_name: str
        timm のモデルの名前
    fetch: bool
        ネットワークを使うか

    Returns
    ----------
    info: dict
        ライセンスの情報
    """
    # 設定の Hugging Face の置き場 (@ の後ろは版) を先に読み, 読めなければ
    # 設定の license を参考に表示 (配布元で確かめていないので分からないもの)
    config = timm_config(model_name)
    hub_id = getattr(config, "hf_hub_id", None) or None
    revision = "main" if hub_id else None
    if hub_id and "@" in hub_id:
        hub_id, revision = hub_id.split("@", 1)
    info = make_info("timm", model_name, hub_id, revision)
    if config is None:
        info["source"] = (
            "読めなかった (timm が無いか, timm の知らないモデルの名前)"
        )
        return info
    if hub_id and fill_from_hub(info, fetch):
        return info
    license_id = getattr(config, "license", None) or None
    if license_id:
        info["license"] = license_id
        info["source"] = (
            "timm のモデルの設定から読んだ (配布元の説明では確かめていない)"
            + (f". {info['source']}" if info["source"] else "")
        )
    elif not info["source"]:
        info["source"] = "読めなかった (timm のモデルの設定に情報が無い)"

    return info


def smp_info(params: dict, fetch: bool) -> dict:
    """
    segmentation_models_pytorch の符号化器の重みのライセンスの情報を作る
    関数

    Parameters
    ----------
    params: dict
        ノードの引数の値 (encoder_name, encoder_weights)
    fetch: bool
        ネットワークを使うか

    Returns
    ----------
    info: dict
        ライセンスの情報
    """
    # timm の符号化器は timm と同じに読む
    encoder = str(params.get("encoder_name") or "resnet34")
    weights = str(params.get("encoder_weights"))
    if encoder.startswith("tu-"):
        info = timm_info(encoder[3:], fetch)
        info["library"] = "smp"
        return info

    # ライブラリが持つ置き場の名前と版から Hugging Face を読む
    smp, _ = import_quietly("segmentation_models_pytorch.encoders")
    settings: Any = {}
    if smp is not None:
        encoders = getattr(smp, "encoders", {})
        settings = (
            encoders.get(encoder, {}).get("pretrained_settings", {})
            .get(weights, {})
        )
    hub_id = settings.get("repo_id") if isinstance(settings, dict) else None
    info = make_info(
        "smp", f"{encoder} ({weights})", hub_id,
        (settings.get("revision") or "main") if hub_id else None
    )
    if hub_id:
        fill_from_hub(info, fetch)
    else:
        info["source"] = (
            "読めなかった (segmentation_models_pytorch が無いか, 符号化器と"
            "重みの組に置き場の情報が無い)"
        )

    return info


def huggingface_info(params: dict, fetch: bool) -> dict:
    """
    Hugging Face のモデルの重みのライセンスの情報を作る関数

    Parameters
    ----------
    params: dict
        ノードの引数の値 (model_name, revision)
    fetch: bool
        ネットワークを使うか

    Returns
    ----------
    info: dict
        ライセンスの情報
    """
    # 手元のディレクトリは配布元が分からないので読まない
    model_name = str(params.get("model_name") or "")
    revision = str(params.get("revision") or "main")
    if not model_name or os.path.isdir(model_name):
        info = make_info("huggingface", model_name)
        info["source"] = (
            "読めなかった (手元のファイルで, 配布元の情報が無い)"
            if model_name else "読めなかった (モデルの名前が空)"
        )
        return info
    info = make_info("huggingface", model_name, model_name, revision)
    fill_from_hub(info, fetch)

    return info


def library_info(spec: UnitSpec, params: dict) -> dict:
    """
    機械で読めるライセンスの情報の無いライブラリの重みの情報を作る関数

    Parameters
    ----------
    spec: UnitSpec
        部品 (torchvision, MONAI など)
    params: dict
        ノードの引数の値

    Returns
    ----------
    info: dict
        分からないものとしての情報
    """
    # モデルの名前と重みの名前が分かれば並べて表示
    names = [
        str(params[key]) for key in ("name", "weights") if params.get(key)
    ]
    info = make_info(spec.source, " ".join(names) or spec.name)
    info["source"] = (
        f"読めなかった ({spec.source} の重みには機械で読めるライセンスの"
        "情報が無い. 配布元のページで確かめる)"
    )

    return info


def finish_info(info: dict) -> dict:
    """
    読んだライセンスから分類, 再配布, 警告と説明を埋める関数

    Parameters
    ----------
    info: dict
        licenses, confirmed, gated, source を持つ情報 (書き換える)

    Returns
    ----------
    info: dict
        status, redistributable, alert, message を埋めた情報
    """
    # 複数のライセンスは最も厳しいもの. 配布元の説明で確かめていないもの
    # (読めなかった) と申請制の重みは, ライセンスに依らず分からないもの
    status = classify_licenses(info["licenses"])
    if not info["confirmed"] or info["gated"]:
        status = "unknown"
    info["status"] = status
    info["redistributable"] = status == "commercial"
    info["alert"] = status != "commercial"

    # 画面にそのまま出す説明 (ライセンス, 分類, 次の一手)
    license_text = info["license"] or "不明"
    message = f"ライセンス: {license_text} ({STATUS_TEXT[status]})"
    if info["gated"]:
        message += ". 申請制の重み (配布元のページで条件を確かめる)"
    elif status == "noncommercial":
        message += ". 研究や評価だけに使い, 商用の製品には使わない"
    elif not info["confirmed"]:
        message += ". 配布元の説明を読めなかったので, 配布元のページで確かめる"
    elif status == "unknown" and info["license"]:
        message += ". 独自の条件か制限つきなので, 配布元の条件を原文で確かめる"
    elif status == "unknown":
        message += ". 配布元のページで条件を確かめる"
    info["message"] = message

    return info


def pretrained_info(
    unit: str, params: Optional[dict] = None, fetch: bool = True
) -> Optional[dict]:
    """
    ノードが読む学習済みの重みのライセンスの情報を返す関数

    ライセンスは配布元 (Hugging Face のモデルの説明など) から読み, 商用で
    使えないものと分からないものは警告 (alert) にする

    Parameters
    ----------
    unit: str
        部品の名前 (hf.AutoModel, timm.Model など)
    params: Optional[dict] = None
        ノードの引数の値
    fetch: bool = True
        覚えた結果が無い場合にネットワークを使うか

    Returns
    ----------
    info: Optional[dict]
        ライセンスの情報 (unit, library, model, hub_id, revision,
        pinned_revision, license, licenses, confirmed, status,
        redistributable, gated, alert, message, source, url), 重みを
        読まない場合は None
    """
    # 重みを読まない部品と引数の値では情報なし
    if params is not None and not isinstance(params, dict):
        raise TypeError("params は引数の名前と値の辞書にしてください")
    spec = get_unit(unit)
    values = {param.name: param.default for param in spec.params}
    values.update(params or {})
    if not loads_weights(spec, values):
        return None

    # 部品の出どころごとに配布元の情報を読み, 分類と説明を付加
    if spec.source == "transformers":
        info = huggingface_info(values, fetch)
    elif spec.name == "timm.Model":
        info = timm_info(str(values.get("model_name") or ""), fetch)
    elif spec.source == "segmentation_models_pytorch" and (
        values.get("encoder_weights") is not None
    ):
        info = smp_info(values, fetch)
    else:
        info = library_info(spec, values)
    info["unit"] = unit
    info = finish_info(info)

    return info
