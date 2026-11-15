import importlib
import importlib.util
import inspect
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from torch import nn

from ml_modelcontrol.registry import (
    REGISTRY, ParamSpec, PortSpec, UnitSpec, apply_notes, describe_params,
    needs_module, params_from_init, ports_from_forward, register
)

# 学習済みの重みを読む引数と, 形の推定で置き換える「読まない」値
OFFLINE_VALUES = {
    "pretrained": False, "pretrained_backbone": False,
    "encoder_weights": None
}


@dataclass
class ClassSource:
    # ライブラリの名前 (入っているかの確認), クラスを集めるモジュール
    package: str
    module: str

    # 部品の名前の頭 (tv など), 出どころ, 分類
    prefix: str
    source: str
    category: str

    # 書き出すコードでモジュールに付ける別名 (tv_ops など)
    alias: str

    # 同じ頭の他のモジュールと名前が重なる場合に挟む名前 (enhance など.
    # kornia.enhance.Normalize のようにする)
    sub: str

    # ライブラリが無い場合も一覧に出す, よく使う部品
    common: list[str] = field(default_factory=list)


# クラスを集めて部品にするライブラリのモジュール
CLASS_SOURCES = [
    ClassSource(
        "torchvision", "torchvision.ops", "tv", "torchvision", "tv_ops",
        "tv_ops", "ops",
        ["DeformConv2d", "SqueezeExcitation", "StochasticDepth",
         "Conv2dNormActivation", "FrozenBatchNorm2d", "MLP"]
    ),
    ClassSource(
        "timm", "timm.layers", "timm", "timm", "timm_layers", "timm_layers",
        "layers", ["Mlp", "Attention", "DropPath", "PatchEmbed", "LayerNorm2d"]
    ),
    ClassSource(
        "torchaudio", "torchaudio.transforms", "audio", "torchaudio",
        "audio_transforms", "ta_transforms", "transforms",
        ["Spectrogram", "MelSpectrogram", "MFCC", "AmplitudeToDB",
         "Resample", "FrequencyMasking", "TimeMasking"]
    ),
    ClassSource(
        "torchaudio", "torchaudio.models", "audio", "torchaudio",
        "audio_models", "ta_models", "models",
        ["Conformer", "Wav2Letter", "ConvTasNet"]
    ),
    ClassSource(
        "einops", "einops.layers.torch", "einops", "einops", "einops_layers",
        "einops_layers", "layers", ["Rearrange", "Reduce", "EinMix"]
    ),
    ClassSource(
        "kornia", "kornia.augmentation", "kornia", "kornia",
        "kornia_augmentation", "kornia_aug", "augmentation",
        ["RandomHorizontalFlip", "RandomResizedCrop", "ColorJitter",
         "Normalize", "Resize"]
    ),
    ClassSource(
        "kornia", "kornia.filters", "kornia", "kornia", "kornia_filters",
        "kornia_filters", "filters", ["GaussianBlur2d", "Sobel"]
    ),
    ClassSource(
        "kornia", "kornia.color", "kornia", "kornia", "kornia_color",
        "kornia_color", "color", ["RgbToGrayscale", "RgbToHsv"]
    ),
    ClassSource(
        "kornia", "kornia.enhance", "kornia", "kornia", "kornia_enhance",
        "kornia_enhance", "enhance", []
    ),
    ClassSource(
        "kornia", "kornia.geometry.transform", "kornia", "kornia",
        "kornia_geometry", "kornia_geometry", "geometry", []
    ),
    ClassSource(
        "monai", "monai.networks.nets", "monai", "monai", "monai_nets",
        "monai_nets", "nets",
        ["UNet", "BasicUNet", "SwinUNETR", "DenseNet121"]
    ),
    ClassSource(
        "monai", "monai.networks.blocks", "monai", "monai", "monai_blocks",
        "monai_blocks", "blocks", ["Convolution", "ResidualUnit"]
    ),
    ClassSource(
        "segmentation_models_pytorch", "segmentation_models_pytorch", "smp",
        "segmentation_models_pytorch", "smp_models", "smp", "models",
        ["Unet", "FPN", "DeepLabV3Plus"]
    )
]

# シグネチャからは分からない, 定義から作れないクラスと理由
UNAVAILABLE_CLASSES = {
    "kornia.RandomDissolving": "diffusers のモデルが必要なので, 今は使えません",
    "kornia.StableDiffusionDissolving": (
        "diffusers のモデルが必要なので, 今は使えません"
    ),
    "monai.BertMixedLayer": (
        "引数に設定のオブジェクトが必要なので, 今は定義から作れません"
    ),
    "monai.Transchex": (
        "作るときに必ず BERT の重みを取得するので, 今は定義から作れません"
    ),
    "monai.MultiModal": (
        "作るときに必ず BERT の重みを取得するので, 今は定義から作れません"
    )
}

# 部品ごとの, 学習済みの重みを読む引数の「読まない」値 (共通の名前以外)
CLASS_OFFLINE = {
    "monai.HoVerNet": {"pretrained_url": None}
}

# ライブラリごとの, 入れる方法の案内 (pip の任意の依存の名前)
EXTRAS = {
    "torchvision": "torchvision", "timm": "timm", "torchaudio": "torchaudio",
    "einops": "einops", "kornia": "kornia", "monai": "monai",
    "segmentation_models_pytorch": "smp"
}

# よく使う部品の説明
CLASS_NOTES: dict[str, dict] = {
    "tv.DeformConv2d": {
        "description": (
            "変形畳み込み層. offset に (バッチ, 2 * カーネルの要素数, 高さ, "
            "幅) のずれをつなぐ"
        )
    },
    "tv.SqueezeExcitation": {
        "description": "チャンネルごとの重みを掛ける注意 (Squeeze-and-"
                       "Excitation)"
    },
    "tv.StochasticDepth": {
        "description": "学習のときに確率 p で入力を 0 にする (残差の枝に使う)"
    },
    "tv.Conv2dNormActivation": {
        "description": "畳み込み, 正規化, 活性化関数をまとめた層"
    },
    "timm.DropPath": {
        "description": "学習のときに確率 drop_prob でサンプルごとに枝を落とす"
    },
    "timm.Mlp": {"description": "全結合, 活性化関数, 全結合の層 (Transformer "
                              "の MLP)"},
    "timm.Attention": {
        "description": "Vision Transformer の自己注意. 入力は (バッチ, "
                       "系列, dim)"
    },
    "timm.PatchEmbed": {
        "description": "画像を小さな区画 (パッチ) に分けて特徴量にする"
    },
    "audio.Spectrogram": {"description": "波形からスペクトログラムを求める"},
    "audio.MelSpectrogram": {
        "description": "波形からメルスペクトログラムを求める. 入力は (..., "
                       "時間) の波形"
    },
    "audio.MFCC": {"description": "波形からメル周波数ケプストラム係数を求める"},
    "audio.AmplitudeToDB": {"description": "振幅か強さをデシベルにする"},
    "einops.Rearrange": {
        "description": (
            "式で次元を並べ替える (例: 'b c h w -> b (h w) c'). 式の中の "
            "大きさは, 一覧に無い引数 (h=2 など) で渡す"
        )
    },
    "einops.Reduce": {
        "description": "式で次元をまとめる (例: 'b c h w -> b c', "
                       "reduction='mean')",
        "params": {
            "reduction": {
                "description": "まとめ方",
                "choices": ["mean", "sum", "max", "min", "prod"]
            }
        }
    },
    "einops.EinMix": {
        "description": "式で重みを掛けて次元を混ぜる (全結合の一般化)"
    },
    "kornia.Normalize": {"description": "チャンネルごとに平均と標準偏差で"
                                        "正規化する"},
    "monai.UNet": {"description": "医用画像の領域分割の U-Net"},
    "smp.Unet": {
        "description": (
            "符号化器 (encoder_name) を選べる U-Net. 学習済みの符号化器は "
            "encoder_weights (imagenet など) で読む"
        )
    }
}


def has_module(name: str) -> bool:
    """
    ライブラリが入っているかを, 読み込まずに確かめる関数

    Parameters
    ----------
    name: str
        ライブラリの名前

    Returns
    ----------
    found: bool
        入っていれば True
    """
    found = importlib.util.find_spec(name) is not None

    return found


def import_quietly(name: str) -> tuple[Any, str]:
    """
    モジュールを, 読み込みの警告を表示せずに読み込む関数

    Parameters
    ----------
    name: str
        モジュールの名前

    Returns
    ----------
    module: Any
        読み込んだモジュール, 読み込めない場合は None
    reason: str
        読み込めない場合の理由, 読み込めた場合は空
    """
    # ライブラリが無い場合と, 読み込みで失敗した場合 (版の食い違いなど)
    package = name.split(".")[0]
    if not has_module(package):
        extra = EXTRAS.get(package, package)
        return None, (
            f"{package} が入っていません (pip install "
            f"\"ml-modelcontrol[{extra}]\")"
        )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            module = importlib.import_module(name)
    except Exception as error:
        return None, (
            f"{name} を読み込めません: {type(error).__name__}: {error}"
        )

    return module, ""


def skipped_class(name: str, cls: Any) -> bool:
    """
    部品にしないクラス (土台, 入れ物, 別名) かを返す関数

    Parameters
    ----------
    name: str
        モジュールで公開された名前
    cls: Any
        その名前の値

    Returns
    ----------
    skipped: bool
        部品にしない場合は True
    """
    # nn.Module のクラス以外, 抽象クラス, 土台と入れ物, 別名は除外
    skipped = (
        not inspect.isclass(cls) or not issubclass(cls, nn.Module)
        or name != cls.__name__ or name.startswith("_")
        or inspect.isabstract(cls) or "Base" in name
        or name.endswith(("Sequential", "List", "Dict"))
    )

    return skipped


def make_class_builder(cls: type) -> Callable[..., nn.Module]:
    """
    引数の値からクラスの部品を作る関数を作る関数

    Parameters
    ----------
    cls: type
        nn.Module のクラス

    Returns
    ----------
    builder: Callable[..., nn.Module]
        引数を受け取って部品を作る関数
    """
    def builder(**values: Any) -> nn.Module:
        """
        引数の値から部品を作る関数

        Parameters
        ----------
        **values: Any
            クラスに渡す引数

        Returns
        ----------
        module: nn.Module
            作った部品
        """
        module = cls(**values)

        return module

    return builder


def register_class(
    cls: type, name: str, category: str, source: str, code_name: str,
    code_import: str, notes: Optional[dict] = None
) -> UnitSpec:
    """
    外部のライブラリのクラスを部品として登録する関数

    引数と口は nn.Module のクラスのシグネチャから取り出す

    Parameters
    ----------
    cls: type
        nn.Module のクラス
    name: str
        部品の名前 (tv.DeformConv2d など)
    category: str
        分類
    source: str
        出どころ (ライブラリの名前)
    code_name: str
        書き出すコードで作る式に使う名前 (tv_ops.DeformConv2d など)
    code_import: str
        書き出すコードの import の行
    notes: Optional[dict] = None
        手で書いた説明と範囲

    Returns
    ----------
    spec: UnitSpec
        登録した部品
    """
    # シグネチャから引数と口を取り出し, 重みを読む引数を推定のために記録
    params, extra = params_from_init(cls)
    inputs, forward_params = ports_from_forward(cls)
    offline = {
        param.name: OFFLINE_VALUES[param.name] for param in params
        if param.name in OFFLINE_VALUES
    }
    offline.update(CLASS_OFFLINE.get(name, {}))
    spec = UnitSpec(
        name=name, category=category, source=source, params=params,
        inputs=inputs, outputs=[PortSpec("out")],
        forward_params=forward_params, extra_params=extra,
        builder=make_class_builder(cls), code_name=code_name,
        code_import=code_import, offline=offline
    )
    apply_notes(spec, notes or {})
    describe_params(spec, cls)

    # Kornia の数の組の引数は, 型ヒントと違い数 1 つも受け取るものが多いので,
    # 数も受け付ける (正しい値を確認で誤りにしないため)
    if source == "kornia":
        for param in spec.params:
            if param.types == ["list"]:
                param.types = ["list", "float"]

    # 別の部品やテンソルを引数に取るクラスと, 辞書の入力を取るクラス
    # (異種グラフの層など) は今は対象外
    if needs_module(cls):
        spec.available = False
        spec.reason = (
            "引数に別の部品 (nn.Module) かテンソルが必要なので, 今は定義から"
            "作れません"
        )
    if any(port.name.endswith("_dict") for port in inputs):
        spec.available = False
        spec.reason = "辞書の入力は, 今は定義からつなげません"
    if name in UNAVAILABLE_CLASSES:
        spec.available = False
        spec.reason = UNAVAILABLE_CLASSES[name]

    # 他の口の後ろの *args は省略できる口 (追加の値)
    if len(inputs) > 1:
        for port in inputs:
            if port.variadic:
                port.optional = True
    register(spec)

    return spec


def register_unavailable(
    name: str, category: str, source: str, reason: str,
    description: str = ""
) -> None:
    """
    入っていないライブラリの部品を, 使えないものとして登録する関数

    Parameters
    ----------
    name: str
        部品の名前
    category: str
        分類
    source: str
        出どころ
    reason: str
        使えない理由
    description: str = ""
        部品の説明

    Returns
    ----------
    None
    """
    register(UnitSpec(
        name=name, category=category, source=source,
        description=description, available=False, reason=reason
    ))


def load_class_source(item: ClassSource) -> None:
    """
    ライブラリのモジュールのクラスを集めて, 部品として登録する関数

    ライブラリが無い場合は, よく使う部品を使えないものとして登録する.
    同じ頭の名前が他のモジュールで登録済みの場合は, 頭の後ろにモジュール
    の名前を挟む (kornia.enhance.Normalize など)

    Parameters
    ----------
    item: ClassSource
        集めるモジュール

    Returns
    ----------
    None
    """
    # 読み込めない場合は, よく使う部品を理由付きで使えないものにして終了
    module, reason = import_quietly(item.module)
    if module is None:
        for name in item.common:
            unit_name = f"{item.prefix}.{name}"
            if unit_name not in REGISTRY:
                register_unavailable(
                    unit_name, item.category, item.source, reason,
                    CLASS_NOTES.get(unit_name, {}).get("description", "")
                )
        return

    # 公開された名前のクラスを順に登録 (取り出せないクラスは除外)
    # 同じ頭の他のモジュールで登録済みの名前は, モジュールの名前を挟む
    parent, _, leaf = item.module.rpartition(".")
    code_import = (
        f"from {parent} import {leaf} as {item.alias}" if parent
        else f"import {item.module} as {item.alias}"
    )
    names = getattr(module, "__all__", None) or dir(module)
    for name in names:
        cls = getattr(module, name, None)
        if skipped_class(name, cls):
            continue
        unit_name = f"{item.prefix}.{name}"
        taken = REGISTRY.get(unit_name)
        if taken is not None and taken.source == item.source and (
            taken.code_import != code_import
        ):
            unit_name = f"{item.prefix}.{item.sub}.{name}"
        try:
            register_class(
                cls, unit_name, item.category, item.source,
                f"{item.alias}.{name}", code_import,
                CLASS_NOTES.get(unit_name)
            )
        except Exception:
            continue


def make_factory_builder(
    factory: Callable[..., nn.Module]
) -> Callable[..., nn.Module]:
    """
    名前でモデルを作る関数を, 部品を作る関数にする関数

    Parameters
    ----------
    factory: Callable[..., nn.Module]
        torchvision.models.get_model や timm.create_model

    Returns
    ----------
    builder: Callable[..., nn.Module]
        引数を受け取ってモデルを作る関数
    """
    def builder(**values: Any) -> nn.Module:
        """
        引数の値からモデルを作る関数

        Parameters
        ----------
        **values: Any
            モデルの名前と, モデルに渡す引数

        Returns
        ----------
        module: nn.Module
            作ったモデル
        """
        module = factory(**values)

        return module

    return builder


def make_torchvision_builder(models: Any) -> Callable[..., nn.Module]:
    """
    torchvision のモデルを名前で作る関数を作る関数

    Parameters
    ----------
    models: Any
        torchvision.models のモジュール

    Returns
    ----------
    builder: Callable[..., nn.Module]
        引数を受け取ってモデルを作る関数
    """
    def builder(name: str, **values: Any) -> nn.Module:
        """
        モデルの名前と引数からモデルを作る関数

        形の推定で加える weights_backbone=None は, その引数を持つモデル
        (検出と領域分割のモデル) にだけ渡す

        Parameters
        ----------
        name: str
            モデルの名前 (resnet18 など)
        **values: Any
            モデルに渡す引数

        Returns
        ----------
        module: nn.Module
            作ったモデル
        """
        accepted = inspect.signature(
            models.get_model_builder(name)
        ).parameters
        if values.get("weights_backbone", 0) is None and (
            "weights_backbone" not in accepted
        ):
            values.pop("weights_backbone")
        module = models.get_model(name, **values)

        return module

    return builder


def load_torchvision_models() -> None:
    """
    torchvision のモデルを名前で作る部品 (tv.Model) を登録する関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 名前の候補はライブラリにあるモデルの一覧
    models, reason = import_quietly("torchvision.models")
    spec = UnitSpec(
        name="tv.Model", category="tv_models", source="torchvision",
        description=(
            "torchvision のモデルを名前 (resnet18, vit_b_16 など) で作る. "
            "入力は (バッチ, 3, 高さ, 幅) の画像"
        ),
        params=[
            ParamSpec(
                "name", "str", required=True, description="モデルの名前",
                suggestions=(
                    sorted(models.list_models()) if models else None
                )
            ),
            ParamSpec(
                "weights", "str", None,
                description=(
                    "学習済みの重み (DEFAULT か IMAGENET1K_V1 など). 空なら"
                    "読まない"
                )
            ),
            ParamSpec(
                "num_classes", "int", None, minimum=1,
                description="分類のクラスの数 (空ならモデルの既定)"
            )
        ],
        inputs=[PortSpec("x", "画像 (バッチ, 3, 高さ, 幅)")],
        outputs=[PortSpec("out")], extra_params=True,
        available=models is not None, reason=reason,
        code_name="tv_models.get_model",
        code_import="from torchvision import models as tv_models",
        offline={"weights": None, "weights_backbone": None}
    )
    if models is not None:
        spec.builder = make_torchvision_builder(models)
    register(spec)


def load_timm_models() -> None:
    """
    timm のモデルを名前で作る部品 (timm.Model) を登録する関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 名前の候補はライブラリにあるモデルの一覧
    timm, reason = import_quietly("timm")
    spec = UnitSpec(
        name="timm.Model", category="timm_models", source="timm",
        description=(
            "timm のモデルを名前 (resnet50, vit_base_patch16_224 など) で"
            "作る. features_only を true にすると途中の特徴量のリストを返す"
        ),
        params=[
            ParamSpec(
                "model_name", "str", required=True,
                description="モデルの名前",
                suggestions=sorted(timm.list_models()) if timm else None
            ),
            ParamSpec(
                "pretrained", "bool", False,
                description="学習済みの重みを読むか (ネットワークを使う)"
            ),
            ParamSpec(
                "num_classes", "int", None, minimum=0,
                description=(
                    "分類のクラスの数 (0 なら分類の層を外す. 空ならモデルの"
                    "既定)"
                )
            ),
            ParamSpec(
                "in_chans", "int", None, minimum=1,
                description="入力のチャンネル数 (空なら 3)"
            ),
            ParamSpec(
                "features_only", "bool", False,
                description=(
                    "途中の特徴量のリストを返すか (検出や領域分割の符号化器に"
                    "使う)"
                )
            ),
            ParamSpec(
                "out_indices", "any", None,
                description=(
                    "features_only の場合に返す段の番号のリスト (例: [1, 2, 3]."
                    " 空なら全て)"
                )
            ),
            ParamSpec(
                "drop_rate", "float", None, minimum=0.0, maximum=1.0,
                description="分類の層の前のドロップアウトの確率 (空なら 0)"
            )
        ],
        inputs=[PortSpec("x", "画像 (バッチ, チャンネル, 高さ, 幅)")],
        outputs=[PortSpec("out")], extra_params=True,
        available=timm is not None, reason=reason,
        code_name="timm.create_model", code_import="import timm",
        offline={"pretrained": False}
    )
    if timm is not None:
        spec.builder = make_factory_builder(timm.create_model)
    register(spec)


def load_library_units() -> None:
    """
    外部のライブラリの部品を登録簿に読み込む関数

    torchvision, timm, torchaudio, einops, Kornia, MONAI,
    segmentation_models_pytorch が対象

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 名前で作るモデルを先に登録 (同じ名前のクラスでの上書きはなし)
    load_torchvision_models()
    load_timm_models()
    for item in CLASS_SOURCES:
        load_class_source(item)
