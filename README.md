# ML-ModelControl

PyTorch (と互換のライブラリ) で作れるモデルの部品を登録簿として持ち, 部品をつないだ定義 (YAML か JSON) から `nn.Module` を組み立てるライブラリ.

- `nn.Linear` のような層, 足し算や連結のような演算, PyTorch Geometric のグラフニューラルネットワークの層, Hugging Face の学習済みのモデルを, 同じ形の部品として扱う
- 分岐と合流のあるモデル (残差のつなぎ方, 複数の入力と出力) を定義できる
- 定義の確認 (部品の名前, 引数, つなぎ方) と, 各部品の出力の形とパラメータ数の推定を, 部品ごとの結果で返す. GUI で部品を組み立てるアプリ (ML-CreateArchitecture) はこれを使う

設計の方針は [docs/DESIGN.md](docs/DESIGN.md) を参照.

## 導入

```bash
git clone https://github.com/victor-von-pooh/ML-ModelControl.git
cd ML-ModelControl
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"                          # 基本
pip install -e ".[dev,all]"    # 外部のライブラリの部品も全て使う場合 (ライブラリごとに [geometric], [timm] なども選べる)
```

## 使い方

モデルの定義を YAML で書く (書き方は [docs/DESIGN.md](docs/DESIGN.md) の「モデルの定義の形」).

```yaml
# model.yaml
format_version: 1
name: residual_mlp
inputs:
  x: {shape: [4, 16]}
nodes:
  fc1: {unit: nn.Linear, params: {in_features: 16, out_features: 16}, inputs: x}
  act: {unit: nn.ReLU, inputs: fc1}
  add: {unit: op.Add, inputs: {xs: [x, act]}}
  head: {unit: nn.Linear, params: {in_features: 16, out_features: 3}, inputs: add}
outputs:
  logits: head
```

```python
from ml_modelcontrol import build_model, describe_units, infer, validate

issues = validate("model.yaml")          # 誤りと注意 (ノードごと)
result = infer("model.yaml")             # ノードごとの出力の形とパラメータ数
model = build_model("model.yaml", seed=0)  # nn.Module
logits = model(x)                        # 入力は定義の順の位置か, 名前のキーワード
units = describe_units()                 # 部品の一覧 (引数と口の説明付き)
```

外部のライブラリの部品は, そのライブラリを入れると使える (無い場合は一覧に「使えない」と理由と入れ方が出る).

| 任意の依存 | 部品 |
| --- | --- |
| `geometric` | PyTorch Geometric のグラフの層 (`pyg.GCNConv`, `pyg.MeanAggregation` など) |
| `huggingface` | Hugging Face のモデル (`hf.AutoModel`, `hf.AutoModelForSequenceClassification`, `hf.AutoModelForImageClassification` など) |
| `torchvision` | torchvision の層 (`tv.DeformConv2d`, `tv.SqueezeExcitation` など) と, 名前で作るモデル (`tv.Model`: `name: resnet18`) |
| `timm` | timm の層 (`timm.Mlp`, `timm.Attention`, `timm.DropPath` など) と, 名前で作るモデル (`timm.Model`: `model_name: vit_base_patch16_224`) |
| `torchaudio` | 音声の変換 (`audio.MelSpectrogram`, `audio.MFCC` など) とモデル (`audio.Conformer` など) |
| `einops` | `einops.Rearrange`, `einops.Reduce`, `einops.EinMix` |
| `kornia` | 画像の拡張, フィルタ, 色の変換 (`kornia.RandomHorizontalFlip`, `kornia.Normalize` など) |
| `monai` | 医用画像のネットワークと部品 (`monai.UNet`, `monai.SwinUNETR` など) |
| `smp` | 画像の領域分割のモデル (`smp.Unet`, `smp.FPN` など. segmentation_models_pytorch) |

学習済みの重みを読む引数 (Hugging Face の `pretrained`, `tv.Model` の `weights`, `timm.Model` の `pretrained`, `smp` の `encoder_weights` など) は, 形の推定 (`infer`) では重みを読まない値に置き換える (組み立てでは書いた値のまま). Hugging Face のモデルは `params` の `model_name` に名前かローカルのパスを書く. 外部のコード (`trust_remote_code`) は使わない.

同梱した例のモデルの定義 (全結合, 残差, CNN, LSTM, Transformer のエンコーダー, GCN, BERT の分類) は `list_examples()` で一覧にし, `load_example("cnn_classifier")` で読み込める. 自分のモデルを書き始めるときの元にできる.

`export_python("model.yaml")` で, 同じ計算をする `nn.Module` のクラスの Python のコードを書き出せる (`state_dict` のキーは `build_model` と同じ).

自作の部品は `register_unit(MyBlock, name="my.Block")` で登録すると, 定義の `unit` に書ける (`forward` が複数の値を返す場合は `outputs=["a", "b"]` で口の名前を付ける).

## 開発

```bash
pytest
```

コードの書き方は [CODING_STYLE.md](CODING_STYLE.md) に従う.

## ライセンス

MIT
