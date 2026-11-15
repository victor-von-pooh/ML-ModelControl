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
import torch
from ml_modelcontrol import (
    build_model, describe_units, dump_spec, export_python, get_unit, infer,
    load_spec, validate
)

issues = validate("model.yaml")          # 誤りと注意 (ノードごと)
result = infer("model.yaml")             # ノードごとの出力の形とパラメータ数
model = build_model("model.yaml", seed=0)  # nn.Module (GraphModel)
logits = model(torch.zeros(4, 16))       # 入力は定義の順の位置か, 名前のキーワード
units = describe_units()                 # 部品の一覧 (引数と口の説明付き. JSON にできる辞書)
unit = get_unit("nn.Linear")             # 1 つの部品 (UnitSpec. 引数は unit.params)
spec = load_spec("model.yaml")           # 定義の辞書 (版を今の版に直したもの)
text = dump_spec(spec)                   # 決まったキーの順の YAML の文字 (パスを渡すと保存)
code = export_python("model.yaml")       # 同じ計算をする Python のコード (下を参照)
```

公開している名前は `GraphModel`, `SpecError`, `build_model`, `describe_spec_units`, `describe_units`, `dump_spec`, `embed_subgraph`, `export_python`, `get_unit`, `infer`, `list_examples`, `load_example`, `load_spec`, `register_unit`, `validate`, `__version__`.

`validate` の 1 件は `{severity, code, message, node, location, subgraph}` の辞書 (`severity` は `error` か `warning`, `code` は `missing_param` などの決まった英字の識別子, `location` は `params.out_features` のような場所. 部分モデルの中の誤りは `node` が `None` で, `subgraph` に部分モデルの名前が入る).

`build_model` と `export_python` は, 誤りのある定義では `SpecError` を投げる (`error.issues` に `validate` と同じ形の誤りの一覧). 定義のファイルを読めない場合は, YAML の誤りなどは `ValueError`, ファイルが無い場合は `FileNotFoundError` (`validate` はどちらも `invalid_spec` の誤りとして返す). 組み立てたモデルの計算の途中で部品が誤りを出した場合は, どのノードで起きたかを誤りに添える (Python 3.11 以上).

`infer` は, 定義に誤りがあっても, 誤りのあるノード (つなぎ先の無い口, 引数の誤り, 部品を作れないものなど) とその先のノードだけを除いて計算する (組み立ての途中でも形を確かめられるように). そのときの `ok` は `false` で, 除いたノードの `status` は `error` (誤りの文付き) か `skipped`, `params` は計算できた部分の合計. `build_model` と `export_python` は誤りのある定義では作らない. 学習済みの重みとクラスの数のように, 形の推定では分からず組み立てで失敗する引数の組み合わせは, `validate` が `invalid_combination` の誤りとして返す.

外部のライブラリの部品は, そのライブラリを入れると使える (無い場合は一覧に「使えない」と理由と入れ方が出る).

| 任意の依存 | 部品 |
| --- | --- |
| `geometric` | PyTorch Geometric のグラフの層 (`pyg.GCNConv`, `pyg.MeanAggregation` など) |
| `huggingface` | Hugging Face のモデル (`hf.AutoModel`, `hf.AutoModelForSequenceClassification`, `hf.AutoModelForImageClassification`, 画像の本体の `hf.AutoVisionModel` など) |
| `torchvision` | torchvision の層 (`tv.DeformConv2d`, `tv.SqueezeExcitation` など) と, 名前で作るモデル (`tv.Model`: `name: resnet18`) |
| `timm` | timm の層 (`timm.Mlp`, `timm.Attention`, `timm.DropPath` など) と, 名前で作るモデル (`timm.Model`: `model_name: vit_base_patch16_224`) |
| `torchaudio` | 音声の変換 (`audio.MelSpectrogram`, `audio.MFCC` など) とモデル (`audio.Conformer` など) |
| `einops` | `einops.Rearrange`, `einops.Reduce`, `einops.EinMix` |
| `kornia` | 画像の拡張, フィルタ, 色の変換 (`kornia.RandomHorizontalFlip`, `kornia.Normalize` など. 拡張と同じ名前の他のクラスは `kornia.enhance.Normalize` のようにモジュールの名前を挟む) |
| `monai` | 医用画像のネットワークと部品 (`monai.UNet`, `monai.SwinUNETR` など) |
| `smp` | 画像の領域分割のモデル (`smp.Unet`, `smp.FPN` など. segmentation_models_pytorch) |

学習済みの重みを読む引数 (Hugging Face の `pretrained`, `tv.Model` の `weights`, `timm.Model` の `pretrained`, `smp` の `encoder_weights` など) は, 形の推定 (`infer`) では重みを読まない値に置き換える (組み立てでは書いた値のまま). Hugging Face のモデルは `params` の `model_name` に名前かローカルのパスを書く. 外部のコード (`trust_remote_code`) は使わない.

同梱した例のモデルの定義 (全結合, 残差, 部分モデル (残差のブロック) を重ねたもの (`stacked_blocks`), CNN, LSTM, Transformer のエンコーダー, GCN, BERT の分類) は `list_examples()` で一覧にし, `load_example("cnn_classifier")` で読み込める. 自分のモデルを書き始めるときの元にできる.

`export_python("model.yaml")` で, 同じ計算をする `nn.Module` のクラスの Python のコードを書き出せる (`state_dict` のキーは `build_model` と同じ). 書き出したコードはこのライブラリが無くても動く (演算 (`op.Add` など) と Hugging Face の部品のクラスの定義をコードに書き写す. 外部のライブラリ (PyTorch Geometric, timm など) の部品を使う場合は, そのライブラリは要る). `export_python(..., standalone=False)` にすると, 書き写さずに `from ml_modelcontrol import ops` などで読み込む. `path` を渡すとファイルに保存し, `class_name` で本体のクラスの名前 (既定 `Model`) を変えられる. 自作の部品は登録したクラスのモジュールから読み込む行を書く (関数の中や, 実行したスクリプトで定義したクラスは書き出せない).

保存したモデルの定義は, 別の定義の中で 1 つの部品 (部分モデル) として使える. `embed_subgraph(spec, "block", "block.yaml")` で定義の `subgraphs` に写し, ノードの `unit` に `sub.block` と書く (部分モデルの入力が全て必須の入力の口 (入力の順に位置で渡す), 出力が出力の口になり, 引数は持たない. 同じ部分モデルを何度使っても重みは別). 元のファイルを参照せずに写すので, 1 つのファイルだけで組み立てられる. 元の定義を直した後に写し直す場合は `replace=True` を付ける. 書き方は [docs/DESIGN.md](docs/DESIGN.md) の「部分モデル」.

```python
from ml_modelcontrol import describe_spec_units, embed_subgraph

spec = embed_subgraph("model.yaml", "block", "block.yaml")  # 写した定義 (辞書)
spec["nodes"]["b1"] = {"unit": "sub.block", "inputs": "add"}
spec["nodes"]["head"]["inputs"] = "b1"  # add と head の間に部分モデルを挟む
units = describe_spec_units(spec)        # 部分モデルの部品の一覧 (describe_units() に加えて使う)
dump_spec(spec, "model.yaml")            # 写した定義を保存
```

自作の部品は `register_unit(MyBlock, name="my.Block")` で登録すると, 定義の `unit` に書ける (`forward` が複数の値を返す場合は `outputs=["a", "b"]` で口の名前を付ける. 属性や辞書のキーで値を返す場合は `outputs={"logits": ["logits"], "last": ["hidden_states", -1]}` のように口ごとの取り出し方を書く. 引数の説明と範囲は `notes={"hidden": {"description": "中間の大きさ", "minimum": 1}}` のように付けられる. 名前の頭に組み込みの部品の頭 (`nn.`, `op.`, `sub.` など) は使えない).

## ML-Found で学習する

組み立てたモデルは普通の `nn.Module` なので, [ML-Found](https://github.com/victor-von-pooh/ML-Found) の `Trainer` にそのまま渡せる. 定義のファイルの置き場をプロジェクトの config に書き, 学習スクリプトで組み立てる. 定義のファイルは実験の出力先に写しておくと, 後から同じモデルを作って `final_model.pt` を読み込める.

- ML-CreateArchitecture で保存した定義は, その置き場 (`~/.ml-createarchitecture/models/<名前>.yaml`) から, または画面の「YAML」のダウンロードで, プロジェクトの中 (下の例では `models/`) に写して使う. 定義を直した場合は写し直す
- 学習に使う Python (ML-Dashboard から起動する場合はプロジェクトの仮想環境) には, ML-Found と一緒にこのライブラリも入れる
- 学習スクリプトは, 実行した時点のプロジェクトの定義のファイルから組み立てる. ML-Dashboard の「同じ設定でもう一度」で起動し直す場合も同じなので, 前の実験と同じモデルにしたい場合は, その実験の `model_spec.yaml` をプロジェクトの定義のファイルに写してから起動する

```yaml
# configs/default/train.yaml (プロジェクト固有の項目)
model:
  spec: models/iris_mlp.yaml    # プロジェクトのディレクトリからの相対パス
```

```python
import shutil
from pathlib import Path

from ml_found import Trainer, load_project_config, start_experiment
from ml_modelcontrol import build_model

PROJECT_DIR = Path(__file__).resolve().parent

cfg, diff = load_project_config(str(PROJECT_DIR))

# 定義から組み立て (誤りがあれば, 実験の出力先を作る前に SpecError で止まる)
spec_path = PROJECT_DIR / cfg.model.spec
model = build_model(spec_path, seed=cfg.experiment.seed)

# 実験を始め, 定義を実験の出力先に残す
experiment = start_experiment(cfg, diff, str(PROJECT_DIR))
shutil.copy(spec_path, experiment.run_dir / "model_spec.yaml")

# step とデータ (train_loader, valid_loader) は, ML-Found の README の
# 「学習スクリプト」と同じくプロジェクトで用意する
trainer = Trainer(model, experiment, step)
history = trainer.fit(train_loader, valid_loader)
```

`experiment.run_dir` は実験の出力先 (`Path`. プロジェクトのディレクトリの `outputs/<experiment.name>/<日付>/<時刻>/`). 後から読み込む場合は, その中の定義と重みを使う.

```python
# 後から読み込む (パスはプロジェクトのディレクトリから)
import torch

run_dir = "outputs/iris/2026-11-15/01-14-54"
model = build_model(f"{run_dir}/model_spec.yaml")
model.load_state_dict(torch.load(f"{run_dir}/final_model.pt"))
```

- 出力が 1 つの定義のモデルはテンソルを返し, 複数の場合は出力の名前の辞書を返す. `step` では `logits = model(x)` か, 複数なら `logits = model(x)["logits"]` のように受け取る
- 入力は定義の `inputs` の順の位置か, 名前のキーワードで渡す
- ML-CreateArchitecture で保存した定義 (YAML) もそのまま使える
- 学習済みの重みを読む部品 (`pretrained: true` など) は, `build_model` のときに重みを取得する (後から読み込むときも, 取得してから `final_model.pt` の重みで上書きする. ネットワークと時間が要る)

## 開発

```bash
pytest
```

コードの書き方は [CODING_STYLE.md](CODING_STYLE.md) に従う.

## ライセンス

MIT
