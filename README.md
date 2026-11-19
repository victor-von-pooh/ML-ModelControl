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
pip install -e .                 # 使うだけの場合
pip install -e ".[dev]"          # テスト (pytest) も動かす場合
pip install -e ".[dev,all]"      # 外部のライブラリの部品も全て使う場合 (ライブラリごとに [geometric], [timm] なども選べる)
```

- PyPI には公開していないので, clone したディレクトリで入れる. 別の環境 (学習のプロジェクトの仮想環境など) に入れる場合は, `pip install -e <clone したディレクトリ>` か `pip install "ml-modelcontrol @ git+https://github.com/victor-von-pooh/ML-ModelControl.git"` (任意の依存は `ml-modelcontrol[geometric] @ git+...`)
- PyTorch は, 先に [PyTorch の手順](https://pytorch.org/get-started/locally/) で環境 (CPU, CUDA) に合わせて入れておくと, 大きな版を入れ直さずに済む
- Python 3.10 から 3.13 で確かめている. PyTorch Geometric は Python 3.15 では読み込めない (ライブラリの側の問題) ので, グラフの部品を使う場合は 3.13 までを使う

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

学習したモデルの一部だけを呼ぶ場合は `model.compute(値の辞書, 出力の名前)` を使う. 値は入力だけでなく途中のノードの出力の口 (定義のつなぎ先と同じく `ノード` か `ノード:口`) にも渡せ, 求める出力に要るノードだけを計算する (重みはそのモデルのまま. 結果は出力の名前の辞書). 例えば, 学習した VAE の例 (`vae`) の復号器だけを潜在変数から呼ぶ場合は `model.compute({"z": z}, "recon")["recon"]`, 2 塔の例 (`two_tower_retrieval`) の品物の埋め込みだけを前もって計算する場合は `model.compute({"item": item}, "item_embedding")`. 要る入力が渡されていない場合は, どの出力にどの入力が要るかを添えた `ValueError` になる. `compute` は書き出したコードには無い (書き出すコードは全体の `forward` だけ).

`validate`, `infer`, `build_model`, `export_python` は, ファイルのパスの代わりに定義の辞書 (`load_spec` で読んだものや, 手で作ったもの) も受け取る. `load_spec` は読み込みと版を直すことだけをし, 誤りは確かめない (確かめるのは `validate`).

公開している名前は `GraphModel`, `SpecError`, `build_model`, `classify_license`, `clear_pretrained_cache`, `describe_spec_units`, `describe_units`, `dump_spec`, `embed_subgraph`, `export_python`, `get_unit`, `infer`, `list_examples`, `load_example`, `load_spec`, `pretrained_info`, `register_unit`, `validate`, `__version__`.

学習済みの重みを読むノードのライセンスは `pretrained_info("hf.AutoModel", {"model_name": "google-bert/bert-base-uncased"})` で読む. 配布元 (Hugging Face のモデルの説明. timm と segmentation_models_pytorch の重みはライブラリが持つ Hugging Face の置き場) から読み, `status` (`commercial`, `noncommercial`, `unknown`), 警告するか (`alert`), 画面に出す説明 (`message`), どこから読んだか (`source`), 推論の一式に写せるか (`redistributable`), 読んだ版 (`pinned_revision`) を返す. 重みを読まない場合は `None`. torchvision の重みには機械で読める情報が無いので `unknown`. ネットワークを使い, 結果はプロセスの中に覚える. 非公開と申請制のモデルは環境変数 `HF_TOKEN` のトークンで読み, `HF_HUB_OFFLINE=1` ではネットワークを使わない. 詳しくは docs/DESIGN.md の「学習済みの重みのライセンス」.

`validate` の 1 件は `{severity, code, message, node, location, subgraph}` の辞書 (`severity` は `error` か `warning`, `code` は `missing_param` などの決まった英字の識別子, `location` は `params.out_features` のような場所. 部分モデルの中の誤りは `node` が `None` で, `subgraph` に部分モデルの名前が入る).

`build_model` と `export_python` は, 誤りのある定義では `SpecError` を投げる (`error.issues` に `validate` と同じ形の誤りの一覧). 定義のファイルを読めない場合は, YAML の誤りやライブラリより新しい版 (`format_version`) などは `ValueError`, ファイルが無い場合は `FileNotFoundError` (`validate` はどちらも `invalid_spec` の誤りとして返す. `infer` と `load_spec` は投げる). 組み立てたモデルの計算の途中で部品が誤りを出した場合は, どのノードで起きたかを誤りに添える (Python 3.11 以上).

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

同梱した例のモデルの定義 (全結合, 残差, 部分モデル (残差のブロック) を重ねたもの (`stacked_blocks`), CNN, LSTM, Transformer のエンコーダー, GCN, BERT の分類, 受け持ちごとの雛形 (VAE, GAN の生成器と識別器, 拡散の雑音の予測, 対照学習の符号化器, 推薦の 2 塔, 強化学習の方策と価値)) は `list_examples()` で一覧にし, `load_example("cnn_classifier")` で読み込める. 受け持ちごとの雛形は, 説明に学習の step で行うことを書く. 自分のモデルを書き始めるときの元にできる.

`export_python("model.yaml")` で, 同じ計算をする `nn.Module` のクラスの Python のコードを書き出せる (`state_dict` のキーは `build_model` と同じ). 書き出したコードはこのライブラリが無くても動く (演算 (`op.Add` など) と Hugging Face の部品のクラスの定義をコードに書き写す. 外部のライブラリ (PyTorch Geometric, timm など) の部品を使う場合は, そのライブラリは要る). `export_python(..., standalone=False)` にすると, 書き写さずに `from ml_modelcontrol import ops` などで読み込む. `path` を渡すとファイルに保存し, `class_name` で本体のクラスの名前 (既定 `Model`) を変えられる. 自作の部品は登録したクラスのモジュールから読み込む行を書く (関数の中や, 実行したスクリプトで定義したクラスは書き出せない).

保存したモデルの定義は, 別の定義の中で 1 つの部品 (部分モデル) として使える. `embed_subgraph(spec, "block", "block.yaml")` で定義の `subgraphs` に写し, ノードの `unit` に `sub.block` と書く (部分モデルの入力が全て必須の入力の口 (入力の順に位置で渡す), 出力が出力の口になり, 引数は持たない. 同じ部分モデルを何度使っても重みは別). 元のファイルを参照せずに写すので, 1 つのファイルだけで組み立てられる. 元の定義を直した後に写し直す場合は `replace=True` を付ける. 書き方は [docs/DESIGN.md](docs/DESIGN.md) の「部分モデル」.

```yaml
# block.yaml (入力 1 つ, 出力 1 つの残差のブロック. 入力の shape は書かなくてよい)
format_version: 1
name: block
inputs:
  x: {}
nodes:
  fc: {unit: nn.Linear, params: {in_features: 16, out_features: 16}, inputs: x}
  act: {unit: nn.ReLU, inputs: fc}
  add: {unit: op.Add, inputs: {xs: [x, act]}}
outputs:
  out: add
```

```python
from ml_modelcontrol import describe_spec_units, dump_spec, embed_subgraph

spec = embed_subgraph("model.yaml", "block", "block.yaml")  # 写した定義 (辞書)
spec["nodes"]["b1"] = {"unit": "sub.block", "inputs": "add"}
spec["nodes"]["head"]["inputs"] = "b1"  # add と head の間に部分モデルを挟む
units = describe_spec_units(spec)        # 部分モデルの部品の一覧 (describe_units() に加えて使う)
dump_spec(spec, "model_with_block.yaml")  # 写した定義を別の名前で保存
```

部分モデルの出力が 1 つなら, 使うノードはノードの名前だけでつなぎ先にできる. 2 つ以上なら「ノードの名前:出力の名前」で選ぶ.

自作の部品は `register_unit(MyBlock, name="my.Block")` で登録すると, 定義の `unit` に書ける (`forward` が複数の値を返す場合は `outputs=["a", "b"]` で口の名前を付ける. 属性や辞書のキーで値を返す場合は `outputs={"logits": ["logits"], "last": ["hidden_states", -1]}` のように口ごとの取り出し方を書く. 引数の説明と範囲は `notes={"hidden": {"description": "中間の大きさ", "minimum": 1}}` のように付けられる. 名前の頭に組み込みの部品の頭 (`nn.`, `op.`, `sub.` など) は使えない).

## ML-Found で学習する

組み立てたモデルは普通の `nn.Module` なので, PyTorch の学習の基盤 [ML-Found](https://github.com/victor-von-pooh/ML-Found) の `Trainer` にそのまま渡せる. 定義のファイルの置き場をプロジェクトの config に書き, 学習スクリプトで組み立てる. 定義のファイルは実験の出力先に写しておくと, 後から同じモデルを作って重みを読み込める.

ML-Found のプロジェクト (`configs/default/train.yaml`, `configs/experiment/train.yaml`, 学習スクリプトを持つディレクトリ) の作り方は, ML-Found の README の「使い方」にある. ここでは, その学習スクリプトでモデルの定義を使う部分を示す.

1. プロジェクトの仮想環境に, ML-Found とこのライブラリを入れる

    ```bash
    pip install "ml-found @ git+https://github.com/victor-von-pooh/ML-Found.git"
    pip install "ml-modelcontrol @ git+https://github.com/victor-von-pooh/ML-ModelControl.git"
    ```

2. モデルの定義をプロジェクトの中 (下の例では `models/`) に置く. 例から始める場合は次のように保存する

    ```python
    from ml_modelcontrol import dump_spec, load_example

    dump_spec(load_example("mlp"), "models/mlp.yaml")  # 16 次元の入力から 3 クラス
    ```

3. 定義の置き場を, プロジェクトの default の config にプロジェクト固有の項目として書く (ML-Found の config には, プロジェクト固有の項目を自由に足せる)

    ```yaml
    # configs/default/train.yaml
    experiment:
      name: mlp
      seed: 0
    model:
      spec: models/mlp.yaml    # プロジェクトのディレクトリからの相対パス
    ```

4. 学習スクリプトで定義から組み立てて学習し, 定義を実験の出力先に写す

    ```python
    # train.py (プロジェクトのディレクトリに置く)
    import shutil
    from pathlib import Path

    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    from ml_found import Trainer, load_project_config, start_experiment
    from ml_modelcontrol import build_model, infer

    PROJECT_DIR = Path(__file__).resolve().parent


    def step(model: nn.Module, batch: tuple, phase: str) -> dict:
        # 1 バッチの損失と指標 (ML-Found が平均して記録する)
        x, y = batch
        logits = model(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        return {"loss": loss, "acc": acc}


    # config (default に experiment と「項目=値」を重ねたもの) と差分
    cfg, diff = load_project_config(str(PROJECT_DIR))

    # 定義から組み立て (引数の不足などの誤りは, 実験の出力先を作る前に
    # SpecError で止まる)
    spec_path = PROJECT_DIR / cfg.model.spec
    model = build_model(spec_path, seed=cfg.experiment.seed)

    # 形の誤り (in_features の食い違いなど) は組み立てでは分からないので,
    # 形の推定で学習を始める前に確かめる
    shapes = infer(spec_path)
    if not shapes["ok"]:
        errors = [
            f"{name}: {node['error']}"
            for name, node in shapes["nodes"].items() if node["status"] == "error"
        ]
        raise SystemExit("形を計算できません\n" + "\n".join(errors))

    # 実験を始め, 定義を実験の出力先に残す
    experiment = start_experiment(cfg, diff, str(PROJECT_DIR))
    shutil.copy(spec_path, experiment.run_dir / "model_spec.yaml")

    # データ (ここでは乱数. 実際はプロジェクトのデータ)
    x = torch.randn(256, 16)
    y = torch.randint(0, 3, (256,))
    train_loader = DataLoader(TensorDataset(x[:192], y[:192]), batch_size=32)
    valid_loader = DataLoader(TensorDataset(x[192:], y[192:]), batch_size=32)

    trainer = Trainer(model, experiment, step)
    history = trainer.fit(train_loader, valid_loader)
    ```

`experiment.run_dir` は実験の出力先 (`Path`. プロジェクトのディレクトリの `outputs/<experiment.name>/<日付>/<時刻>/`). ML-Found は学習した重みを `state_dict` として `final_model.pt` に保存するので, 後から読み込む場合は, 写した定義で組み立ててから重みを読み込む.

```python
# 後から読み込む (パスはプロジェクトのディレクトリから. 日時は実験ごとに違う)
import torch
from ml_modelcontrol import build_model

run_dir = "outputs/mlp/2026-11-15/01-14-54"
model = build_model(f"{run_dir}/model_spec.yaml")
model.load_state_dict(torch.load(f"{run_dir}/final_model.pt"))
```

- 出力が 1 つの定義のモデルはテンソルを返し, 複数の場合は出力の名前の辞書を返す. `step` では `logits = model(x)` か, 複数なら `logits = model(x)["logits"]` のように受け取る
- 入力は定義の `inputs` の順の位置か, 名前のキーワードで渡す
- 学習済みの重みを読む部品 (`pretrained: true` など) は, `build_model` のときに重みを取得する (後から読み込むときも, 取得してから `final_model.pt` の重みで上書きする. ネットワークと時間が要る)
- 学習スクリプトは, 実行した時点のプロジェクトの定義のファイルから組み立てる. 前の実験と同じモデルで学習し直したい場合は, その実験の `model_spec.yaml` をプロジェクトの定義のファイルに写してから起動する

GUI のアプリと合わせて使う場合 (どちらも任意).

- ML-CreateArchitecture (このライブラリの部品を画面で組み立てるアプリ) で保存した定義は, その置き場 (`~/.ml-createarchitecture/models/<名前>.yaml`) から, または画面の「YAML」のダウンロードで, プロジェクトの中に写して使う. 定義を直した場合は写し直す
- [ML-Dashboard](https://github.com/victor-von-pooh/ML-Dashboard) (ML-Found の実験を見て, 学習を起動するアプリ) から起動する場合は, プロジェクトの仮想環境にこのライブラリも入れておく. 「同じ設定でもう一度」で起動し直す場合も, 上と同じく実行した時点のプロジェクトの定義のファイルから組み立てる

## 開発

```bash
pytest
```

コードの書き方は [CODING_STYLE.md](CODING_STYLE.md) に従う.

## ライセンス

MIT
