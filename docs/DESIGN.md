# 設計

ML-ModelControl の設計の方針をまとめる.

## 目的

PyTorch (と互換のライブラリ) で作れるモデルの部品を登録簿として持ち, 部品をつないだ定義から `nn.Module` を組み立てる.
モデルを Python で書かずに, 定義 (YAML か JSON) を書くか, GUI (ML-CreateArchitecture) で部品を置いてつなぐだけで作れるようにする.

- 部品の単位は 1 つに決めない. `nn.Linear` のような層, 足し算や連結のような演算, PyTorch Geometric のグラフニューラルネットワークの層, Hugging Face, torchvision, timm などの学習済みのモデルを, 同じ登録簿で同じように扱う
- GUI が必要とすること (部品の一覧と引数の説明, つなぎ方の確認, 各部品の出力の形, パラメータ数) は, このライブラリが返す. GUI は描画と編集だけを受け持つ
- 学習はこのライブラリの対象外. 組み立てたモデルは普通の `nn.Module` なので, ML-Found などで学習できる

## 用語

| 用語 | 意味 |
| --- | --- |
| 部品 (unit) | 登録簿にある, モデルの 1 つの単位. 名前 (`nn.Linear`, `op.Add`, `pyg.GCNConv`, `hf.AutoModel` など), 引数, 入力と出力の口を持つ |
| ノード (node) | モデルの定義の中で使う部品の 1 つ. 名前 (id), 部品の名前, 引数の値, 入力のつなぎ先を持つ |
| 口 (port) | 部品の入力と出力の名前. `nn.Linear` は入力 `input`, 出力 `out`. GCN の層は入力 `x`, `edge_index`, `edge_weight` (省略可) |
| モデルの定義 (graph) | モデルの入力, ノード, 出力をまとめたもの. YAML か JSON で保存する |

## モデルの定義の形

ノードとつなぎ方のグラフ (分岐と合流のある有向非巡回グラフ) にする.
順に並べるだけのモデルも, 前のノードにつなぐだけなので同じ形で書ける.

```yaml
format_version: 1
name: residual_mlp
description: 残差のつなぎ方を持つ全結合のモデル
inputs:
  x: {shape: [4, 16]}                  # 形の推定に使う例の形 (書いたとおりの形)
nodes:
  fc1:
    unit: nn.Linear
    params: {in_features: 16, out_features: 16}
    inputs: {input: x}                 # 口の名前: つなぎ先
    ui: {x: 120, y: 80}                # GUI の位置など (組み立てでは使わない)
  act:
    unit: nn.ReLU
    inputs: fc1                        # 必須の口が 1 つの部品は口の名前を省略可
  add:
    unit: op.Add
    inputs: {xs: [x, act]}             # 可変の数の口はリスト (順に意味がある)
  head:
    unit: nn.Linear
    params: {in_features: 16, out_features: 3}
    inputs: add
outputs:
  logits: head
```

- つなぎ先は「入力の名前」か「ノードの名前」. 出力が複数の部品は「ノードの名前:口の名前」(例: `lstm:output`) で選ぶ. 出力が複数の部品を口の名前なしで参照するとエラー
- `inputs` の口は書く順に意味は無い. 省略できる口 (GCN の `edge_weight` など) は書かなくてよい. `inputs: fc1` の省略形は, 必須の口がちょうど 1 つの部品だけで使え, その口につなぐ
- `params` は部品を作るときの引数 (`__init__`), `forward_params` は呼び出すときの設定の値 (`nn.MultiheadAttention` の `need_weights` など). どちらも JSON の値 (数, 文字, 真偽値, null, それらのリスト) だけを書ける
- `outputs` が 1 つなら, 組み立てたモデルはテンソルを返す. 2 つ以上なら名前をキーにした辞書を返す. 組み立てたモデルは, 入力を定義の順の位置か, 名前のキーワードで受け取る
- `ui` (ノードごとと, 一番上) と `description` は自由に書け, 組み立てでは使わず, 保存では残す. それ以外の知らない項目はエラーにする (書き間違いを見つけるため)

### 部分モデル (subgraphs)

保存したモデルの定義を, 別の定義の中で 1 つの部品として使える. 定義の一番上の `subgraphs` に名前と定義を書き, ノードの `unit` に `sub.<名前>` と書く.

```yaml
subgraphs:
  block:                               # 部分モデルの名前
    description: 残差のブロック
    inputs: {x: {}}                    # 口になる (形は書かなくてよい)
    nodes:
      fc: {unit: nn.Linear, params: {in_features: 16, out_features: 16}, inputs: x}
      add: {unit: op.Add, inputs: {xs: [x, fc]}}
    outputs: {y: add}
nodes:
  block1: {unit: sub.block, inputs: x}
  block2: {unit: sub.block, inputs: block1}   # 同じ部分モデルを何度も使える (重みは別)
```

- 部分モデルの入力は入力の口 (全て必須. 部分モデルの入力の順に位置で渡す), 出力は出力の口になる (出力が 2 つ以上なら辞書のキーで取り出す). 引数は持たない. 書き出すコードのクラスの名前は, 部分モデルの名前を `_` で区切った語の頭を大文字にしたもの (`res_block` は `ResBlock`. 識別子にならない場合は頭に `Subgraph` を付ける)
- 部分モデルの定義は, `format_version` と `subgraphs` を除いて普通の定義と同じ形. 入力の `shape` は書かなくてよく, 書いても形の推定には使わない (形は前のノードから決まる). 部分モデルの中で別の部分モデル (`sub.<名前>`, 同じ `subgraphs` の中のもの) を使え, 互いに使い合う循環は誤り
- **参照ではなく写す**: 部分モデルは, 元の定義のファイルを参照せず, 定義の中に写して持つ. 1 つのファイルだけで組み立てられ (学習のスクリプトに渡すのも 1 つ), 元のファイルを後から変えても, 保存した重み (`state_dict`) と合わなくなることがない. 写すには `embed_subgraph(spec, 名前, 写す定義)` を使う. 写す定義の部分モデルも同じ名前のまま一緒に写す (同じ名前で中身の違うものがあれば誤り). 元の定義を直した後に写し直す場合は `replace=True` で置き換える
- 組み立てたモデルでは, 部分モデルのノードは中に `GraphModel` を持つ. `state_dict` のキーは `nodes.block1.nodes.fc.weight` のようになる
- 確認では, 部分モデルの中の誤りを `node` が `None`, `subgraph` が部分モデルの名前, `location` が `subgraphs.<名前>.nodes.<ノード>.<場所>` (ノードに付かない誤りは `subgraphs.<名前>.<場所>`) の誤りとして返す. それを使うノードは使えない部品 (`unavailable_unit`) の誤り, 定義に無い部分モデルを使うノードは `unknown_subgraph` の誤りになる. 使われない部分モデル (本体から使う部分モデルをたどって届かないもの) は注意. どこからも使われない部分モデルでも, 中に誤りがあれば定義全体の誤りになり, 組み立てと書き出しはできない
- 形の推定は, 部分モデルを使うノードについて, その出力の口ごとの形とパラメータ数 (中の全てのノードの合計) を返す (中のノードごとの形は返さない). 部分モデルの中の誤りは, それを使うノードとその先だけを止める
- 書き出すコードでは, 部分モデルごとに 1 つのクラスを書く
- GUI のために, 定義の中の部分モデルの部品の一覧を `describe_spec_units(spec)` で返す (`describe_units` と同じ形で, 部分モデルの部品だけ. `describe_units()` の一覧に加えて使う). 分類は `sub`

### 入力の例

- `shape` は書いたとおりの形 (バッチの次元も含む. グラフのデータでは `x` は `[ノード数, 特徴量]`, `edge_index` は `[2, エッジ数]`). 形の推定と確認にだけ使い, 組み立てたモデルは任意の形を受け取る
- `dtype` は `float32` (既定), `int64` など. 単語の番号, `edge_index`, グラフの番号 (`batch`) は `int64`
- `fill` は例の値の作り方で, `zeros` (既定), `ones` (注意のマスクなど), `arange` (位置の番号など)
- 例の形には 1 を使わない方がよい. 1 の次元はブロードキャストで他の大きさに合ってしまい, 形の誤りを見逃す (BatchNorm も学習時はバッチ 1 で失敗する)

### 名前の規則

- 入力, ノード, 出力の名前は, 英字か `_` で始まり, 英数字と `_` だけからなる名前 (Python の予約語を除く)
- 入力とノードは 1 つの名前の空間で, 同じ名前は使えない
- 組み立てたモデルの部品の入れ物 (`nn.ModuleDict`) の属性と同じ名前 (`type`, `float`, `training`, `keys`, `update` など) はノードに使えない. ノードの名前はそのまま部品の名前になり, `state_dict` のキー (`nodes.fc1.weight` など) にも使われる. そのため, ノードの名前を変えると, 前に保存した重みは読み込めない

### この形にした理由

- **グラフにする**: 残差のつなぎ方, Transformer の注意の分岐, GNN の `x` と `edge_index` のように, 実用のモデルの多くは一直線ではない. 一直線のリスト (`nn.Sequential`) から始めると, 後で形を変えることになる
- **ノードごとに入力のつなぎ先を書く**: エッジを別の一覧にするより, 1 つのノードを読めば何が入ってくるか分かり, 手で書いても読みやすい. GUI のエッジとは 1 対 1 に変換できる. GUI の位置はノードの `ui` に持つので, ノードの名前を変えても位置が迷子にならない
- **定義のファイルにする (コードを生成しない)**: 定義は検証, 差分, 再編集ができ, GUI と行き来できる. 他のプロジェクトにコードとして持っていきたい場合のために, 定義から Python のコードを書き出す機能も用意する

### 版

`format_version` は整数. 読み込むときに古い版から順に今の版に直し, ライブラリより新しい版はエラーにする. 保存するときはキーの順を決めて書き, ノードは書いた順のまま残す (GUI で保存しても差分が小さいように).

## 登録簿

部品ごとに, 名前, 分類, 出どころ, 説明, 引数 (型, 既定値, 必須か, 説明, 範囲, 選択肢), 入力と出力の口, 使えるかどうか (使えない場合は理由) を持つ.

| 出どころ | 名前の形 | 中身 |
| --- | --- | --- |
| `torch.nn` | `nn.<クラス名>` | `torch.nn` の層. 引数と入力の口はシグネチャから取り出し, よく使うものには説明と範囲を付ける (入れ物, 損失関数, 遅延初期化の `Lazy*` は除く) |
| 演算 | `op.<名前>` | `Add`, `Multiply`, `Concat`, `Stack`, `Reshape`, `Permute`, `Transpose`, `Mean`, `Select` (タプルの要素), `GetAttr` (出力の属性) など. `torch.nn.functional` は使わず, 全てモジュールとして作る |
| PyTorch Geometric | `pyg.<クラス名>` | `torch_geometric.nn` の畳み込みの層と集約の層. ライブラリが入っている場合だけ使える |
| Hugging Face | `hf.AutoModel`, `hf.AutoModelForImageClassification` など | モデルの名前 (またはローカルのパス) から読み込む. 文 (`input_ids`) を受け取る種類と画像 (`pixel_values`) を受け取る種類がある. 重みを読むか, 設定だけから作るかを選べる. ライブラリが入っている場合だけ使える |
| torchvision | `tv.<クラス名>`, `tv.Model` | `torchvision.ops` の層 (`DeformConv2d`, `SqueezeExcitation`, `StochasticDepth` など) と, 名前で作るモデル (`tv.Model`. `resnet18` などを `torchvision.models.get_model` で作る) |
| timm | `timm.<クラス名>`, `timm.Model` | `timm.layers` の層 (`Mlp`, `Attention`, `DropPath` など) と, 名前で作るモデル (`timm.Model`. `timm.create_model` で作る) |
| torchaudio | `audio.<クラス名>` | `torchaudio.transforms` の変換 (`MelSpectrogram`, `MFCC` など) と `torchaudio.models` のモデル (`Conformer`, `Wav2Letter` など) |
| einops | `einops.<クラス名>` | `Rearrange`, `Reduce`, `EinMix` (式で次元を並べ替え, まとめる) |
| Kornia | `kornia.<クラス名>` | 画像の拡張 (`kornia.augmentation`), フィルタ, 色の変換, 明るさの調整, 幾何の変換の層 |
| MONAI | `monai.<クラス名>` | 医用画像のネットワーク (`monai.networks.nets`. `UNet`, `SwinUNETR` など) と部品 (`monai.networks.blocks`) |
| segmentation_models_pytorch | `smp.<クラス名>` | 画像の領域分割のモデル (`Unet`, `FPN`, `DeepLabV3Plus` など. 符号化器を名前で選ぶ) |
| 部分モデル | `sub.<名前>` | 定義の `subgraphs` に書いた部分モデル (定義ごとに違う. 登録簿には入れない) |
| 自作 | 登録した名前 | `register_unit` で `nn.Module` のクラスを登録する (Python のコードから. モデルの定義からは登録できない). 出力の口は, タプルの順の名前のリストか, 口ごとの取り出し方 (番号か, 属性かキーの名前の並び) の辞書で付ける |

### 口

- 入力の口は, テンソル (必須), 省略できるテンソル, 可変の数 (順に意味があるリスト, 口の名前は `xs`) のどれか. 手で書いた口の一覧があればそれを使い, 無ければ `forward` のシグネチャから取り出す (既定値の無い引数は必須, 既定値が `None` の引数は省略できる口, `*args` は可変の数. 真偽値や数の既定値を持つ引数は口ではなく `forward_params`)
- 呼び出しでは, `*args` より前の口と位置だけの引数の口は位置で (省略した場合は `None`), 可変の数の口は位置で並べ, それ以外の口と `forward_params` はキーワードで渡す. 真偽値を受け取る引数は, 既定値が `None` でも口ではなく `forward_params` にする
- 出力の口は, 1 つのテンソルを返すものは `out`. タプルや属性を持つ値を返すものは, 口ごとに取り出し方 (番号か属性の名前の並び) を持つ. 例えば `nn.LSTM` は `output` (`[0]`), `h_n` (`[1][0]`), `c_n` (`[1][1]`), `nn.MultiheadAttention` は `output` (`[0]`) と `weights` (`[1]`, `None` の場合もある)
- 引数の種類は型ヒントから決める (`int` と組の両方を受け取る `kernel_size` などは任意, `Literal` は選択肢). 種類が任意の引数も, 型ヒントの Union を分けて受け取れる種類 (`int`, `float`, `bool`, `str`, `list`, `null`. 例: `kernel_size` は int と list) を `types` に持ち, 確認で値の種類を確かめる (テンソルや関数を含む型ヒントは確かめない). 説明と範囲は, ML-Found の Optimizer の登録簿と同じ考え方で, シグネチャから自動で取り出したものに, よく使うものだけ手で書いた説明を重ねる. 説明の無い引数は, 引数の名前ごとの説明 (日本語. `in_channels` など多くのライブラリで同じ意味のもの) を使い, それも無ければクラスの docstring (Google 形式, NumPy 形式, Sphinx 形式) の説明をライブラリの言葉 (英語) のまま使う (長いものは 300 文字で切る)
- 別の部品を引数に取るクラス (`nn.TransformerEncoder` など) は, 今は「使えない」として一覧に出す
- 入っていないライブラリの部品は, 一覧に「使えない (理由: ライブラリが無い)」として返す. GUI で選べないことを示せる

## 確認, 形の推定, 組み立て

- **確認** (`validate`): 定義の形, 名前, 部品の名前, 引数 (名前の誤り, 必須の不足, 範囲), つなぎ先の存在, 口, 循環, 使われないノードを調べる. 最初の誤りで止めず, 全ての誤りと注意を返す (GUI でノードごとに示すため). 1 件は `{node, location, code, severity, message}` (`location` は `params.out_features` や `inputs.edge_index`, `code` は決まった英字の識別子, `severity` は `error` か `warning`). ネットワークは使わない
- **組み立て** (`build_model`): `nn.Module` (`GraphModel`) を返す. ノードの部品を `nn.ModuleDict` に持ち, 前もって決めた順 (トポロジカル順. 同じ順位は定義に書いた順) に計算する. 重みは `state_dict` で保存して読み込む (モデルごとの `torch.save` は, 読み込みに `weights_only=False` が要る). 乱数の種を指定できる. 出力に届かないノードも組み立てて計算する (確認では注意)
- **形の推定** (`infer`): 入力の例 (`shape`, `dtype`, `fill`) から値を作り, cpu で `eval()` と勾配なしで, ノードを 1 つずつ計算する. 各ノードの出力の口ごとの形と型 (`None` も可), パラメータ数 (全体と学習するもの. 共有した重みは 1 回だけ数える) を返す. 失敗したノードはエラーを, その先のノードは「前のノードの失敗で計算できない」を返す. 大きすぎるモデルでメモリを使い切らないように, 先に (meta デバイスで) パラメータ数を数え, 上限 (既定 2 億) を超える場合は計算せずにパラメータ数だけを返す. 形 (`shape`) の無い入力がある場合は計算しない. 定義に誤りがあっても, 組み立ての途中で形を確かめられるように, 誤りのあるノード (つなぎ先の無い口, 引数の誤り, 部品を作れないものなど) とその先のノードだけを除いて計算する (誤りのあるノードは確認の誤りの文を持つ `error`, その先は `skipped`, `ok` は `false`). ノードに付かない誤り (入力の書き方など. 出力が無いことと出力のつなぎ先の誤りは除く) がある場合は計算しない. 結果は `ok`, `issues`, `nodes` (ノードごとの `status` (`ok`, `error`, `skipped`), `outputs`, `params`, `trainable`, `error`), `outputs`, `params`, `trainable`, `estimated_only`
- **書き出し** (`export_python`): 同じ計算をする `nn.Module` のクラスの Python のコードを書き出す. `state_dict` のキーは `build_model` と同じにする. 出力の口の名前での取り出しは, 組み立てたモデルと同じにする. Hugging Face の出力 (辞書) は `.get` (無いキーは `None`) で書き, 型の分からない自作の部品の出力は, 辞書ならキー, それ以外は属性で取り出す関数 (`_pick`) をコードに置いて使う. 部品を作る行とノードを呼ぶ行は, 79 文字を超える場合に引数を折り返す (出力の辞書の行と, とても長い名前の行は超えることがある)

## 安全

- 部品は登録簿の名前だけで選ぶ. モデルの定義に import のパスやクラスを書く方法は作らない. 自作の部品の登録は Python のコードから行う
- YAML は `yaml.safe_load` で読む. 引数は JSON の値だけ
- Hugging Face のモデルの外部のコード (`trust_remote_code`) は既定で使わず, モデルの定義からは有効にできない (ライブラリを使う側のコードから設定する)
- 確認 (`validate`) はネットワークを使わない. Hugging Face のモデルの設定が手元に無い場合は, 確認できないことを注意で返す

## 外部のライブラリ

- 外部のライブラリは全て任意の依存 (`pip install "ml-modelcontrol[geometric]"` のようにライブラリごとに選ぶか, `[all]` で全て). 入っていない環境でも, 残りの部品は使える. 入っていないライブラリは, よく使う部品だけを「使えない」として理由付きで一覧に出す
- 部品の取り込み方は 2 つ. (1) **クラスを集める**: ライブラリのモジュールの `nn.Module` のクラスを, PyTorch Geometric と同じくシグネチャから引数と口を取り出して部品にする (土台のクラス, 入れ物, 別名は除く. 引数に別の部品が必要なクラスは「使えない」). (2) **名前で作る**: torchvision と timm のモデルは数が多く, クラスより名前で選ぶ方が分かりやすいので, モデルの名前を引数に持つ 1 つの部品にする. 名前の候補 (`suggestions`) は一覧に付けるが, 候補以外も書ける (新しいモデルのため)
- 部品の名前は `<頭>.<クラス名>`. 頭はライブラリごとに決め (`tv` は torchvision, `audio` は torchaudio, `smp` は segmentation_models_pytorch, 他はライブラリの名前), 分類は `<頭>_<モジュール>` (`tv_ops`, `kornia_filters` など). 同じライブラリの複数のモジュールに同じ名前のクラスがある場合 (`kornia.augmentation.Normalize` と `kornia.enhance.Normalize` など), 先に集めたモジュールのクラスを `kornia.Normalize`, 後のモジュールのクラスを `kornia.enhance.Normalize` のようにモジュールの名前を挟んで区別する
- 名前で作る部品のモデルの名前の引数は, ライブラリの関数の引数の名前に合わせる (`tv.Model` は `name`, `timm.Model` と Hugging Face は `model_name`. 書き出すコードがそのまま関数の呼び出しになるため). Hugging Face の部品の名前は transformers のクラスの名前に合わせる. ただし `hf.AutoVisionModel` は, 画像を受け取る `AutoModel` (ViT などの本体) を文のモデルと区別するための名前
- 書き出すコードでは, ライブラリのモジュールを別名 (`from torchvision import ops as tv_ops` など) で読み込み, 違うライブラリの同じ名前のクラスが重ならないようにする
- 学習済みの重みを読む引数 (`pretrained`, `weights`, `encoder_weights` など) を部品ごとに持ち (`offline`), 形の推定ではその引数を「読まない」値に置き換える (`tv.Model` は `weights` と, 検出と領域分割のモデルの `weights_backbone`) (形は重みに依らず, 大きな重みの取得を避けるため). 組み立て (`build_model`) では書いた値のまま使う
- Hugging Face のモデルを名前で読む場合はネットワークを使う (形の推定でも設定の取得に使う). 確認では使わない

## 段階

| 段階 | 内容 |
| --- | --- |
| 1 | 登録簿 (`torch.nn` と演算, 自作の部品の登録), モデルの定義 (名前の規則, `ui`), 確認, 組み立て, 形の推定 |
| 2 | PyTorch Geometric と Hugging Face の部品, Python のコードの書き出し |
| 3 | ML-CreateArchitecture (GUI) から使う API の整備 |
| 4 | ML-Found との連携 (定義をプロジェクトの config から読んで学習する). 今は README の手順 (学習スクリプトで `build_model` を呼び, 定義を出力先に写す) で行い, ML-Found に専用の仕組み (定義の自動の保存や `run.json` への記録) を入れるかは, 使ってみてから決める |
