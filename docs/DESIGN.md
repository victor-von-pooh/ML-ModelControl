# 設計

ML-ModelControl の設計の方針をまとめる.

## 目的

PyTorch (と互換のライブラリ) で作れるモデルの部品を登録簿として持ち, 部品をつないだ定義から `nn.Module` を組み立てる.
モデルを Python で書かずに, 定義 (YAML か JSON) を書くか, GUI (ML-CreateArchitecture) で部品を置いてつなぐだけで作れるようにする.

- 部品の単位は 1 つに決めない. `nn.Linear` のような層, 足し算や連結のような演算, PyTorch Geometric のグラフニューラルネットワークの層, Hugging Face の学習済みのモデルを, 同じ登録簿で同じように扱う
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
inputs:
  x: {shape: [1, 16]}                  # 形の推定に使う例の形 (先頭はバッチ)
nodes:
  fc1:
    unit: nn.Linear
    params: {in_features: 16, out_features: 16}
    inputs: {input: x}                 # 口の名前: つなぎ先
  act:
    unit: nn.ReLU
    inputs: {input: fc1}
  add:
    unit: op.Add
    inputs: {inputs: [x, act]}         # 可変の数の口はリスト
  head:
    unit: nn.Linear
    params: {in_features: 16, out_features: 3}
    inputs: {input: add}
outputs:
  logits: head
```

- つなぎ先は「入力の名前」か「ノードの名前」. 出力が複数の部品は「ノードの名前:口の名前」(例: `lstm:output`) で選ぶ
- `inputs` の口は部品の定義の順に並べる必要は無い. 省略できる口 (GCN の `edge_weight` など) は書かなくてよい
- 入力が 1 つの口だけの部品では, `inputs: fc1` のように口の名前を省略して書ける
- `outputs` が 1 つなら, 組み立てたモデルはテンソルを返す. 2 つ以上なら名前をキーにした辞書を返す
- 入力の `shape` と `dtype` (既定は `float32`) は, 形の推定と確認にだけ使う. 組み立てたモデルは任意の形を受け取る

### この形にした理由

- **グラフにする**: 残差のつなぎ方, Transformer の注意の分岐, GNN の `x` と `edge_index` のように, 実用のモデルの多くは一直線ではない. 一直線のリスト (`nn.Sequential`) から始めると, 後で形を変えることになる
- **ノードごとに入力のつなぎ先を書く**: エッジを別の一覧にするより, 1 つのノードを読めば何が入ってくるか分かり, 手で書いても読みやすい. GUI のエッジとは 1 対 1 に変換できる
- **定義のファイルにする (コードを生成しない)**: 定義は検証, 差分, 再編集ができ, GUI と行き来できる. 他のプロジェクトにコードとして持っていきたい場合のために, 定義から Python のコードを書き出す機能も用意する

## 登録簿

部品ごとに, 名前, 分類, 出どころ, 説明, 引数 (型, 既定値, 必須か, 説明, 範囲, 選択肢), 入力と出力の口, 使えるかどうか (使えない場合は理由) を持つ.

| 出どころ | 名前の形 | 中身 |
| --- | --- | --- |
| `torch.nn` | `nn.<クラス名>` | `torch.nn` の層. 引数と入力の口はシグネチャから取り出し, よく使うものには説明と範囲を付ける |
| 演算 | `op.<名前>` | `Add`, `Multiply`, `Concat`, `Reshape`, `Permute`, `Select` (タプルの要素), `GetAttr` (出力の属性) など. `torch.nn.functional` は使わず, 全てモジュールとして作る |
| PyTorch Geometric | `pyg.<クラス名>` | `torch_geometric.nn` の畳み込みの層と集約の層. ライブラリが入っている場合だけ使える |
| Hugging Face | `hf.AutoModel` など | モデルの名前 (またはローカルのパス) から読み込む. 重みを読むか, 設定だけから作るかを選べる. ライブラリが入っている場合だけ使える |
| 自作 | 登録した名前 | `register_unit` で `nn.Module` のクラスを登録する |

- 引数の説明と範囲は, ML-Found の Optimizer の登録簿と同じ考え方で, シグネチャから自動で取り出したものに, よく使うものだけ手で書いた説明を重ねる
- 入力の口は `forward` のシグネチャから取り出す. 出力の口は, 1 つのテンソルを返すものは `out`, タプルを返すもの (`nn.LSTM` など) は手で書いた名前 (`output`, `state` など)
- 入っていないライブラリの部品は, 一覧に「使えない (理由: ライブラリが無い)」として返す. GUI で選べないことを示せる

## 確認, 形の推定, 組み立て

- **確認** (`validate`): 部品の名前, 引数 (名前の誤り, 必須の不足, 範囲), つなぎ先の存在, 必須の口, 循環, 使われないノードを調べ, 誤りと注意をノードの名前付きの一覧で返す (最初の誤りで止めない. GUI でノードごとに示すため)
- **形の推定** (`infer`): 入力の例の形でゼロのテンソルを作り, 組み立てたモデルに cpu で 1 回通して, 各ノードの出力の形と型, パラメータ数を返す. 失敗したノードの名前とエラーも返す
- **組み立て** (`build_model`): `nn.Module` (`GraphModel`) を返す. ノードの部品を `nn.ModuleDict` に持ち, 前もって決めた順 (トポロジカル順) に計算する. `torch.save` と `state_dict` が普通に使える
- **書き出し** (`export_python`): 同じ計算をする `nn.Module` のクラスの Python のコードを書き出す

## 外部のライブラリ

- PyTorch Geometric と Hugging Face の transformers は任意の依存 (`pip install "ml-modelcontrol[geometric]"`, `[huggingface]`). 入っていない環境でも, 残りの部品は使える
- Hugging Face のモデルを名前で読む場合はネットワークを使う. 確認と形の推定では, 重みを読まずに設定だけから作ることもできる

## 段階

| 段階 | 内容 |
| --- | --- |
| 1 | 登録簿 (`torch.nn` と演算), モデルの定義, 確認, 組み立て, 形の推定 |
| 2 | PyTorch Geometric と Hugging Face の部品, 自作の部品の登録, Python のコードの書き出し |
| 3 | ML-CreateArchitecture (GUI) から使う API の整備 |
| 4 | ML-Found との連携 (定義をプロジェクトの config から読んで学習する) |
