# ML-ModelControl

PyTorch (と互換のライブラリ) で作れるモデルの部品を登録簿として持ち, 部品をつないだ定義 (YAML か JSON) から `nn.Module` を組み立てるライブラリ.

- `nn.Linear` のような層, 足し算や連結のような演算, PyTorch Geometric のグラフニューラルネットワークの層, Hugging Face の学習済みのモデルを, 同じ形の部品として扱う
- 分岐と合流のあるモデル (残差のつなぎ方, 複数の入力と出力) を定義できる
- 定義の確認 (部品の名前, 引数, つなぎ方) と, 各部品の出力の形とパラメータ数の推定を, 部品ごとの結果で返す. GUI で部品を組み立てるアプリ (ML-CreateArchitecture) はこれを使う

設計の方針は [docs/DESIGN.md](docs/DESIGN.md) を参照. 開発中のため, 使い方は実装に合わせて追記する.

## 導入

```bash
git clone https://github.com/victor-von-pooh/ML-ModelControl.git
cd ML-ModelControl
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"                          # 基本
pip install -e ".[dev,geometric,huggingface]"    # PyTorch Geometric と Hugging Face の部品も使う場合
```

## 開発

```bash
pytest
```

コードの書き方は [CODING_STYLE.md](CODING_STYLE.md) に従う.

## ライセンス

MIT
