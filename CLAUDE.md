# CLAUDE.md

ML-ModelControl で作業するときの方針をまとめる.

## このリポジトリの目的

PyTorch (と互換のライブラリ) で作れるモデルの部品を登録簿として持ち, 部品をつないだ定義から `nn.Module` を組み立てるライブラリ. 設計は [docs/DESIGN.md](docs/DESIGN.md) にまとめる.

- GUI のアプリ (ML-CreateArchitecture) が必要とする情報 (部品の一覧と説明, 確認の結果, 形の推定) はこのライブラリが返す. 画面に固有のものは入れない
- 学習は対象外 (組み立てたモデルは ML-Found などで学習する)

## コードの書き方

- [CODING_STYLE.md](CODING_STYLE.md) に従う (日本語の NumPy 形式の docstring, コメントは体言止め, 79 文字以内, `torch.nn.functional` を使わないなど)
- ドキュメントとコードのコメントでは, 「機械学習を全て GUI 上で行う」という目的のために開発しているリポジトリ (今は ML-Found, ML-Dashboard, ML-ModelControl, ML-CreateArchitecture, ML_tester) 以外に触れない. ローカルのディレクトリ構成やパスも書かない. 外部のライブラリ (PyTorch Geometric, Hugging Face の transformers など) には触れてよい

## 進める流れ

1. 方針を変える場合や新しい機能を加える場合は, 先に docs/DESIGN.md を直す
2. テストを書き, `pytest` を通す. 任意の依存 (PyTorch Geometric, transformers) が無い環境でも, 残りのテストは通るようにする
3. **セルフレビュー**: 複数のエージェントを並行して動かし, バグ, 設計との食い違い, CODING_STYLE.md への準拠を確認する
4. 指摘を直し, 直した内容と残した指摘をユーザーに報告する

## git

- 意味のある変更ごとにブランチを切って PR を作る. `main` に直接プッシュしない
- CI が通ったことを確かめてからマージする. 自分の PR は GitHub の仕様で Approve できないので, 確かめた内容をレビューのコメントに残してからマージする
- コミットとプッシュ, PR のマージは基本的にユーザーが行う. ユーザーが `[Commit] <日時>までなら...` の prefix を付けて指示した場合は, その日時まで Claude の判断で行ってよい
- コミットメッセージと PR のタイトルは日本語の 1 行
