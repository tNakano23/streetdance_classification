# streetdance_classification/sandbox/make_dataset

元動画からのデータ生成、学習用のjsonデータ作成のためのスクリプトです。

## ファイルの構成と機能

- `_modules.py`
  - `1_cli_clipper.py`、`2_cli_combiner.py`、`2.5_gui_app.py`で使用する共通のmoduleが記述されています。
- `1_cli_clipper.py`
  - 動画をフレームに分割して保存するスクリプトです。
  - 各パラメタはエントリポイントから直接指定して下さい。
  - モードが２つ存在します。
    - beatモード: 音楽のビートに合わせてフレームを抽出（beat推定はmadmomによる）
    - intervalモード: 一定時間ごとにフレームを抽出（interval_sec 秒ごと）
- `2_cli_combiner.py`
  - 抽出したフレームを複数枚組み合わせて1枚の画像にするスクリプトです。
  - 各パラメタはエントリポイントから直接指定して下さい。
    - ngram_n: 何枚のフレームを1枚の画像にまとめるか
    - layout: "格子状" または "横一列" または "縦一列"
    - cols: 格子の列数（layoutが"格子状"のときのみ有効｜行数はなるべく四角形になるように補完）
- `2.5_gui_app.py`
  - `1_cli_clipper.py` と `2_cli_combiner.py` の機能を統合したGUIアプリです。
- `3_make_datasetjson.py`
  - 動画や画像のファイルを読み込み、Qwen形式の学習用JSONを生成するスクリプトです。
  - コマンドライン引数で、入力パス、出力パス、対応表のパス、拡張子["jpg", "mp4"]、データ分割の割合、ランダムシードを指定可能です。
  - 扱う引数については、スクリプト内の parse_args関数 も参照ください。
