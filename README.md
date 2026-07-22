# what is this???

ストリートダンスおよびコンテンポラリーダンスの動画・画像から、ダンスの「ジャンル」や「具体的な動き（ムーブ）」を自動で識別・分類するための、大規模言語・視覚モデル（VLM）をベースとした研究開発リポジトリです。本プロジェクトは卒業研究の成果物をベースに構築されています。

動画からビート（拍）に合わせたフレーム抽出、複数フレームの画像結合によるデータセット作成、Qwenファミリーをベースとしたモデルのファインチューニング、および実験結果の推論・評価までの一連のパイプラインを提供します。

---

## 主な機能

* **データセット自動生成 (`sandbox/make_dataset`)**: `madmom`を用いた音楽のビート同期、または固定時間間隔（インターバル）でのフレーム抽出に対応。複数フレームを1枚の画像（格子状・一列）にパッキングする機能を搭載。
* **推論マネジメント (`sandbox/inf`)**: YAMLファイルベースの設定管理と、Streamlitによる設定GUI。Qwen2.5-VLなどの最新VLMに対するバッチ推論の実行。
* **ファインチューニング (`sandbox/finetune`)**: LoRAを用いた効率的なビジョン・言語タスクへの追加学習環境の構築。
* **評価・可視化 (`sandbox/eval`)**: 推論結果（CSV）の分析。正解率（Accuracy）やF1スコアの算出、LaTeX用テーブル（TeX形式）の自動出力、Streamlitによるヒートマップ等の視覚化。

---

## ディレクトリ構成

```text

streetdance_classification/
├── data/                  # データセットおよび実験結果の格納領域（Google Driveで公開しています。）
│   ├── frames/            # 1_cli_clipper.py で動画から抽出されたフレーム画像
│   ├── dataset/           # 2_cli_combiner.py で結合された学習・推論用画像
│   ├── dataset_json/      # 3_make_datasetjson.py で生成されたQwen形式のJSONファイル
│   ├── result/            # qwen_inf.py で出力された推論結果のCSV
│   ├── result_eval/       # ev.py で出力されたスコア評価テキスト（summary_simple.txt等）
│   └── dance_choreo_label_mapping.tsv  # ジャンルやムーブ（動き）のID対応表
├── models/                # 成果物としての学習済みモデル（デモ用）（Google Driveで公開しています。）
├── sandbox/               # 研究開発用コアスクリプト群
│   ├── make_dataset/      # フレーム抽出・画像結合・JSON作成
│   ├── inf/               # 推論実行・設定生成
│   ├── finetune/          # ファインチューニング用スクリプト・環境定義
│   └── eval/              # スコア評価およびStreamlitによる結果の可視化
├── .gitignore             # Git管理対象外の設定
├── README.md              # 本ファイル
└── uv.lock                # パッケージ管理用のロックファイル（uv推奨）


```


---
## 動作環境・セットアップ

### 仮想環境について

本プロジェクトではパッケージマネージャーとして [uv](https://github.com/astral-sh/uv) を使用しています。

`uv` は、Rust製で開発された極めて高速なPythonのパッケージマネージャーです。

各`sandbox/**/script/`内に `uv.lock`または`pyproject.toml`（またはその両方）が入っているので、以下のコマンドで仮想環境の構築と同期をして下さい。


```bash
uv sync

```

### データのクローン
本研究で作成したスクリプトは、以下のコマンドを実行することでクローンできます。

```bash

git clone "https://github.com/tNakano23/streetdance_classification.git"
cd streetdance_classification

```


### データのダウンロード

本研究で使用したデータ(data)および学習済み重み(models)は、Google Driveで公開しています。
以下のコマンドを実行することで、環境に直接ダウンロードして展開できます。

1. gdownのインストール

    ```bash
    pip install gdown
    ```


1. 学習済みモデルのダウンロードと展開 (7GB)
    
    一時的に`tmp_models.zip`として落とし、`models/`フォルダを新規作成してそこに直接解凍します。
    
    ```bash
    gdown "https://drive.google.com/uc?id=1NaFFZdTfnZ61ltrubkp-iwICSqEkL6Ck" -O tmp_models.zip
    mkdir -p models
    unzip -j tmp_models.zip -d models/
    rm tmp_models.zip
    ```
    
1. データセットのダウンロードと展開 (34GB)
    
    一時的に`tmp_data.zip`として落とし、`data/`フォルダを新規作成してそこに直接解凍します。
    
    ```bash
    gdown "https://drive.google.com/uc?id=1RsT8fuaKbNsgqB5Ryh0ruNqa_x33j329" -O tmp_data.zip
    mkdir -p data
    unzip -j tmp_data.zip -d data/
    rm tmp_data.zip
    ```


---

## 各モジュールの詳細と使用方法

### 1. データセット作成 (`sandbox/make_dataset`)

元動画から、VLMの学習・推論に適したデータおよびQwen形式の学習用JSONを生成します。

* **`1_cli_clipper.py`**: 動画をフレーム分割して保存します。
	* `beatモード`: 音楽のビートに合わせてフレームを抽出（`madmom`を使用）。
	* `intervalモード`: 指定した秒数（`interval_sec`）ごとにフレームを抽出。


* **`2_cli_combiner.py`**: 抽出した複数フレームを1枚の画像に結合します。
	* `ngram_n`: まとめるフレーム数。
	* `layout`: `"格子状"`, `"横一列"`, `"縦一列"` から選択（格子状の列数 `cols` も指定可能）。


* **`2.5_gui_app.py`**: 上記2つの機能を統合したGUIアプリケーションです。
* **`3_make_datasetjson.py`**: 画像や動画ファイルを読み込み、Qwen形式のデータセット用JSONを出力します。

---

### 2. モデル学習 (`sandbox/finetune`)

1. **ベースリポジトリのクローン**
    本スクリプト群を動かす前に、以下の外部リポジトリをクローンして環境を構築してください。
    
    ```bash
    git clone "https://github.com/2U1/Qwen-VL-Series-Finetune.git"
    ```

2. **環境構築・実行**
    `pyproject.toml` および `uv.lock` はクローン前に格納してあります。
    
    ```bash
    uv sync
    ```
    環境を構築後、以下のシェルスクリプトを参考にファインチューニングを実行します。
    * `script/my_finetune_lora_vision_*.sh` （画像・LoRA全層対象）
    * `script/my_finetune_video_*.sh` （動画対象）


---


### 3. モデル推論 (`sandbox/inf`)

作成したデータセットとモデルを用いて推論を行います。

* **設定の作成**: `_make_config.py`（Streamlit）および `launch_config_server.py` を用いて、推論に必要なYAMLファイルを直感的に生成できます。
* **推論の実行**: `qwen_inf.py` を実行し、YAMLの設定に従ってバッチ推論を行います。（詳細は`sandbox/inf/README.md`を参照ください。）

---

### 4. スコア評価と可視化 (`sandbox/eval`)

出力された推論結果（CSV）を基に、定量的なパフォーマンス評価を行います。

- **`ev.py`**: ディレクトリ内の結果CSVを一括評価し、2つのテキストファイルを生成します。
	- `summary_detail.txt`: 全体スコアに加え、各ダンスクラス（例: `Ballet Jazz_Fouette` 等）ごとの回答分布や詳細なメトリクスを出力。
	- `summary_simple.txt`: 論文・報告書にそのまま貼り付けられる **LaTeX（tex）の表形式** で、タスクごとの正解率（acc）とF1スコアを出力。

実行例は以下のとおりです。

```bash
uv run --project sandbox/eval sandbox/eval/script/ev.py data/result

```


- **`vis.py`**: `summary_detail.txt` を基に、正解数のヒートマップやクラスごとの傾向をブラウザ上で視覚的に分析できるStreamlitアプリです。

実行例は以下のとおりです。

```bash
streamlit run sandbox/eval/script/vis.py

```



---

## 最後に

本プロジェクトでは一部のデータをdemoとして公開しております。
データ元は [AIST Dance Video Database](https://aistdancedb.ongaaccel.jp/) です。貴重なデータセットの公開に深く感謝申し上げます。
