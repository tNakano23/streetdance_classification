# streetdance_classification/sandbox/inf

モデル推論のためのディレクトリです。

## ファイルの構成と機能

- `_make_config.py`
  - qwen_inf.pyで用いるyamlファイルを生成するためのファイルです。streamlit上で動きます。
- `launch_config_server.py`
  - _make_configをsubprocessから起動するためのプログラムです。
- `qwen_inf.py`
  - あらかじめ作成されたyamlファイルをもとに、推論を行います。


## yamlの内容

"""
experiment:
  method_name: demo     #・・・(任意)プロジェクトの名前を設定します。
  json_path: data/dataset_json/beat_clip/test.json      #・・・推論に用いるjsonパスを指定します。/sandbox/inf/configs/presets.jsonより指定プリセット指定できます。
  data_path: data/dataset/beat_clip      #・・・推論に用いるデータパスを指定します。/sandbox/inf/configs/presets.jsonより指定プリセット指定できます。
  output_dir: data_result      #・・・推論結果(csv)を保存するパスを指定します。
  content_type: image      #・・・["image" or "video"] prompt処理やdatasetの選択に関わります。
  prompt_type: basic      #・・・プロンプトを指定します。
  batch_save_size: 100
  max_errors: 100      #・・・この数以上のエラーが出たら、その時点で推論処理を中断します。
  debug_mode: true      #・・・デバッグ時はtrueにすることで、指定ステップで処理を中断することができます。
  debug_limit: 100      #・・・debug_mode←true　でないと機能しません。
  seed: 42      #・・・randomシードの値を指定します。
model_config:
  model_name: Qwen/Qwen2.5-VL-3B-Instruct      #・・・使用するQwenモデルを選択します。
  load_lora: true      #・・・trueの場合、loraを読み込みます。falseの場合、非追加学習モデルを使用します。
  lora_path: models/demo/Qwen2.5-VL-3B-Instruct/checkpoint-14740      #・・・load_lora←true　でないと機能しません。
gen_params:
  max_new_tokens: 2048
  do_sample: false
prompts:        #・・・content_type←"video"の場合、 "この動画のダンスのジャンルと～～"に置き換わります。
  basic: '# Role

    あなたは世界レベルのストリートダンス・コンテンポラリーダンスの専門審査員です。

    # Task

    この画像のダンスのジャンルと、動きの名前を教えてください。'
"""