# streetdance_classification/sandbox/eval

csvのスコア評価のためのスクリプトです。

## ファイルの構成と機能

- `ev.py`
  - """""EXAMPLE"""""""""""""""""""""""""""""
  - `uv run --project sandbox/eval sandbox/eval/script/ev.py data/result`
  - """""""""""""""""""""""""""""""""""""""""
  - 引数に `inf/script/qwen_inf.py` で出力したcsvが含まれるディレクトリを指定します。
  - このスクリプトから出力されるファイルは2つです。
    - `summary_simple.txt`
      - texの表形式に対応した簡易なスコア表示です。
      - タスクごとに正解率(acc)、F1スコア(f1)が表示されます。
        - """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        - filename         & g_acc & m_acc & gm_acc&  g_f1 &  m_f1 & gm_f1
        - 20260529_163602	 & 100.00 & 90.00 & 90.00 & 100.00 & 86.33 & 87.06
        - """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    - `summary_detail.txt`
      - タスクごとの正解率(acc)、F1スコア(f1)に加え、クラスごとの回答の分布が表示されます。
      - 各クラスの分析に便利なファイルです。
        - """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        - CSV: 20260529_163602_demo.csv
        - Valid samples (Keyword Matched): 100 / 101
        - Metrics (acc, f1):
        - [Genre] 100.00% & 100.00%
        - [Move]  90.00% & 86.33%
        - [GM]    90.00% & 87.06%
        - 
        - [GM per class]
        - Ballet Jazz_Fouette: acc=100.00%, f1=100.00%, n=1 | [【Ballet Jazz_Fouette:1件】]
        - Ballet Jazz_grand pas de chat: acc=100.00%, f1=100.00%, n=1 | [【Ballet Jazz_grand pas de chat:1件】]
        - Ballet Jazz_pas de chat: acc=100.00%, f1=100.00%, n=1 | [【Ballet Jazz_pas de chat:1件】]
        - Break_3 step: acc=100.00%, f1=100.00%, n=1 | [【Break_3 step:1件】]
        - ...
        - """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
- `vis.py`
  - """""EXAMPLE"""""""""""""""""""""""""""""
  - `streamlit run sandbox/eval/script/vis.py`
  - """""""""""""""""""""""""""""""""""""""""
  - summary_detail.txt を視覚化するスクリプトです。Streamlitを用います。 
  - 正解数のHeatMapやクラスごとの回答など詳細を確認できます。
