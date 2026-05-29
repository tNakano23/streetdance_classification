import csv
import datetime
import glob
import os
import re
import shutil
import sys
from collections import Counter

import numpy as np
import pandas as pd
import tqdm
from sklearn.metrics import accuracy_score, f1_score


# =====================
# 1. キーワードマッチングによる抽出
# =====================
def extract_by_keywords(text, genre_list, move_list):
    if text is None or not isinstance(text, str):
        return None, None

    pred_l1 = None
    pred_l2 = None

    for g in sorted(genre_list, key=len, reverse=True):
        if g in text:
            pred_l1 = g
            break

    for m in sorted(move_list, key=len, reverse=True):
        if m in text:
            pred_l2 = m
            break

    return pred_l1, pred_l2

# =====================
# 2. ラベル情報の読み込み
# =====================
def get_label_lists(tsv_path):
    genre_set = set()
    move_set = set()
    valid_gm_set = set()

    with open(tsv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            g = row["GENRE_NAME"]
            m = row["CHOREOGRAPHY_NAME"]
            genre_set.add(g)
            move_set.add(m)
            valid_gm_set.add(f"{g}_{m}")

    return list(genre_set), list(move_set), valid_gm_set


# =====================
# 3. 統合処理メイン
# =====================
def process_and_evaluate(pred_csv, genre_list, move_list, valid_gm_set, out_dir):
    data = []
    with open(pred_csv, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for irow, row in enumerate(reader):
            if len(row) < 3:
                continue

            gt_l1, gt_l2, l3_text = row[0], row[1], row[2]
            pred_l1, pred_l2 = extract_by_keywords(l3_text, genre_list, move_list)

            data.append(
                {
                    "INDEX": irow,
                    "gt_l1": gt_l1,
                    "gt_l2": gt_l2,
                    "pred_l1": pred_l1,
                    "pred_l2": pred_l2,
                    "l3_text": l3_text,
                }
            )

    df = pd.DataFrame(data).fillna("none")
    df["gt_gm"] = df["gt_l1"] + "_" + df["gt_l2"]
    df["pred_gm"] = df["pred_l1"] + "_" + df["pred_l2"]

    # 評価
    before_count = len(df)
    df_valid = df[
        df["gt_gm"].isin(valid_gm_set) & df["pred_gm"].isin(valid_gm_set)
    ].reset_index(drop=True)
    after_count = len(df_valid)

    if after_count == 0:
        return None, None

    results = {
        "genre": {
            "acc": accuracy_score(df_valid["gt_l1"], df_valid["pred_l1"]),
            "f1": f1_score(df_valid["gt_l1"], df_valid["pred_l1"], average="macro"),
        },
        "move": {
            "acc": accuracy_score(df_valid["gt_l2"], df_valid["pred_l2"]),
            "f1": f1_score(df_valid["gt_l2"], df_valid["pred_l2"], average="macro"),
        },
        "gm": {
            "acc": accuracy_score(df_valid["gt_gm"], df_valid["pred_gm"]),
            "f1": f1_score(df_valid["gt_gm"], df_valid["pred_gm"], average="macro"),
        },
    }

    gm_per_class = {}
    for cls in sorted(df_valid["gt_gm"].unique()):
        sub = df_valid[df_valid["gt_gm"] == cls]
        gm_per_class[cls] = {
            "acc": (sub["gt_gm"] == sub["pred_gm"]).sum() / len(sub),
            "f1": f1_score(
                (df_valid["gt_gm"] == cls),
                (df_valid["pred_gm"] == cls),
                zero_division=0,
            ),
            "support": len(sub),
            "breakdown": dict(Counter(sub["pred_gm"])),
        }

    # 評価済み個別CSVの保存
    base_name = os.path.basename(pred_csv).replace(".csv", "")
    df_valid.to_csv(
        os.path.join(out_dir, f"{base_name}_evaluated.csv"),
        index=False,
        encoding="utf-8",
    )

    # Simple用データ (一行)
    filename = os.path.basename(pred_csv)

    report_simple = (
        f"{filename:<15.15}\t & "
        f"{results['genre']['acc'] * 100:5.2f} & "
        f"{results['move']['acc'] * 100:5.2f} & "
        f"{results['gm']['acc'] * 100:5.2f} & "
        f"{results['genre']['f1'] * 100:5.2f} & "
        f"{results['move']['f1'] * 100:5.2f} & "
        f"{results['gm']['f1'] * 100:5.2f}"
    )

    # Detail用データ
    report_detail = [
        f"CSV: {filename}",
        f"Valid samples (Keyword Matched): {after_count} / {before_count}",
        f"Metrics (acc, f1):",
        f"  [Genre] {results['genre']['acc'] * 100:.2f}% & {results['genre']['f1'] * 100:.2f}%",
        f"  [Move]  {results['move']['acc'] * 100:.2f}% & {results['move']['f1'] * 100:.2f}%",
        f"  [GM]    {results['gm']['acc'] * 100:.2f}% & {results['gm']['f1'] * 100:.2f}%\n",
        "[GM per class]",
    ]
    for cls, v in gm_per_class.items():
        bd_str = ", ".join(
            [
                f"【{p}:{c}件】"
                for p, c in sorted(v["breakdown"].items(), key=lambda x: -x[1])
            ]
        )
        report_detail.append(
            f"  {cls}: acc={v['acc'] * 100:.2f}%, f1={v['f1'] * 100:.2f}%, n={v['support']} | [{bd_str}]"
        )

    return report_simple, "\n".join(report_detail)


# =====================
# 0. エントリポイント
# =====================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python unified_evaluate.py <input_dir_path>")
        sys.exit(1)

    INPUT_DIR = sys.argv[1]
    LABEL_TSV = "/home/tanaka_n/dance_mm/data/dance_choreo_label_mapping.tsv"

    genre_list, move_list, valid_gm_set = get_label_lists(LABEL_TSV)

    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M")
    OUT_DIR = os.path.join(
        "data/result_eval", timestamp
    )
    os.makedirs(OUT_DIR, exist_ok=True)

    csv_files = sorted(
        glob.glob(os.path.join(INPUT_DIR, "**", "*.csv"), recursive=True)
    )

    if not csv_files:
        print("No CSV files found in the input directory.")
        sys.exit(1)

    # 保存先ファイルのパス
    simple_path = os.path.join(OUT_DIR, "summary_simple.txt")
    detail_path = os.path.join(OUT_DIR, "summary_detail.txt")

    simple_list = [
        "filename         & g_acc & m_acc & gm_acc&  g_f1 &  m_f1 & gm_f1"
    ]  # ヘッダー

    with open(detail_path, "w", encoding="utf-8") as df:
        for csv_path in tqdm.tqdm(csv_files):
            try:
                res_s, res_d = process_and_evaluate(
                    csv_path, genre_list, move_list, valid_gm_set, OUT_DIR
                )
                if res_s:
                    simple_list.append(res_s)
                    df.write(res_d + "\n")
                    df.write("=" * 50 + "\n\n")
                    df.write(res_d + "\n")
                    df.write("=" * 50 + "\n\n")
            except Exception as e:
                print(f"Error processing {csv_path}: {e}")

    # Simple版をまとめて書き出し
    with open(simple_path, "w", encoding="utf-8") as sf:
        sf.write("\n".join(simple_list))

    print(f"Done! \nSimple: {simple_path}\nDetail: {detail_path}")

