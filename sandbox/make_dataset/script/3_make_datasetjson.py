import os
import json
import argparse
import csv
import random
from glob import glob
from datetime import datetime
"""
動画や画像のファイルを読み込み、Qwen形式の学習用JSONを生成するスクリプト
- コマンドライン引数で、入力パス、出力パス、対応表のパス、拡張子["jpg", "mp4"]、データ分割の割合、ランダムシードを指定可能
- 扱う引数については、parse_args関数も参照ください

*******************************************************************************************************
[EXAMPLE USAGE]
python sandbox/make_dataset/script/3_make_datasetjson.py \
--in_path "./data/dataset" \
--out_path "./data/dataset_json" \
--mapping_tsv "./data/dance_choreo_label_mapping.tsv" \
--ext "jpg"  \
--train_ratio 0.8 \
--val_ratio 0.1 \
--seed 42
*******************************************************************************************************
"""

def parse_args():
    """コマンドライン引数のパースと値の整合性チェック"""
    parser = argparse.ArgumentParser(description="ダンスデータセットからVLMファインチューニング用JSONを生成・分割するスクリプト")
    parser.add_argument(
        "--in_path", 
        type=str, 
        default="../data/dataset", 
        help="入力ルートディレクトリのパス (構造: in_path/{method_name}/data)"
    )
    parser.add_argument(
        "--out_path", 
        type=str, 
        default="../data/dataset_json", 
        help="出力先ルートディレクトリのパス"
    )
    parser.add_argument(
        "--mapping_tsv", 
        type=str, 
        required=True,
        help="ジャンル・動きの対応表(TSV)のパス"
    )
    parser.add_argument(
        "--ext", 
        type=str, 
        default="jpg", 
        choices=["jpg", "mp4"], 
        help="対象ファイルの拡張子 (jpg または mp4)"
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.8, help="訓練データの割合"
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.1, help="検証データの割合"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="シャッフル用のランダムシード"
    )
    
    args = parser.parse_args()

    # === 🚨 割合(ratio)の整合性バリデーション ===
    # 浮動小数点の演算誤差を考慮し、微小な許容誤差(1e-5)を持たせて判定
    if args.train_ratio + args.val_ratio > 1.0 + 1e-5:
        parser.error(
            f"エラー: --train_ratio ({args.train_ratio}) と --val_ratio ({args.val_ratio}) "
            f"の合計値 ({args.train_ratio + args.val_ratio}) が 1.0 を超えています。"
        )
    
    if args.train_ratio < 0 or args.val_ratio < 0:
        parser.error("エラー: 各 ratio に負の値を指定することはできません。")

    return args


# --- 以下、前回の関数群（省略せずに再掲） ---

def print_summary(files, image_dir, mapping_tsv, method_name):
    """処理開始前のログ出力"""
    print("\n" + "="*40)
    print(f" 🚀 処理開始: {method_name}")
    print("="*40)
    print(f"- データディレクトリ : {image_dir}")
    print(f"- 対応表(TSV)        : {mapping_tsv}")
    print(f"- 検出ファイル数     : {len(files)}")
    print("-" * 40)


def load_mapping(tsv_path):
    """TSVファイルを読み込み、IDからラベルへのマッピング辞書を作成する"""
    genre_mapping = {}
    choreo_mapping = {}
    with open(tsv_path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for idx, row in enumerate(reader):
            if idx == 0 or not row:
                continue
            genre_mapping[row[0]] = row[1]
            choreo_mapping[f"{row[0]}_{row[2]}"] = row[3]
    return genre_mapping, choreo_mapping


def build_vlm_samples(files, genre_mapping, choreo_mapping, ext, question):
    """ファイルリストからQwen形式の学習用データ（辞書のリスト）を構築する"""
    content_tag = "video" if ext == "mp4" else "image"
    samples = []

    for file in files:
        file_basename = os.path.basename(file)
        parts = file_basename.split(".")[0].split("_")
        
        if len(parts) <= max(0, 5):
            print(f"⚠️ 警告: ファイル名フォーマットが不正なためスキップします: {file_basename}")
            continue

        genre_id, pre_choreo_id = parts[0], parts[5]
        choreo_id = f"{genre_id}_{pre_choreo_id}"

        genre_label = genre_mapping.get(genre_id, genre_id)
        choreo_label = choreo_mapping.get(choreo_id, choreo_id)

        img_file_for_json = os.path.join(*file.split(os.sep)[-3:])

        sample = {
            "id": file_basename.rsplit(f".{ext}", 1)[0],
            content_tag: img_file_for_json,
            "genre": genre_label,
            "move": choreo_label,
            "conversations": [
                {
                    "from": "human",
                    "value": f"<{content_tag}>\n{question}"
                },
                {
                    "from": "gpt",
                    "value": f"ジャンルは{genre_label}で、動きの名前は{choreo_label}です。"
                }
            ]
        }
        samples.append(sample)
    
    return samples


def split_and_save_json(data, method_output_dir, train_ratio, val_ratio, seed):
    """データを train/val/test に分割して、それぞれのJSONファイルとして保存する"""
    random.seed(seed)
    random.shuffle(data)

    total = len(data)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))

    splits = {
        "train.json": data[:train_end],
        "val.json": data[train_end:val_end],
        "test.json": data[val_end:]
    }

    print(f"📊 データ分割結果 (Total: {total})")
    for filename, split_data in splits.items():
        save_path = os.path.join(method_output_dir, filename)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(split_data, f, ensure_ascii=False, indent=2)
        print(f"  - {filename}: {len(split_data)} 件 -> {save_path}")


def main():
    args = parse_args()
    
    question = "この画像のダンスのジャンルと、動きの名前を教えてください。"
    if args.ext == "mp4":
        question = question.replace("画像", "動画")

    if not os.path.exists(args.mapping_tsv):
        print(f"❌ エラー: 対応表が見つかりません: {args.mapping_tsv}")
        return

    genre_mapping, choreo_mapping = load_mapping(args.mapping_tsv)

    if not os.path.exists(args.in_path):
        print(f"❌ エラー: 入力ディレクトリが存在しません: {args.in_path}")
        return
        
    method_dirs = [
        d for d in glob(os.path.join(args.in_path, "*")) 
        if os.path.isdir(d)
    ]

    if not method_dirs:
        print(f"⚠️ 警告: {args.in_path} 内にサブディレクトリ（手法）が見つかりませんでした。")
        return

    for method_dir in method_dirs:
        method_name = os.path.basename(method_dir)
        
        files = glob(os.path.join(method_dir, "**", f"*.{args.ext}"), recursive=True)
        if not files:
            print(f"⚠️ {method_name}: 対象ファイル ({args.ext}) が見つからないためスキップします。")
            continue

        files = sorted(files)
        print_summary(files, method_dir, args.mapping_tsv, method_name)

        samples = build_vlm_samples(files, genre_mapping, choreo_mapping, args.ext, question)

        if not samples:
            print(f"⚠️ {method_name}: 有効なサンプルが生成されませんでした。")
            continue

        method_output_dir = os.path.join(args.out_path, method_name)
        os.makedirs(method_output_dir, exist_ok=True)

        split_and_save_json(
            samples, 
            method_output_dir, 
            args.train_ratio, 
            args.val_ratio, 
            args.seed
        )
        print(f"✅ {method_name} の処理が完了しました。\n")


if __name__ == "__main__":
    main()

    """
    [EXAMPLE USAGE]
    python sandbox/make_dataset/script/3_make_datasetjson.py \
    --in_path "./data/dataset" \
    --out_path "./data/dataset_json" \
    --mapping_tsv "./data/dance_choreo_label_mapping.tsv" \
    --ext "jpg"  \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --seed 42
    """