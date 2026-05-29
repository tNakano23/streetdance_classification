import os
import glob
from tqdm import tqdm
from _modules import ImageComposer

"""
抽出したフレームを複数枚組み合わせて1枚の画像にするスクリプト
- ngram_n: 何枚のフレームを1枚の画像にまとめるか
- layout: "格子状" または "横一列" または "縦一列"
- cols: 格子の列数（layoutが"格子状"のときのみ有効｜行数はなるべく四角形になるように補完）
"""

def run_combiner(frames_root: str, output_root: str, ngram_n: int = 9, layout: str = "格子状", cols: int = 3):
    composer = ImageComposer()
    os.makedirs(output_root, exist_ok=True)
    
    # フレームが入っているサブディレクトリを探索
    sub_dirs = [os.path.join(frames_root, d) for d in os.listdir(frames_root) if os.path.isdir(os.path.join(frames_root, d))]
    
    for s_dir in tqdm(sub_dirs, desc="フォルダ毎に結合中"):
        img_paths = glob.glob(os.path.join(s_dir, "*.jpg")) + glob.glob(os.path.join(s_dir, "*.png"))
        img_paths = sorted(img_paths, key=composer.get_seconds_from_name)
        
        if len(img_paths) < ngram_n:
            print(f"⚠️ 画像が足りません ({os.path.basename(s_dir)}): {len(img_paths)}枚 / 必要{ngram_n}枚")
            continue
            
        # N枚ずつのチャンクを作って結合
        for i in range(len(img_paths) - ngram_n + 1):
            chunk = img_paths[i:i+ngram_n]
            
            try:
                out_img = composer.compose(
                    img_paths=chunk,
                    layout=layout,
                    enable_label=True, #各フレームに時系列（順序）ラベルをつける
                    corner="左上", #順序ラベルの位置（"左上"、"左下"、"右上"、"右下"）
                    cols=cols,
                    bg_color=(0, 0, 0) # 背景色（RGB）
                )
                
                # 保存処理
                dir_name = os.path.basename(s_dir)
                save_dir = os.path.join(output_root, dir_name)
                os.makedirs(save_dir, exist_ok=True)
                
                save_path = os.path.join(save_dir, f"{dir_name}_comb_{i:04d}.jpg")
                out_img.save(save_path)
            except Exception as e:
                print(f"❌ 結合失敗 ({s_dir} index {i}): {e}")

if __name__ == "__main__":
    # FRAMES_DIR   = "./data/frames/beat_clip"
    # COMBINED_DIR = "./data/dataset/beat_clip"
    FRAMES_DIR   = "./data/frames"
    COMBINED_DIR = "./data/98_dast"
    
    # 9枚の格子状(3x3)に結合
    run_combiner(FRAMES_DIR, COMBINED_DIR, ngram_n=9, layout="格子状", cols=3)