import glob
import os
from tqdm import tqdm
from _modules import VideoProcessor

"""
動画をフレームに分割して保存するスクリプト
- beatモード: 音楽のビートに合わせてフレームを抽出（beat推定はmadmomによる）
- intervalモード: 一定時間ごとにフレームを抽出（interval_sec 秒ごと）
"""

def run_clipper(input_dir: str, output_dir: str, mode: str = "beat", interval_sec: float = 0.5):
    """
    mode: "beat"（音楽のビート自動同期） または "interval"（一定時間おき）
    """
    # 拡張子は大文字小文字を考慮
    video_paths = glob.glob(os.path.join(input_dir, "**/*.mp4"), recursive=True)
    video_paths = sorted(video_paths)
    
    print(f"🎥 検出された動画数: {len(video_paths)}本")
    
    for path in tqdm(video_paths, desc="動画処理中"):
        try:
            if mode == "beat":
                print(f"\n🎵 ビートを解析中: {os.path.basename(path)}")
                seconds = VideoProcessor.extract_beat_times(path)
                interval_mode = False
            else:
                # [開始, 終了(仮), ステップ]
                seconds = [0.0, 9999.0, interval_sec]
                interval_mode = True
                
            saved_dir = VideoProcessor.save_frames(path, seconds, output_dir, interval_mode=interval_mode)
            print(f"✅ フレーム保存完了: {saved_dir}")
        except Exception as e:
            print(f"❌ エラーが発生しました ({os.path.basename(path)}): {e}")

if __name__ == "__main__":
    # INPUT_DIR = "./data/VIDEOS"
    INPUT_DIR = "./data/dataset/video_ft"
    OUTPUT_DIR = "./data/frames"
    RUN_MODE = "beat" # "beat" または "interval"
    # INTERVAL_SEC = 0.5 # intervalモードのときの秒数間隔

    run_clipper(INPUT_DIR, OUTPUT_DIR, mode=RUN_MODE)