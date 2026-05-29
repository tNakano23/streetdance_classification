import os
import re
import math
import random
import cv2
import tqdm
from datetime import datetime
from typing import List, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont
from moviepy import VideoFileClip
from madmom.features.beats import RNNBeatProcessor, BeatTrackingProcessor

# ==========================================
# 1. ビート解析 & フレーム切り出しモジュール
# ==========================================
class VideoProcessor:
    @staticmethod
    def extract_beat_times(video_path: str) -> List[float]:
        """動画から音声を一時的に抽出し、ビートの位置（秒）をリストで返す"""
        temp_wav = f"temp_{datetime.now().strftime('%Y%m%d%H%M%S')}.wav"
        try:
            clip = VideoFileClip(video_path)
            clip.audio.write_audiofile(temp_wav, logger=None)
            clip.close()
            
            proc = RNNBeatProcessor()
            act = proc(temp_wav)
            processor = BeatTrackingProcessor(fps=100)
            beat_times = processor(act).tolist()
            return beat_times
        finally:
            if os.path.exists(temp_wav):
                os.remove(temp_wav)

    @staticmethod
    def save_frames(video_path: str, seconds_list: List[float], output_root: str, interval_mode: bool = False):
        """指定された秒数、または等間隔でフレームを抽出して保存する"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"動画を開けませんでした: {video_path}")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        video_basename = os.path.basename(video_path).replace(".mp4", "")
        output_dir = os.path.join(output_root, f"{video_basename}_{datetime.now().strftime('%Y%m%d%H%M')}")
        os.makedirs(output_dir, exist_ok=True)
        
        # ターゲットとなるフレーム番号をセット化
        if interval_mode:
            # seconds_list を [開始秒, 終了秒, ステップ秒] として扱う
            start, end, step = seconds_list[0], min(seconds_list[1], duration), seconds_list[2]
            target_seconds = []
            while start < end:
                target_seconds.append(start)
                start += step
            target_frames = {int(fps * s) for s in target_seconds if s < duration}
        else:
            target_frames = {int(fps * s) for s in seconds_list if s < duration}
            
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx in target_frames:
                current_second = frame_idx / fps
                output_path = os.path.join(output_dir, f"{video_basename}_{current_second:.2f}s.jpg")
                cv2.imwrite(output_path, frame)
            frame_idx += 1
            
        cap.release()
        return output_dir

# ==========================================
# 2. 画像結合 & ラベリングモジュール
# ==========================================
class ImageComposer:
    SEC_PATTERN = re.compile(r"_(\d+(?:\.\d+)?)s")

    @classmethod
    def get_seconds_from_name(cls, path: str) -> float:
        m = cls.SEC_PATTERN.search(os.path.basename(path))
        return float(m.group(1)) if m else float('inf')

    @staticmethod
    def compute_grid(n: int, cols: Optional[int] = None) -> Tuple[int, int]:
        if n <= 0: return (0, 0)
        if cols and cols > 0:
            return cols, math.ceil(n / cols)
        c = math.ceil(math.sqrt(n))
        return c, math.ceil(n / c)

    @staticmethod
    def draw_label(draw: ImageDraw.ImageDraw, box: Tuple[int, int, int, int], index: int, corner: str, bg_color: Tuple[int,int,int]):
        x, y, w, h = box
        text = f"({index})"
        size = int(min(w, h)//10)
        font="sandbox/make_dataset/font/ariali.ttf"
        
        try:
            font = ImageFont.truetype(font, size)
        except OSError:
            font = ImageFont.load_default()
            
        txt_bbox = draw.textbbox((0, 0), text, font=font, anchor='lt')
        tw, th = txt_bbox[2] - txt_bbox[0], txt_bbox[3] - txt_bbox[1]
        pad = max(4, th // 6)
        
        if corner == '左上': tx, ty = x + pad, y + pad
        elif corner == '右上': tx, ty = x + w - tw - pad, y + pad
        elif corner == '左下': tx, ty = x + pad, y + h - th - pad
        else: tx, ty = x + w - tw - pad, y + h - th - pad
        
        # 反転色でアウトラインを描画して視認性を確保
        outline = (255, 255, 255) if sum(bg_color)/3 < 128 else (0, 0, 0)
        fill = (0, 0, 0) if outline == (255, 255, 255) else (255, 255, 255)
        
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                draw.text((tx + dx, ty + dy), text, font=font, fill=outline)
        draw.text((tx, ty), text, font=font, fill=fill)

    def compose(self, img_paths: List[str], layout: str, enable_label: bool, corner: str, cols: Optional[int] = None, bg_color=(255,255,255)) -> Image.Image:
        # 秒数でソート
        img_paths = sorted(img_paths, key=self.get_seconds_from_name)
        images = [Image.open(p).convert('RGB') for p in img_paths]
        
        n = len(images)
        if n == 0: raise ValueError("画像がありません")
        
        if layout == '縦一列':
            max_w = max(im.width for im in images)
            resized = [im.resize((max_w, int(im.height * (max_w / im.width))), Image.LANCZOS) for im in images]
            out = Image.new('RGB', (max_w, sum(im.height for im in resized)), bg_color)
            y = 0
            draw = ImageDraw.Draw(out)
            for idx, im in enumerate(resized, start=1):
                out.paste(im, (0, y))
                if enable_label: self.draw_label(draw, (0, y, im.width, im.height), idx, corner, bg_color)
                y += im.height
                
        elif layout == '横一列':
            max_h = max(im.height for im in images)
            resized = [im.resize((int(im.width * (max_h / im.height)), max_h), Image.LANCZOS) for im in images]
            out = Image.new('RGB', (sum(im.width for im in resized), max_h), bg_color)
            x = 0
            draw = ImageDraw.Draw(out)
            for idx, im in enumerate(resized, start=1):
                out.paste(im, (x, 0))
                if enable_label: self.draw_label(draw, (x, 0, im.width, im.height), idx, corner, bg_color)
                x += im.width
                
        else:  # 格子状
            grid_cols, grid_rows = self.compute_grid(n, cols)
            cell_w = min(im.width for im in images)
            cell_h = min(im.height for im in images)
            
            out = Image.new('RGB', (grid_cols * cell_w, grid_rows * cell_h), bg_color)
            draw = ImageDraw.Draw(out)
            
            for i, im in enumerate(images):
                r, c = i // grid_cols, i % grid_cols
                # アスペクト維持縮小パディング
                scale = min(cell_w / im.width, cell_h / im.height)
                nw, nh = int(im.width * scale), int(im.height * scale)
                rim = im.resize((nw, nh), Image.LANCZOS)
                
                cx, cy = c * cell_w, r * cell_h
                ox, oy = (cell_w - nw) // 2, (cell_h - nh) // 2
                out.paste(rim, (cx + ox, cy + oy))
                
                if enable_label:
                    self.draw_label(draw, (cx, cy, cell_w, cell_h), i + 1, corner, bg_color)
        return out