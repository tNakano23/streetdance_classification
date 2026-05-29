import os
import json
import csv
import yaml
import torch
import random
import numpy as np
import tqdm
import datetime
import cv2
from typing import Dict, List
from peft import PeftModel
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
)
from qwen_vl_utils import process_vision_info

# =================================================================
# Utilities
# =================================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def get_video_metadata(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    meta = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": cap.get(cv2.CAP_PROP_FRAME_COUNT),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    meta["duration_sec"] = meta["frame_count"] / meta["fps"] if meta["fps"] > 0 else 0
    return meta

def save_csv(data_list, path):
    if not data_list: return
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["ans_genre", "ans_move", "output_text", "path"])
        writer.writerows(data_list)

# =================================================================
# Core Classes
# =================================================================
class DatasetManager:
    """メタデータの管理とレジューム機能を担当"""
    def __init__(self, json_path: str, cache_dir: str = ".cache"):
        self.json_path = json_path
        self.cache_path = os.path.join(cache_dir, f"meta_{os.path.basename(json_path)}")
        os.makedirs(cache_dir, exist_ok=True)

    def prepare_metadata(self, dataset: List[Dict], data_root: str, content_type: str):
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "r") as f:
                return json.load(f)
        
        print("Pre-calculating video metadata...")
        meta_cache = {}
        for data in tqdm.tqdm(dataset, desc="Extracting Video Metadata"):
            item_id = data.get(content_type)
            if item_id and item_id not in meta_cache:
                full_path = os.path.join(data_root, item_id)
                meta_cache[item_id] = get_video_metadata(full_path)
        
        with open(self.cache_path, "w") as f:
            json.dump(meta_cache, f)
        return meta_cache

    def get_processed_count(self, csv_path: str) -> int:
        if not os.path.exists(csv_path):
            return 0
        with open(csv_path, "r", encoding="utf-8") as f:
            return max(0, sum(1 for _ in f) - 1)

class VLMInferenceEngine:
    """YAMLの model_config を受け取って初期化する推論エンジン"""
    def __init__(self, config: Dict):
        self.model_name = config.get('model_name')
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        load_lora = config.get('load_lora', False)
        lora_path = config.get('lora_path')
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        print(f"Loading model: {self.model_name}")
        if "Qwen2.5" in self.model_name:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name, torch_dtype="auto", device_map="auto"
            )
        elif "Qwen3.5" in self.model_name:
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_name, quantization_config=bnb_config, device_map="auto"
            )
        else:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_name, torch_dtype="auto", device_map="auto"
            )

        if load_lora and lora_path:
            print(f"Applying LoRA: {lora_path}")
            self.model = PeftModel.from_pretrained(self.model, lora_path)
        
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(self.model_name)

    def prepare_messages(self, conversations: List[Dict], content_path: str, content_type: str, video_meta: Dict = None):
        messages = []
        for c in conversations:
            role = 'user' if c['from'] == 'human' else 'assistant'
            content_list = []
            raw_text = c['value']
            
            if f'<{content_type}>' in raw_text:
                if content_type == "video" and video_meta:
                    content_list.append({
                        "type": "video", "video": content_path, 
                        "max_pixels": 1280 * 720, "fps": video_meta['fps']
                    })
                elif content_type == "image":
                    content_list.append({"type": "image", "image": content_path})
                
                text = raw_text.replace(f'<{content_type}>', '').strip()
                if text: content_list.append({"type": "text", "text": text})
            else:
                content_list.append({"type": "text", "text": raw_text})
            
            messages.append({"role": role, "content": content_list})
        return messages

    def infer(self, messages: List[Dict], video_meta: Dict = None, **gen_kwargs):
        try:
            text_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            
            # None の場合はキーを含めないようにする配慮
            proc_kwargs = {
                "text": [text_prompt], "images": image_inputs, "videos": video_inputs,
                "padding": True, "return_tensors": "pt"
            }
            if video_meta:
                proc_kwargs["video_metadata"] = video_meta

            inputs = self.processor(**proc_kwargs).to(self.device)
            output_ids = self.model.generate(**inputs, **gen_kwargs)
            
            generated_ids = [out[len(ins):] for ins, out in zip(inputs.input_ids, output_ids)]
            output_text = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            return output_text[0].split("</think>")[-1].strip()
        except Exception as e:
            raise e

# =================================================================
# Main Pipeline
# =================================================================
def main(config_path: str = "configs/0_qwen_config.yaml"):
    # 1. YAMLのロード (すべての設定はここから取得)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    exp_cfg = cfg['experiment']
    set_seed(exp_cfg.get('seed', 42))
    
    # 出力パスの設定
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = exp_cfg.get('output_dir', 'results')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{timestamp}_{exp_cfg['method_name']}.csv")

    # 2. データとメタデータの準備
    with open(exp_cfg['json_path'], "r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    ds_manager = DatasetManager(exp_cfg['json_path'])
    meta_cache = ds_manager.prepare_metadata(dataset, exp_cfg['data_path'], exp_cfg['content_type'])
    start_idx = ds_manager.get_processed_count(out_path)
    
    if start_idx > 0:
        print(f"Resuming from index: {start_idx}")

    # 3. 推論エンジンの初期化 (YAMLのmodel_configを丸投げ)
    engine = VLMInferenceEngine(cfg['model_config'])
    
    # 4. 推論ループ
    results = []
    error_count = 0
    prompt_tpl = cfg['prompts'].get(exp_cfg['prompt_type'], 'basic')
    
    subset = dataset[start_idx:]
    pbar = tqdm.tqdm(subset, desc=f"Inferring [{exp_cfg['method_name']}]")
    
    for i, data in enumerate(pbar):
        if exp_cfg['debug_mode'] and i >= exp_cfg['debug_limit']:
            print(f"\n[DEBUG] 制限件数({exp_cfg['debug_limit']})到達。終了します。")
            break

        try:
            item_file = data.get(exp_cfg['content_type'])
            full_path = os.path.join(exp_cfg['data_path'], item_file)
            
            # プロンプトの上書きと変換
            convs = [{"from": "human", "value": f"{prompt_tpl}\n<{exp_cfg['content_type']}>"}]
            v_meta = meta_cache.get(item_file)
            
            messages = engine.prepare_messages(convs, full_path, exp_cfg['content_type'], v_meta)
            
            # 推論 (YAMLのgen_paramsを展開して渡す)
            output = engine.infer(messages, video_meta=v_meta, **cfg.get('gen_params', {}))
            results.append([data.get("genre"), data.get("move"), output, full_path])

        except Exception as e:
            error_count += 1
            print(f"\n[Error at {item_file}] {str(e)}")
            if error_count >= exp_cfg['max_errors']:
                print(f"!!! 許容エラー数({exp_cfg['max_errors']})超過。強制終了します。 !!!")
                break
            continue

        # 5. バッチ保存
        if len(results) >= exp_cfg['batch_save_size']:
            save_csv(results, out_path)
            results = []

    # 残りの保存
    if results:
        save_csv(results, out_path)
    print(f"Done. Results saved to {out_path}")

if __name__ == "__main__":
    # 実行時はYAMLパスのみを指定
    main(config_path="sandbox/inf/configs/0_qwen_config.yaml")