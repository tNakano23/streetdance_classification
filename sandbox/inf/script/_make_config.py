import datetime
import json
import os

import streamlit as st
import yaml

# ==========================================
# 初期設定 & ファイル読み込み
# ==========================================

label_choices = """
[genre]:[move1,move2,move3,move4,move5,move6,move7,move8,move9,move10]
Break : indian step,twist,two step,roll two step,up rock,battle rock,6 step,3 step,cc,back cc
Pop : fresno,walk out,loft,hand wave,body wave,neck-o-flex,flex,walk,old man,roll
Lock : lock,twirl,point,crap,pacing,stop and go,rock steady,skeeter rabbit,scooby doo,cross hand
Middle Hip-hop : roger rabbit,rock the board,charleston,club,brooklyn,slide,box,broolklyn bounce,running man,popcorn
LA style Hip-hop : slide,paddbre,ball change,shamrock,babysitter,break down,roger rabbit,club,box,walk
House : loose legs,paddbre,side kick,swirl,train,farmer,chase,shuffle,heel step,back skip
Waack : twirl,open,pose,walk,shake,four corner,swap,punk,grab,turn
Krump : stomp,chest pops,arm swing,jabs,bang outs,kick back,buck hop,whip,Tick,focus
Street Jazz : positions des bras,positions des pieds,plie,jump,tendu,chaines,passe balance,paddbre,chasse,contraction
Ballet Jazz : chaines,pique,Fouette,paddbre,pas de chat,grand pas de chat,grand jete,passe,entrelace,airplane
"""


st.set_page_config(page_title="VLM Config 券売機", layout="wide")

PRESETS_PATH = "sandbox/inf/configs/presets.json"
try:
    with open(PRESETS_PATH, "r", encoding="utf-8") as f:
        PRESETS = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    PRESETS = {
        "data": {"default.json": ["default_path"]},
        "lora_model": {"image": [], "video": []},
    }

PROMPTS = {
    "basic":"# Role\nあなたは世界レベルのストリートダンス・コンテンポラリーダンスの専門審査員です。\n# Task\nこの画像のダンスのジャンルと、動きの名前を教えてください。",
    "choice_given":f"# Role\nあなたは世界レベルのストリートダンス・コンテンポラリーダンスの専門審査員です。\n# Task\nこの画像のダンスのジャンルと、動きの名前を教えてください。\nなお、以下からひとつを選択してください。{label_choices}",
}

MODELS = [
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen2.5-VL-32B-Instruct",
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "Qwen/Qwen2.5-VL-3B-Instruct",
]

st.title("🎫 VLM 推論コンフィグ 券売機")
st.markdown("推論スクリプトに渡す `0_qwen_config.yaml` を生成します。")


# ==========================================
# ファイル操作関数
# ==========================================
def backup_and_save_yaml(config_dict):
    target_dir = "sandbox/inf/configs"
    os.makedirs(target_dir, exist_ok=True)
    target_file = os.path.join(target_dir, "0_qwen_config.yaml")

    if os.path.exists(target_file):
        with open(target_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        first_line = lines[0].strip()
        if first_line.startswith("# "):
            raw_time = first_line.replace("# ", "")
            safe_time = raw_time.replace(":", "").replace("-", "")
            timestamp = (
                f"{safe_time[:8]}-{safe_time[8:]}"
                if len(safe_time) == 12
                else safe_time
            )
        else:
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")

        try:
            old_yaml = yaml.safe_load("".join(lines[1:]))
            method_name = old_yaml.get("experiment", {}).get("method_name", "unknown")
        except:
            method_name = "unknown"

        backup_name = f"{timestamp}_{method_name}_qwen_config.yaml"
        backup_path = os.path.join(target_dir, backup_name)
        os.rename(target_file, backup_path)
        st.toast(f"既存の設定ファイルをバックアップしました: {backup_name}")

    current_time_str = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M")
    with open(target_file, "w", encoding="utf-8") as f:
        f.write(f"# {current_time_str}\n")
        yaml.dump(
            config_dict,
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )

    return target_file


# ==========================================
# GUI レイアウト
# ==========================================
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📁 1. データ & 実行設定")
    method_name = st.text_input("Method Name", value="dance_expert")
    content_type = st.radio("Content Type", options=["video", "image"], horizontal=True)

    json_options = list(PRESETS["data"].keys())
    json_path = st.selectbox("JSON Path", options=json_options)
    data_path_options = PRESETS["data"].get(json_path, [])
    data_path = st.selectbox("Data Path", options=data_path_options)
    output_dir = st.text_input("Output Dir", value=PRESETS.get("OUTPUT_DIR", "data_result"))

    st.divider()
    batch_save_size = st.number_input(
        "Batch Save Size (保存間隔)", min_value=1, value=100, step=10
    )
    max_errors = st.number_input(
        "Max Errors (許容エラー数)", min_value=1, value=100, step=10
    )

with col2:
    st.subheader("🤖 2. モデル & デバッグ設定")
    model_name = st.selectbox("Model Name", options=MODELS)

    load_lora = st.toggle("Load LoRA (LoRAを使用する)", value=False)
    lora_options = PRESETS["lora_model"].get(content_type, ["設定なし"])
    lora_path = st.selectbox("LoRA Path", options=lora_options, disabled=not load_lora)

    st.divider()
    debug_mode = st.toggle("Debug Mode", value=True)
    debug_limit = st.number_input(
        "Debug Limit (推論件数上限)", min_value=1, value=10, disabled=not debug_mode
    )

    st.markdown("**Seed Value**")
    seed_col1, seed_col2 = st.columns([1, 1])
    with seed_col1:
        seed_choice = st.radio(
            "Seed設定",
            options=["42", "Custom"],
            horizontal=True,
            label_visibility="collapsed",
        )
    with seed_col2:
        if seed_choice == "Custom":
            seed_val = st.number_input(
                "カスタムSeed値", value=0, step=1, label_visibility="collapsed"
            )
        else:
            seed_val = 42

with col3:
    st.subheader("💬 3. 推論パラメーター & プロンプト")

    max_new_tokens = st.selectbox("Max New Tokens", options=[2048, 1024, 512, 256, 128])
    do_sample = st.checkbox("Do Sample", value=False)

    st.divider()
    prompt_type = st.selectbox(
        "Prompt Type (プロンプト種別)", options=list(PROMPTS.keys())
    )
    if content_type == "video":
        true_prompt = PROMPTS[prompt_type].replace("この画像の", "この動画の")
    else:
        true_prompt = PROMPTS[prompt_type]
    st.markdown(
        f"""
        <div style='background-color: #f0f2f6; padding: 10px; border-radius: 5px; font-size: 0.85em; white-space: pre-wrap; color: #31333F;'>
            {true_prompt}
        </div>
    """,
        unsafe_allow_html=True,
    )

# ==========================================
# YAML生成アクション
# ==========================================
st.markdown("---")
if st.button("✨ YAML作成 (configs/0_qwen_config.yaml)", use_container_width=True):
    config_dict = {
        "experiment": {
            "method_name": method_name,
            "json_path": json_path,
            "data_path": data_path,
            "output_dir": output_dir,
            "content_type": content_type,
            "prompt_type": prompt_type,
            "batch_save_size": batch_save_size,
            "max_errors": max_errors,
            "debug_mode": debug_mode,
            "debug_limit": debug_limit if debug_mode else None,
            "seed": seed_val,
        },
        "model_config": {
            "model_name": model_name,
            "load_lora": load_lora,
            "lora_path": lora_path if load_lora else None,
        },
        "gen_params": {"max_new_tokens": max_new_tokens, "do_sample": do_sample},
        "prompts": {prompt_type: true_prompt},
    }

    try:
        saved_path = backup_and_save_yaml(config_dict)
        st.success(f"✅ 設定ファイルを保存しました！\n\n保存先: `{saved_path}`")
    except Exception as e:
        st.error(f"エラーが発生しました: {e}")
