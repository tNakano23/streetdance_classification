import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from _modules import VideoProcessor, ImageComposer

"""
1_cli_clipper.py と 2_cli_combiner.py の機能を統合したGUIアプリ
"""

class IntegratedApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Video Frame & Image Combiner Tool")
        self.geometry("700x500")
        
        self.composer = ImageComposer()
        self.selected_images = []
        
        # タブコントロールの作成
        notebook = ttk.Notebook(self)
        notebook.pack(fill='both', expand=True)
        
        tab1 = ttk.Frame(notebook)
        tab2 = ttk.Frame(notebook)
        
        notebook.add(tab1, text="1. 動画フレーム切り出し")
        notebook.add(tab2, text="2. 画像結合・アノテーション")
        
        self.setup_tab1(tab1)
        self.setup_tab2(tab2)

    # --- タブ1: 切り出し設定 ---
    def setup_tab1(self, master):
        lbl = ttk.Label(master, text="動画からビート、または等間隔でフレームを抽出します。", font=("Meiriyo", 10, "bold"))
        lbl.pack(pady=10)
        
        # モード選択
        frm_mode = ttk.LabelFrame(master, text="抽出モード")
        frm_mode.pack(fill='x', padx=20, pady=10)
        self.clip_mode = tk.StringVar(value="beat")
        ttk.Radiobutton(frm_mode, text="音楽のビートに同期", variable=self.clip_mode, value="beat").pack(side='left', padx=20, pady=5)
        ttk.Radiobutton(frm_mode, text="等間隔（秒指定）", variable=self.clip_mode, value="interval").pack(side='left', padx=20, pady=5)
        
        self.ent_interval = ttk.Entry(frm_mode, width=5)
        self.ent_interval.insert(0, "0.5")
        self.ent_interval.pack(side='left', padx=5)
        ttk.Label(frm_mode, text="秒ごと").pack(side='left')

        # 実行ボタン
        btn_run = ttk.Button(master, text="動画ファイルを選択して実行 🏃", command=self.process_video)
        btn_run.pack(pady=30, ipadx=20, ipady=10)

    def process_video(self):
        v_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4")])
        if not v_path: return
        
        out_root = filedialog.askdirectory(title="フレーム保存先フォルダを選択")
        if not out_root: return
        
        try:
            if self.clip_mode.get() == "beat":
                seconds = VideoProcessor.extract_beat_times(v_path)
                im_mode = False
            else:
                seconds = [0.0, 9999.0, float(self.ent_interval.get())]
                im_mode = True
                
            res_dir = VideoProcessor.save_frames(v_path, seconds, out_root, interval_mode=im_mode)
            messagebox.showinfo("完了", f"フレームの抽出が完了しました！\n保存先: {res_dir}")
        except Exception as e:
            messagebox.showerror("エラー", str(e))

    # --- タブ2: 結合設定 ---
    def setup_tab2(self, master):
        # ファイル選択
        frm_files = ttk.LabelFrame(master, text="画像の選択")
        frm_files.pack(fill='both', expand=True, padx=20, pady=10)
        
        btn_add = ttk.Button(frm_files, text="画像を追加...", command=self.add_images)
        btn_add.grid(row=0, column=0, padx=5, pady=5)
        btn_clear = ttk.Button(frm_files, text="クリア", command=self.clear_images)
        btn_clear.grid(row=0, column=1, padx=5, pady=5)
        
        self.listbox = tk.Listbox(frm_files, height=6)
        self.listbox.grid(row=1, column=0, columnspan=3, sticky='nsew', padx=5, pady=5)
        frm_files.columnconfigure(2, weight=1)
        frm_files.rowconfigure(1, weight=1)
        
        # 配置・アノテーション設定
        frm_opt = ttk.LabelFrame(master, text="配置・ラベル設定")
        frm_opt.pack(fill='x', padx=20, pady=5)
        
        ttk.Label(frm_opt, text="レイアウト:").grid(row=0, column=0, padx=5, pady=5)
        self.var_layout = tk.StringVar(value="格子状")
        ttk.Combobox(frm_opt, values=["格子状", "縦一列", "横一列"], textvariable=self.var_layout, state="readonly", width=10).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(frm_opt, text="列数(格子用):").grid(row=0, column=2, padx=5, pady=5)
        self.var_cols = tk.StringVar(value="3")
        ttk.Entry(frm_opt, textvariable=self.var_cols, width=4).grid(row=0, column=3, padx=5, pady=5)
        
        self.var_label = tk.BooleanVar(value=True)
        ttk.Checkbutton(frm_opt, text="順序ラベルを付与", variable=self.var_label).grid(row=0, column=4, padx=10, pady=5)

        btn_save = ttk.Button(master, text="結合して保存する 💾", command=self.save_combined)
        btn_save.pack(pady=10, ipady=5)

    def add_images(self):
        paths = filedialog.askopenfilenames(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        if not paths: return
        self.selected_images.extend(paths)
        for p in paths:
            self.listbox.insert('end', os.path.basename(p))

    def clear_images(self):
        self.selected_images.clear()
        self.listbox.delete(0, 'end')

    def save_combined(self):
        if not self.selected_images:
            messagebox.showwarning("警告", "画像が選択されていません。")
            return
            
        save_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")])
        if not save_path: return
        
        try:
            cols = int(self.var_cols.get()) if self.var_cols.get().isdigit() else None
            out = self.composer.compose(
                img_paths=self.selected_images,
                layout=self.var_layout.get(),
                enable_label=self.var_label.get(),
                corner="左上",
                cols=cols,
                bg_color=(255, 255, 255)
            )
            out.save(save_path)
            messagebox.showinfo("成功", f"結合画像を保存しました:\n{save_path}")
        except Exception as e:
            messagebox.showerror("エラー", str(e))

if __name__ == "__main__":
    App = IntegratedApp()
    App.mainloop()