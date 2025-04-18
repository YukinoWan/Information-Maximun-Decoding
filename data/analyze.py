

# import h5py

# with h5py.File("/home/zhen/Information-Maximun-Decoding/data/AVQA/AVQA/AVQA_extracted_features/AVQA_audio_PANNs_feat.h5", "r") as f:
#     def show_structure(name, obj):
#         print(f"{name} ({type(obj)})")
#     f.visititems(show_structure)

# with h5py.File("/home/zhen/Information-Maximun-Decoding/data/AVQA/AVQA/AVQA_extracted_features/AVQA_audio_PANNs_feat.h5", "r") as f:
#     ids = f["ids"][:]
#     features = f["vlaudio_features"][:]

# print("📌 ids.shape:", ids.shape)
# print("📌 features.shape:", features.shape)

# print("✅ First 5 IDs:", ids[:5])
# print("✅ First feature shape:", features[0].shape)

import csv
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ========== CONFIG ==========
CSV_PATH = "./AVQA/AVQA/AVQA_dataset/avqa_download_urls.csv"          # ← 你的 CSV 文件路径
BASE_DIR = "AVQA_Audio"                  # 输出根目录
NUM_THREADS = 10                 # 并发线程数
SAMPLE_RATE = 16000                 # 音频采样率
MONO = True                         # 是否转换为单声道
# ============================

# 创建 train/test 文件夹
os.makedirs(os.path.join(BASE_DIR, "train"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "test"), exist_ok=True)

# 读取 CSV 内容为任务列表
tasks = []
with open(CSV_PATH, "r") as f:
    reader = csv.reader(f)
    next(reader)  # 跳过标题
    for row in reader:
        if len(row) < 3:
            continue
        video_id, start_sec, split = row
        start_sec = int(start_sec)
        split = split.strip().lower()
        split = "test" if split.startswith("test") else "train"
        output_path = os.path.join(BASE_DIR, split, f"{video_id}_{start_sec}.wav")
        tasks.append((video_id, start_sec, split, output_path))

# 下载函数（核心逻辑）
def download_audio(video_id, start_sec, split, output_path):
    if os.path.exists(output_path):
        return f"✔ Skipped {output_path} (exists)"

    post_args = f"-ss {start_sec} -ar {SAMPLE_RATE}"
    if MONO:
        post_args += " -ac 1"

    cmd = [
        "yt-dlp", "-f", "bestaudio", "--extract-audio", "--audio-format", "wav",
        "--postprocessor-args", post_args,
        f"https://www.youtube.com/watch?v={video_id}",
        "-o", output_path
    ]

    try:
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if result.returncode != 0:
            print(f"❌ Failed {video_id}")
            return f"❌ Failed {video_id}"
        return f"✅ Downloaded {output_path}"
    except Exception as e:
        print(f"❌ Exception for {video_id}: {e}")
        return f"❌ Exception for {video_id}: {e}"

# 并发执行任务并显示进度条
results = []
with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
    futures = {executor.submit(download_audio, *task): task for task in tasks}
    for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
        results.append(future.result())

# 输出结果
for line in results:
    print(line)

