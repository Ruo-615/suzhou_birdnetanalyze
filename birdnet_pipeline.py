import csv
import re
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, filtfilt
# ================== 配置区 ==================
BASE_DIR = Path(__file__).parent

# 使用data/ 下的目录
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw-audio"
FILTERED_DIR = DATA_DIR / "filtered_audio"
RESULT_DIR = DATA_DIR / "result"
RESULT_ZH_DIR = DATA_DIR / "result_zh"

SPECIES_MAP_CSV = BASE_DIR / "suzhou_species_zh.csv"

FILTERED_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_ZH_DIR.mkdir(parents=True, exist_ok=True)

# BirdNET经纬度参数（苏州）
LAT = 31.30
LON = 120.62

# week 自动推断开关：True=自动从文件名推断；False=用固定值
AUTO_WEEK = True
WEEK_FALLBACK = 2  # 推断失败时使用

MIN_CONF = 0.3#置信度阈值
SENSITIVITY = 1.8
OVERLAP = 0.5
THREADS = 4

# 高通滤波
HP_CUTOFF = 500  # Hz
FILTER_ORDER = 4

AUDIO_EXTS = {".wav", ".WAV"}
# ============================================


def highpass_filter(data: np.ndarray, sr: int, cutoff: float = 500.0) -> np.ndarray:
    nyq = 0.5 * sr
    b, a = butter(FILTER_ORDER, cutoff / nyq, btype="high")
    return filtfilt(b, a, data).astype(np.float32)


def list_audio_files(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix in AUDIO_EXTS]
    files.sort()
    return files


def extract_datetime_from_filename(name: str) -> datetime | None:
    """
    支持从文件名中提取 12位时间戳 YYYYMMDDHHMM
    示例：YD01_202601091059_21.wav -> 2026-01-09 10:59
    """
    m = re.search(r"(\d{12})", name)
    if not m:
        return None
    ts = m.group(1)
    try:
        return datetime.strptime(ts, "%Y%m%d%H%M")
    except ValueError:
        return None


def infer_week_from_folder(folder: Path, fallback: int) -> int:
    """
    用文件夹里的“第一个音频文件”推断 week（ISO week: 1~53）
    """
    files = list_audio_files(folder)
    if not files:
        return fallback

    dt = extract_datetime_from_filename(files[0].name)
    if dt is None:
        return fallback

    # ISO 周：1~53
    return int(dt.isocalendar().week)


def step1_filter_audio():
    print("Step 1: 高通滤波（500 Hz）")

    wav_files = list_audio_files(RAW_DIR)
    if not wav_files:
        print(f"未找到音频文件：{RAW_DIR}")
        return

    for wav in wav_files:
        data, sr = sf.read(wav)

        # 立体声转单声道
        if isinstance(data, np.ndarray) and data.ndim > 1:
            data = np.mean(data, axis=1)

        filtered = highpass_filter(np.asarray(data, dtype=np.float32), sr, HP_CUTOFF)

        out_path = FILTERED_DIR / wav.name
        sf.write(out_path, filtered, sr)
        print(f"处理完成: {wav.name}")

    print("高通滤波完成\n")

#余睿泽
def step2_run_birdnet():
    print("Step 2: BirdNET 鸟类识别")

    wav_files = list_audio_files(FILTERED_DIR)
    if not wav_files:
        print(f"filtered_audio 中没有 wav 文件：{FILTERED_DIR}")
        return

    week = infer_week_from_folder(FILTERED_DIR, WEEK_FALLBACK) if AUTO_WEEK else WEEK_FALLBACK
    if AUTO_WEEK:
        dt = extract_datetime_from_filename(wav_files[0].name)
        if dt is not None:
            print(f"自动推断 week：{week}（来自文件 {wav_files[0].name} -> {dt.date()}）")
        else:
            print(f"自动推断 week 失败，使用 fallback：{week}")
    else:
        print(f"使用固定 week：{week}")

    cmd = [
        "python",
        "-m",
        "birdnet_analyzer.analyze",
        str(FILTERED_DIR),          # INPUT 是位置参数（文件夹）
        "--output", str(RESULT_DIR),
        "--lat", str(LAT),
        "--lon", str(LON),
        "--week", str(week),
        "--min_conf", str(MIN_CONF),
        "--sensitivity", str(SENSITIVITY),
        "--overlap", str(OVERLAP),
        "--threads", str(THREADS),
        "--rtype", "csv",
    ]

    print("运行命令：")
    print(" ".join(cmd))
    print()

    subprocess.run(cmd, check=True)
    print("✓ BirdNET 识别完成\n")


def load_species_map(csv_path: Path) -> dict[str, str]:
    """
    读取 英文名 → 中文名 映射表
    CSV 格式：
    english_name,chinese_name
    Eurasian Tree Sparrow,树麻雀
    """
    mapping: dict[str, str] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            en = (row.get("english_name") or "").strip().lower()
            zh = (row.get("chinese_name") or "").strip()
            if en:
                mapping[en] = zh
    return mapping


def normalize_common_name(name: str) -> str:
    return (name or "").strip().lower()


def step3_translate_to_zh():
    """
    把 result/*.BirdNET.results.csv 转成 result_zh/
    - 增加 Chinese name 列
    - 并把 Chinese name 放在 File 前面（交换位置）
    """
    print("Step 3: 识别结果转中文")

    if not SPECIES_MAP_CSV.exists():
        print(f"未找到映射表：{SPECIES_MAP_CSV}")
        return

    mapping = load_species_map(SPECIES_MAP_CSV)

    result_files = sorted(RESULT_DIR.glob("*.BirdNET.results.csv"))
    if not result_files:
        print(f"未找到识别结果 CSV：{RESULT_DIR}")
        return

    hit = 0
    miss = 0
    miss_counter = Counter()

    for fp in result_files:
        with fp.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            in_fields = reader.fieldnames or []

        # 输出字段
        if "File" in in_fields:
            idx = in_fields.index("File")
            out_fields = in_fields[:idx] + ["Chinese name"] + in_fields[idx:]
        else:
            out_fields = in_fields + ["Chinese name"]

        for r in rows:
            common = r.get("Common name", "")
            key = normalize_common_name(common)
            zh = mapping.get(key, "")
            r["Chinese name"] = zh

            if zh:
                hit += 1
            else:
                miss += 1
                if common:
                    miss_counter[common] += 1

        out_path = RESULT_ZH_DIR / fp.name
        with out_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=out_fields)
            writer.writeheader()
            writer.writerows(rows)

    print(f"已处理 {len(result_files)} 个结果文件，输出到：{RESULT_ZH_DIR}\n")
    print(f"匹配统计：命中 {hit} 条，未命中 {miss} 条\n")

    if miss_counter:
        print("未命中的 Common name（出现次数最多的前 20 个）：")
        for name, cnt in miss_counter.most_common(20):
            print(f"{cnt:>6}  {name}")
        print("\n提示：这些名字要么不在 suzhou_species_zh.csv，要么写法不一致（空格/大小写/别名）。")

    print()


def main():
    step1_filter_audio()
    step2_run_birdnet()
    step3_translate_to_zh()
    print("全流程完成")


if __name__ == "__main__":
    main()
