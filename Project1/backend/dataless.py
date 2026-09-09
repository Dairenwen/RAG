import csv
import random
from pathlib import Path
from Src.config import data_path, dataless_path

# 用于精简测试集的函数，从原始数据集中随机采样部分数据行，并将其保存到目标目录中。

def sample_csv_files():
    source_dir = Path(data_path)
    target_dir = Path(dataless_path)
    target_dir.mkdir(parents=True, exist_ok=True)

    for csv_file in source_dir.glob("*.csv"):
        with open(csv_file, "r", encoding="utf-8-sig", newline="") as src:
            rows = list(csv.reader(src))

        if rows:
            sample_size = max(1, len(rows) // 1000)
            sampled_rows = random.sample(rows, min(sample_size, len(rows)))
        else:
            sampled_rows = []

        with open(target_dir / csv_file.name, "w", encoding="utf-8", newline="") as dst:
            print(f"已经采样 {len(sampled_rows)} 行数据到 {target_dir / csv_file.name}")
            csv.writer(dst).writerows(sampled_rows)


if __name__ == "__main__":
    sample_csv_files()
