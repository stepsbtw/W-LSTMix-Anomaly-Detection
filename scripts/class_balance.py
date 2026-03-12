import argparse
import os
from dataclasses import dataclass

import pandas as pd


@dataclass
class BalanceStats:
    files: int = 0
    rows: int = 0
    normal: int = 0
    anomaly: int = 0

    def add(self, other_stats: "BalanceStats") -> None:
        self.files += other_stats.files
        self.rows += other_stats.rows
        self.normal += other_stats.normal
        self.anomaly += other_stats.anomaly

    @property
    def valid(self) -> int:
        return self.normal + self.anomaly

    @property
    def anomaly_pct(self) -> float:
        if self.valid == 0:
            return 0.0
        return 100.0 * self.anomaly / self.valid


def read_table(path: str) -> pd.DataFrame:
    if path.endswith(".csv"):
        return pd.read_csv(path)
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {path}")


def count_file(path: str, label_col: str) -> BalanceStats:
    df = read_table(path)
    stats = BalanceStats(files=1, rows=len(df))

    if label_col not in df.columns:
        stats.missing = len(df)
        return stats

    labels = pd.to_numeric(df[label_col], errors="coerce")
    missing_mask = labels.isna()
    stats.missing = int(missing_mask.sum())

    valid = labels[~missing_mask]
    stats.normal = int((valid == 0).sum())
    stats.anomaly = int((valid == 1).sum())
    return stats


def print_summary(title: str, stats: BalanceStats) -> None:
    print(title)
    print(f"  Files             : {stats.files}")
    print(f"  Total             : {stats.rows}")
    print(f"  Normal (0)  : {stats.normal}")
    print(f"  Anomaly (1) : {stats.anomaly}")
    print(f"  Anomaly rate      : {stats.anomaly_pct:.2f}%")


def class_balance(dataset_path: str, label_col: str = "label") -> None:
    overall = BalanceStats()

    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(f"Dataset folder not found: {dataset_path}")

    region_names = [
        name for name in sorted(os.listdir(dataset_path))
        if os.path.isdir(os.path.join(dataset_path, name))
    ]

    if not region_names:
        print(f"No region folders found in: {dataset_path}")
        return

    print(f"Scanning: {dataset_path}")
    print(f"Label column: {label_col}\n")

    for region in region_names:
        region_dir = os.path.join(dataset_path, region)
        region_stats = BalanceStats()

        for fname in sorted(os.listdir(region_dir)):
            fpath = os.path.join(region_dir, fname)
            if not os.path.isfile(fpath):
                continue
            if not (fname.endswith(".csv") or fname.endswith(".parquet")):
                continue

            fstats = count_file(fpath, label_col)
            region_stats.add(fstats)

        overall.add(region_stats)
        print_summary(f"[{region}]", region_stats)
        print()

    print_summary("[OVERALL]", overall)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="dataset_labeled",
        help="Path to labeled dataset root (contains region subfolders).",
    )
    args = parser.parse_args()

    class_balance(args.input, args.label_col)