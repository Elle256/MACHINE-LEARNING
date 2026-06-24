# -*- coding: utf-8 -*-
"""
analyze_results.py
-------------------
Đọc CSV logs và vẽ các biểu đồ so sánh:
  1. FL (IID) vs FL (Non-IID) vs Centralized  →  MSE / CI / rm² over rounds
  2. Communication rounds vs Test MSE trade-off
  3. Phân phối kinase families per client (Non-IID visualization)

Usage:
    python analyze_results.py --results_dir ./results --output_dir ./figures
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

# ─── Style ───────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="tab10", font_scale=1.15)
COLORS = sns.color_palette("tab10")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="./results")
    parser.add_argument("--output_dir", default="./figures")
    parser.add_argument(
        "--runs", nargs="*",
        help="Tên run cụ thể (bỏ qua để tự detect tất cả CSV trong results_dir)"
    )
    return parser.parse_args()


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_all_runs(results_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load tất cả CSV trong results_dir."""
    csvs = sorted(results_dir.glob("*.csv"))
    runs = {}
    for csv in csvs:
        name = csv.stem
        try:
            df = pd.read_csv(csv)
            runs[name] = df
            print(f"  Loaded: {name}  ({len(df)} rows)")
        except Exception as e:
            print(f"  Skip {csv.name}: {e}")
    return runs


def classify_run(name: str) -> dict:
    """Phân loại run từ tên file."""
    info = {
        "mode": "centralized" if "centralized" in name else "fedavg",
        "dataset": "kiba" if "kiba" in name else "davis",
        "partition": "iid" if "iid" in name and "non_iid" not in name else
                     ("non_iid" if "non_iid" in name else
                      ("dirichlet" if "dirichlet" in name else "unknown")),
        "x_col": "epoch" if "centralized" in name else "round",
    }
    return info


def _get_label(name: str) -> str:
    info = classify_run(name)
    if info["mode"] == "centralized":
        return "Centralized"
    partition_map = {"iid": "FedAvg (IID)", "non_iid": "FedAvg (Non-IID)", "dirichlet": "FedAvg (Dirichlet)"}
    return partition_map.get(info["partition"], name)


# ─── Plot helpers ─────────────────────────────────────────────────────────────

def smooth(series: pd.Series, window: int = 5) -> pd.Series:
    return series.rolling(window, min_periods=1, center=True).mean()


def plot_metric_comparison(
    runs: Dict[str, pd.DataFrame],
    metric: str,
    output_path: Path,
    smooth_window: int = 5,
    title: Optional[str] = None,
):
    """Vẽ 1 metric cho tất cả runs trên cùng 1 plot."""
    fig, ax = plt.subplots(figsize=(9, 5))

    for i, (name, df) in enumerate(runs.items()):
        info = classify_run(name)
        x_col = info["x_col"]

        if x_col not in df.columns or metric not in df.columns:
            continue

        x = df[x_col]
        y = smooth(df[metric], smooth_window)
        label = _get_label(name)

        ls = "--" if info["mode"] == "centralized" else "-"
        ax.plot(x, y, ls, color=COLORS[i % len(COLORS)], label=label, linewidth=2)

        # Shade raw behind smooth
        ax.fill_between(
            x, df[metric].rolling(smooth_window, min_periods=1, center=True).min(),
            df[metric].rolling(smooth_window, min_periods=1, center=True).max(),
            alpha=0.08, color=COLORS[i % len(COLORS)]
        )

    metric_labels = {"test_mse": "Test MSE ↓", "test_ci": "CI ↑", "test_rm2": "rm² ↑"}
    ax.set_xlabel("Round / Epoch", fontsize=12)
    ax.set_ylabel(metric_labels.get(metric, metric), fontsize=12)
    ax.set_title(title or f"{metric} over training", fontsize=13, fontweight="bold")
    ax.legend(frameon=True, fontsize=10)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=10))
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def plot_rounds_vs_mse(
    runs: Dict[str, pd.DataFrame],
    output_path: Path,
    target_mse_thresholds: List[float] = None,
):
    """
    Plot: số rounds cần thiết để đạt các ngưỡng MSE khác nhau.
    Trả lời câu hỏi: communication rounds vs accuracy trade-off.
    """
    if target_mse_thresholds is None:
        # Lấy tự động từ dữ liệu
        all_mse = []
        for df in runs.values():
            if "test_mse" in df.columns:
                all_mse.extend(df["test_mse"].dropna().tolist())
        if not all_mse:
            return
        min_mse = min(all_mse)
        target_mse_thresholds = [
            min_mse * 1.05,
            min_mse * 1.10,
            min_mse * 1.20,
            min_mse * 1.50,
        ]

    fig, ax = plt.subplots(figsize=(9, 5))

    bar_data = {}
    for name, df in runs.items():
        info = classify_run(name)
        if "round" not in df.columns or "test_mse" not in df.columns:
            continue
        label = _get_label(name)
        rounds_needed = []
        for thresh in target_mse_thresholds:
            reached = df[df["test_mse"] <= thresh]
            rounds_needed.append(reached["round"].min() if len(reached) > 0 else np.nan)
        bar_data[label] = rounds_needed

    x = np.arange(len(target_mse_thresholds))
    width = 0.8 / max(len(bar_data), 1)

    for i, (label, rounds) in enumerate(bar_data.items()):
        offset = (i - len(bar_data) / 2 + 0.5) * width
        bars = ax.bar(x + offset, rounds, width * 0.9,
                      label=label, color=COLORS[i % len(COLORS)], alpha=0.85)
        for bar, val in zip(bars, rounds):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.5,
                        f"{int(val)}", ha="center", va="bottom", fontsize=8)

    thresh_labels = [f"MSE ≤ {t:.3f}" for t in target_mse_thresholds]
    ax.set_xticks(x)
    ax.set_xticklabels(thresh_labels, fontsize=9)
    ax.set_ylabel("Rounds to reach threshold", fontsize=12)
    ax.set_title("Communication Rounds vs Accuracy Trade-off", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def plot_final_metrics_bar(
    runs: Dict[str, pd.DataFrame],
    output_path: Path,
):
    """Bar chart so sánh final metrics (last N rounds/epochs average)."""
    metrics = ["test_mse", "test_ci", "test_rm2"]
    metric_labels = ["MSE ↓", "CI ↑", "rm² ↑"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))

    for ax, metric, mlabel in zip(axes, metrics, metric_labels):
        labels, values = [], []
        for name, df in runs.items():
            if metric not in df.columns:
                continue
            last_n = df[metric].tail(10).mean()
            labels.append(_get_label(name))
            values.append(last_n)

        colors = [COLORS[i % len(COLORS)] for i in range(len(labels))]
        bars = ax.bar(labels, values, color=colors, alpha=0.85, edgecolor="white")

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(values) * 0.01,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax.set_title(mlabel, fontsize=12, fontweight="bold")
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))

    plt.suptitle("Final Performance Comparison (avg last 10)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def plot_train_loss(
    runs: Dict[str, pd.DataFrame],
    output_path: Path,
):
    """Train loss convergence cho tất cả runs."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (name, df) in enumerate(runs.items()):
        info = classify_run(name)
        x_col = info["x_col"]
        if x_col not in df.columns or "train_loss" not in df.columns:
            continue
        y = smooth(df["train_loss"], window=5)
        ax.plot(df[x_col], y, label=_get_label(name),
                color=COLORS[i % len(COLORS)], linewidth=2,
                linestyle="--" if info["mode"] == "centralized" else "-")

    ax.set_xlabel("Round / Epoch")
    ax.set_ylabel("Train Loss (MSE)")
    ax.set_title("Training Loss Convergence", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[Analyzer] Loading runs from: {results_dir}\n")
    all_runs = load_all_runs(results_dir)

    if not all_runs:
        print("Không tìm thấy CSV nào. Chạy experiments trước.")
        return

    # Filter theo args.runs nếu được chỉ định
    if args.runs:
        all_runs = {k: v for k, v in all_runs.items() if any(r in k for r in args.runs)}

    print(f"\nVẽ biểu đồ cho {len(all_runs)} runs...\n")

    # 1. MSE over rounds/epochs
    plot_metric_comparison(
        all_runs, "test_mse",
        output_dir / "mse_comparison.png",
        title="Test MSE: FL (IID) vs FL (Non-IID) vs Centralized",
    )

    # 2. CI over rounds/epochs
    plot_metric_comparison(
        all_runs, "test_ci",
        output_dir / "ci_comparison.png",
        title="Concordance Index: FL vs Centralized",
    )

    # 3. rm² over rounds/epochs
    plot_metric_comparison(
        all_runs, "test_rm2",
        output_dir / "rm2_comparison.png",
        title="rm²: FL vs Centralized",
    )

    # 4. Train loss convergence
    plot_train_loss(all_runs, output_dir / "train_loss.png")

    # 5. Rounds vs accuracy trade-off (chỉ FL runs)
    fl_runs = {k: v for k, v in all_runs.items() if "centralized" not in k}
    if fl_runs:
        plot_rounds_vs_mse(fl_runs, output_dir / "rounds_vs_mse.png")

    # 6. Final metrics bar chart
    plot_final_metrics_bar(all_runs, output_dir / "final_metrics_bar.png")

    print(f"\n✓ Tất cả biểu đồ đã lưu tại: {output_dir}\n")


if __name__ == "__main__":
    main()
