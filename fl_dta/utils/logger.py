# -*- coding: utf-8 -*-
"""
utils/logger.py
----------------
Logger hỗ trợ:
  - CSV file (luôn bật)
  - WandB (tuỳ chọn)
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


class Logger:
    """
    Ghi log metrics ra CSV và optionally WandB.

    Parameters
    ----------
    results_dir : thư mục lưu kết quả
    run_name    : tên run (dùng làm tên file/wandb run)
    use_wandb   : bật WandB không
    project     : WandB project name
    config      : dict config để log lên WandB
    """

    def __init__(
        self,
        results_dir: str = "./results",
        run_name: Optional[str] = None,
        use_wandb: bool = False,
        project: str = "fl-dta",
        config: Optional[dict] = None,
    ):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        if run_name is None:
            run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = run_name

        # CSV setup
        self.csv_path = self.results_dir / f"{run_name}.csv"
        self._csv_initialized = False

        # Config snapshot
        if config:
            cfg_path = self.results_dir / f"{run_name}_config.json"
            cfg_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

        # WandB
        self.use_wandb = use_wandb
        if use_wandb:
            try:
                import wandb
                wandb.init(project=project, name=run_name, config=config or {})
                self.wandb = wandb
            except ImportError:
                print("WandB không được cài, bỏ qua WandB logging.")
                self.use_wandb = False

    def log_round(self, rnd: int, train_loss: float, metrics: Dict[str, float]):
        """Log một FL round."""
        row = {"round": rnd, "train_loss": train_loss, **metrics}
        self._write_csv(row)
        if self.use_wandb:
            self.wandb.log(row, step=rnd)

    def log_epoch(self, epoch: int, train_loss: float, metrics: Dict[str, float]):
        """Log một centralized training epoch."""
        row = {"epoch": epoch, "train_loss": train_loss, **metrics}
        self._write_csv(row)
        if self.use_wandb:
            self.wandb.log(row, step=epoch)

    def finish(self):
        if self.use_wandb:
            self.wandb.finish()
        print(f"\n✓ Log đã lưu tại: {self.csv_path}")

    def _write_csv(self, row: dict):
        mode = "a" if self._csv_initialized else "w"
        with open(self.csv_path, mode, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not self._csv_initialized:
                writer.writeheader()
                self._csv_initialized = True
            writer.writerow(row)


def make_run_name(args) -> str:
    """Tạo run name từ args để dễ phân biệt các experiment."""
    parts = [
        args.dataset,
        getattr(args, "mode", "fedavg"),
        f"c{args.num_clients}",
        getattr(args, "partition", "iid"),
    ]
    return "_".join(parts) + "_" + datetime.now().strftime("%m%d_%H%M")
