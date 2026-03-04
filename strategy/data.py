# strategy/data.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd


@dataclass
class PriceDataConfig:
    price_dir: Path = Path("data/prices")
    price_col: str = "adjusted_close"   # your csv has adjusted_close
    out_col: str = "close_adj"


def load_prices(cfg: PriceDataConfig = PriceDataConfig()) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load per-ticker csvs from cfg.price_dir into:
      px: DataFrame(index=date, columns=ticker, values=adjusted close)
      ret_df: daily simple returns
    """
    files = sorted(Path(cfg.price_dir).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No price files found in {Path(cfg.price_dir).resolve()}")

    dfs = []
    for fp in files:
        df = pd.read_csv(fp)
        df["date"] = pd.to_datetime(df["date"])
        if cfg.price_col not in df.columns:
            raise ValueError(f"Missing {cfg.price_col} in {fp.name}")
        df = df.rename(columns={cfg.price_col: cfg.out_col})
        df["tic"] = fp.stem
        dfs.append(df[["date", "tic", cfg.out_col]])

    daily = pd.concat(dfs, ignore_index=True).sort_values(["tic", "date"])
    px = daily.pivot(index="date", columns="tic", values=cfg.out_col).sort_index()
    ret_df = px.pct_change().sort_index()
    return px, ret_df