from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class EqualWeightConfig:
    rebalance_freq: str = "MS"  # month-start
    min_valid_price: int = 1
    allow_partial_universe: bool = True


def _month_start_rebalance_dates(px_index: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    month_starts = pd.Series(index=px_index, data=1.0).resample(freq).first().index
    positions = px_index.searchsorted(month_starts, side="left")

    rebalance_dates = []
    for pos in positions:
        if pos >= len(px_index):
            continue
        d = px_index[pos]
        if rebalance_dates and d == rebalance_dates[-1]:
            continue
        rebalance_dates.append(d)

    return pd.DatetimeIndex(rebalance_dates)


def build_weights_equal_weight(
    px: pd.DataFrame,
    cfg: EqualWeightConfig = EqualWeightConfig(),
) -> pd.DataFrame:
    if px.empty:
        return pd.DataFrame(columns=px.columns, dtype=float)

    px = px.sort_index()
    rebalance_dates = _month_start_rebalance_dates(px.index, cfg.rebalance_freq)
    all_tickers = list(px.columns)
    rows = {}

    for d in rebalance_dates:
        prices_d = px.loc[d]
        tradable_mask = prices_d.notna()
        if not cfg.allow_partial_universe and not bool(tradable_mask.all()):
            continue

        tradable = tradable_mask[tradable_mask].index
        if len(tradable) < cfg.min_valid_price or len(tradable) == 0:
            continue

        w = pd.Series(0.0, index=all_tickers, dtype=float)
        w.loc[tradable] = 1.0 / float(len(tradable))
        total = float(w.sum())
        if total > 0:
            w = w / total
        if len(tradable) > 0 and not np.isclose(w.sum(), 1.0):
            w = w / (w.sum() + 1e-12)

        rows[pd.Timestamp(d)] = w.to_dict()

    W = pd.DataFrame.from_dict(rows, orient="index")
    if W.empty:
        return pd.DataFrame(columns=all_tickers, dtype=float)

    W = W.reindex(columns=all_tickers).sort_index()
    W.index.name = "rebalance_date"
    return W.astype(float)
