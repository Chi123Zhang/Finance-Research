# strategy/momentum_ivol.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class MomentumIVOLConfig:
    topn: int = 3
    mom_window: int = 63
    vol_window: int = 20
    weight_cap: float = 0.7


def month_start_rebalance_dates(px_index: pd.DatetimeIndex) -> pd.Series:
    """Return Series index=month(period), value=first trading day timestamp."""
    return (
        pd.Series(px_index, index=px_index)
        .groupby(px_index.to_period("M"))
        .min()
        .sort_values()
    )


def build_weights_momentum_ivol(
    px: pd.DataFrame,
    ret_df: pd.DataFrame,
    cfg: MomentumIVOLConfig = MomentumIVOLConfig(),
) -> pd.DataFrame:
    """
    Build rebalance-day weights:
      - signal: 3m momentum
      - pick topn
      - weight by inverse realized vol
      - cap weight_cap then renormalize
    Returns wide weights DataFrame: index=rebalance_date, columns=tickers
    """
    reb_dates = month_start_rebalance_dates(px.index)

    mom = px.pct_change(cfg.mom_window)
    vol = ret_df.rolling(cfg.vol_window).std() * np.sqrt(252)

    all_tics = list(px.columns)
    weights = {}

    for d in reb_dates.values:
        if d not in mom.index or d not in vol.index:
            continue

        mom_d = mom.loc[d].dropna()
        vol_d = vol.loc[d].replace(0, np.nan).dropna()
        common = mom_d.index.intersection(vol_d.index)
        if len(common) < max(3, cfg.topn):
            continue

        mom_rank = mom_d.loc[common].sort_values(ascending=False)
        picked = mom_rank.head(min(cfg.topn, len(mom_rank))).index.tolist()
        if not picked:
            continue

        v = vol_d.loc[picked].dropna()
        picked = v.index.tolist()
        if not picked:
            continue

        w = (1.0 / v)
        w = w / w.sum()
        w = w.clip(upper=cfg.weight_cap)
        w = w / w.sum()

        row = {tic: 0.0 for tic in all_tics}
        for tic, val in w.items():
            row[tic] = float(val)
        weights[pd.Timestamp(d)] = row

    W = pd.DataFrame(weights).T.sort_index()
    W.index.name = "rebalance_date"
    return W
