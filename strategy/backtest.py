# strategy/backtest.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Dict

import numpy as np
import pandas as pd


@dataclass
class BacktestConfig:
    fee_rate: float = 0.0005   # per 1.0 turnover (rough)
    lag_weights: int = 1       # critical: avoid look-ahead
    annualization: int = 252


def backtest_weights_returns(
    portfolio_weights_df: pd.DataFrame,
    ret_df: pd.DataFrame,
    cfg: BacktestConfig = BacktestConfig(),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    portfolio_weights_df: index=date (rebalance days), columns=tickers, values=weights
    ret_df: index=date (daily), columns=tickers, values=daily simple returns
    Returns:
      result: DataFrame with gross_ret, net_ret, turnover, nav
      W_exec: executed daily weights after ffill + lag
    """
    trade_index = ret_df.index

    W_daily = portfolio_weights_df.reindex(trade_index).ffill().fillna(0.0)
    W_exec = W_daily.shift(cfg.lag_weights).fillna(0.0)

    port_ret_gross = (W_exec * ret_df.fillna(0.0)).sum(axis=1)

    turnover = W_exec.diff().abs().sum(axis=1).fillna(0.0)
    port_ret_net = port_ret_gross - cfg.fee_rate * turnover

    nav = (1 + port_ret_net).cumprod()
    nav = nav / nav.iloc[0]

    result = pd.DataFrame({
        "gross_ret": port_ret_gross,
        "net_ret": port_ret_net,
        "turnover": turnover,
        "nav": nav,
    })
    return result, W_exec


def perf_from_returns(daily_ret: pd.Series, annualization: int = 252) -> Dict[str, float]:
    r = daily_ret.fillna(0.0)
    nav = (1 + r).cumprod()
    nav = nav / nav.iloc[0]

    ann_ret = nav.iloc[-1] ** (annualization / len(nav)) - 1
    ann_vol = r.std(ddof=0) * np.sqrt(annualization)
    sharpe = ann_ret / (ann_vol + 1e-12)

    dd = nav / nav.cummax() - 1
    return {
        "Annualized Return": float(ann_ret),
        "Annualized Vol": float(ann_vol),
        "Sharpe (rf=0)": float(sharpe),
        "Max Drawdown": float(dd.min()),
    }


@dataclass
class VolTargetConfig:
    target_vol: float = 0.20
    lookback: int = 20
    max_leverage: float = 1.5
    annualization: int = 252


def vol_target_returns(daily_ret: pd.Series, cfg: VolTargetConfig = VolTargetConfig()) -> pd.Series:
    r = daily_ret.fillna(0.0)
    roll_vol = r.rolling(cfg.lookback).std(ddof=0) * np.sqrt(cfg.annualization)
    scale = (cfg.target_vol / (roll_vol + 1e-12)).clip(upper=cfg.max_leverage)
    scale = scale.fillna(0.0)
    return r * scale
