# strategy/similarity.py
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import pandas as pd

from chunk.index import SimilarityIndex
from llm.explain_similarity import explain_trade_signal


@dataclass
class SimilarityStrategyConfig:
    k: int = 20
    sim_min: float = 0.75
    weight_cap: float = 0.6
    total_gross_exposure: float = 1.0  # kept for backward compatibility
    past_only: bool = True

    # new controls
    same_symbol_only: bool = True
    min_hit_rate: float = 0.55
    min_p50: float = 0.0
    min_month_signal_sum: float = 0.0
    max_month_gross_exposure: float = 1.0


def _pred_from_neighbors(neigh: pd.DataFrame) -> Optional[Dict[str, float]]:
    """
    neigh: output of SimilarityIndex.query() with columns including
    similarity and label_next_month_ret.
    """
    neigh = neigh.dropna(subset=["label_next_month_ret"]).copy()
    if len(neigh) < 5:
        return None

    sims = neigh["similarity"].astype(float).clip(lower=0)
    w = sims / (sims.sum() + 1e-12)
    rets = neigh["label_next_month_ret"].astype(float)
    hit_rate = float((rets > 0).mean())

    return {
        "mu": float((w * rets).sum()),
        "p10": float(rets.quantile(0.10)),
        "p50": float(rets.quantile(0.50)),
        "p90": float(rets.quantile(0.90)),
        "top1_sim": float(sims.max()),
        "confidence": float(sims.mean()),
        "hit_rate": hit_rate,
        "n_used": int(len(neigh)),
    }


def build_monthly_weights_similarity(
    idx: SimilarityIndex,
    px_index: pd.DatetimeIndex,
    cfg: SimilarityStrategyConfig = SimilarityStrategyConfig(),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build month-start weights using similarity-based forecasts.

    Returns:
      W_wide: index=rebalance_date, columns=tickers
      pred_df: per (symbol, month) diagnostics
    """
    reb_dates = (
        pd.Series(px_index, index=px_index)
        .groupby(px_index.to_period("M"))
        .min()
        .sort_values()
    )

    symbols = sorted(idx.monthly_chunks["symbol"].unique())
    months = sorted(idx.monthly_chunks["month"].unique())

    weights_rows = []
    preds_rows = []

    for mon in months:
        period = pd.Period(mon)
        if period not in reb_dates.index:
            continue
        reb_date = reb_dates.loc[period]

        raw = {}

        for sym in symbols:
            neigh = pd.DataFrame()
            evidence = {}
            pred = None
            try:
                neigh, evidence = idx.query(
                    sym,
                    mon,
                    k=cfg.k,
                    past_only=cfg.past_only,
                    same_symbol_only=cfg.same_symbol_only,
                )
            except Exception:
                raw[sym] = 0.0
            else:
                pred = _pred_from_neighbors(neigh)
                if (
                    pred is None
                    or pred["top1_sim"] < cfg.sim_min
                    or pred["hit_rate"] < cfg.min_hit_rate
                    or pred["p50"] <= cfg.min_p50
                ):
                    raw[sym] = 0.0
                else:
                    # robust signal: median upside vs tail downside
                    downside = abs(pred["p10"]) + 1e-6
                    strength = max(pred["p50"], 0.0) * pred["confidence"] / downside
                    raw[sym] = float(max(strength, 0.0))

            signal = explain_trade_signal(sym, mon, neigh, evidence, max_bullets=5)
            rationale_list = signal.get("rationale", [])
            if not isinstance(rationale_list, list):
                rationale_list = [str(rationale_list)]
            try:
                assert isinstance(rationale_list, list)
            except Exception:
                rationale_list = []

            row = {
                "rebalance_date": reb_date,
                "month": mon,
                "symbol": sym,
                "action": signal.get("action", "HOLD"),
                "score": float(signal.get("score", 0.0)),
                "expected_return": float(signal.get("expected_return", float("nan"))),
                "uncertainty": float(signal.get("uncertainty", float("nan"))),
                "risk_mdd": float(signal.get("risk_mdd", float("nan"))),
                "rationale": json.dumps(rationale_list, ensure_ascii=False),
            }
            if pred is not None:
                row.update(pred)
            preds_rows.append(row)

        s = sum(raw.values())
        if s <= 1e-12:
            continue

        # do NOT force full investment
        gross_exposure = min(cfg.max_month_gross_exposure, max(cfg.min_month_signal_sum, s))

        # scale raw weights to chosen gross exposure
        w = {k: (gross_exposure * v / s) for k, v in raw.items()}

        # cap then renormalize to keep same gross exposure
        w = {k: min(v, cfg.weight_cap) for k, v in w.items()}
        s2 = sum(w.values())
        if s2 > 1e-12:
            w = {k: (gross_exposure * v / s2) for k, v in w.items()}
        else:
            w = {k: 0.0 for k in w}

        for sym, ww in w.items():
            weights_rows.append({
                "rebalance_date": reb_date,
                "symbol": sym,
                "weight": float(ww),
            })

    W_long = pd.DataFrame(weights_rows)
    pred_df = pd.DataFrame(preds_rows)

    if W_long.empty:
        return pd.DataFrame(), pred_df

    W_wide = (
        W_long.pivot(index="rebalance_date", columns="symbol", values="weight")
        .fillna(0.0)
        .sort_index()
    )
    return W_wide, pred_df
