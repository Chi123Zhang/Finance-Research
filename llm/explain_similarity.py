from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd


def _as_float(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return float("nan")
    if np.isnan(v):
        return float("nan")
    return v


def _fmt(x: Any, ndigits: int = 4) -> str:
    v = _as_float(x)
    if np.isnan(v):
        return "NaN"
    return f"{v:.{ndigits}f}"


def _weighted_mean(series: pd.Series, weights: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce").clip(lower=0.0)
    mask = s.notna() & w.notna()
    if not mask.any():
        return float("nan")
    s = s[mask]
    w = w[mask]
    wsum = float(w.sum())
    if wsum <= 0:
        return float("nan")
    return float((s * (w / wsum)).sum())


def explain_trade_signal(symbol, month, res, evidence, *, max_bullets=5) -> dict:
    """
    Build a deterministic explanation only from `res` + `evidence`.
    """
    if not isinstance(res, pd.DataFrame):
        res = pd.DataFrame()
    if not isinstance(evidence, dict):
        evidence = {}

    max_bullets = int(max(3, max_bullets))
    ret_th = 1e-6

    pred = evidence.get("pred", {}) if isinstance(evidence.get("pred", {}), dict) else {}
    p10 = _as_float(pred.get("p10"))
    p50 = _as_float(pred.get("p50"))
    p90 = _as_float(pred.get("p90"))
    mean = _as_float(pred.get("mean"))

    expected_return = p50
    if np.isnan(expected_return):
        expected_return = mean

    if np.isnan(expected_return):
        action = "HOLD"
    elif expected_return > ret_th:
        action = "BUY"
    elif expected_return < -ret_th:
        action = "SELL"
    else:
        action = "HOLD"

    if np.isnan(p10) or np.isnan(p90):
        uncertainty = float("nan")
    else:
        uncertainty = float(p90 - p10)

    if np.isnan(expected_return):
        score = 0.0
    elif np.isnan(uncertainty) or abs(uncertainty) <= 1e-12:
        score = float(expected_return)
    else:
        score = float(expected_return / (uncertainty + 1e-12))

    if {"label_next_month_mdd", "similarity"}.issubset(set(res.columns)):
        risk_mdd = _weighted_mean(res["label_next_month_mdd"], res["similarity"])
    else:
        risk_mdd = float("nan")

    bullets: List[str] = []

    if not (np.isnan(p10) and np.isnan(p50) and np.isnan(p90)):
        bullets.append(
            "Pred quantiles from neighbors: "
            f"p10={_fmt(p10)}, p50={_fmt(p50)}, p90={_fmt(p90)}; "
            f"expected_return={_fmt(expected_return)}."
        )
    elif not np.isnan(mean):
        bullets.append(f"Pred mean from neighbors: mean={_fmt(mean)}; expected_return fallback={_fmt(expected_return)}.")

    top1_row = None
    if not res.empty:
        if "rank" in res.columns:
            try:
                top1_row = res.sort_values("rank").iloc[0]
            except Exception:
                top1_row = res.iloc[0]
        else:
            top1_row = res.iloc[0]

    top1 = evidence.get("top1", {}) if isinstance(evidence.get("top1", {}), dict) else {}
    if top1_row is not None or top1:
        top1_sym = top1.get("symbol", None)
        top1_mon = top1.get("month", None)
        top1_sim = _as_float(top1.get("similarity"))
        top1_ret = float("nan")
        top1_mdd = float("nan")
        if top1_row is not None:
            if top1_sym is None and "symbol" in top1_row.index:
                top1_sym = top1_row.get("symbol")
            if top1_mon is None and "month" in top1_row.index:
                top1_mon = top1_row.get("month")
            if np.isnan(top1_sim) and "similarity" in top1_row.index:
                top1_sim = _as_float(top1_row.get("similarity"))
            if "label_next_month_ret" in top1_row.index:
                top1_ret = _as_float(top1_row.get("label_next_month_ret"))
            if "label_next_month_mdd" in top1_row.index:
                top1_mdd = _as_float(top1_row.get("label_next_month_mdd"))

        bullets.append(
            "Top1 neighbor: "
            f"symbol={top1_sym}, month={top1_mon}, similarity={_fmt(top1_sim)}, "
            f"next_ret={_fmt(top1_ret)}, next_mdd={_fmt(top1_mdd)}."
        )

    curve = evidence.get("curve", {}) if isinstance(evidence.get("curve", {}), dict) else {}
    metrics = curve.get("metrics", {}) if isinstance(curve.get("metrics", {}), dict) else {}
    if metrics:
        corr = _as_float(metrics.get("curve_corr"))
        mse = _as_float(metrics.get("curve_mse"))
        mdd_q = _as_float(metrics.get("mdd_query"))
        mdd_t = _as_float(metrics.get("mdd_top1"))
        pieces = []
        if not np.isnan(corr):
            pieces.append(f"corr={_fmt(corr)}")
        if not np.isnan(mse):
            pieces.append(f"mse={_fmt(mse)}")
        if not np.isnan(mdd_q):
            pieces.append(f"mdd_query={_fmt(mdd_q)}")
        if not np.isnan(mdd_t):
            pieces.append(f"mdd_top1={_fmt(mdd_t)}")
        if pieces:
            bullets.append("Curve similarity metrics: " + ", ".join(pieces) + ".")

    if {"label_next_month_ret", "similarity"}.issubset(set(res.columns)):
        ret_wmean = _weighted_mean(res["label_next_month_ret"], res["similarity"])
        hit_rate = float(
            (pd.to_numeric(res["label_next_month_ret"], errors="coerce") > 0.0).mean()
        ) if len(res) > 0 else float("nan")
        sim_avg = _as_float(pd.to_numeric(res["similarity"], errors="coerce").mean())
        bullets.append(
            "Neighbor distribution: "
            f"n={int(len(res))}, sim_avg={_fmt(sim_avg)}, hit_rate={_fmt(hit_rate)}, "
            f"sim_weighted_ret={_fmt(ret_wmean)}, sim_weighted_mdd={_fmt(risk_mdd)}."
        )

    bullets.append(
        f"Decision rule: ret_th={ret_th:.0e}, action={action}, score={_fmt(score)}, "
        f"expected_return={_fmt(expected_return)}, uncertainty={_fmt(uncertainty)}."
    )

    if len(bullets) < 3:
        bullets.append(
            f"Query context: symbol={symbol}, month={month}, neighbors={int(len(res))}."
        )

    if len(bullets) < 3:
        bullets.append("Available evidence is limited; missing fields were skipped deterministically.")

    rationale = bullets[:max_bullets]

    return {
        "action": action,
        "score": float(score),
        "expected_return": float(expected_return) if not np.isnan(expected_return) else float("nan"),
        "uncertainty": float(uncertainty) if not np.isnan(uncertainty) else float("nan"),
        "risk_mdd": float(risk_mdd) if not np.isnan(risk_mdd) else float("nan"),
        "rationale": rationale,
    }
