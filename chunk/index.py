# chunk/index.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


@dataclass
class IndexConfig:
    artifacts_dir: Path = Path("chunk/artifacts")
    chunks_filename: str = "monthly_chunks.parquet"
    feat_matrix_filename: str = "monthly_feat_matrix.npy"

    metric: str = "cosine"
    default_n_neighbors: int = 50


def _as_month_str(x: str) -> str:
    """
    Expect 'YYYY-MM'. We keep it as string so lexicographic ordering works.
    """
    if not isinstance(x, str):
        x = str(x)
    # minimal validation
    if len(x) != 7 or x[4] != "-":
        raise ValueError(f"month must be like 'YYYY-MM', got: {x}")
    return x


def _curve_metrics(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = min(len(a), len(b))
    if n == 0:
        return {"curve_corr": np.nan, "curve_mse": np.nan, "len_days_used": 0}

    a = a[:n]
    b = b[:n]

    a = a / (a[0] if a[0] != 0 else 1.0)
    b = b / (b[0] if b[0] != 0 else 1.0)

    corr = float(np.corrcoef(a, b)[0, 1]) if n >= 2 else np.nan
    mse = float(np.mean((a - b) ** 2))

    def mdd_pos(x):
        peak = np.maximum.accumulate(x)
        dd = x / peak - 1.0
        i = int(np.argmin(dd))
        return float(dd[i]), i

    mdd_a, pos_a = mdd_pos(a)
    mdd_b, pos_b = mdd_pos(b)

    return {
        "curve_corr": corr,
        "curve_mse": mse,
        "mdd_query": mdd_a,
        "mdd_day_query": pos_a,
        "mdd_top1": mdd_b,
        "mdd_day_top1": pos_b,
        "len_days_used": int(n),
    }


class SimilarityIndex:
    """
    Pattern-RAG index for monthly chunks.
    """

    def __init__(self, monthly_chunks: pd.DataFrame, X: np.ndarray, nn: NearestNeighbors, cfg: IndexConfig):
        self.monthly_chunks = monthly_chunks
        self.X = X
        self.nn = nn
        self.cfg = cfg

        # lookup will be built by _build_lookup()
        self._key_to_idx = {}

    @classmethod
    def load(cls, cfg: Optional[IndexConfig] = None) -> "SimilarityIndex":
        cfg = cfg or IndexConfig()
        chunks_path = cfg.artifacts_dir / cfg.chunks_filename
        feat_path = cfg.artifacts_dir / cfg.feat_matrix_filename

        if not chunks_path.exists():
            raise FileNotFoundError(f"Missing {chunks_path.resolve()}")
        if not feat_path.exists():
            raise FileNotFoundError(f"Missing {feat_path.resolve()}")

        monthly_chunks = pd.read_parquet(chunks_path)
        X = np.load(feat_path)

        # ensure month is str
        monthly_chunks = monthly_chunks.copy()
        monthly_chunks["month"] = monthly_chunks["month"].astype(str)

        nn = NearestNeighbors(n_neighbors=cfg.default_n_neighbors, metric=cfg.metric)
        nn.fit(X)

        obj = cls(monthly_chunks=monthly_chunks, X=X, nn=nn, cfg=cfg)
        obj._build_lookup()
        return obj

    def _build_lookup(self) -> None:
        self._key_to_idx = {}
        for i, r in self.monthly_chunks[["symbol", "month"]].reset_index(drop=True).iterrows():
            self._key_to_idx[(r["symbol"], r["month"])] = int(i)

    def get_index(self, symbol: str, month: str) -> int:
        month = _as_month_str(month)
        key = (symbol, month)
        if key not in self._key_to_idx:
            raise KeyError(f"Cannot find chunk for {symbol} {month}.")
        return self._key_to_idx[key]

    def query(
        self,
        symbol: str,
        month: str,
        k: int = 10,
        past_only: bool = True,
        same_symbol_only: bool = False,
        exclude_self: bool = True,
        n_candidates: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Returns:
          res: DataFrame of top-k neighbors with similarity + labels
          evidence: dict containing prediction + top1 curve + curve metrics
        """
        month = _as_month_str(month)
        q_idx = self.get_index(symbol, month)
        q_vec = self.X[q_idx].reshape(1, -1)

        # retrieve a bigger candidate pool, then filter
        n_candidates = n_candidates or max(k * 8, 80)
        n_candidates = min(n_candidates, len(self.monthly_chunks))

        dist, idx = self.nn.kneighbors(q_vec, n_neighbors=n_candidates)
        idx = idx.flatten()
        dist = dist.flatten()
        sim = 1.0 - dist  # cosine distance -> similarity

        out = []
        for i, s in zip(idx, sim):
            if exclude_self and int(i) == q_idx:
                continue

            r = self.monthly_chunks.iloc[int(i)]
            # optionally restrict neighbors to the same symbol
            if same_symbol_only and str(r["symbol"]) != symbol:
                continue
            # avoid future leakage: only use months strictly before query month
            if past_only and str(r["month"]) >= month:
                continue

            out.append((int(i), float(s)))
            if len(out) >= k:
                break

        if len(out) == 0:
            raise ValueError(
                f"No neighbors found for {symbol} {month}. "
                f"Try past_only=False or increase n_candidates."
            )

        rows = []
        for rank, (i, s) in enumerate(out, start=1):
            r = self.monthly_chunks.iloc[i]
            rows.append({
                "rank": rank,
                "similarity": s,
                "symbol": r["symbol"],
                "month": r["month"],
                "start_date": r.get("start_date", None),
                "end_date": r.get("end_date", None),
                "feat_month_ret": r.get("feat_month_ret", np.nan),
                "feat_month_vol": r.get("feat_month_vol", np.nan),
                "feat_month_mdd": r.get("feat_month_mdd", np.nan),
                "label_next_month_ret": r.get("label_next_month_ret", np.nan),
                "label_next_month_mdd": r.get("label_next_month_mdd", np.nan),
            })
        res = pd.DataFrame(rows)

        # prediction: similarity-weighted mean + distribution quantiles
        neigh_ret = res["label_next_month_ret"].astype(float)
        w = res["similarity"].astype(float).clip(lower=0)
        w = w / (w.sum() + 1e-12)

        pred_mean = float((w * neigh_ret).sum())
        pred_p10 = float(neigh_ret.quantile(0.10))
        pred_p50 = float(neigh_ret.quantile(0.50))
        pred_p90 = float(neigh_ret.quantile(0.90))

        q_row = self.monthly_chunks.iloc[q_idx]
        top1_row = self.monthly_chunks.iloc[out[0][0]]

        q_curve = np.asarray(q_row["curve_cum"], dtype=float)
        t_curve = np.asarray(top1_row["curve_cum"], dtype=float)

        q_curve = q_curve / (q_curve[0] if q_curve[0] != 0 else 1.0)
        t_curve = t_curve / (t_curve[0] if t_curve[0] != 0 else 1.0)

        curve_stats = _curve_metrics(q_curve, t_curve)

        evidence = {
            "query": {"symbol": symbol, "month": month},
            "top1": {
                "symbol": str(top1_row["symbol"]),
                "month": str(top1_row["month"]),
                "similarity": float(out[0][1]),
            },
            "pred": {
                "mean": pred_mean,
                "p10": pred_p10,
                "p50": pred_p50,
                "p90": pred_p90,
            },
            "curve": {
                "q_curve": q_curve,
                "top1_curve": t_curve,
                "metrics": curve_stats,
            },
        }

        return res, evidence


if __name__ == "__main__":
    idx = SimilarityIndex.load()
    res, evidence = idx.query("AAPL", "2024-11", k=10, past_only=True, same_symbol_only=True)
    print(res.head(10))
    print("Pred:", evidence["pred"])
    print("Top1:", evidence["top1"])
    print("Curve metrics:", evidence["curve"]["metrics"])