# chunk/build.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class BuildConfig:
    """
    Build monthly chunks from processed daily panel data.

    Expected input files: CSVs under proc_dir (one per symbol),
    each with at least: date, symbol, log_ret
    and optionally: volatility_20d, volume_change, ps_ratio, pe_ratio, rev_growth_qoq,
    real_rate, yield_curve, unemployment_change, close, is_trading_day.
    """
    proc_dir: Path = Path("data/processed_company_dataset")
    out_dir: Path = Path("chunk/artifacts")

    # If True and column exists, keep rows where is_trading_day == 1
    trading_day_only: bool = True

    # Base columns we will try to keep (missing ones are ignored)
    base_cols: Tuple[str, ...] = (
        "date", "symbol", "log_ret", "volatility_20d", "volume_change",
        "ps_ratio", "pe_ratio", "rev_growth_qoq",
        "real_rate", "yield_curve", "unemployment_change",
        "close", "is_trading_day",
    )

    # Feature columns used for embedding (we will keep only existing cols)
    feature_cols: Optional[List[str]] = None

    # Output filenames
    chunks_filename: str = "monthly_chunks.parquet"
    feat_matrix_filename: str = "monthly_feat_matrix.npy"
    feat_mu_filename: str = "feat_mu.csv"
    feat_sd_filename: str = "feat_sd.csv"


def _max_drawdown_from_nav(nav: np.ndarray) -> float:
    if len(nav) == 0:
        return float("nan")
    peak = np.maximum.accumulate(nav)
    dd = nav / peak - 1.0
    return float(np.min(dd))


def _trend_slope(y: np.ndarray) -> float:
    """Simple standardized slope proxy for trend strength."""
    if len(y) < 3:
        return float("nan")
    x = np.arange(len(y), dtype=float)
    x = (x - x.mean()) / (x.std() + 1e-12)
    y = (y - y.mean()) / (y.std() + 1e-12)
    return float((x * y).mean())


def load_processed_panel(cfg: BuildConfig) -> pd.DataFrame:
    files = sorted(Path(cfg.proc_dir).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No csv files found in {Path(cfg.proc_dir).resolve()}")

    dfs = []
    for fp in files:
        df = pd.read_csv(fp)
        if "date" not in df.columns:
            raise ValueError(f"Missing 'date' in {fp}")
        df["date"] = pd.to_datetime(df["date"])
        dfs.append(df)

    panel = (
        pd.concat(dfs, ignore_index=True)
        .sort_values(["symbol", "date"])
        .reset_index(drop=True)
    )

    # keep only columns that exist
    use_cols = [c for c in cfg.base_cols if c in panel.columns]
    panel = panel[use_cols].copy()

    if cfg.trading_day_only and "is_trading_day" in panel.columns:
        panel = panel[panel["is_trading_day"] == 1].copy()

    panel["month"] = panel["date"].dt.to_period("M").astype(str)
    return panel


def build_monthly_chunks(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Convert daily panel -> monthly chunks (one row per symbol-month).
    Includes:
      - curve_logret (list)
      - curve_cum (list, normalized cumulative starting at 1)
      - feat_* engineered features
      - ctx_* context snapshots (macro + fundamentals where present)
    """
    rows = []

    for (sym, mon), g in panel.groupby(["symbol", "month"], sort=True):
        g = g.sort_values("date")
        logret = g["log_ret"].astype(float).to_numpy()

        # monthly cumulative curve (start at 1)
        nav = np.exp(np.cumsum(np.nan_to_num(logret, nan=0.0)))
        nav = np.insert(nav, 0, 1.0)

        month_ret = float(np.exp(np.nansum(logret)) - 1.0)
        month_vol = (
            float(np.nanstd(np.expm1(logret), ddof=0) * np.sqrt(252))
            if len(logret) > 1 else float("nan")
        )
        month_mdd = _max_drawdown_from_nav(nav)
        slope = _trend_slope(nav[1:])

        mid = max(1, len(logret) // 2)
        ret_first = float(np.exp(np.nansum(logret[:mid])) - 1.0)
        ret_second = float(np.exp(np.nansum(logret[mid:])) - 1.0)

        vol20_start = float(g["volatility_20d"].iloc[0]) if "volatility_20d" in g.columns else float("nan")
        vol20_end = float(g["volatility_20d"].iloc[-1]) if "volatility_20d" in g.columns else float("nan")
        vol20_chg = (vol20_end - vol20_start) if (np.isfinite(vol20_start) and np.isfinite(vol20_end)) else float("nan")

        # macro context: mean/end/chg
        ctx = {}
        for c in ["real_rate", "yield_curve", "unemployment_change"]:
            if c in g.columns:
                s = g[c].astype(float)
                ctx[f"{c}_mean"] = float(np.nanmean(s))
                ctx[f"{c}_end"] = float(s.iloc[-1])
                ctx[f"{c}_chg"] = float(s.iloc[-1] - s.iloc[0])

        # fundamentals snapshot: end-of-month values
        fctx = {}
        for c in ["pe_ratio", "ps_ratio", "rev_growth_qoq", "volume_change"]:
            if c in g.columns:
                fctx[f"{c}_end"] = float(g[c].astype(float).iloc[-1])

        rows.append({
            "symbol": sym,
            "month": mon,
            "start_date": g["date"].iloc[0],
            "end_date": g["date"].iloc[-1],
            "n_days": int(len(g)),
            "curve_logret": logret.tolist(),
            "curve_cum": nav.tolist(),

            # features
            "feat_month_ret": month_ret,
            "feat_month_vol": month_vol,
            "feat_month_mdd": month_mdd,
            "feat_trend_slope": slope,
            "feat_ret_first_half": ret_first,
            "feat_ret_second_half": ret_second,
            "feat_vol20_chg": vol20_chg,

            # context
            **{f"ctx_{k}": v for k, v in ctx.items()},
            **{f"ctx_{k}": v for k, v in fctx.items()},
        })

    monthly = pd.DataFrame(rows).sort_values(["symbol", "month"]).reset_index(drop=True)
    return monthly


def add_next_month_labels(monthly: pd.DataFrame) -> pd.DataFrame:
    monthly = monthly.sort_values(["symbol", "month"]).reset_index(drop=True)
    monthly["label_next_month_ret"] = monthly.groupby("symbol")["feat_month_ret"].shift(-1)
    monthly["label_next_month_mdd"] = monthly.groupby("symbol")["feat_month_mdd"].shift(-1)
    return monthly


def build_feature_matrix(monthly: pd.DataFrame, feature_cols: Optional[List[str]] = None):
    """
    Create standardized feature matrix Xz (numpy array) and store per-row vectors in monthly['feat_vec'].
    Returns: (monthly_with_feat_vec, Xz, mu_series, sd_series, used_feature_cols)
    """
    if feature_cols is None:
        feature_cols = [
            "feat_month_ret",
            "feat_month_vol",
            "feat_month_mdd",
            "feat_trend_slope",
            "feat_ret_first_half",
            "feat_ret_second_half",
            "feat_vol20_chg",
            "ctx_yield_curve_end",
            "ctx_yield_curve_chg",
            "ctx_real_rate_end",
            "ctx_real_rate_chg",
            "ctx_unemployment_change_end",
            "ctx_unemployment_change_chg",
            "ctx_pe_ratio_end",
            "ctx_ps_ratio_end",
            "ctx_rev_growth_qoq_end",
            "ctx_volume_change_end",
        ]

    used = [c for c in feature_cols if c in monthly.columns]
    if not used:
        raise ValueError("No feature columns found in monthly chunks.")

    X = monthly[used].astype(float).copy()

    # fill missing with per-column median (robust)
    X = X.apply(lambda s: s.fillna(s.median()), axis=0)

    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0).replace(0, 1.0)
    Xz = ((X - mu) / sd).to_numpy()

    monthly = monthly.copy()
    monthly["feat_vec"] = list(Xz)

    return monthly, Xz, mu, sd, used


def save_artifacts(
    monthly: pd.DataFrame,
    Xz: np.ndarray,
    mu: pd.Series,
    sd: pd.Series,
    cfg: BuildConfig,
) -> None:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    chunks_path = cfg.out_dir / cfg.chunks_filename
    feat_path = cfg.out_dir / cfg.feat_matrix_filename
    mu_path = cfg.out_dir / cfg.feat_mu_filename
    sd_path = cfg.out_dir / cfg.feat_sd_filename

    monthly.to_parquet(chunks_path, index=False)
    np.save(feat_path, Xz)

    mu.to_csv(mu_path)
    sd.to_csv(sd_path)


def run_build(cfg: Optional[BuildConfig] = None) -> pd.DataFrame:
    """
    End-to-end build:
      processed csvs -> panel -> monthly chunks -> labels -> feature matrix -> save artifacts
    Returns the final monthly_chunks dataframe.
    """
    cfg = cfg or BuildConfig()

    panel = load_processed_panel(cfg)
    monthly = build_monthly_chunks(panel)
    monthly = add_next_month_labels(monthly)
    monthly, Xz, mu, sd, used = build_feature_matrix(monthly, cfg.feature_cols)

    save_artifacts(monthly, Xz, mu, sd, cfg)

    print(f"[OK] Built monthly chunks: {monthly.shape}")
    print(f"[OK] Feature matrix shape: {Xz.shape} | features used: {len(used)}")
    print(f"[OK] Saved to: {cfg.out_dir.resolve()}")
    return monthly


if __name__ == "__main__":
    run_build()