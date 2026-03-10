from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def load_monthly_data(
    chunks_path: str = "chunk/sample/monthly_chunks.parquet",
    x_path: str = "chunk/sample/monthly_feat_matrix.npy",
):
    monthly_chunks = pd.read_parquet(chunks_path)
    X = np.load(x_path)
    return monthly_chunks, X


def build_knn_index(X, n_neighbors: int = 50):
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
    nn.fit(X)
    return nn


def query_similar(monthly_chunks, X, nn, symbol: str, month: str, k: int = 10, exclude_self: bool = True):
    mask = (monthly_chunks["symbol"] == symbol) & (monthly_chunks["month"] == month)
    if mask.sum() == 0:
        raise ValueError(f"Cannot find chunk for {symbol} {month}. Check available months.")

    q_idx = int(np.where(mask.to_numpy())[0][0])
    q_vec = X[q_idx].reshape(1, -1)

    dist, idx = nn.kneighbors(q_vec, n_neighbors=max(k + 5, 20))
    idx = idx.flatten()
    dist = dist.flatten()
    sim = 1 - dist

    out = []
    for i, s in zip(idx, sim):
        if exclude_self and i == q_idx:
            continue
        out.append((i, float(s)))
        if len(out) >= k:
            break

    rows = []
    for rank, (i, s) in enumerate(out, start=1):
        r = monthly_chunks.iloc[i]
        rows.append({
            "rank": rank,
            "similarity": s,
            "symbol": r.get("symbol"),
            "month": r.get("month"),
            "start_date": r.get("start_date"),
            "end_date": r.get("end_date"),
            "feat_month_ret": r.get("feat_month_ret", np.nan),
            "feat_month_vol": r.get("feat_month_vol", np.nan),
            "feat_month_mdd": r.get("feat_month_mdd", np.nan),
            "feat_trend_slope": r.get("feat_trend_slope", np.nan),
            "label_next_month_ret": r.get("label_next_month_ret", np.nan),
        })
    res = pd.DataFrame(rows)

    q_row = monthly_chunks.iloc[q_idx]
    top1_i = out[0][0]
    top1_row = monthly_chunks.iloc[top1_i]

    q_curve = np.array(q_row["curve_cum"], dtype=float)
    t_curve = np.array(top1_row["curve_cum"], dtype=float)

    q_curve = q_curve / (q_curve[0] if q_curve[0] != 0 else 1.0)
    t_curve = t_curve / (t_curve[0] if t_curve[0] != 0 else 1.0)

    evidence = {
        "query": {"symbol": symbol, "month": month},
        "top1": {"symbol": top1_row["symbol"], "month": top1_row["month"], "similarity": out[0][1]},
        "q_curve": q_curve,
        "top1_curve": t_curve,
        "q_idx": q_idx,
        "top1_idx": int(top1_i),
    }
    return res, evidence


def build_compare_table(monthly_chunks: pd.DataFrame, evidence: dict) -> pd.DataFrame:
    q_idx = evidence["q_idx"]
    t_idx = evidence["top1_idx"]
    q_row = monthly_chunks.iloc[q_idx]
    t_row = monthly_chunks.iloc[t_idx]

    wanted = [c for c in monthly_chunks.columns if c.startswith("ctx_")]
    wanted += [
        "feat_month_ret", "feat_month_vol", "feat_month_mdd", "feat_trend_slope",
        "feat_ret_first_half", "feat_ret_second_half", "feat_vol20_chg",
    ]
    wanted = [c for c in wanted if c in monthly_chunks.columns]

    compare = pd.DataFrame({
        "Query": [q_row[c] for c in wanted],
        "Top1": [t_row[c] for c in wanted],
    }, index=wanted)

    compare["Query"] = pd.to_numeric(compare["Query"], errors="coerce")
    compare["Top1"] = pd.to_numeric(compare["Top1"], errors="coerce")
    return compare


def build_segments(monthly_chunks: pd.DataFrame, res: pd.DataFrame, query_symbol: str, query_month: str):
    q = monthly_chunks[(monthly_chunks["symbol"] == query_symbol) & (monthly_chunks["month"] == query_month)]
    if q.empty:
        raise ValueError(f"Query not found in monthly_chunks: {query_symbol} {query_month}")

    qrow = q.iloc[0]
    seg2 = {
        "symbol": query_symbol,
        "start": pd.to_datetime(qrow["start_date"]),
        "end": pd.to_datetime(qrow["end_date"]),
    }

    seg1_row = None
    for _, row in res.iterrows():
        if (row["symbol"] == query_symbol) and (row["month"] == query_month):
            continue
        seg1_row = row
        break

    if seg1_row is None:
        raise ValueError("No suitable historical segment found in res.")

    seg1 = {
        "symbol": seg1_row["symbol"],
        "start": pd.to_datetime(seg1_row["start_date"]),
        "end": pd.to_datetime(seg1_row["end_date"]),
    }
    return seg1, seg2


def build_llm_table(seg1, seg2, compare: pd.DataFrame):
    rows = []

    r1 = {
        "segment_id": 1,
        "company": f"{seg1['symbol']} (Top1)",
        "start_date": seg1["start"].strftime("%Y-%m-%d"),
        "end_date": seg1["end"].strftime("%Y-%m-%d"),
    }

    r2 = {
        "segment_id": 2,
        "company": f"{seg2['symbol']} (Query)",
        "start_date": seg2["start"].strftime("%Y-%m-%d"),
        "end_date": seg2["end"].strftime("%Y-%m-%d"),
    }

    for feat in compare.index:
        r1[feat] = float(compare.loc[feat, "Top1"])
        r2[feat] = float(compare.loc[feat, "Query"])

    rows.append(r1)
    rows.append(r2)
    return pd.DataFrame(rows)


def build_market_prompt(segment_table_md: str) -> str:
    return f"""
You are a quantitative market strategist writing an institutional-style market commentary.

Write analysis for TWO market segments.

IMPORTANT RULES
Output plain text only. Bullet points using "-" are allowed.
Do not output reasoning steps.
Use only numerical values from the table.
Round numbers to 2 decimals.
Write numbers as decimals (do not convert to percentages).
Follow the structure exactly.

You may include qualitative macro interpretation such as:

- monetary policy conditions
- discount rate environment
- investor risk appetite
- technology sector sentiment
- institutional capital flows into large-cap technology stocks

Do not introduce numerical values that are not present in the table.

Segment mapping
Segment 1 = Top1 similar segment
Segment 2 = Query segment

STRUCTURE

Segment X: company, start_date to end_date

Paragraph 1 — Market Regime
Describe the macroeconomic regime and investor sentiment during the period.
Mention technology sector positioning and general market stability.

Paragraph 2 — Price Dynamics
Explain price behavior using feat_month_ret and the overall trend.

Paragraph 3 — Macro Conditions
Interpret the macro indicators:

Real interest rate ≈ ctx_real_rate_mean
Yield curve ≈ ctx_yield_curve_mean
Unemployment change ≈ ctx_unemployment_change_mean

Explain what these indicators imply about discount rates and economic stability.

Paragraph 4 — Equity Market Implications
Explain how this macro regime influences investor positioning,
risk appetite, and capital flows toward large-cap technology stocks.

Paragraph 5 — Market Statistics
Introduce the statistical profile, then list:

- Monthly return ≈ feat_month_ret
- Monthly volatility ≈ feat_month_vol
- Maximum drawdown ≈ feat_month_mdd
- Trend slope ≈ feat_trend_slope

Paragraph 6 — Risk-Return Interpretation
Explain the asset’s momentum strength, downside risk,
and overall risk-return profile.

IMPORTANT FOR SEGMENT 2

Do NOT repeat Segment 1 wording.
Highlight differences in market behavior, especially trend slope
(feat_trend_slope) and volatility.

FINAL PARAGRAPH (Segment 2 only)

Explain why Segment 2 is similar to Segment 1.

This explanation must be at least 60 words and must:

- compare macro indicators
  (ctx_real_rate_mean, ctx_yield_curve_mean, ctx_unemployment_change_mean)

- compare market indicators
  (feat_month_ret, feat_month_vol, feat_month_mdd, feat_trend_slope)

- explain the economic mechanism behind the similarity,
  such as shared discount-rate environments,
  synchronized capital flows,
  or comparable volatility regimes.

Write Segment 1 first, then Segment 2.

Data:
{segment_table_md}
""".strip()


def extract_final_output(text: str) -> str:
    if "</think>" in text:
        text = text.split("</think>")[-1]
    if "<<<BEGIN_FINAL>>>" in text and "<<<END_FINAL>>>" in text:
        text = text.split("<<<BEGIN_FINAL>>>", 1)[1].split("<<<END_FINAL>>>", 1)[0]
    return text.strip()


def build_prompt_from_query(query_symbol: str, query_month: str, k: int = 10):
    monthly_chunks, X = load_monthly_data()
    nn = build_knn_index(X)
    res, evidence = query_similar(monthly_chunks, X, nn, query_symbol, query_month, k=k)
    compare = build_compare_table(monthly_chunks, evidence)
    seg1, seg2 = build_segments(monthly_chunks, res, query_symbol, query_month)
    segment_df = build_llm_table(seg1, seg2, compare)
    segment_table_md = segment_df.to_markdown(index=False)
    prompt = build_market_prompt(segment_table_md)

    return {
        "res": res,
        "evidence": evidence,
        "compare": compare,
        "seg1": seg1,
        "seg2": seg2,
        "segment_df": segment_df,
        "segment_table_md": segment_table_md,
        "prompt": prompt,
    }

