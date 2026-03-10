#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/Chi123Zhang/Finance-Research/blob/main/LLM%20reason.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[205]:


# =========================
# 0) Setup: clone repo
# =========================
REPO_URL = "https://github.com/Chi123Zhang/Finance-Research.git"
REPO_DIR = "Finance-Research"

from pathlib import Path
import os

if not Path(REPO_DIR).exists():
    get_ipython().system('git clone {REPO_URL}')

get_ipython().run_line_magic('cd', '{REPO_DIR}')
get_ipython().system('ls -lh')


# In[206]:


# =========================
# 1) Load monthly_chunks + feature matrix
# =========================
import pandas as pd
import numpy as np

CHUNKS_PATH = Path("chunk/sample/monthly_chunks.parquet")
X_PATH      = Path("chunk/sample/monthly_feat_matrix.npy")

assert CHUNKS_PATH.exists(), f"Missing: {CHUNKS_PATH}"
assert X_PATH.exists(), f"Missing: {X_PATH}"

monthly_chunks = pd.read_parquet(CHUNKS_PATH)
X = np.load(X_PATH)

print("monthly_chunks:", monthly_chunks.shape)
print("X:", X.shape)
monthly_chunks.head()


# In[207]:


# =========================
# 2) Build cosine KNN index
# =========================
from sklearn.neighbors import NearestNeighbors

# cosine distance: smaller is more similar; we will convert to similarity = 1 - dist
nn = NearestNeighbors(n_neighbors=50, metric="cosine")
nn.fit(X)

print("✅ KNN index built.")


# In[208]:


# =========================
# 3) Query function: find Top-K similar months + build evidence + build compare table
# =========================
import numpy as np

def query_similar(symbol: str, month: str, k: int = 10, exclude_self: bool = True):
    """
    symbol: e.g. "AAPL"
    month:  e.g. "2024-11"
    returns:
      res: DataFrame with top-k similar rows
      evidence: dict contains query/top1 meta + curves
    """
    mask = (monthly_chunks["symbol"] == symbol) & (monthly_chunks["month"] == month)
    if mask.sum() == 0:
        raise ValueError(f"Cannot find chunk for {symbol} {month}. Check available months.")

    q_idx = int(np.where(mask.to_numpy())[0][0])
    q_vec = X[q_idx].reshape(1, -1)

    # get more neighbors then filter self
    dist, idx = nn.kneighbors(q_vec, n_neighbors=max(k + 5, 20))
    idx = idx.flatten()
    dist = dist.flatten()
    sim = 1 - dist  # cosine similarity

    out = []
    for i, s in zip(idx, sim):
        if exclude_self and i == q_idx:
            continue
        out.append((i, float(s)))
        if len(out) >= k:
            break

    # build result table
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
            # some common feature cols (if exist)
            "feat_month_ret": r.get("feat_month_ret", np.nan),
            "feat_month_vol": r.get("feat_month_vol", np.nan),
            "feat_month_mdd": r.get("feat_month_mdd", np.nan),
            "feat_trend_slope": r.get("feat_trend_slope", np.nan),
            "label_next_month_ret": r.get("label_next_month_ret", np.nan),
        })
    res = pd.DataFrame(rows)

    # evidence curves (used for plotting / debugging)
    q_row = monthly_chunks.iloc[q_idx]
    top1_i = out[0][0]
    top1_row = monthly_chunks.iloc[top1_i]

    q_curve = np.array(q_row["curve_cum"], dtype=float)
    t_curve = np.array(top1_row["curve_cum"], dtype=float)

    # normalize start=1
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
    """
    Create the Query vs Top1 feature compare table like your screenshot.
    index: feature name
    cols:  Query, Top1
    """
    q_idx = evidence["q_idx"]
    t_idx = evidence["top1_idx"]
    q_row = monthly_chunks.iloc[q_idx]
    t_row = monthly_chunks.iloc[t_idx]

    # choose columns: ctx_* plus selected feat_*
    wanted = [c for c in monthly_chunks.columns if c.startswith("ctx_")]
    wanted += [
        "feat_month_ret", "feat_month_vol", "feat_month_mdd", "feat_trend_slope",
        "feat_ret_first_half", "feat_ret_second_half", "feat_vol20_chg",
    ]
    # keep only existing
    wanted = [c for c in wanted if c in monthly_chunks.columns]

    compare = pd.DataFrame({
        "Query": [q_row[c] for c in wanted],
        "Top1":  [t_row[c] for c in wanted],
    }, index=wanted)

    # make sure numeric where possible
    compare["Query"] = pd.to_numeric(compare["Query"], errors="coerce")
    compare["Top1"] = pd.to_numeric(compare["Top1"], errors="coerce")

    return compare


# In[209]:


# =========================
# 4) Pick which (symbol, month) you want -> generate compare.csv automatically
# =========================
QUERY_SYMBOL = "AAPL"
QUERY_MONTH  = "2024-11"
K = 10

res, evidence = query_similar(QUERY_SYMBOL, QUERY_MONTH, k=K)
display(res)

compare = build_compare_table(monthly_chunks, evidence)
display(compare)

# save compare.csv into repo
OUT_COMPARE = Path("chunk/sample/compare.csv")
compare.to_csv(OUT_COMPARE)
print("✅ saved:", OUT_COMPARE, " shape:", compare.shape)


# In[210]:


print(compare.columns)


# In[211]:


print(res.columns)
print(res.head())


# In[212]:


import pandas as pd

# 1) Segment 2：严格用 query_symbol + query_month 在 monthly_chunks 里定位
q = monthly_chunks[(monthly_chunks["symbol"] == QUERY_SYMBOL) & (monthly_chunks["month"] == QUERY_MONTH)]
if q.empty:
    raise ValueError(f"Query not found in monthly_chunks: {QUERY_SYMBOL} {QUERY_MONTH}")

qrow = q.iloc[0]
seg2 = {
    "symbol": QUERY_SYMBOL,
    "start": pd.to_datetime(qrow["start_date"]),
    "end": pd.to_datetime(qrow["end_date"]),
}

# 2) Segment 1：从 res 里挑“历史最相似”，排除同 symbol & 同 month（避免拿自己/同月）
seg1_row = None
for _, row in res.iterrows():
    if (row["symbol"] == QUERY_SYMBOL) and (row["month"] == QUERY_MONTH):
        continue
    # 你也可以只排除同 month：if row["month"] == QUERY_MONTH: continue
    seg1_row = row
    break

if seg1_row is None:
    raise ValueError("No suitable historical segment found in res.")

seg1 = {
    "symbol": seg1_row["symbol"],
    "start": pd.to_datetime(seg1_row["start_date"]),
    "end": pd.to_datetime(seg1_row["end_date"]),
}

print("Segment 1:", seg1)
print("Segment 2:", seg2)


# In[213]:


# =========================
# 5) Build Segments (two companies, same month)
# =========================

segments = [
    {
        "segment_id": 1,
        "symbol": seg1["symbol"],
        "start_date": seg1["start"].strftime("%Y-%m-%d"),
        "end_date": seg1["end"].strftime("%Y-%m-%d"),
    },
    {
        "segment_id": 2,
        "symbol": seg2["symbol"],
        "start_date": seg2["start"].strftime("%Y-%m-%d"),
        "end_date": seg2["end"].strftime("%Y-%m-%d"),
    },
]

print("Segments:\n", segments)


print(f"""
Segment 1:
{seg1['symbol']} {seg1['start'].strftime('%Y-%m-%d')} → {seg1['end'].strftime('%Y-%m-%d')}

Segment 2:
{seg2['symbol']} {seg2['start'].strftime('%Y-%m-%d')} → {seg2['end'].strftime('%Y-%m-%d')}
""")


# In[214]:


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
segment_df = build_llm_table(seg1, seg2, compare)
segment_df = build_llm_table(seg1, seg2, compare)

print(segment_df)


# In[215]:


segment_table_md = segment_df.to_markdown(index=False)
print(segment_table_md)


# In[216]:


from pathlib import Path

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

prompt = build_market_prompt(segment_table_md)
Path("prompt.txt").write_text(prompt, encoding="utf-8")
print("prompt saved")


# In[217]:


from huggingface_hub import hf_hub_download

repo_id = "unsloth/Qwen3.5-35B-A3B-GGUF"
filename = "Qwen3.5-35B-A3B-Q4_K_M.gguf"

local_path = hf_hub_download(
    repo_id=repo_id,
    filename=filename
)


# In[218]:


# =========================
# 8) (Optional) Run local LLM with llama.cpp -> analysis.md
#    如果你还没有 gguf 模型，先跳过这一格
# =========================
from pathlib import Path

# Build llama.cpp if needed
if not Path("llama.cpp").exists():
    get_ipython().system('git clone https://github.com/ggerganov/llama.cpp')

get_ipython().run_line_magic('cd', 'llama.cpp')
get_ipython().system('cmake -B build -DGGML_CUDA=ON')
get_ipython().system('cmake --build build -j')
get_ipython().run_line_magic('cd', '..')

# >>> IMPORTANT: Set your local gguf model path here <<<
MODEL_PATH = "/root/.cache/huggingface/hub/models--unsloth--Qwen3.5-35B-A3B-GGUF/snapshots/0da7d49832a73abe50f0c89070971b59dad0039d/Qwen3.5-35B-A3B-Q4_K_M.gguf"

assert Path(MODEL_PATH).exists(), f"❌ MODEL_PATH not found: {MODEL_PATH}"



# In[219]:


get_ipython().system('./llama.cpp/build/bin/llama-cli    -m "{MODEL_PATH}"    -f prompt.txt    -c 8192    -ngl 50    --temp 0.0    2>&1 | tee analysis.md')


# In[222]:


from pathlib import Path

raw = Path("analysis.md").read_text(errors="ignore")


def extract_final_output(text: str) -> str:

    if "</think>" in text:
        text = text.split("</think>")[-1]

    if "<<<BEGIN_FINAL>>>" in text and "<<<END_FINAL>>>" in text:
        text = text.split("<<<BEGIN_FINAL>>>", 1)[1].split("<<<END_FINAL>>>", 1)[0]

    return text.strip()


final_text = extract_final_output(raw)

Path("final.md").write_text(final_text, encoding="utf-8")

print("saved final.md, chars =", len(final_text))

print("\n--- preview ---\n")
print("\n".join(final_text.splitlines()[:120]))


# In[223]:


import os
print(os.getcwd())

