from strategy.data import load_prices
from strategy.momentum_ivol import build_weights_momentum_ivol
from strategy.backtest import backtest_weights_returns, perf_from_returns, vol_target_returns
from chunk.index import SimilarityIndex
from strategy.similarity import build_monthly_weights_similarity

px, ret_df = load_prices()

# Baseline
W_base = build_weights_momentum_ivol(px, ret_df)
res_base, _ = backtest_weights_returns(W_base, ret_df)
print("BASE:", perf_from_returns(res_base["net_ret"]))

# Similarity
idx = SimilarityIndex.load()
W_sim, pred_df = build_monthly_weights_similarity(idx, px.index)
print("pred_df rows:", len(pred_df), "W_sim shape:", getattr(W_sim, "shape", None))

res_sim, _ = backtest_weights_returns(W_sim, ret_df)
print("SIM:", perf_from_returns(res_sim["net_ret"]))

# Vol-target on similarity
vt = vol_target_returns(res_sim["net_ret"])
print("SIM_VT:", perf_from_returns(vt))