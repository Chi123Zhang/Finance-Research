from strategy.data import load_prices
from strategy.equal_weight import build_weights_equal_weight, EqualWeightConfig
from strategy.backtest import backtest_weights_returns, perf_from_returns, vol_target_returns
from chunk.index import SimilarityIndex
from strategy.similarity import build_monthly_weights_similarity

# load prices
px, ret_df = load_prices()

# baseline strategy
W_base = build_weights_equal_weight(px, cfg=EqualWeightConfig())
res_base, _ = backtest_weights_returns(W_base, ret_df)
print("BASE (EqualWeight Monthly Rebalance):", perf_from_returns(res_base["net_ret"]))

# similarity strategy
idx = SimilarityIndex.load()
W_sim, pred_df = build_monthly_weights_similarity(idx, px.index)
res_sim, _ = backtest_weights_returns(W_sim, ret_df)
print("SIM:", perf_from_returns(res_sim["net_ret"]))

# vol targeting on sim
vt = vol_target_returns(res_sim["net_ret"])
print("SIM_VT:", perf_from_returns(vt))
