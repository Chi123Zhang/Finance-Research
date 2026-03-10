# from strategy.data import load_prices
# from strategy.equal_weight import build_weights_equal_weight, EqualWeightConfig
# from strategy.backtest import backtest_weights_returns, perf_from_returns, vol_target_returns
# from chunk.index import SimilarityIndex
# from strategy.similarity import build_monthly_weights_similarity

# px, ret_df = load_prices()

# # Baseline
# W_base = build_weights_equal_weight(px, cfg=EqualWeightConfig())
# res_base, _ = backtest_weights_returns(W_base, ret_df)
# print("BASE (EqualWeight Monthly Rebalance):", perf_from_returns(res_base["net_ret"]))

# # Similarity
# idx = SimilarityIndex.load()
# W_sim, pred_df = build_monthly_weights_similarity(idx, px.index)
# print("pred_df rows:", len(pred_df), "W_sim shape:", getattr(W_sim, "shape", None))

# res_sim, _ = backtest_weights_returns(W_sim, ret_df)
# print("SIM:", perf_from_returns(res_sim["net_ret"]))

# # Vol-target on similarity
# vt = vol_target_returns(res_sim["net_ret"])
# print("SIM_VT:", perf_from_returns(vt))
# run_all.py
from strategy.data import load_prices
from strategy.equal_weight import build_weights_equal_weight, EqualWeightConfig
from strategy.backtest import (
    backtest_weights_returns,
    perf_from_returns,
    vol_target_returns,
)
from chunk.index import SimilarityIndex
from strategy.similarity import build_monthly_weights_similarity
from llm.llm_call import explain_similarity_with_llm


def main():
    px, ret_df = load_prices()

    # Baseline
    W_base = build_weights_equal_weight(px, cfg=EqualWeightConfig())
    res_base, _ = backtest_weights_returns(W_base, ret_df)
    print("BASE (EqualWeight Monthly Rebalance):", perf_from_returns(res_base["net_ret"]))

    # Similarity retrieval
    idx = SimilarityIndex.load()
    W_sim, pred_df = build_monthly_weights_similarity(idx, px.index)
    print("pred_df rows:", len(pred_df), "W_sim shape:", getattr(W_sim, "shape", None))

    # LLM explanation
    report = explain_similarity_with_llm(pred_df=pred_df, output_dir="llm")
    print("\n===== LLM / REPORT OUTPUT =====\n")
    print(report[:3000])  # 只打印前3000字符，避免终端太长

    # Backtest similarity strategy
    res_sim, _ = backtest_weights_returns(W_sim, ret_df)
    print("SIM:", perf_from_returns(res_sim["net_ret"]))

    # Vol-target on similarity
    vt = vol_target_returns(res_sim["net_ret"])
    print("SIM_VT:", perf_from_returns(vt))


if __name__ == "__main__":
    main()
