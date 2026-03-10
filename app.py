import traceback
import pandas as pd
import gradio as gr

from strategy.data import load_prices
from strategy.backtest import (
    backtest_weights_returns,
    perf_from_returns,
    vol_target_returns,
)
from strategy.equal_weight import build_weights_equal_weight, EqualWeightConfig
from chunk.index import SimilarityIndex
from strategy.similarity import build_monthly_weights_similarity
from llm.reasoning import build_prompt_from_query
from llm.llm_call import run_local_llm


# ----------------------------
# Config
# ----------------------------
DEFAULT_QUERY_SYMBOL = "AAPL"
DEFAULT_QUERY_MONTH = "2024-11"
DEFAULT_K = 10

DEFAULT_REPO_ID = "unsloth/Qwen2.5-7B-Instruct-GGUF"
DEFAULT_FILENAME = "Qwen2.5-7B-Instruct-Q4_K_M.gguf"


# ----------------------------
# Simple in-memory cache
# ----------------------------
_CACHE = {
    "px": None,
    "ret_df": None,
    "idx": None,
    "W_base": None,
    "base_perf": None,
    "W_sim": None,
    "pred_df": None,
    "sim_perf": None,
    "vt": None,
    "vt_perf": None,
}


def fmt_perf(perf: dict) -> str:
    if not isinstance(perf, dict):
        return str(perf)
    try:
        return (
            f"annualized return = {perf.get('Annualized Return', float('nan')):.4f}, "
            f"annualized vol = {perf.get('Annualized Vol', float('nan')):.4f}, "
            f"sharpe = {perf.get('Sharpe (rf=0)', float('nan')):.4f}, "
            f"max drawdown = {perf.get('Max Drawdown', float('nan')):.4f}"
        )
    except Exception:
        return str(perf)


def preview_dataframe(df: pd.DataFrame, n_rows: int = 5, max_cols: int = 6) -> str:
    if df is None or df.empty:
        return "No rows available."

    out = df.head(n_rows).copy()
    if out.shape[1] > max_cols:
        out = out.iloc[:, :max_cols]

    return out.to_string(index=False)


def preview_text(text: str, max_chars: int = 1800) -> str:
    if text is None:
        return ""
    text = str(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n...[truncated]..."


def get_data_and_returns(logs):
    if _CACHE["px"] is None or _CACHE["ret_df"] is None:
        logs.append("Loading prices and returns from source.")
        px, ret_df = load_prices()
        _CACHE["px"] = px
        _CACHE["ret_df"] = ret_df
    else:
        logs.append("Using cached prices and returns.")

    px = _CACHE["px"]
    ret_df = _CACHE["ret_df"]

    logs.append(f"Price data shape: {getattr(px, 'shape', None)}")
    logs.append(f"Return data shape: {getattr(ret_df, 'shape', None)}")
    return px, ret_df


def get_similarity_index(logs):
    if _CACHE["idx"] is None:
        logs.append("Loading similarity index from disk.")
        _CACHE["idx"] = SimilarityIndex.load()
    else:
        logs.append("Using cached similarity index.")
    return _CACHE["idx"]


def run_baseline(logs):
    px, ret_df = get_data_and_returns(logs)

    if _CACHE["W_base"] is None or _CACHE["base_perf"] is None:
        logs.append("Running equal-weight baseline.")
        W_base = build_weights_equal_weight(px, cfg=EqualWeightConfig())
        res_base, _ = backtest_weights_returns(W_base, ret_df)
        base_perf = perf_from_returns(res_base["net_ret"])

        _CACHE["W_base"] = W_base
        _CACHE["base_perf"] = base_perf
    else:
        logs.append("Using cached baseline results.")

    return _CACHE["base_perf"]


def run_similarity(logs):
    px, ret_df = get_data_and_returns(logs)
    idx = get_similarity_index(logs)

    if (
        _CACHE["W_sim"] is None
        or _CACHE["pred_df"] is None
        or _CACHE["sim_perf"] is None
    ):
        logs.append("Building similarity-based weights.")
        W_sim, pred_df = build_monthly_weights_similarity(idx, px.index)

        logs.append(f"Retrieved similarity rows: {len(pred_df)}")
        logs.append(f"Similarity weight matrix shape: {getattr(W_sim, 'shape', None)}")

        logs.append("Backtesting similarity strategy.")
        res_sim, _ = backtest_weights_returns(W_sim, ret_df)
        sim_perf = perf_from_returns(res_sim["net_ret"])

        _CACHE["W_sim"] = W_sim
        _CACHE["pred_df"] = pred_df
        _CACHE["sim_perf"] = sim_perf
    else:
        logs.append("Using cached similarity results.")

    return _CACHE["sim_perf"], _CACHE["pred_df"]


def run_vol_target(logs):
    sim_perf, pred_df = run_similarity(logs)

    if _CACHE["vt"] is None or _CACHE["vt_perf"] is None:
        logs.append("Applying volatility targeting to similarity returns.")
        px, ret_df = get_data_and_returns(logs)
        W_sim = _CACHE["W_sim"]
        res_sim, _ = backtest_weights_returns(W_sim, ret_df)
        vt = vol_target_returns(res_sim["net_ret"])
        vt_perf = perf_from_returns(vt)

        _CACHE["vt"] = vt
        _CACHE["vt_perf"] = vt_perf
    else:
        logs.append("Using cached vol-target results.")

    return _CACHE["vt_perf"], pred_df


def run_llm_prompt(logs, query_symbol=DEFAULT_QUERY_SYMBOL, query_month=DEFAULT_QUERY_MONTH, k=DEFAULT_K):
    logs.append(
        f"Building LLM prompt for symbol={query_symbol}, month={query_month}, k={k}."
    )
    pkg = build_prompt_from_query(query_symbol, query_month, k=k)
    logs.append("LLM prompt package built successfully.")
    logs.append(f"LLM compare table shape: {getattr(pkg.get('compare'), 'shape', None)}")
    logs.append(
        f"LLM segment table shape: {getattr(pkg.get('segment_df'), 'shape', None)}"
    )
    return pkg


def run_llm_analysis(
    logs,
    query_symbol=DEFAULT_QUERY_SYMBOL,
    query_month=DEFAULT_QUERY_MONTH,
    k=DEFAULT_K,
    repo_id=DEFAULT_REPO_ID,
    filename=DEFAULT_FILENAME,
):
    logs.append(
        f"Building full LLM analysis for symbol={query_symbol}, month={query_month}, k={k}."
    )
    logs.append(f"Using Hugging Face model: {repo_id} / {filename}")

    pkg = build_prompt_from_query(query_symbol, query_month, k=k)
    logs.append("Prompt package built. Running local LLM inference.")

    out = run_local_llm(
        prompt=pkg["prompt"],
        repo_id=repo_id,
        filename=filename,
    )

    logs.append(f"Local LLM return code: {out.get('returncode')}")
    logs.append(f"Resolved model path: {out.get('model_path')}")
    logs.append(f"Saved raw analysis to: {out.get('analysis_path')}")
    return pkg, out


def build_help_message():
    return """## Finance Research Demo

You can ask things like:

- `help`
- `run all`
- `summary`
- `show baseline`
- `show similarity`
- `show vol target`
- `compare strategies`
- `show top rows`
- `what did the similarity model retrieve?`
- `build llm prompt`
- `show llm prompt`
- `market prompt`
- `generate llm analysis`
- `run llm`
- `market commentary`

This demo supports simple natural-language commands and returns the corresponding strategy outputs.
"""


def route_message(user_message: str) -> str:
    msg = user_message.lower().strip()

    if any(x in msg for x in ["help", "what can you do", "commands"]):
        return "help"

    if any(x in msg for x in ["generate llm analysis", "run llm", "market commentary", "generate commentary"]):
        return "llm_analysis"

    if any(x in msg for x in ["run all", "summary", "overall", "full pipeline"]):
        return "run_all"

    if any(x in msg for x in ["compare", "comparison", "compare strategies"]):
        return "compare"

    if any(x in msg for x in ["baseline", "equal weight", "equal-weight"]):
        return "baseline"

    if any(
        x in msg
        for x in [
            "vol target",
            "vol-target",
            "volatility target",
            "volatility targeting",
        ]
    ):
        return "vol_target"

    if any(
        x in msg
        for x in [
            "top rows",
            "show rows",
            "retrieved rows",
            "similarity rows",
            "what did the similarity model retrieve",
        ]
    ):
        return "rows"

    if any(x in msg for x in ["similarity", "similar strategy", "similarity strategy"]):
        return "similarity"

    if any(
        x in msg
        for x in [
            "llm prompt",
            "build llm prompt",
            "show llm prompt",
            "market prompt",
        ]
    ):
        return "llm_prompt"

    return "default"


def run_pipeline(user_message, chat_history):
    chat_history = chat_history or []
    logs = []

    if not user_message or not str(user_message).strip():
        return chat_history, "Please enter a message.", ""

    command = route_message(user_message)
    logs.append(f"User command routed to: {command}")

    try:
        if command == "help":
            final_reply = build_help_message()

        elif command == "baseline":
            base_perf = run_baseline(logs)
            final_reply = "\n".join(
                [
                    "## Equal-Weight Baseline",
                    f"- {fmt_perf(base_perf)}",
                    "",
                    "This is the benchmark strategy used for comparison.",
                ]
            )

        elif command == "similarity":
            sim_perf, pred_df = run_similarity(logs)
            final_reply = "\n".join(
                [
                    "## Similarity Strategy",
                    f"- {fmt_perf(sim_perf)}",
                    "",
                    "This strategy builds weights from retrieved similar historical patterns.",
                ]
            )

        elif command == "vol_target":
            vt_perf, pred_df = run_vol_target(logs)
            final_reply = "\n".join(
                [
                    "## Similarity + Vol Target",
                    f"- {fmt_perf(vt_perf)}",
                    "",
                    "This version applies volatility targeting on top of the similarity strategy.",
                ]
            )

        elif command == "rows":
            sim_perf, pred_df = run_similarity(logs)
            final_reply = "\n".join(
                [
                    "## Retrieved Similarity Rows",
                    "```text",
                    preview_dataframe(pred_df, n_rows=5, max_cols=6),
                    "```",
                    "",
                    "Only a compact preview is shown here so the chat layout stays readable.",
                ]
            )

        elif command == "llm_prompt":
            llm_pkg = run_llm_prompt(
                logs,
                query_symbol=DEFAULT_QUERY_SYMBOL,
                query_month=DEFAULT_QUERY_MONTH,
                k=DEFAULT_K,
             )

            final_reply = "\n".join(
                [
                    "## LLM Market Prompt",
                    "",
                    "### Segment Table Preview",
                    "```text",
                    preview_dataframe(llm_pkg["segment_df"], n_rows=5, max_cols=8),
                    "```",
                    "",
                    "### Prompt Preview",
                    "```text",
                    preview_text(llm_pkg["prompt"], max_chars=1800),
                    "```",
                ]
            )

        elif command == "llm_analysis":
           llm_pkg, llm_out = run_llm_analysis(
               logs,
               query_symbol=DEFAULT_QUERY_SYMBOL,
               query_month=DEFAULT_QUERY_MONTH,
               k=DEFAULT_K,
               repo_id=DEFAULT_REPO_ID,
               filename=DEFAULT_FILENAME,
            )

            final_text = llm_out.get("final_text", "").strip()

            final_reply = "\n".join(
                [
                    "## LLM Market Commentary",
                    "",
                    final_text if final_text else "LLM returned no final text.",
                ]
            )

        elif command == "compare" or command == "run_all":
            base_perf = run_baseline(logs)
            sim_perf, pred_df = run_similarity(logs)
            vt_perf, _ = run_vol_target(logs)
            llm_pkg = run_llm_prompt(
                logs,
                query_symbol=DEFAULT_QUERY_SYMBOL,
                query_month=DEFAULT_QUERY_MONTH,
                k=DEFAULT_K,
            )

            final_reply = "\n".join(
                [
                    "## Strategy Comparison",
                    f"- **Baseline:** {fmt_perf(base_perf)}",
                    f"- **Similarity:** {fmt_perf(sim_perf)}",
                    f"- **Similarity + Vol Target:** {fmt_perf(vt_perf)}",
                    "",
                    "## Retrieved Similarity Rows (Preview)",
                    "```text",
                    preview_dataframe(pred_df, n_rows=5, max_cols=6),
                    "```",
                    "",
                    "## LLM Segment Table (Preview)",
                    "```text",
                    preview_dataframe(llm_pkg["segment_df"], n_rows=5, max_cols=8),
                    "```",
                    "",
                    "## LLM Prompt (Preview)",
                    "```text",
                    preview_text(llm_pkg["prompt"], max_chars=1200),
                    "```",
                ]
            )

        else:
            base_perf = run_baseline(logs)
            sim_perf, pred_df = run_similarity(logs)
            vt_perf, _ = run_vol_target(logs)

            final_reply = "\n".join(
                [
                    "## Strategy Summary",
                    f"- **Baseline:** {fmt_perf(base_perf)}",
                    f"- **Similarity:** {fmt_perf(sim_perf)}",
                    f"- **Similarity + Vol Target:** {fmt_perf(vt_perf)}",
                    "",
                    "I did not detect a specific command, so I returned the overall summary.",
                    "Try `help` to see supported prompts.",
                ]
            )

    except Exception as e:
        final_reply = f"## Pipeline Error\n\n{str(e)}"
        logs.append("Pipeline failed.")
        logs.append(traceback.format_exc())

    chat_history.append({"role": "user", "content": user_message})
    chat_history.append({"role": "assistant", "content": final_reply})

    pipeline_text = "\n".join(logs)
    return chat_history, pipeline_text, ""


def clear_chat():
    return [], "New session started. Send a message to see pipeline details.", ""


with gr.Blocks(title="Finance Research Demo") as demo:
    gr.Markdown("# Finance Research Demo")
    gr.Markdown(
        "Ask about baseline, similarity strategy, volatility targeting, retrieved rows, the LLM market prompt, or generate full LLM commentary."
    )

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                height=500,
                label="Conversation",
                render_markdown=True,
                type="messages",
            )

            msg = gr.Textbox(
                label="Your message",
                placeholder="Example: compare strategies / build llm prompt / generate llm analysis",
            )

            with gr.Row():
                send_btn = gr.Button("Send", variant="primary")
                clear_btn = gr.Button("Clear Chat")

        with gr.Column(scale=2):
            pipeline_box = gr.Textbox(
                label="Pipeline Details",
                value="New session started. Send a message to see pipeline details.",
                lines=30,
                interactive=False,
            )

    send_btn.click(
        fn=run_pipeline,
        inputs=[msg, chatbot],
        outputs=[chatbot, pipeline_box, msg],
    )

    msg.submit(
        fn=run_pipeline,
        inputs=[msg, chatbot],
        outputs=[chatbot, pipeline_box, msg],
    )

    clear_btn.click(
        fn=clear_chat,
        inputs=[],
        outputs=[chatbot, pipeline_box, msg],
    )

demo.launch()
