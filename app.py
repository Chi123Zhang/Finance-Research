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


def run_pipeline(user_message, chat_history):
    chat_history = chat_history or []
    logs = []
    reply_parts = []

    try:
        logs.append("Step 1: Loading prices and returns.")
        px, ret_df = load_prices()
        logs.append(f"Loaded price data shape: {getattr(px, 'shape', None)}")
        logs.append(f"Loaded return data shape: {getattr(ret_df, 'shape', None)}")

        logs.append("Step 2: Running equal-weight baseline.")
        W_base = build_weights_equal_weight(px, cfg=EqualWeightConfig())
        res_base, _ = backtest_weights_returns(W_base, ret_df)
        base_perf = perf_from_returns(res_base["net_ret"])
        logs.append(f"Baseline metrics: {base_perf}")

        logs.append("Step 3: Loading similarity index.")
        idx = SimilarityIndex.load()
        logs.append("Similarity index loaded.")

        logs.append("Step 4: Building similarity-based weights.")
        W_sim, pred_df = build_monthly_weights_similarity(idx, px.index)
        logs.append(f"pred_df rows: {len(pred_df)}")
        logs.append(f"W_sim shape: {getattr(W_sim, 'shape', None)}")

        logs.append("Step 5: Backtesting similarity strategy.")
        res_sim, _ = backtest_weights_returns(W_sim, ret_df)
        sim_perf = perf_from_returns(res_sim["net_ret"])
        logs.append(f"Similarity metrics: {sim_perf}")

        logs.append("Step 6: Applying vol target.")
        vt = vol_target_returns(res_sim["net_ret"])
        vt_perf = perf_from_returns(vt)
        logs.append(f"Vol-target metrics: {vt_perf}")

        reply_parts.append("## Strategy Summary")
        reply_parts.append(f"- Baseline: `{base_perf}`")
        reply_parts.append(f"- Similarity: `{sim_perf}`")
        reply_parts.append(f"- Similarity + Vol Target: `{vt_perf}`")

        if isinstance(pred_df, pd.DataFrame) and not pred_df.empty:
            reply_parts.append("\n## Retrieved Similarity Rows")
            try:
                reply_parts.append(pred_df.head(5).to_markdown(index=False))
            except Exception:
                reply_parts.append(pred_df.head(5).to_string(index=False))

        final_reply = "\n".join(reply_parts)

    except Exception as e:
        final_reply = f"Error while running pipeline:\n\n{str(e)}"
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

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                height=500,
                label="Conversation",
                type="messages",
            )
            msg = gr.Textbox(
                label="Your message",
                placeholder="Type your message here...",
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
