from huggingface_hub import hf_hub_download
from llama_cpp import Llama


def run_local_llm(prompt):

    model_path = hf_hub_download(
        repo_id="bartowski/Qwen2.5-7B-Instruct-GGUF",
        filename="Qwen2.5-7B-Instruct-Q4_K_M.gguf"
    )

    llm = Llama(
        model_path=model_path,
        n_ctx=8192,
        n_gpu_layers=-1
    )

    output = llm(
        prompt,
        max_tokens=512,
        temperature=0.2
    )

    text = output["choices"][0]["text"]

    return {
        "raw_text": text,
        "final_text": text,
        "returncode": 0,
        "model_path": model_path
    }
