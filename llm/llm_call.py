from pathlib import Path
from huggingface_hub import hf_hub_download
from llama_cpp import Llama


def run_local_llm(
    prompt: str,
    context_size: int = 8192,
    max_tokens: int = 512,
    temperature: float = 0.2,
):
    model_path = hf_hub_download(
        repo_id="bartowski/Qwen2.5-7B-Instruct-GGUF",
        filename="Qwen2.5-7B-Instruct-Q4_K_M.gguf",
    )

    llm = Llama(
        model_path=model_path,
        n_ctx=context_size,
        n_gpu_layers=-1,   # Apple Silicon 尽量上 Metal
        verbose=False,
    )

    output = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    text = output["choices"][0]["text"]

    work_dir = Path(".")
    prompt_path = work_dir / "prompt.txt"
    analysis_path = work_dir / "analysis.md"

    prompt_path.write_text(prompt, encoding="utf-8")
    analysis_path.write_text(text, encoding="utf-8")

    return {
        "raw_text": text,
        "final_text": text,
        "returncode": 0,
        "prompt_path": str(prompt_path),
        "analysis_path": str(analysis_path),
        "model_path": str(model_path),
    }
