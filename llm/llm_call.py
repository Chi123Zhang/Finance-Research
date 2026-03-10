from pathlib import Path
import subprocess

from llm.reasoning import extract_final_output


def run_local_llm(
    prompt: str,
    model_path: str,
    llama_cli_path: str = "llama.cpp/build/bin/llama-cli",
    work_dir: str = ".",
    context_size: int = 8192,
    ngl: int = 50,
    temperature: float = 0.0,
):
    work_dir = Path(work_dir)
    prompt_path = work_dir / "prompt.txt"
    analysis_path = work_dir / "analysis.md"

    prompt_path.write_text(prompt, encoding="utf-8")

    cmd = [
        str(llama_cli_path),
        "-m", str(model_path),
        "-f", str(prompt_path),
        "-c", str(context_size),
        "-ngl", str(ngl),
        "--temp", str(temperature),
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(work_dir),
    )

    raw_text = result.stdout + "\n" + result.stderr
    analysis_path.write_text(raw_text, encoding="utf-8")

    final_text = extract_final_output(raw_text)
    return {
        "raw_text": raw_text,
        "final_text": final_text,
        "returncode": result.returncode,
        "prompt_path": str(prompt_path),
        "analysis_path": str(analysis_path),
    }
