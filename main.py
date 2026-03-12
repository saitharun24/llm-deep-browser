import sys
import os
from langchain_ollama import ChatOllama
from tools.react import run_research

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["OLLAMA_GPU_LAYERS"] = "999"

FAST_MODEL = "gemma3:4b"  # plan generation, summarization, simple checks
SMART_MODEL = "qwen3.5:9b"  # final synthesis, complex reasoning

def browse(question, fast_llm, smart_llm):
    answer = run_research(fast_llm, smart_llm, question)
    # print(f"Total tokens used: {answer[1]}")
    return answer

def get_llm(model: str, temperature: float = 0.7, port: int = 11434):
    return ChatOllama(
        model=model,
        base_url=f"http://localhost:{port}",
        temperature=temperature,
        num_ctx=8192,       # ← explicit context window
        num_predict=2048,   # ← max tokens to generate
        reasoning=False,
    )

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py '<question>'")
        sys.exit(1)

    fast_llm = get_llm(FAST_MODEL, temperature=0.3, port=11434)  # ← low temp for structured tasks
    smart_llm = get_llm(SMART_MODEL, temperature=0.7, port=11434)  # ← separate instance
    print(browse(sys.argv[1], fast_llm, smart_llm))


# Use this incase you are choosing to run both the models in seperate ports
# # Terminal 1 — fast model
# ollama serve
#
# # Terminal 2 — smart model
# set OLLAMA_HOST = 0.0.0.0: 11435 & & ollama serve