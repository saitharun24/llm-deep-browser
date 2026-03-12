# llm-deep-browser

A local, GPU-accelerated deep research pipeline powered by local LLMs via Ollama. Given any question, it autonomously plans, searches the web, crawls pages, extracts content, and synthesizes a comprehensive answer — all running on your own hardware with no API keys or cloud dependencies.

---

## How It Works

The pipeline uses a **dual-phase ReAct architecture**:

1. **Parallel ReAct** — The smart LLM generates a multi-step execution plan. Independent steps run concurrently using `ThreadPoolExecutor`. Results from earlier steps are passed as inputs to dependent steps.
2. **Iterative ReAct (fallback)** — If planning fails, the agent falls back to a classic step-by-step ReAct loop with tool selection, observation, and retry logic.

Both phases synthesize a final answer using only observed data — never from LLM training memory.

```
Question
   │
   ▼
[Plan Generation] ──────────────────────────────────────────────┐
   │                                                             │
   ▼                                                          fallback
[Parallel Tool Execution]                                        │
   ├── current_datetime                                          ▼
   ├── web_search ──► search ──► rank ──► crawl ──► extract   [Iterative ReAct Loop]
   ├── stock_price                                               │
   └── llm_knowledge                                            │
   │                                                             │
   └──────────────────────────┬──────────────────────────────────┘
                               ▼
                      [Synthesis with smart LLM]
                               │
                               ▼
                           Final Answer
```

---

## Features

- **No API keys** — fully local via Ollama
- **Dual-model setup** — fast model for cheap tasks, smart model for reasoning and synthesis
- **Parallel execution** — independent research steps run simultaneously
- **Multi-signal URL ranking** — TF-IDF similarity, domain authority, recency, keyword density
- **Robust extraction** — Trafilatura with BeautifulSoup fallback for JS-heavy pages
- **Headless browser fallback** — Playwright kicks in when requests are blocked
- **NSE stock data** — live Indian stock prices with symbol auto-correction
- **Repeat detection** — prevents the agent from calling the same tool twice
- **Placeholder detection** — forces retry if LLM hallucinates `[insert ...]` style answers

---

## Project Structure

```
llm-deep-browser/
├── main.py                     # Entry point
├── config.py                   # (reserved for future config)
├── requirements.txt
│
├── pipeline/
│   ├── research_agent.py       # Web research orchestrator
│   ├── aggregator.py           # Final answer synthesis
│   └── chunker.py              # Text chunking with overlap
│
── prompts/
│   ├── prompts.py              # Contains all the prompts

└── tools/
    ├── react.py                # ReAct agent — parallel + iterative
    ├── search.py               # DuckDuckGo search
    ├── ranker.py               # Multi-signal URL ranker
    ├── crawler.py              # HTTP + headless page fetcher
    ├── extractor.py            # HTML content extractor
    ├── summarizer.py           # Per-chunk LLM summarizer
    └── getstockprice.py        # NSE India live stock prices
```

---

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) installed and running
- NVIDIA GPU recommended (Ollama uses GPU automatically if CUDA is available)

### Install dependencies

```bash
pip install requests playwright beautifulsoup4 trafilatura duckduckgo-search scikit-learn numpy langchain-core langchain-ollama
playwright install chromium
```

---

## Setup

### 1. Pull models

```bash
ollama pull gemma3:4b       # fast model
ollama pull qwen3.5:9b      # smart model (or any capable model)
```

### 2. Run two Ollama instances (for true parallelism)

**Terminal 1 — fast model (port 11434):**
```bash
ollama serve
```

**Terminal 2 — smart model (port 11435):**
```bash
# Windows
set OLLAMA_HOST=0.0.0.0:11435 && ollama serve

# Linux/macOS
OLLAMA_HOST=0.0.0.0:11435 ollama serve
```
*Note: It is not necessary but good to have, if not ollama will resort to using the models sequentially either by loading both (in case RAM supports) else loads each one by one when being called.*

### 3. Run a query

```bash
python main.py "What are the reasons for NIFTY's decline today?"
```

---

## Configuration

Edit the model names and ports at the top of `main.py`:

```python
FAST_MODEL  = "gemma3:4b"    # used for: summarization, input validation, query enrichment
SMART_MODEL = "qwen3.5:9b"   # used for: planning, reasoning, final synthesis

fast_llm  = get_llm(FAST_MODEL,  temperature=0.3, port=11434)
smart_llm = get_llm(SMART_MODEL, temperature=0.7, port=11435)
```

### Model assignment rationale

| Task | Model | Reason |
|---|---|---|
| Chunk summarization | `fast_llm` | High volume, simple task |
| Input validation | `fast_llm` | Structured JSON, low complexity |
| Query enrichment | `fast_llm` | Short text transformation |
| Execution planning | `smart_llm` | Multi-step reasoning |
| Tool selection | `smart_llm` | Nuanced decision making |
| Final synthesis | `smart_llm` | Quality matters most here |

---

## Tools

| Tool | Description |
|---|---|
| `current_datetime` | Returns current date and time — never guessed |
| `web_search` | DuckDuckGo search → rank → crawl → extract → summarize |
| `llm_knowledge` | Uses LLM's own knowledge for timeless facts (math, history, definitions) |
| `stock_price` | Live NSE India stock data with symbol auto-correction via difflib |

---

## GPU Acceleration

Ollama uses your GPU automatically if CUDA drivers are installed. Verify:

```bash
nvidia-smi        # confirm GPU is visible
ollama ps         # shows VRAM usage when a model is loaded
```

If Ollama falls back to CPU, set before launching:

```bash
set CUDA_VISIBLE_DEVICES=0
set OLLAMA_GPU_LAYERS=999
ollama serve
```

Monitor VRAM usage in real time:

```bash
nvidia-smi -l 1
```

---

## Integrating with Open WebUI

`main.py` exposes a `browse(question, fast_llm, smart_llm)` function that can be wired into any Open WebUI pipeline. Import and call it from your pipeline handler:

```python
from main import browse, get_llm

fast_llm  = get_llm("gemma3:4b",  temperature=0.3, port=11434)
smart_llm = get_llm("qwen2.5:7b", temperature=0.7, port=11435)

answer = browse("What is the weather in Chennai today?", fast_llm, smart_llm)
```

*Disclaimer:- Segment not yet tested*

---

## Troubleshooting

**Ollama not using GPU**
Run `nvidia-smi` to confirm drivers are installed. Set `OLLAMA_GPU_LAYERS=999` before `ollama serve`.

**Playwright not found**
Run `playwright install chromium` after pip install.

**NSE stock prices returning errors**
NSE requires a session cookie from the homepage before API calls. This is handled automatically — if it fails, it's likely a temporary NSE rate limit. Retry after a few seconds.

**Empty or short answers**
Increase `chunk_size` in `chunker.py` (default 1500 words) or raise the summary count threshold in `research_agent.py` (default 6 summaries).

---

## Developed By
A. Sai Tharun
