from tools.search import search_web
from tools.ranker import rank_urls
from tools.crawler import fetch_html
from tools.extractor import extract_content
from tools.summarizer import summarize_chunk
from pipeline.chunker import chunk_text
from pipeline.aggregator import synthesize_answer
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

def run_web_research(fast_llm, smart_llm, question):
    results = search_web(question)
    if not results:
        return "No search results found."

    ranked = rank_urls(question, results)
    urls_to_fetch = ranked[:8]

    summaries = []
    summaries_lock = threading.Lock()
    done = threading.Event()

    def process_and_summarize(r):
        url = r.get("url", "")
        try:
            # Step 1: Fetch
            html = fetch_html(url)
            if not html or done.is_set(): return

            # Step 2: Extract
            text = extract_content(html)
            if not text or done.is_set(): return

            # Step 3: Chunk
            chunks = chunk_text(text, 4)
            if not chunks or done.is_set(): return

            # Step 4: Summarize each chunk immediately
            with ThreadPoolExecutor(max_workers=len(chunks)) as chunk_executor:
                chunk_futures = {
                    chunk_executor.submit(summarize_chunk, fast_llm, c): c
                    for c in chunks
                }
                for future in as_completed(chunk_futures):
                    if done.is_set():
                        chunk_executor.shutdown(wait=False, cancel_futures=True)
                        return
                    try:
                        result = future.result()
                        if result:
                            with summaries_lock:
                                summaries.append(result)
                                enough = len(summaries) >= 8  # ← check inside lock
                            if enough:                          # ← act outside lock
                                done.set()
                                chunk_executor.shutdown(wait=False, cancel_futures=True)
                                return
                    except Exception as e:
                        print(f"[WARN] Chunk summarization failed: {e}")

        except Exception as e:
            print(f"[WARN] Skipping {r['url']}: {e}")

    with ThreadPoolExecutor(max_workers=len(urls_to_fetch)) as executor:
        futures = {
            executor.submit(process_and_summarize, r): r
            for r in ranked[:8]
        }
        for future in as_completed(futures):
            if done.is_set():
                executor.shutdown(wait=False, cancel_futures=True)
                break
            try:
                future.result()
            except Exception as e:
                print(f"[WARN] URL processing failed: {e}")

    if not summaries:
        print(f"[WARN] No summaries extracted for: {question}")
        return ["No results found."]

    print(f"[INFO] Synthesizing {len(summaries)} summaries for: {question}")
    return synthesize_answer(smart_llm, summaries, question)