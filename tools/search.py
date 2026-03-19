from ddgs import DDGS

def search_web(query, max_results=20):
    if not query or not query.strip():
        raise ValueError("query is mandatory")
    try:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query.strip(), max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                })

        # Filter out results with missing URLs
        results = [r for r in results if r["url"]]

        if not results:
            print(f"[WARN] No search results for query: '{query}'")

        print(f"[Search] {len(results)} results for: '{query}'")
        return results

    except Exception as e:
        print(f"[WARN] Search failed for '{query}': {e}")
        return []