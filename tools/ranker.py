from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from urllib.parse import urlparse
import re

# ─── Signal Weights ────────────────────────────────────────────────────────────
WEIGHTS = {
    "tfidf_similarity":  0.30,  # ← reduced slightly
    "domain_authority":  0.25,  # ← increased, more domains now
    "url_cleanliness":   0.10,
    "snippet_length":    0.05,  # ← reduced
    "keyword_density":   0.15,
    "recency":           0.15,  # ← increased for news
}

# ─── High/Low Authority Domains ───────────────────────────────────────────────
HIGH_AUTHORITY_DOMAINS = {
    # ── Tech & General News ────────────────────────────────────────────────────
    "wikipedia.org", "github.com", "stackoverflow.com",
    "docs.python.org", "arxiv.org", "nature.com",
    "techcrunch.com", "wired.com", "theverge.com",
    "towardsdatascience.com", "medium.com", "dev.to",
    "arstechnica.com", "engadget.com", "zdnet.com",
    "thenextweb.com", "venturebeat.com", "mashable.com",
    "cnet.com", "gizmodo.com", "digitaltrends.com",

    # ── International News ─────────────────────────────────────────────────────
    "reuters.com", "bbc.com", "bbc.co.uk",
    "apnews.com", "bloomberg.com", "ft.com",
    "theguardian.com", "nytimes.com", "washingtonpost.com",
    "wsj.com", "economist.com", "foreignpolicy.com",
    "aljazeera.com", "dw.com", "france24.com",
    "euronews.com", "abc.net.au", "cbc.ca",

    # ── Indian News ────────────────────────────────────────────────────────────
    "thehindu.com", "hindustantimes.com", "ndtv.com",
    "timesofindia.com", "indianexpress.com", "livemint.com",
    "businessstandard.com", "financialexpress.com",
    "economictimes.indiatimes.com", "indiatoday.in",
    "news18.com", "firstpost.com", "scroll.in",
    "thewire.in", "theprint.in", "outlookindia.com",
    "dnaindia.com", "deccanherald.com", "tribuneindia.com",

    # ── Finance & Markets ──────────────────────────────────────────────────────
    "moneycontrol.com", "nseindia.com", "bseindia.com",
    "investing.com", "marketwatch.com", "cnbc.com",
    "forbes.com", "fortune.com", "barrons.com",
    "seekingalpha.com", "fool.com", "zerodha.com",
    "groww.in", "tickertape.in", "screener.in",

    # ── Product & Hardware Reviews ─────────────────────────────────────────────
    "amazon.com", "amazon.in", "flipkart.com",
    "tomshardware.com", "pcmag.com", "rtings.com",
    "gsmarena.com", "anandtech.com", "notebookcheck.net",
    "91mobiles.com", "gadgets360.com", "digit.in",

    # ── Government & Official ──────────────────────────────────────────────────
    "gov.in", "nic.in", "rbi.org.in",
    "sebi.gov.in", "mca.gov.in", "incometax.gov.in",
    "who.int", "un.org", "worldbank.org",
    "imf.org", "oecd.org",

    # ── Science & Research ─────────────────────────────────────────────────────
    "nature.com", "science.org", "pubmed.ncbi.nlm.nih.gov",
    "scholar.google.com", "researchgate.net", "ieee.org",
    "acm.org", "springer.com", "sciencedirect.com",
}

LOW_AUTHORITY_DOMAINS = {
    "pinterest.com", "quora.com", "yahoo.com",
    "ask.com", "answers.com",
}

RECENCY_SIGNALS = [
    r"\b202[3-9]\b",            # recent years
    r"\b(today|latest|new|updated|current|now)\b",
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b",
]

def get_domain(url: str) -> str:
    """Extract clean domain from URL."""
    try:
        return urlparse(url).netloc.lower().replace("www.", "")
    except Exception:
        return ""

def score_domain_authority(url: str) -> float:
    domain = get_domain(url)
    if not domain:
        return 0.5

    if domain in HIGH_AUTHORITY_DOMAINS:
        return 1.0
    if domain in LOW_AUTHORITY_DOMAINS:
        return 0.1

    parts = domain.split(".")
    for i in range(1, len(parts)):
        parent = ".".join(parts[i:])
        if parent in HIGH_AUTHORITY_DOMAINS:
            return 0.9
        if parent in LOW_AUTHORITY_DOMAINS:
            return 0.1

    if domain.endswith((".edu", ".gov")):
        return 0.85
    if domain.endswith(".org"):
        return 0.7

    return 0.5

def score_url_cleanliness(url: str) -> float:
    try:
        parsed = urlparse(url)
        path = parsed.path

        # Penalize tracking params, session ids, excessive depth
        penalties = 0
        if len(parsed.query) > 50: penalties += 0.3  # long query strings
        if path.count("/") > 5: penalties += 0.2  # too deep
        if re.search(r"\d{8,}", path): penalties += 0.2  # long numeric IDs
        if re.search(r"(session|tracking|utm)", parsed.query): penalties += 0.2
        return max(0.0, 1.0 - penalties)
    except Exception:
        return 0.5


def score_snippet_length(snippet: str) -> float:
    length = len(snippet.strip())
    if length == 0:
        return 0.0
    return min(1.0, length / 200)


def score_keyword_density(query: str, title: str, snippet: str) -> float:
    query_words = set(re.findall(r'\b\w+\b', query.lower()))
    if not query_words:
        return 0.0
    text = (title + " " + snippet).lower()
    text_words = re.findall(r'\b\w+\b', text)
    if not text_words:
        return 0.0
    matches = sum(1 for w in text_words if w in query_words)
    return min(1.0, matches / len(query_words))


def score_recency(snippet: str) -> float:
    snippet_lower = snippet.lower()
    matches = sum(1 for p in RECENCY_SIGNALS if re.search(p, snippet_lower))
    return min(1.0, matches / len(RECENCY_SIGNALS))


def rank_urls(query: str, results: list) -> list:
    if not results:
        return []

    # ── TF-IDF similarity ──────────────────────────────────────────────────────
    docs = [query] + [r["title"] + " " + r["snippet"] for r in results]
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(docs)
    tfidf_scores = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()

    # ── Combine all signals ────────────────────────────────────────────────────
    scored = []
    for i, r in enumerate(results):
        url     = r.get("url", "")
        title   = r.get("title", "")
        snippet = r.get("snippet", "")

        signals = {
            "tfidf_similarity": float(tfidf_scores[i]),
            "domain_authority": score_domain_authority(url),
            "url_cleanliness":  score_url_cleanliness(url),
            "snippet_length":   score_snippet_length(snippet),
            "keyword_density":  score_keyword_density(query, title, snippet),
            "recency":          score_recency(snippet),
        }

        # Weighted sum
        final_score = sum(
            signals[signal] * weight
            for signal, weight in WEIGHTS.items()
        )

        scored.append((r, final_score))

    # ── Sort by final score ────────────────────────────────────────────────────
    scored.sort(key=lambda x: x[1], reverse=True)

    print(f"[Rank] Top results for: '{query}'")
    for r, score in scored[:8]:
        print(f"  {score:.3f} — {r.get('url', '')[:80]}")

    return [r for r, _ in scored]