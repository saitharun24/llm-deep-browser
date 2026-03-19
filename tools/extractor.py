import trafilatura
from trafilatura.settings import use_config

# Custom config to extract more content
config = use_config()
config.set("DEFAULT", "MIN_EXTRACTED_SIZE", "100")          # lower minimum content size
config.set("DEFAULT", "MIN_OUTPUT_SIZE", "100")             # lower minimum output size
config.set("DEFAULT", "MIN_OUTPUT_COMM_SIZE", "100")        # lower comment size threshold
config.set("DEFAULT", "MIN_EXTRACTED_COMM_SIZE", "100")     # lower extracted comment size
config.set("DEFAULT", "MIN_DUPLCHECK_SIZE", "100")          # lower duplicate check size
config.set("DEFAULT", "MAX_REPETITIONS", "5")               # allow more repetitions

def extract_content(html: str) -> str | None:
    if not html:
        return None

     # ← Guard against URL strings being passed instead of HTML
    stripped = html.strip()
    if stripped.startswith("http://") or stripped.startswith("https://"):
        print(f"[WARN] extract_content received a URL instead of HTML: {stripped[:100]}")
        return None

    # ← Guard against non-HTML content
    if "<" not in stripped:
        print(f"[WARN] extract_content received non-HTML content: {stripped[:100]}")
        return None

    # ── Attempt 1: Trafilatura with relaxed settings ───────────────────────────
    result = trafilatura.extract(
        html,
        config=config,
        include_comments=False,
        include_tables=True,
        include_links=False,
        include_images=False,
        no_fallback=False,          # ← allow fallback extractors
        favor_recall=True,          # ← extract more content, sacrifice precision
        deduplicate=True,
        output_format="txt",
    )

    if result and len(result.strip()) > 200:
        return result

    # ── Attempt 2: Trafilatura with bare minimum filtering ────────────────────
    print("[Extract] Attempt 1 too short, trying bare extraction...")
    result = trafilatura.extract(
        html,
        config=config,
        include_comments=False,
        include_tables=True,
        include_links=False,
        no_fallback=False,
        favor_recall=True,
        deduplicate=False,
        output_format="txt",
    )

    if result and len(result.strip()) > 200:
        print(result)
        return result

    # ── Attempt 3: BeautifulSoup fallback for heavily structured pages ─────────
    print("[Extract] Trafilatura failed, falling back to BeautifulSoup...")
    return extract_with_bs4(html)


def extract_with_bs4(html: str) -> str | None:
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")

        # Remove noise tags
        for tag in soup(["script", "style", "nav", "footer", "header",
                         "aside", "advertisement", "iframe", "noscript",
                         "form", "button", "input", "meta", "link"]):
            tag.decompose()

        # ── Use only leaf-level content tags to avoid duplicate text ──────────
        # e.g. <section><p>text</p></section> — only extract <p>, not <section>
        BLOCK_TAGS = {"p", "h1", "h2", "h3", "h4", "h5", "h6","li", "td", "th", "blockquote", "pre", "code"}

        seen = set()  # ← deduplicate identical lines
        lines = []

        for tag in soup.find_all(BLOCK_TAGS):
            # Skip if this tag is nested inside another content tag
            if any(parent.name in BLOCK_TAGS for parent in tag.parents):
                continue

            text = tag.get_text(separator=" ", strip=True)
            if len(text) > 30 and text not in seen:  # ← skip short + duplicate
                seen.add(text)
                lines.append(text)

        if not lines:
            # Last resort: raw text dump
            raw = soup.get_text("\n", True)
            return raw if len(raw) > 200 else None

        result = "\n\n".join(lines)
        return result if len(result) > 200 else None

    except Exception as e:
        print(f"[WARN] BeautifulSoup extraction failed: {e}")
        return None