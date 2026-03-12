import requests
import random
import time
from playwright.sync_api import sync_playwright, ViewportSize
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_3_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3.1 Safari/605.1.15",
]

BROWSER_HEADERS = {
    "Accept":                    "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language":           "en-US,en;q=0.9",
    "Accept-Encoding":           "gzip, deflate, br",
    "Connection":                "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest":            "document",
    "Sec-Fetch-Mode":            "navigate",
    "Sec-Fetch-Site":            "none",
    "Sec-Fetch-User":            "?1",
    "Cache-Control":             "max-age=0",
}

# Status codes that are worth retrying
RETRY_STATUS_CODES = [429, 500, 502, 503, 504]

# Status codes that mean we're blocked — no point retrying, go headless
BLOCKED_STATUS_CODES = [400, 401, 403, 406, 407, 429]

# Block resource types that waste bandwidth and add no text content
BLOCKED_RESOURCE_TYPES = ["image", "font", "media", "stylesheet"]
BLOCKED_EXTENSIONS = "**/*.{png,jpg,jpeg,gif,svg,ico,woff,woff2,ttf,mp4,webp,css}"


def fetch_html_requests(url: str) -> str | None:
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=RETRY_STATUS_CODES,
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.mount("http://",  HTTPAdapter(max_retries=retry))

    headers = {
        **BROWSER_HEADERS,
        "User-Agent": random.choice(USER_AGENTS),  # ← rotate per request
    }

    try:
        response = session.get(url, headers=headers, timeout=10, allow_redirects=True)

        if response.status_code in BLOCKED_STATUS_CODES:
            print(f"[WARN] {response.status_code} blocked by {url}, trying headless...")
            return None

        response.raise_for_status()
        return response.text

    except requests.exceptions.Timeout:
        print(f"[WARN] Timeout: {url}")
        return None
    except requests.exceptions.ConnectionError:
        print(f"[WARN] Connection error: {url}")
        return None
    except Exception as e:
        print(f"[WARN] Request failed {url}: {e}")
        return None


def fetch_html_headless(url: str) -> str | None:
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent=random.choice(USER_AGENTS),  # ← rotate here too
                viewport=ViewportSize(width=1920, height=1080),
                ignore_https_errors=True,
                extra_http_headers={
                    "Accept-Language": "en-US,en;q=0.9",
                }
            )
            page = context.new_page()

            # Block images/fonts/media to speed up loading
            page.route(BLOCKED_EXTENSIONS,lambda route: route.abort())

            try:
                page.goto(url, timeout=20000, wait_until="domcontentloaded")
                return page.content()
            except Exception as e:
                print(f"[WARN] Headless failed {url}: {e}")
                return None
            finally:
                context.close()
                browser.close()

    except Exception as e:
        print(f"[WARN] Playwright error {url}: {e}")
        return None


def fetch_html(url: str) -> str | None:
    # Try fast requests first, fall back to headless if blocked
    html = fetch_html_requests(url)
    if not html:
        print(f"[INFO] Falling back to headless for {url}")
        html = fetch_html_headless(url)

    # ← Sanity check — make sure we got actual HTML back
    if html and "<" not in html:
        print(f"[WARN] fetch_html returned non-HTML for {url}: {html[:100]}")
        return None
    return html