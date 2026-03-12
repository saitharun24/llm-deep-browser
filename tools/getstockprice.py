import requests
import difflib
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
    "Connection": "keep-alive",
}


def create_nse_session() -> requests.Session:
    """Create a session with NSE cookies initialized."""
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retry))

    try:
        # Hit homepage once to get session cookies
        session.get("https://www.nseindia.com", headers=NSE_HEADERS, timeout=10)
    except Exception as e:
        print(f"[WARN] NSE session init failed: {e}")

    return session


def lookup_symbol(session: requests.Session, symbol: str) -> tuple[str, bool]:
    """
    Verify symbol exists on NSE using existing session.
    Returns (correct_symbol, was_corrected).
    """
    try:
        search_url = f"https://www.nseindia.com/api/search/autocomplete?q={symbol}"
        response = session.get(search_url, headers=NSE_HEADERS, timeout=10)

        if response.status_code != 200:
            return symbol, False

        data = response.json()
        symbols = data.get("symbols", [])

        if not symbols:
            return symbol, False

        candidates = [s.get("symbol", "").upper() for s in symbols if s.get("symbol")]
        if not candidates:
            return symbol, False

        # Exact match
        if symbol.upper() in candidates:
            return symbol.upper(), False

        # Closest match
        matches = difflib.get_close_matches(symbol.upper(), candidates, n=1, cutoff=0.4)
        if matches:
            return matches[0], True

        # Fallback to first result
        return candidates[0], True

    except Exception as e:
        print(f"[WARN] Symbol lookup failed: {e}")
        return symbol, False


def tool_stock_price(input: str) -> str:
    """Fetch stock price from NSE India API with symbol validation."""
    symbol = input.strip().upper()

    # ── Single session for both lookup and fetch ───────────────────────────────
    session = create_nse_session()

    # ── Step 1: Validate / correct symbol ─────────────────────────────────────
    corrected_symbol, was_corrected = lookup_symbol(session, symbol)

    if was_corrected:
        print(f"[INFO] Symbol '{symbol}' corrected to '{corrected_symbol}'")
        correction_note = f"(Note: '{symbol}' was corrected to '{corrected_symbol}')\n"
        symbol = corrected_symbol
    else:
        correction_note = ""

    # ── Step 2: Fetch stock data using same session ────────────────────────────
    try:
        url = f"https://www.nseindia.com/api/quote-equity?symbol={symbol}"
        response = session.get(url, headers=NSE_HEADERS, timeout=10)

        if response.status_code != 200:
            return f"Failed to fetch data for {symbol}. Status: {response.status_code}"

        data = response.json()

        info       = data.get("info", {})
        price_info = data.get("priceInfo", {})
        meta       = data.get("metadata", {})

        name       = info.get("companyName", symbol)
        last_price = price_info.get("lastPrice", "N/A")
        change     = price_info.get("change", "N/A")
        pct_change = price_info.get("pChange", "N/A")
        open_price = price_info.get("open", "N/A")
        close      = price_info.get("previousClose", "N/A")
        day_high   = price_info.get("intraDayHighLow", {}).get("max", "N/A")
        day_low    = price_info.get("intraDayHighLow", {}).get("min", "N/A")
        week_high  = price_info.get("weekHighLow", {}).get("max", "N/A")
        week_low   = price_info.get("weekHighLow", {}).get("min", "N/A")
        series     = meta.get("series", "N/A")

        return (
            f"{correction_note}"
            f"Stock: {name} ({symbol}) | Series: {series}\n"
            f"Last Price:       ₹{last_price}\n"
            f"Change:           ₹{change} ({pct_change}%)\n"
            f"Open:             ₹{open_price}\n"
            f"Previous Close:   ₹{close}\n"
            f"Day High/Low:     ₹{day_high} / ₹{day_low}\n"
            f"52-Week High/Low: ₹{week_high} / ₹{week_low}\n"
        )

    except requests.exceptions.Timeout:
        return f"Request timed out for symbol: {symbol}"
    except requests.exceptions.ConnectionError:
        return f"Could not connect to NSE API for symbol: {symbol}"
    except Exception as e:
        return f"Error fetching stock data for {symbol}: {e}"