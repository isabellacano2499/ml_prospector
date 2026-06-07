from pathlib import Path

CENSUS_OUTPUT = Path(__file__).parent.parent / "latino_re_engine" / "data" / "output" / "latino_market_zip_2024.csv"

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

MIN_SCORE = 40
MIN_POPULATION = 1000
MAX_AGENTS_PER_ZIP = 25

DELAY_MIN = 3.0
DELAY_MAX = 7.0

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,es;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
}

STATE_ZIP_RANGES = {
    "TX": list(range(750, 800)),
    "FL": list(range(320, 350)),
    "CA": list(range(900, 962)),
    "AZ": list(range(850, 866)),
    "NM": list(range(870, 885)),
    "NY": list(range(100, 150)),
    "NJ": list(range(70, 90)),
}
