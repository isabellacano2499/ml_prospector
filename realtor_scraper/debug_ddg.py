"""Test DuckDuckGo HTML search for Instagram handles."""
import requests, re

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}
resp = requests.get(
    "https://html.duckduckgo.com/html/",
    params={"q": "site:instagram.com Lisa Munoz realtor Texas"},
    headers=headers,
    timeout=15,
)
print("DDG status:", resp.status_code, "length:", len(resp.text))
handles = re.findall(r"instagram\.com/([A-Za-z0-9_.]{3,30})", resp.text)
skip = {"p", "reel", "explore", "accounts", "stories", "tv", "reels", "about", "privacy", "legal", "help"}
clean = [h for h in handles if h.lower() not in skip and not h.startswith("_")]
print("Handles found:", clean[:10])
print()
print(resp.text[:1000])
