"""Quick debug: test Google search + Instagram fetch for a known realtor."""
import sys, requests, re
sys.path.insert(0, '.')
from instagram.finder import find_instagram, make_session

sess = make_session()

# 1. Test raw Google search
url    = "https://www.google.com/search"
params = {"q": 'site:instagram.com "Lisa Munoz" realtor Texas', "num": 5, "hl": "en"}
resp   = sess.get(url, params=params, timeout=15)
print(f"Google status: {resp.status_code}, length: {len(resp.text)}")

if resp.status_code == 200:
    handles = re.findall(r'instagram\.com/([A-Za-z0-9_.]{3,30})(?:[/?"]|$)', resp.text)
    skip = {"p", "reel", "explore", "accounts", "stories", "tv", "reels", "about", "privacy"}
    clean = [h for h in handles if h.lower() not in skip and not h.startswith("_")]
    print(f"Handles found: {clean[:5]}")
    if not clean:
        print("No handles. First 1000 chars of response:")
        print(resp.text[:1000])
elif resp.status_code == 429:
    print("429 - Google rate limited")
else:
    print(f"HTTP {resp.status_code}")
    print(resp.text[:500])

# 2. Full Instagram lookup
print("\n--- Full find_instagram test ---")
result = find_instagram("Lisa", "Munoz", "Texas", sess)
print("Result:", result)
