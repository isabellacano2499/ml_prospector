"""Debug Instagram search step by step."""
import sys, requests, re, time
sys.path.insert(0, ".")
from instagram.finder import _duckduckgo_search_handle, _fetch_ig_profile, make_session

sess = make_session()

# Step 1: Test DDG search directly
print("=== Step 1: DuckDuckGo search ===")
handle = _duckduckgo_search_handle("Lisa", "Munoz", "Texas", sess)
print(f"Handle from DDG: {handle!r}")

# Step 2: If handle found, test Instagram fetch
if handle:
    print(f"\n=== Step 2: Fetch Instagram profile @{handle} ===")
    time.sleep(2)
    profile = _fetch_ig_profile(handle, sess)
    print(f"Profile data: {profile}")
else:
    print("\nNo handle found. Testing manual DDG request...")
    resp = sess.get(
        "https://html.duckduckgo.com/html/",
        params={"q": 'site:instagram.com "Lisa Munoz" realtor Texas'},
        timeout=15,
    )
    print(f"DDG status: {resp.status_code}, length: {len(resp.text)}")
    handles = re.findall(r"instagram\.com/([A-Za-z0-9_.]{3,30})", resp.text)
    print(f"Raw handles: {handles[:10]}")
    print("\nFirst 500 chars:")
    print(resp.text[:500])

# Step 3: Test Instagram profile fetch for a known handle
print("\n=== Step 3: Test Instagram fetch for known handle ===")
time.sleep(2)
test_profile = _fetch_ig_profile("findyouraustin", sess)
print(f"findyouraustin profile: {test_profile}")
