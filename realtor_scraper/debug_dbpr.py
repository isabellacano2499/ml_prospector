import sys
import requests
from bs4 import BeautifulSoup

sys.stdout.reconfigure(encoding="utf-8")

s = requests.Session()
s.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})

# === FLORIDA: Check DBPR instant public records page ===
print("=== DBPR Instant Public Records ===")
r = s.get("https://www2.myfloridalicense.com/instant-public-records/", timeout=30)
print(f"Status: {r.status_code}")
soup = BeautifulSoup(r.text, "lxml")
# Look for download links
for a in soup.find_all("a", href=True):
    href = a["href"]
    text = a.get_text(strip=True)
    if any(k in text.lower() or k in href.lower() for k in
           ("real estate", "download", "csv", "excel", "zip", ".txt", "license")):
        print(f"  {repr(text):60s}  {href}")

# Also print all links
print("\nAll links:")
for a in soup.find_all("a", href=True):
    text = a.get_text(strip=True)
    if text:
        print(f"  {repr(text):50s}  {a['href']}")

# === TEXAS: Try Texas Open Data Portal ===
print("\n\n=== Texas Open Data Portal (TREC licensees) ===")
# The Socrata API for Texas open data
r2 = s.get(
    "https://data.texas.gov/api/views/metadata/v1",
    params={"q": "TREC real estate license"},
    timeout=20
)
print(f"Status: {r2.status_code}")
if r2.ok:
    print(r2.text[:2000])
