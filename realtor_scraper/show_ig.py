import sys, pandas as pd
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")

# Read the most recent output file
output_dir = Path("output")
files = sorted(output_dir.glob("mmi_enriched_*.csv"), key=lambda p: p.stat().st_mtime)
latest = files[-1]
print(f"Reading: {latest.name}\n")

df = pd.read_csv(latest)
cols = ["full_name","ig_handle","ig_followers","ig_posts","ig_bio","ig_spanish_signals","ig_realtor_signals","zillow_url","email_zillow","phone_zillow","zillow_total_sales"]
available = [c for c in cols if c in df.columns]
print(df[available].to_string())
