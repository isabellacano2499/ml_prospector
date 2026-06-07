import pandas as pd
from config import CENSUS_OUTPUT, MIN_SCORE, MIN_POPULATION, STATE_ZIP_RANGES


def _derive_state(zip_code: int) -> str:
    z = str(zip_code).zfill(5)
    if z[:3] in ("006", "007", "009"):
        return "PR"
    prefix = int(z[:3])
    for state, ranges in STATE_ZIP_RANGES.items():
        if prefix in ranges:
            return state
    return "OTHER"


def load_target_zips(states: list[str]) -> pd.DataFrame:
    df = pd.read_csv(CENSUS_OUTPUT)
    df["state_derived"] = df["zip_code"].apply(_derive_state)
    df["zip_code"] = df["zip_code"].astype(str).str.zfill(5)

    filtered = df[
        (df["state_derived"].isin(states))
        & (df["overall_score"] >= MIN_SCORE)
        & (df["total_population"] >= MIN_POPULATION)
    ].copy()

    filtered = filtered.sort_values("overall_score", ascending=False)

    cols = [
        "zip_code", "state_derived", "total_population",
        "hispanic_pct", "overall_score", "first_home_buyer_score",
        "latino_market_score", "median_household_income",
        "spanish_home_pct", "employment_rate",
    ]
    return filtered[cols].reset_index(drop=True)
