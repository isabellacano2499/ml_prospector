"""
Language feature engineering — Spanish-speaker signals for marketing targeting.
"""

import numpy as np
import pandas as pd

from src.features.demographic import _safe_div


def add_language_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    lang_total = df.get("language_households_total", pd.Series(np.nan, index=df.index))
    spanish_well = df.get("spanish_speak_english_well", pd.Series(np.nan, index=df.index))
    spanish_not_well = df.get("spanish_speak_english_not_well", pd.Series(np.nan, index=df.index))

    # All Spanish-speaking households
    df["spanish_households"] = spanish_well.fillna(0) + spanish_not_well.fillna(0)
    df["spanish_home_pct"] = _safe_div(df["spanish_households"], lang_total)

    # Limited English proficiency (LEP) — primary Spanish-language marketing target
    df["lep_spanish_pct"] = _safe_div(spanish_not_well, lang_total)

    # English-comfortable Spanish speakers — bilingual segment
    df["bilingual_spanish_pct"] = _safe_div(spanish_well, lang_total)

    # Spanish marketing opportunity:
    # high LEP = high need for Spanish-language loan officers
    # weighted: LEP households matter more than bilingual for outreach urgency
    df["spanish_marketing_opportunity"] = (
        df["lep_spanish_pct"].fillna(0) * 0.65
        + df["bilingual_spanish_pct"].fillna(0) * 0.35
    )

    return df
