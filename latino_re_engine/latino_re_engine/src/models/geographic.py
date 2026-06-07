"""
Pydantic models for validated Census records.
These are used for row-level validation after raw API data is cleaned.
"""

from typing import Optional
from pydantic import BaseModel, Field, field_validator


class GeoRecord(BaseModel):
    """Base geographic identifier fields shared by all levels."""

    name: Optional[str] = None
    state: Optional[str] = None
    state_fips: Optional[str] = None


class ZIPRecord(GeoRecord):
    zip_code: str = Field(..., min_length=5, max_length=5)
    county: Optional[str] = None

    # Core demographics
    total_population: Optional[int] = None
    hispanic_population: Optional[int] = None
    male_total: Optional[int] = None
    female_total: Optional[int] = None
    hispanic_male: Optional[int] = None
    hispanic_female: Optional[int] = None

    # Age — male
    male_18_19: Optional[int] = None
    male_20: Optional[int] = None
    male_21: Optional[int] = None
    male_22_24: Optional[int] = None
    male_25_29: Optional[int] = None
    male_30_34: Optional[int] = None
    male_35_39: Optional[int] = None
    male_40_44: Optional[int] = None
    male_45_49: Optional[int] = None
    male_50_54: Optional[int] = None

    # Age — female
    female_18_19: Optional[int] = None
    female_20: Optional[int] = None
    female_21: Optional[int] = None
    female_22_24: Optional[int] = None
    female_25_29: Optional[int] = None
    female_30_34: Optional[int] = None
    female_35_39: Optional[int] = None
    female_40_44: Optional[int] = None
    female_45_49: Optional[int] = None
    female_50_54: Optional[int] = None

    # Citizenship
    citizenship_total: Optional[int] = None
    naturalized_citizens: Optional[int] = None
    non_citizens: Optional[int] = None

    # Marital status
    marital_total: Optional[int] = None
    male_never_married: Optional[int] = None
    male_married: Optional[int] = None
    male_separated: Optional[int] = None
    male_divorced: Optional[int] = None
    female_never_married: Optional[int] = None
    female_married: Optional[int] = None
    female_separated: Optional[int] = None
    female_divorced: Optional[int] = None

    # Education
    education_total: Optional[int] = None
    hs_diploma: Optional[int] = None
    ged: Optional[int] = None
    some_college_lt1yr: Optional[int] = None
    some_college_gt1yr: Optional[int] = None
    associates_degree: Optional[int] = None
    bachelors_degree: Optional[int] = None
    masters_degree: Optional[int] = None
    professional_degree: Optional[int] = None
    doctorate_degree: Optional[int] = None

    # Income
    median_household_income: Optional[float] = None
    hispanic_median_income: Optional[float] = None
    income_lt10k: Optional[int] = None
    income_10_15k: Optional[int] = None
    income_15_20k: Optional[int] = None
    income_20_25k: Optional[int] = None
    income_25_30k: Optional[int] = None
    income_30_35k: Optional[int] = None
    income_35_40k: Optional[int] = None
    income_40_45k: Optional[int] = None
    income_45_50k: Optional[int] = None
    income_50_60k: Optional[int] = None
    income_60_75k: Optional[int] = None
    income_75_100k: Optional[int] = None
    income_100_125k: Optional[int] = None
    income_125_150k: Optional[int] = None
    income_150_200k: Optional[int] = None
    income_200k_plus: Optional[int] = None

    # Employment
    employment_universe: Optional[int] = None
    in_labor_force: Optional[int] = None
    civilian_employed: Optional[int] = None
    civilian_unemployed: Optional[int] = None
    not_in_labor_force: Optional[int] = None

    # Industry
    industry_total: Optional[int] = None
    industry_construction_m: Optional[int] = None
    industry_manufacturing_m: Optional[int] = None
    industry_transportation_m: Optional[int] = None
    industry_finance_re_m: Optional[int] = None
    industry_professional_m: Optional[int] = None
    industry_healthcare_edu_m: Optional[int] = None
    industry_hospitality_m: Optional[int] = None
    industry_construction_f: Optional[int] = None
    industry_manufacturing_f: Optional[int] = None
    industry_transportation_f: Optional[int] = None
    industry_finance_re_f: Optional[int] = None
    industry_professional_f: Optional[int] = None
    industry_healthcare_edu_f: Optional[int] = None
    industry_hospitality_f: Optional[int] = None

    # Housing
    occupied_housing_total: Optional[int] = None
    owner_occupied: Optional[int] = None
    renter_occupied: Optional[int] = None
    median_home_value: Optional[float] = None
    median_gross_rent: Optional[float] = None
    median_rent_income_pct: Optional[float] = None

    # Language
    language_households_total: Optional[int] = None
    spanish_speak_english_well: Optional[int] = None
    spanish_speak_english_not_well: Optional[int] = None

    # Migration
    migration_universe: Optional[int] = None
    same_house_1yr: Optional[int] = None
    moved_within_county: Optional[int] = None
    moved_diff_county_same_state: Optional[int] = None
    moved_diff_state: Optional[int] = None
    moved_from_abroad: Optional[int] = None

    @field_validator("hispanic_population", mode="before")
    @classmethod
    def cap_hispanic_at_total(cls, v, info):
        return v

    class Config:
        extra = "ignore"


class CountyRecord(ZIPRecord):
    """County-level record — same fields as ZIP, different geo identifier."""

    zip_code: Optional[str] = None  # type: ignore[assignment]
    county_fips: Optional[str] = None
    county_name: Optional[str] = None


class StateRecord(GeoRecord):
    """State-level record — subset of fields (no ZIP-specific data)."""

    total_population: Optional[int] = None
    hispanic_population: Optional[int] = None
    median_household_income: Optional[float] = None
    hispanic_median_income: Optional[float] = None
    owner_occupied: Optional[int] = None
    renter_occupied: Optional[int] = None
    median_home_value: Optional[float] = None
    median_gross_rent: Optional[float] = None

    class Config:
        extra = "ignore"
