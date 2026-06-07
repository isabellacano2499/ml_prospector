"""
ACS 5-Year variable codes mapped to internal column names.
Validate codes: https://api.census.gov/data/{year}/acs/acs5/variables.json
"""

# Census API sentinel values that represent "data not available"
CENSUS_SENTINEL_VALUES: set[int] = {-666666666, -999999999, -888888888, -222222222}

VARIABLE_GROUPS: dict[str, dict[str, str]] = {
    "core_demographics": {
        "B01003_001E": "total_population",
        "B03003_003E": "hispanic_population",
        "B01001_002E": "male_total",
        "B01001_026E": "female_total",
        "B01001I_002E": "hispanic_male",
        "B01001I_017E": "hispanic_female",
    },
    "age_male": {
        "B01001_007E": "male_18_19",
        "B01001_008E": "male_20",
        "B01001_009E": "male_21",
        "B01001_010E": "male_22_24",
        "B01001_011E": "male_25_29",
        "B01001_012E": "male_30_34",
        "B01001_013E": "male_35_39",
        "B01001_014E": "male_40_44",
        "B01001_015E": "male_45_49",
        "B01001_016E": "male_50_54",
    },
    "age_female": {
        "B01001_031E": "female_18_19",
        "B01001_032E": "female_20",
        "B01001_033E": "female_21",
        "B01001_034E": "female_22_24",
        "B01001_035E": "female_25_29",
        "B01001_036E": "female_30_34",
        "B01001_037E": "female_35_39",
        "B01001_038E": "female_40_44",
        "B01001_039E": "female_45_49",
        "B01001_040E": "female_50_54",
    },
    "citizenship": {
        "B05001_001E": "citizenship_total",
        "B05001_005E": "naturalized_citizens",
        "B05001_006E": "non_citizens",
    },
    # B12001 has exactly 13 rows (_001E–_013E). Structure:
    # 001=total, 002=male_total, 003=m_never_married, 004=m_married,
    # 005=m_separated, 006=m_widowed, 007=m_divorced,
    # 008=female_total, 009=f_never_married, 010=f_married,
    # 011=f_separated, 012=f_widowed, 013=f_divorced
    "marital_status": {
        "B12001_001E": "marital_total",
        "B12001_003E": "male_never_married",
        "B12001_004E": "male_married",
        "B12001_005E": "male_separated",
        "B12001_007E": "male_divorced",
        "B12001_009E": "female_never_married",
        "B12001_010E": "female_married",
        "B12001_011E": "female_separated",
        "B12001_013E": "female_divorced",
    },
    "education": {
        "B15003_001E": "education_total",
        "B15003_017E": "hs_diploma",
        "B15003_018E": "ged",
        "B15003_019E": "some_college_lt1yr",
        "B15003_020E": "some_college_gt1yr",
        "B15003_021E": "associates_degree",
        "B15003_022E": "bachelors_degree",
        "B15003_023E": "masters_degree",
        "B15003_024E": "professional_degree",
        "B15003_025E": "doctorate_degree",
    },
    "income": {
        "B19013_001E": "median_household_income",
        "B19013I_001E": "hispanic_median_income",
        "B19001_002E": "income_lt10k",
        "B19001_003E": "income_10_15k",
        "B19001_004E": "income_15_20k",
        "B19001_005E": "income_20_25k",
        "B19001_006E": "income_25_30k",
        "B19001_007E": "income_30_35k",
        "B19001_008E": "income_35_40k",
        "B19001_009E": "income_40_45k",
        "B19001_010E": "income_45_50k",
        "B19001_011E": "income_50_60k",
        "B19001_012E": "income_60_75k",
        "B19001_013E": "income_75_100k",
        "B19001_014E": "income_100_125k",
        "B19001_015E": "income_125_150k",
        "B19001_016E": "income_150_200k",
        "B19001_017E": "income_200k_plus",
    },
    "employment": {
        "B23025_001E": "employment_universe",
        "B23025_002E": "in_labor_force",
        "B23025_004E": "civilian_employed",
        "B23025_005E": "civilian_unemployed",
        "B23025_007E": "not_in_labor_force",
    },
    # C24030: Sex by Industry — male sub-totals (female adds ~same offset +17)
    # These codes cover total employed by industry sector (male only; female is +17 offset)
    "industry": {
        "C24030_001E": "industry_total",
        "C24030_004E": "industry_construction_m",
        "C24030_005E": "industry_manufacturing_m",
        "C24030_008E": "industry_transportation_m",
        "C24030_011E": "industry_finance_re_m",
        "C24030_012E": "industry_professional_m",
        "C24030_013E": "industry_healthcare_edu_m",
        "C24030_014E": "industry_hospitality_m",
        "C24030_021E": "industry_construction_f",
        "C24030_022E": "industry_manufacturing_f",
        "C24030_025E": "industry_transportation_f",
        "C24030_028E": "industry_finance_re_f",
        "C24030_029E": "industry_professional_f",
        "C24030_030E": "industry_healthcare_edu_f",
        "C24030_031E": "industry_hospitality_f",
    },
    "housing": {
        "B25003_001E": "occupied_housing_total",
        "B25003_002E": "owner_occupied",
        "B25003_003E": "renter_occupied",
        "B25077_001E": "median_home_value",
        "B25064_001E": "median_gross_rent",
        "B25071_001E": "median_rent_income_pct",
    },
    "language": {
        "B16002_001E": "language_households_total",
        "B16002_003E": "spanish_speak_english_well",
        "B16002_004E": "spanish_speak_english_not_well",
    },
    # B07003: Geographic Mobility by Sex — available at ZCTA, county, and state.
    # Structure: 001=total, 002=male_total, 003=m_same_house, 004=m_within_county,
    # 005=m_diff_county_same_state, 006=m_diff_state, 007=m_from_abroad,
    # 008=female_total, 009=f_same_house, 010=f_within_county,
    # 011=f_diff_county_same_state, 012=f_diff_state, 013=f_from_abroad
    "migration": {
        "B07003_001E": "migration_universe",
        "B07003_003E": "same_house_1yr_m",
        "B07003_004E": "moved_within_county_m",
        "B07003_005E": "moved_diff_county_same_state_m",
        "B07003_006E": "moved_diff_state_m",
        "B07003_007E": "moved_from_abroad_m",
        "B07003_009E": "same_house_1yr_f",
        "B07003_010E": "moved_within_county_f",
        "B07003_011E": "moved_diff_county_same_state_f",
        "B07003_012E": "moved_diff_state_f",
        "B07003_013E": "moved_from_abroad_f",
    },
}

# Flat maps built from VARIABLE_GROUPS
CODE_TO_COLUMN: dict[str, str] = {}
for _group in VARIABLE_GROUPS.values():
    CODE_TO_COLUMN.update(_group)

COLUMN_TO_CODE: dict[str, str] = {v: k for k, v in CODE_TO_COLUMN.items()}

def get_variable_groups_for_level(_geographic_level: str) -> dict[str, dict[str, str]]:
    return dict(VARIABLE_GROUPS)
