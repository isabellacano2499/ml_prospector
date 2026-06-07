"""Maps ZIP codes to city and county using pgeocode (GeoNames offline database)."""
import pgeocode
from functools import lru_cache

_nomi = None


def _get_nomi():
    global _nomi
    if _nomi is None:
        _nomi = pgeocode.Nominatim("us")
    return _nomi


@lru_cache(maxsize=2000)
def zip_to_city_county(zip_code: str) -> tuple[str, str]:
    """Returns (city, county) for a ZIP. Falls back to ('', '') if unknown."""
    nomi = _get_nomi()
    result = nomi.query_postal_code(str(zip_code).zfill(5))
    if result is None:
        return "", ""
    city = result.get("place_name") or ""
    county = result.get("county_name") or ""
    county = str(county).replace(" County", "").strip()
    return str(city), str(county)


def zip_list_to_cities(zip_codes: list[str]) -> dict[str, tuple[str, str]]:
    """Returns {zip_code: (city, county)} for a list of ZIPs."""
    return {z: zip_to_city_county(z) for z in zip_codes}
