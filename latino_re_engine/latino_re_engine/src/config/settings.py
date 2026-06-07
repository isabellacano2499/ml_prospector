from pathlib import Path
from typing import Literal
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


_ENV_FILE = Path(__file__).parent.parent.parent / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    census_api_key: str = Field(..., description="Census Bureau API key")
    acs_year: int = Field(default=2022)
    acs_year_prev: int = Field(default=2019, description="Year for growth comparison")
    acs_dataset: str = Field(default="acs5")
    geographic_level: Literal["zip", "county", "state"] = Field(default="zip")

    output_dir: Path = Field(default=Path("./data/output"))
    raw_data_dir: Path = Field(default=Path("./data/raw"))

    batch_size: int = Field(default=500, description="Records per processing batch")
    variable_batch_size: int = Field(default=40, description="ACS vars per API request")
    request_timeout: int = Field(default=60)
    max_retries: int = Field(default=5)
    log_level: str = Field(default="INFO")

    @model_validator(mode="after")
    def create_directories(self) -> "Settings":
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        return self


settings = Settings()
