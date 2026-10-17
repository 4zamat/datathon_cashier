"""Environment configuration, with no dotenv dependency."""

import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    api_key: SecretStr
    data_dir: Path = PROJECT_ROOT / "data"
    rules_path: Path = PROJECT_ROOT / "rules" / "rules.yaml"
    pipeline_class: str = "pipeline.dummy:DummyPipeline"
    processing_timeout_sec: float = Field(default=270, gt=0, le=270)
    max_upload_mb: int = Field(default=64, ge=1)
    max_pending_jobs: int = Field(default=16, ge=1)

    @field_validator("api_key")
    @classmethod
    def nonempty_key(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ValueError("API_KEY must be set and nonempty")
        return value

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @classmethod
    def from_env(cls) -> "Settings":
        values = {
            field: os.environ[field.upper()]
            for field in cls.model_fields
            if field.upper() in os.environ
        }
        if "api_key" not in values:
            raise ValueError("Set API_KEY before starting the service")
        return cls.model_validate(values)
