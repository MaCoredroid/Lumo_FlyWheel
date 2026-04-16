"""Configuration loaded via pydantic-settings."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RELEASE_READINESS_",
        env_file=".env",
        extra="ignore",
    )

    title: str = Field(default="Release Readiness Report")
    source: Literal["fs", "env"] = Field(default="env")
    fs_path: Path = Field(default=Path("records.json"))
