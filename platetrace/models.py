"""Validated inputs and evidence-based output shapes."""
import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class VehicleRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    plate: str = Field(max_length=20)
    jurisdiction: str = Field(max_length=80)
    vin: str = Field(default="", max_length=17)
    make: str = Field(default="", max_length=80)
    model: str = Field(default="", max_length=80)
    year: int | None = Field(default=None, ge=1886, le=2100)
    fuel: str = Field(default="", max_length=50)
    color: str = Field(default="", max_length=50)


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    plate: str = Field(min_length=1, max_length=20)
    jurisdiction: str = Field(min_length=2, max_length=80)
    vin: str = Field(default="", max_length=17)
    make: str = Field(default="", max_length=80)
    model: str = Field(default="", max_length=80)
    year: int | None = Field(default=None, ge=1886, le=2100)
    provider: Literal["openrouter", "ollama", "demo"] = "ollama"
    model_id: str = Field(default="qwen3:8b", min_length=1, max_length=200)
    api_key: str = Field(default="", max_length=500, repr=False)
    purpose: str = Field(default="public vehicle research", max_length=100)
    objective: str = Field(
        default="Find publicly available vehicle details, specifications, and recall information.",
        max_length=2000,
    )
    authorized: bool = False
    source_urls: list[str] = Field(default_factory=list, max_length=12)
    records: list[VehicleRecord] = Field(default_factory=list, max_length=500)
    enable_terminal: bool = False
    use_memory: bool = True
    max_steps: int = Field(default=12, ge=2, le=40)

    @field_validator("plate")
    @classmethod
    def clean_plate(cls, value):
        value = value.strip().upper()
        if not re.fullmatch(r"[A-Z0-9 -]{1,20}", value) or not re.search(r"[A-Z0-9]", value):
            raise ValueError("Use letters, numbers, spaces, or hyphens for the plate.")
        return value

    @field_validator("jurisdiction")
    @classmethod
    def clean_jurisdiction(cls, value):
        value = value.strip()
        if len(value) < 2:
            raise ValueError("Include the country and state/province where the plate was issued.")
        return value

    @field_validator("vin")
    @classmethod
    def clean_vin(cls, value):
        value = value.strip().upper()
        if value and not re.fullmatch(r"[A-HJ-NPR-Z0-9]{17}", value):
            raise ValueError("A VIN must contain 17 letters and digits (excluding I, O, and Q).")
        return value

    @field_validator("source_urls")
    @classmethod
    def bounded_urls(cls, values):
        from urllib.parse import urlsplit
        for value in values:
            url = urlsplit(value)
            if len(value) > 2000 or url.scheme not in ("https", "http") or not url.hostname:
                raise ValueError("Source URLs must be complete HTTP or HTTPS links.")
            if url.username or url.password:
                raise ValueError("Source URLs must not include credentials.")
        return values

    @model_validator(mode="after")
    def authorized_scope(self):
        if not self.authorized:
            raise ValueError("Confirm that this is authorized public vehicle research.")
        sensitive = re.compile(
            r"\b(owner'?s? (?:name|address|identity|phone)|identify (?:the )?owner|"
            r"who owns|home address|track (?:the |a |this )?(?:person|driver|owner)|"
            r"live location|current location|movement history|doxx?)\b", re.IGNORECASE
        )
        if sensitive.search(self.objective):
            raise ValueError("PlateTrace supports vehicle facts and recalls, not private identities or tracking.")
        return self


class Finding(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(max_length=200)
    detail: str = Field(max_length=4000)
    source_ids: list[str] = Field(min_length=1, max_length=20)


class Report(BaseModel):
    model_config = ConfigDict(extra="forbid")
    summary: str = Field(min_length=1, max_length=6000)
    findings: list[Finding] = Field(default_factory=list, max_length=30)
    limitations: list[str] = Field(default_factory=list, max_length=30)
    next_steps: list[str] = Field(default_factory=list, max_length=20)

    @field_validator("limitations", "next_steps")
    @classmethod
    def bounded_items(cls, values):
        if any(len(value) > 2000 for value in values):
            raise ValueError("Report entries must be under 2000 characters.")
        return values


class FinishReport(Report):
    """Optional reusable lessons accompany a report, separate from vehicle findings."""

    research_lessons: list[Annotated[str, Field(strict=True, min_length=1, max_length=1000)]] = Field(
        default_factory=list, max_length=5,
    )
