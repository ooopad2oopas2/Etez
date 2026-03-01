#!/usr/bin/env python3
"""
Etez — Institutional trend tracking backend inspired by Elona-style contracts.

Single-file application that exposes:
- FastAPI service for institutions and metric snapshots
- In-memory store with aggregation helpers
- Pydantic models for institutions, snapshots and aggregates
- CLI for quick demos and server control

All names are prefixed with ETZ_ where useful to avoid clashes with other apps.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Tuple

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, HttpUrl, validator


# -----------------------------------------------------------------------------
# Constants and configuration
# -----------------------------------------------------------------------------

ETZ_APP_NAME: str = "Etez"
ETZ_APP_VERSION: str = "1.0.0"
ETZ_DEFAULT_HOST: str = "127.23.57.91"
ETZ_DEFAULT_PORT: int = 8097
ETZ_DEFAULT_STATE_FILE: str = "etez_state.json"
ETZ_TIMEZONE: timezone = timezone.utc
ETZ_MAX_SNAPSHOTS_PER_INSTITUTION: int = 50_000

ETZ_MIN_RISK_SCORE: int = 0
ETZ_MAX_RISK_SCORE: int = 100

ETZ_DEFAULT_ROLLING_WINDOWS_MINUTES: Tuple[int, ...] = (60, 240, 1440)


# -----------------------------------------------------------------------------
# Region and risk tier enumerations
# -----------------------------------------------------------------------------


class ETZRegion(str, Enum):
    """Geographic region for an institution.

    This is intentionally small and Elona-style abstract, not tied to
    any particular jurisdiction list used in other projects.
    """

    GLOBAL = "global"
    NA = "north_america"
    EU = "europe"
    APAC = "apac"
    LATAM = "latam"
    MENA = "mena"
    AFRICA = "africa"
    OTHER = "other"


class ETZRiskTier(str, Enum):
    """Discrete risk tiers for institutions.

    Used to bucket institutions for metrics and allocation.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


def etz_risk_tier_from_score(score: int) -> ETZRiskTier:
    """Map numeric score to ETZRiskTier.

    The mapping is intentionally simple but centralized to keep changes trivial.
    """

    if score < 25:
        return ETZRiskTier.LOW
    if score < 50:
        return ETZRiskTier.MEDIUM
    if score < 75:
        return ETZRiskTier.HIGH
    return ETZRiskTier.EXTREME


def etz_risk_band_label(score: int) -> str:
    """Return a human-friendly label for a numeric risk score."""

    tier = etz_risk_tier_from_score(score)
    if tier is ETZRiskTier.LOW:
        return "calm"
    if tier is ETZRiskTier.MEDIUM:
        return "watch"
    if tier is ETZRiskTier.HIGH:
        return "stress"
    return "critical"


# -----------------------------------------------------------------------------
# Pydantic models
# -----------------------------------------------------------------------------


class ETZInstitution(BaseModel):
    """Institution configuration in the system."""

    id: str = Field(
        default_factory=lambda: f"etz-inst-{uuid.uuid4().hex[:16]}",
        description="Stable identifier used for snapshots and lookups.",
    )
    legal_name: str = Field(..., min_length=2, max_length=256)
    short_name: str = Field(..., min_length=1, max_length=32)
    region: ETZRegion = Field(default=ETZRegion.GLOBAL)
    base_currency: str = Field(default="USD", min_length=3, max_length=3)
    risk_score: int = Field(
        default=10,
        ge=ETZ_MIN_RISK_SCORE,
        le=ETZ_MAX_RISK_SCORE,
        description="0-100 numeric score, higher = riskier.",
    )
    is_active: bool = Field(default=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(ETZ_TIMEZONE))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(ETZ_TIMEZONE))
    website: Optional[HttpUrl] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @validator("legal_name")
    def _etz_validate_legal_name(cls, v: str) -> str:
        if v.strip() != v:
            raise ValueError("legal_name must not have leading or trailing whitespace")
        return v

    @validator("short_name")
    def _etz_validate_short_name(cls, v: str) -> str:
        if " " in v:
            raise ValueError("short_name must be a compact code, no spaces")
        return v.upper()

    @validator("base_currency")
    def _etz_validate_currency(cls, v: str) -> str:
        if not v.isalpha():
            raise ValueError("base_currency must be alphabetic ISO 4217-like code")
        return v.upper()

    @validator("metadata", pre=True)
    def _etz_metadata_default(cls, v: Any) -> Dict[str, Any]:
        if v is None:
            return {}
        if not isinstance(v, dict):
            raise ValueError("metadata must be a JSON object")
        return v


class ETZSnapshot(BaseModel):
    """Time-series data point for an institution.

    This tracks flows and balances that can be aggregated over windows.
    """

    institution_id: str
    captured_at: datetime = Field(
        default_factory=lambda: datetime.now(ETZ_TIMEZONE),
        description="Capture timestamp in UTC.",
    )
    exposure_notional: float = Field(
        ...,
        description="Gross exposure in base_currency units.",
