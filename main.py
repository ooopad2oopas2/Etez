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
