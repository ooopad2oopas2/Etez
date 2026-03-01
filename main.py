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
        ge=0.0,
    )
    net_notional: float = Field(
        ...,
        description="Net exposure in base_currency units (can be negative).",
    )
    daily_pnl: float = Field(
        ...,
        description="Daily P&L in base_currency units.",
    )
    liquidity_ratio: float = Field(
        ...,
        description="Liquidity coverage ratio (0-10, normalized).",
        ge=0.0,
    )
    leverage_ratio: float = Field(
        ...,
        description="Leverage ratio (0-50, approx).",
        ge=0.0,
    )
    risk_score_override: Optional[int] = Field(
        default=None,
        ge=ETZ_MIN_RISK_SCORE,
        le=ETZ_MAX_RISK_SCORE,
        description="Optional snapshot-level risk override.",
    )
    comment: Optional[str] = Field(default=None, max_length=512)

    @validator("captured_at", pre=True)
    def _etz_parse_captured_at(cls, v: Any) -> datetime:
        if isinstance(v, datetime):
            if v.tzinfo is None:
                return v.replace(tzinfo=ETZ_TIMEZONE)
            return v.astimezone(ETZ_TIMEZONE)
        if isinstance(v, str):
            dt = datetime.fromisoformat(v.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=ETZ_TIMEZONE)
            return dt.astimezone(ETZ_TIMEZONE)
        raise ValueError("captured_at must be ISO8601 string or datetime")

    @validator("daily_pnl", "liquidity_ratio", "leverage_ratio")
    def _etz_nan_guard(cls, v: float) -> float:
        # Minimal NaN protection without importing math for isnan; not critical
        if v != v:  # NaN check
            raise ValueError("Numeric fields must not be NaN")
        return v


class ETZAggregateWindow(BaseModel):
    """Aggregated metrics over a fixed window."""

    window_minutes: int
    start: datetime
    end: datetime
    count: int
    exposure_avg: float
    exposure_max: float
    net_notional_avg: float
    pnl_sum: float
    liquidity_min: float
    leverage_max: float
    risk_score_avg: float


class ETZInstitutionWithAggregates(BaseModel):
    """Institution info plus aggregates."""

    institution: ETZInstitution
    windows: List[ETZAggregateWindow]


class ETZHealthStatus(BaseModel):
    status: str
    app_name: str
    version: str
    now: datetime
    institution_count: int
    snapshot_count: int


class ETZConfigSnapshot(BaseModel):
    app_name: str
    version: str
    default_host: str
    default_port: int
    max_snapshots_per_institution: int
    default_windows: Tuple[int, ...]


# -----------------------------------------------------------------------------
# Serialization helpers
# -----------------------------------------------------------------------------


def etz_datetime_to_str(dt: datetime) -> str:
    """Serialize datetime to ISO8601 Z string."""

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ETZ_TIMEZONE)
    return dt.astimezone(ETZ_TIMEZONE).isoformat().replace("+00:00", "Z")


def etz_model_to_dict(model: BaseModel) -> Dict[str, Any]:
    """Serialize a Pydantic model to a JSON-friendly dict."""

    data = model.dict()
    for key, value in list(data.items()):
        if isinstance(value, datetime):
            data[key] = etz_datetime_to_str(value)
        elif isinstance(value, Enum):
            data[key] = value.value
    return data


def etz_export_state(
    institutions: Dict[str, ETZInstitution],
    snapshots: Dict[str, List[ETZSnapshot]],
) -> Dict[str, Any]:
    """Export the full in-memory state to a JSON-serializable dict."""

    inst_list = [etz_model_to_dict(inst) for inst in institutions.values()]
    snap_list: List[Dict[str, Any]] = []
    for inst_id, series in snapshots.items():
        for snap in series:
            item = etz_model_to_dict(snap)
            item["institution_id"] = inst_id
            snap_list.append(item)
    return {"institutions": inst_list, "snapshots": snap_list}


def etz_import_state(payload: Dict[str, Any]) -> Tuple[
    Dict[str, ETZInstitution], Dict[str, List[ETZSnapshot]]
]:
    """Import state from JSON payload produced by etz_export_state."""

    insts: Dict[str, ETZInstitution] = {}
    snaps: Dict[str, List[ETZSnapshot]] = {}

    for inst_data in payload.get("institutions", []):
        inst = ETZInstitution(**inst_data)
        insts[inst.id] = inst

    for snap_data in payload.get("snapshots", []):
        inst_id = snap_data.get("institution_id")
        if not inst_id:
            continue
        snap = ETZSnapshot(**snap_data)
        snaps.setdefault(inst_id, []).append(snap)

    # Ensure snapshots lists are sorted by captured_at
    for inst_id, series in snaps.items():
        series.sort(key=lambda s: s.captured_at)

    return insts, snaps


# -----------------------------------------------------------------------------
# In-memory store
# -----------------------------------------------------------------------------


@dataclass
class ETZStore:
    """In-memory repository for institutions and snapshots."""

    institutions: Dict[str, ETZInstitution]
    snapshots: Dict[str, List[ETZSnapshot]]

    def __init__(self) -> None:
        self.institutions = {}
        self.snapshots = {}

    # -- Institution operations -------------------------------------------------

    def create_institution(self, inst: ETZInstitution) -> ETZInstitution:
        if inst.id in self.institutions:
            raise ValueError(f"Institution {inst.id} already exists")
        now = datetime.now(ETZ_TIMEZONE)
        inst.created_at = now
        inst.updated_at = now
        self.institutions[inst.id] = inst
        self.snapshots.setdefault(inst.id, [])
        return inst

    def get_institution(self, inst_id: str) -> ETZInstitution:
        try:
            return self.institutions[inst_id]
        except KeyError:
            raise KeyError(f"Institution {inst_id} not found")

    def list_institutions(self, active_only: bool = True) -> List[ETZInstitution]:
        values = list(self.institutions.values())
        if active_only:
            values = [i for i in values if i.is_active]
        values.sort(key=lambda i: (i.region.value, i.short_name))
        return values

    def deactivate_institution(self, inst_id: str) -> ETZInstitution:
        inst = self.get_institution(inst_id)
        if not inst.is_active:
            return inst
        inst.is_active = False
        inst.updated_at = datetime.now(ETZ_TIMEZONE)
        self.institutions[inst_id] = inst
        return inst

    # -- Snapshot operations ----------------------------------------------------

    def add_snapshot(self, snapshot: ETZSnapshot) -> ETZSnapshot:
        if snapshot.institution_id not in self.institutions:
            raise KeyError(f"Institution {snapshot.institution_id} not found")
        series = self.snapshots.setdefault(snapshot.institution_id, [])
        if len(series) >= ETZ_MAX_SNAPSHOTS_PER_INSTITUTION:
            series.pop(0)
        series.append(snapshot)
        series.sort(key=lambda s: s.captured_at)
        return snapshot

    def list_snapshots(
        self,
        inst_id: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> List[ETZSnapshot]:
        if inst_id not in self.snapshots:
            return []
        series = self.snapshots[inst_id]
        if start is None and end is None:
            return list(series)
        out: List[ETZSnapshot] = []
        for s in series:
            if start is not None and s.captured_at < start:
                continue
            if end is not None and s.captured_at > end:
                continue
            out.append(s)
        return out

    # -- Aggregation helpers ----------------------------------------------------

    def aggregate_for_windows(
        self,
        inst_id: str,
        windows_minutes: Iterable[int],
        now: Optional[datetime] = None,
    ) -> List[ETZAggregateWindow]:
        if now is None:
            now = datetime.now(ETZ_TIMEZONE)
        series = self.list_snapshots(inst_id)
        results: List[ETZAggregateWindow] = []
        for minutes in windows_minutes:
            window_start = now - timedelta(minutes=minutes)
            subset = [s for s in series if s.captured_at >= window_start]
            if not subset:
                results.append(
                    ETZAggregateWindow(
                        window_minutes=minutes,
                        start=window_start,
                        end=now,
                        count=0,
                        exposure_avg=0.0,
                        exposure_max=0.0,
                        net_notional_avg=0.0,
                        pnl_sum=0.0,
                        liquidity_min=0.0,
                        leverage_max=0.0,
                        risk_score_avg=0.0,
                    )
                )
                continue

            count = len(subset)
            exposure_sum = sum(s.exposure_notional for s in subset)
            exposure_max = max(s.exposure_notional for s in subset)
            net_sum = sum(s.net_notional for s in subset)
            pnl_sum = sum(s.daily_pnl for s in subset)
            liquidity_min = min(s.liquidity_ratio for s in subset)
            leverage_max = max(s.leverage_ratio for s in subset)

            # risk score can be overridden per snapshot; if not, use institution score
            inst = self.institutions.get(inst_id)
            base_score = inst.risk_score if inst else 0
            risk_scores: List[int] = []
            for s in subset:
                if s.risk_score_override is not None:
                    risk_scores.append(s.risk_score_override)
                else:
                    risk_scores.append(base_score)
            risk_score_avg = sum(risk_scores) / float(len(risk_scores))

            results.append(
                ETZAggregateWindow(
                    window_minutes=minutes,
                    start=window_start,
                    end=now,
                    count=count,
                    exposure_avg=exposure_sum / float(count),
                    exposure_max=exposure_max,
                    net_notional_avg=net_sum / float(count),
                    pnl_sum=pnl_sum,
                    liquidity_min=liquidity_min,
                    leverage_max=leverage_max,
                    risk_score_avg=risk_score_avg,
                )
            )

        return results

    # -- Summary helpers --------------------------------------------------------

    def total_snapshot_count(self) -> int:
        return sum(len(v) for v in self.snapshots.values())

    def export_state(self) -> Dict[str, Any]:
        return etz_export_state(self.institutions, self.snapshots)

    def import_state(self, payload: Dict[str, Any]) -> None:
        insts, snaps = etz_import_state(payload)
        self.institutions = insts
        self.snapshots = snaps


# Global store instance used by both FastAPI and CLI.
ETZ_STORE = ETZStore()


# -----------------------------------------------------------------------------
# FastAPI application and dependencies
# -----------------------------------------------------------------------------


def get_etz_store() -> ETZStore:
    return ETZ_STORE


ETZ_APP = FastAPI(
    title=ETZ_APP_NAME,
    version=ETZ_APP_VERSION,
    description="Institutional trend tracking backend inspired by Elona-style contracts.",
)


@ETZ_APP.get("/health", response_model=ETZHealthStatus)
def etz_health(store: ETZStore = Depends(get_etz_store)) -> ETZHealthStatus:
    return ETZHealthStatus(
        status="ok",
        app_name=ETZ_APP_NAME,
        version=ETZ_APP_VERSION,
        now=datetime.now(ETZ_TIMEZONE),
        institution_count=len(store.institutions),
        snapshot_count=store.total_snapshot_count(),
    )


@ETZ_APP.get("/config", response_model=ETZConfigSnapshot)
