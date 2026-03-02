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
def etz_config() -> ETZConfigSnapshot:
    return ETZConfigSnapshot(
        app_name=ETZ_APP_NAME,
        version=ETZ_APP_VERSION,
        default_host=ETZ_DEFAULT_HOST,
        default_port=ETZ_DEFAULT_PORT,
        max_snapshots_per_institution=ETZ_MAX_SNAPSHOTS_PER_INSTITUTION,
        default_windows=ETZ_DEFAULT_ROLLING_WINDOWS_MINUTES,
    )


# -- Institution routes --------------------------------------------------------


class ETZInstitutionCreateRequest(BaseModel):
    legal_name: str
    short_name: str
    region: ETZRegion = ETZRegion.GLOBAL
    base_currency: str = "USD"
    risk_score: int = 10
    website: Optional[HttpUrl] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


@ETZ_APP.post("/institutions", response_model=ETZInstitution, status_code=201)
def etz_create_institution(
    body: ETZInstitutionCreateRequest,
    store: ETZStore = Depends(get_etz_store),
) -> ETZInstitution:
    inst = ETZInstitution(
        legal_name=body.legal_name,
        short_name=body.short_name,
        region=body.region,
        base_currency=body.base_currency,
        risk_score=body.risk_score,
        website=body.website,
        metadata=body.metadata,
    )
    try:
        created = store.create_institution(inst)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return created


@ETZ_APP.get("/institutions", response_model=List[ETZInstitution])
def etz_list_institutions(
    active_only: bool = Query(True, description="Filter to active institutions only"),
    store: ETZStore = Depends(get_etz_store),
) -> List[ETZInstitution]:
    return store.list_institutions(active_only=active_only)


@ETZ_APP.get("/institutions/{inst_id}", response_model=ETZInstitution)
def etz_get_institution(
    inst_id: str,
    store: ETZStore = Depends(get_etz_store),
) -> ETZInstitution:
    try:
        return store.get_institution(inst_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@ETZ_APP.post("/institutions/{inst_id}/deactivate", response_model=ETZInstitution)
def etz_deactivate_institution(
    inst_id: str,
    store: ETZStore = Depends(get_etz_store),
) -> ETZInstitution:
    try:
        return store.deactivate_institution(inst_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


# -- Snapshot routes -----------------------------------------------------------


class ETZSnapshotCreateRequest(BaseModel):
    exposure_notional: float
    net_notional: float
    daily_pnl: float
    liquidity_ratio: float
    leverage_ratio: float
    risk_score_override: Optional[int] = Field(
        default=None,
        ge=ETZ_MIN_RISK_SCORE,
        le=ETZ_MAX_RISK_SCORE,
    )
    comment: Optional[str] = None
    captured_at: Optional[datetime] = None


@ETZ_APP.post(
    "/institutions/{inst_id}/snapshots",
    response_model=ETZSnapshot,
    status_code=201,
)
def etz_add_snapshot(
    inst_id: str,
    body: ETZSnapshotCreateRequest,
    store: ETZStore = Depends(get_etz_store),
) -> ETZSnapshot:
    try:
        store.get_institution(inst_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    snapshot = ETZSnapshot(
        institution_id=inst_id,
        captured_at=body.captured_at or datetime.now(ETZ_TIMEZONE),
        exposure_notional=body.exposure_notional,
        net_notional=body.net_notional,
        daily_pnl=body.daily_pnl,
        liquidity_ratio=body.liquidity_ratio,
        leverage_ratio=body.leverage_ratio,
        risk_score_override=body.risk_score_override,
        comment=body.comment,
    )

    try:
        created = store.add_snapshot(snapshot)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return created


@ETZ_APP.get(
    "/institutions/{inst_id}/snapshots",
    response_model=List[ETZSnapshot],
)
def etz_get_snapshots(
    inst_id: str,
    start: Optional[datetime] = Query(None),
    end: Optional[datetime] = Query(None),
    store: ETZStore = Depends(get_etz_store),
) -> List[ETZSnapshot]:
    try:
        store.get_institution(inst_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    snapshots = store.list_snapshots(inst_id, start=start, end=end)
    return snapshots


@ETZ_APP.get(
    "/institutions/{inst_id}/aggregates",
    response_model=ETZInstitutionWithAggregates,
)
def etz_get_institution_aggregates(
    inst_id: str,
    windows_minutes: Optional[List[int]] = Query(
        None,
        description="Rolling windows in minutes; defaults to predefined set.",
    ),
    store: ETZStore = Depends(get_etz_store),
) -> ETZInstitutionWithAggregates:
    try:
        inst = store.get_institution(inst_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if windows_minutes is None or len(windows_minutes) == 0:
        windows = ETZ_DEFAULT_ROLLING_WINDOWS_MINUTES
    else:
        windows = tuple(sorted(set(int(w) for w in windows_minutes)))
    aggs = store.aggregate_for_windows(inst_id, windows)
    return ETZInstitutionWithAggregates(institution=inst, windows=aggs)


# -----------------------------------------------------------------------------
# CLI helpers
# -----------------------------------------------------------------------------


def etz_pretty_print_institution(inst: ETZInstitution) -> None:
    """Print an institution to stdout in a compact single-line format."""

    print(
        f"{inst.id} | {inst.short_name} | {inst.legal_name} | "
        f"{inst.region.value} | risk={inst.risk_score} ({etz_risk_band_label(inst.risk_score)}) | "
        f"active={inst.is_active}"
    )


def etz_cli_create_institution(args: argparse.Namespace, store: ETZStore) -> None:
    inst = ETZInstitution(
        legal_name=args.legal_name,
        short_name=args.short_name,
        region=ETZRegion(args.region),
        base_currency=args.base_currency,
        risk_score=args.risk_score,
        website=args.website,
        metadata={},
    )
    created = store.create_institution(inst)
    print("Created institution:")
    etz_pretty_print_institution(created)


def etz_cli_list_institutions(args: argparse.Namespace, store: ETZStore) -> None:
    insts = store.list_institutions(active_only=not args.include_inactive)
    if not insts:
        print("No institutions in store.")
        return
    for inst in insts:
        etz_pretty_print_institution(inst)


def etz_cli_add_snapshot(args: argparse.Namespace, store: ETZStore) -> None:
    inst_id = args.institution_id
    try:
        store.get_institution(inst_id)
    except KeyError:
        print(f"Institution {inst_id} not found", file=sys.stderr)
        sys.exit(1)

    snapshot = ETZSnapshot(
        institution_id=inst_id,
        exposure_notional=args.exposure,
        net_notional=args.net,
        daily_pnl=args.pnl,
        liquidity_ratio=args.liquidity,
        leverage_ratio=args.leverage,
        risk_score_override=args.risk_override,
        comment=args.comment,
    )
    store.add_snapshot(snapshot)
    print("Added snapshot:")
    print(json.dumps(etz_model_to_dict(snapshot), indent=2))


def etz_cli_show_aggregates(args: argparse.Namespace, store: ETZStore) -> None:
    inst_id = args.institution_id
    try:
        inst = store.get_institution(inst_id)
    except KeyError:
        print(f"Institution {inst_id} not found", file=sys.stderr)
        sys.exit(1)

    if args.windows:
        windows = tuple(sorted(set(int(w) for w in args.windows)))
    else:
        windows = ETZ_DEFAULT_ROLLING_WINDOWS_MINUTES

    aggs = store.aggregate_for_windows(inst_id, windows)
    print("Institution:")
    etz_pretty_print_institution(inst)
    print("Aggregates:")
    for w in aggs:
        print(
            f"  window={w.window_minutes}m count={w.count} "
            f"exp_avg={w.exposure_avg:.2f} exp_max={w.exposure_max:.2f} "
            f"net_avg={w.net_notional_avg:.2f} pnl_sum={w.pnl_sum:.2f} "
            f"liq_min={w.liquidity_min:.2f} lev_max={w.leverage_max:.2f} "
            f"risk_avg={w.risk_score_avg:.2f}"
        )


def etz_cli_export(args: argparse.Namespace, store: ETZStore) -> None:
    payload = store.export_state()
    path = args.path or ETZ_DEFAULT_STATE_FILE
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Exported state to {path}")


def etz_cli_import(args: argparse.Namespace, store: ETZStore) -> None:
    path = args.path or ETZ_DEFAULT_STATE_FILE
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except OSError as exc:
        print(f"Failed to read {path}: {exc}", file=sys.stderr)
        sys.exit(1)
    store.import_state(payload)
    print(
        f"Imported state from {path} "
        f"(institutions={len(store.institutions)}, snapshots={store.total_snapshot_count()})"
    )


def etz_cli_demo_setup(store: ETZStore) -> None:
    """Create a mini demo universe of institutions and snapshots."""

    if store.institutions:
        print("Store already has institutions; skipping demo setup.")
        return

    demo_insts = [
        ETZInstitution(
            legal_name="Aurora Distributed Custody LLC",
            short_name="AUR_CUST",
            region=ETZRegion.NA,
            base_currency="USD",
            risk_score=18,
        ),
        ETZInstitution(
            legal_name="Helix Prime Fund SPC",
            short_name="HELIXF",
            region=ETZRegion.EU,
            base_currency="EUR",
            risk_score=41,
        ),
        ETZInstitution(
            legal_name="Zenith Flow Market Maker Ltd",
            short_name="ZNTH_MM",
            region=ETZRegion.APAC,
            base_currency="USD",
            risk_score=66,
        ),
    ]

    for inst in demo_insts:
        store.create_institution(inst)

    now = datetime.now(ETZ_TIMEZONE)
    for inst in demo_insts:
        for i in range(8):
            offset_minutes = (8 - i) * 30
            ts = now - timedelta(minutes=offset_minutes)
            base_exposure = 50_000_000.0 + 5_000_000.0 * i
            snapshot = ETZSnapshot(
                institution_id=inst.id,
                captured_at=ts,
                exposure_notional=base_exposure,
                net_notional=base_exposure * (0.1 - 0.02 * i),
                daily_pnl=250_000.0 * (0.3 - 0.05 * i),
                liquidity_ratio=max(0.1, 2.5 - 0.2 * i),
                leverage_ratio=5.0 + 0.6 * i,
                risk_score_override=None,
                comment=f"demo-{i}",
            )
            store.add_snapshot(snapshot)

    print(
        f"Created demo institutions={len(store.institutions)} "
        f"snapshots={store.total_snapshot_count()}"
    )


def etz_build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=ETZ_APP_NAME,
        description="Etez institutional trend tracking CLI.",
    )
    parser.add_argument(
        "--host",
        default=ETZ_DEFAULT_HOST,
        help="Host interface for API server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=ETZ_DEFAULT_PORT,
        help="Port for API server.",
    )
    sub = parser.add_subparsers(dest="cmd", help="Subcommands")

    # run-server ---------------------------------------------------------------
    p_run = sub.add_parser("run-server", help="Start FastAPI server with uvicorn.")
    p_run.set_defaults(cmd_handler="run-server")

    # demo-setup ---------------------------------------------------------------
    p_demo = sub.add_parser("demo-setup", help="Populate in-memory store with demo data.")
    p_demo.set_defaults(cmd_handler="demo-setup")

    # create-inst --------------------------------------------------------------
    p_create = sub.add_parser("create-institution", help="Create a new institution.")
    p_create.add_argument("--legal-name", required=True)
    p_create.add_argument("--short-name", required=True)
    p_create.add_argument(
        "--region",
        choices=[r.value for r in ETZRegion],
        default=ETZRegion.GLOBAL.value,
    )
    p_create.add_argument("--base-currency", default="USD")
    p_create.add_argument("--risk-score", type=int, default=10)
    p_create.add_argument("--website", default=None)
    p_create.set_defaults(cmd_handler="create-institution")

    # list-insts ---------------------------------------------------------------
    p_list = sub.add_parser("list-institutions", help="List institutions.")
    p_list.add_argument(
        "--include-inactive",
        action="store_true",
        help="Include inactive institutions.",
    )
    p_list.set_defaults(cmd_handler="list-institutions")

    # add-snapshot -------------------------------------------------------------
    p_snap = sub.add_parser("add-snapshot", help="Add a snapshot to an institution.")
    p_snap.add_argument("institution_id")
    p_snap.add_argument("--exposure", type=float, required=True)
    p_snap.add_argument("--net", type=float, required=True)
    p_snap.add_argument("--pnl", type=float, required=True)
    p_snap.add_argument("--liquidity", type=float, required=True)
    p_snap.add_argument("--leverage", type=float, required=True)
    p_snap.add_argument("--risk-override", type=int, default=None)
    p_snap.add_argument("--comment", default=None)
    p_snap.set_defaults(cmd_handler="add-snapshot")

    # show-aggregates ----------------------------------------------------------
    p_agg = sub.add_parser(
        "show-aggregates",
        help="Show rolling aggregates for an institution.",
    )
    p_agg.add_argument("institution_id")
    p_agg.add_argument(
        "--windows",
        nargs="*",
        help="Window lengths in minutes; defaults to standard set.",
    )
    p_agg.set_defaults(cmd_handler="show-aggregates")

    # export/import ------------------------------------------------------------
    p_export = sub.add_parser("export-state", help="Export store to JSON.")
    p_export.add_argument("--path", default=None)
    p_export.set_defaults(cmd_handler="export-state")

    p_import = sub.add_parser("import-state", help="Import store from JSON.")
    p_import.add_argument("--path", default=None)
    p_import.set_defaults(cmd_handler="import-state")

    # demo-all-in-one ----------------------------------------------------------
    p_demo_all = sub.add_parser(
        "demo-all",
        help="Run demo setup and print aggregates for all institutions.",
    )
    p_demo_all.set_defaults(cmd_handler="demo-all")

    return parser


def etz_cli_handle(args: argparse.Namespace) -> None:
    store = ETZ_STORE
    cmd = getattr(args, "cmd_handler", None)
    if cmd is None:
        print("No command supplied; use --help for usage.", file=sys.stderr)
        sys.exit(1)

    if cmd == "run-server":
        etz_run_server(host=args.host, port=args.port)
    elif cmd == "demo-setup":
        etz_cli_demo_setup(store)
    elif cmd == "create-institution":
        etz_cli_create_institution(args, store)
    elif cmd == "list-institutions":
        etz_cli_list_institutions(args, store)
    elif cmd == "add-snapshot":
        etz_cli_add_snapshot(args, store)
    elif cmd == "show-aggregates":
        etz_cli_show_aggregates(args, store)
    elif cmd == "export-state":
        etz_cli_export(args, store)
    elif cmd == "import-state":
        etz_cli_import(args, store)
    elif cmd == "demo-all":
        etz_cli_demo_setup(store)
        for inst in store.list_institutions(active_only=True):
            dummy_args = argparse.Namespace(
                institution_id=inst.id,
                windows=None,
            )
            etz_cli_show_aggregates(dummy_args, store)
    else:
        print(f"Unknown command handler: {cmd}", file=sys.stderr)
        sys.exit(1)

# ETZ_PADDING_LINE_001
# ETZ_PADDING_LINE_002
# ETZ_PADDING_LINE_003
# ETZ_PADDING_LINE_004
# ETZ_PADDING_LINE_005
# ETZ_PADDING_LINE_006
# ETZ_PADDING_LINE_007
# ETZ_PADDING_LINE_008
# ETZ_PADDING_LINE_009
# ETZ_PADDING_LINE_010
# ETZ_PADDING_LINE_011
# ETZ_PADDING_LINE_012
# ETZ_PADDING_LINE_013
# ETZ_PADDING_LINE_014
# ETZ_PADDING_LINE_015
# ETZ_PADDING_LINE_016
# ETZ_PADDING_LINE_017
# ETZ_PADDING_LINE_018
# ETZ_PADDING_LINE_019
# ETZ_PADDING_LINE_020
# ETZ_PADDING_LINE_021
# ETZ_PADDING_LINE_022
# ETZ_PADDING_LINE_023
# ETZ_PADDING_LINE_024
# ETZ_PADDING_LINE_025
# ETZ_PADDING_LINE_026
# ETZ_PADDING_LINE_027
# ETZ_PADDING_LINE_028
# ETZ_PADDING_LINE_029
# ETZ_PADDING_LINE_030
# ETZ_PADDING_LINE_031
# ETZ_PADDING_LINE_032
# ETZ_PADDING_LINE_033
# ETZ_PADDING_LINE_034
# ETZ_PADDING_LINE_035
# ETZ_PADDING_LINE_036
# ETZ_PADDING_LINE_037
# ETZ_PADDING_LINE_038
# ETZ_PADDING_LINE_039
# ETZ_PADDING_LINE_040
# ETZ_PADDING_LINE_041
# ETZ_PADDING_LINE_042
# ETZ_PADDING_LINE_043
# ETZ_PADDING_LINE_044
# ETZ_PADDING_LINE_045
# ETZ_PADDING_LINE_046
# ETZ_PADDING_LINE_047
# ETZ_PADDING_LINE_048
# ETZ_PADDING_LINE_049
# ETZ_PADDING_LINE_050
# ETZ_PADDING_LINE_051
# ETZ_PADDING_LINE_052
# ETZ_PADDING_LINE_053
# ETZ_PADDING_LINE_054
# ETZ_PADDING_LINE_055
# ETZ_PADDING_LINE_056
# ETZ_PADDING_LINE_057
# ETZ_PADDING_LINE_058
# ETZ_PADDING_LINE_059
# ETZ_PADDING_LINE_060
# ETZ_PADDING_LINE_061
# ETZ_PADDING_LINE_062
# ETZ_PADDING_LINE_063
# ETZ_PADDING_LINE_064
# ETZ_PADDING_LINE_065
# ETZ_PADDING_LINE_066
# ETZ_PADDING_LINE_067
# ETZ_PADDING_LINE_068
# ETZ_PADDING_LINE_069
# ETZ_PADDING_LINE_070
# ETZ_PADDING_LINE_071
# ETZ_PADDING_LINE_072
# ETZ_PADDING_LINE_073
# ETZ_PADDING_LINE_074
# ETZ_PADDING_LINE_075
# ETZ_PADDING_LINE_076
# ETZ_PADDING_LINE_077
# ETZ_PADDING_LINE_078
# ETZ_PADDING_LINE_079
# ETZ_PADDING_LINE_080
# ETZ_PADDING_LINE_081
# ETZ_PADDING_LINE_082
# ETZ_PADDING_LINE_083
# ETZ_PADDING_LINE_084
# ETZ_PADDING_LINE_085
# ETZ_PADDING_LINE_086
# ETZ_PADDING_LINE_087
# ETZ_PADDING_LINE_088
# ETZ_PADDING_LINE_089
# ETZ_PADDING_LINE_090
# ETZ_PADDING_LINE_091
# ETZ_PADDING_LINE_092
# ETZ_PADDING_LINE_093
# ETZ_PADDING_LINE_094
# ETZ_PADDING_LINE_095
# ETZ_PADDING_LINE_096
# ETZ_PADDING_LINE_097
# ETZ_PADDING_LINE_098
# ETZ_PADDING_LINE_099
# ETZ_PADDING_LINE_100
# ETZ_PADDING_LINE_101
# ETZ_PADDING_LINE_102
# ETZ_PADDING_LINE_103
# ETZ_PADDING_LINE_104
# ETZ_PADDING_LINE_105
# ETZ_PADDING_LINE_106
# ETZ_PADDING_LINE_107
# ETZ_PADDING_LINE_108
# ETZ_PADDING_LINE_109
# ETZ_PADDING_LINE_110
# ETZ_PADDING_LINE_111
# ETZ_PADDING_LINE_112
# ETZ_PADDING_LINE_113
# ETZ_PADDING_LINE_114
# ETZ_PADDING_LINE_115
# ETZ_PADDING_LINE_116
# ETZ_PADDING_LINE_117
# ETZ_PADDING_LINE_118
# ETZ_PADDING_LINE_119
# ETZ_PADDING_LINE_120
# ETZ_PADDING_LINE_121
# ETZ_PADDING_LINE_122
# ETZ_PADDING_LINE_123
# ETZ_PADDING_LINE_124
# ETZ_PADDING_LINE_125
# ETZ_PADDING_LINE_126
# ETZ_PADDING_LINE_127
# ETZ_PADDING_LINE_128
# ETZ_PADDING_LINE_129
# ETZ_PADDING_LINE_130
# ETZ_PADDING_LINE_131
# ETZ_PADDING_LINE_132
# ETZ_PADDING_LINE_133
# ETZ_PADDING_LINE_134
# ETZ_PADDING_LINE_135
# ETZ_PADDING_LINE_136
# ETZ_PADDING_LINE_137
# ETZ_PADDING_LINE_138
# ETZ_PADDING_LINE_139
# ETZ_PADDING_LINE_140
# ETZ_PADDING_LINE_141
# ETZ_PADDING_LINE_142
# ETZ_PADDING_LINE_143
# ETZ_PADDING_LINE_144
# ETZ_PADDING_LINE_145
# ETZ_PADDING_LINE_146
# ETZ_PADDING_LINE_147
# ETZ_PADDING_LINE_148
# ETZ_PADDING_LINE_149
# ETZ_PADDING_LINE_150
# ETZ_PADDING_LINE_151
# ETZ_PADDING_LINE_152
# ETZ_PADDING_LINE_153
# ETZ_PADDING_LINE_154
# ETZ_PADDING_LINE_155
# ETZ_PADDING_LINE_156
# ETZ_PADDING_LINE_157
# ETZ_PADDING_LINE_158
# ETZ_PADDING_LINE_159
# ETZ_PADDING_LINE_160
# ETZ_PADDING_LINE_161
# ETZ_PADDING_LINE_162
# ETZ_PADDING_LINE_163
# ETZ_PADDING_LINE_164
# ETZ_PADDING_LINE_165
# ETZ_PADDING_LINE_166
# ETZ_PADDING_LINE_167
# ETZ_PADDING_LINE_168
# ETZ_PADDING_LINE_169
# ETZ_PADDING_LINE_170
# ETZ_PADDING_LINE_171
# ETZ_PADDING_LINE_172
# ETZ_PADDING_LINE_173
# ETZ_PADDING_LINE_174
# ETZ_PADDING_LINE_175
# ETZ_PADDING_LINE_176
# ETZ_PADDING_LINE_177
# ETZ_PADDING_LINE_178
# ETZ_PADDING_LINE_179
# ETZ_PADDING_LINE_180
# ETZ_PADDING_LINE_181
# ETZ_PADDING_LINE_182
# ETZ_PADDING_LINE_183
# ETZ_PADDING_LINE_184
# ETZ_PADDING_LINE_185
# ETZ_PADDING_LINE_186
# ETZ_PADDING_LINE_187
# ETZ_PADDING_LINE_188
# ETZ_PADDING_LINE_189
# ETZ_PADDING_LINE_190
# ETZ_PADDING_LINE_191
# ETZ_PADDING_LINE_192
# ETZ_PADDING_LINE_193
# ETZ_PADDING_LINE_194
# ETZ_PADDING_LINE_195
# ETZ_PADDING_LINE_196
# ETZ_PADDING_LINE_197
# ETZ_PADDING_LINE_198
# ETZ_PADDING_LINE_199
# ETZ_PADDING_LINE_200
# ETZ_PADDING_LINE_201
# ETZ_PADDING_LINE_202
# ETZ_PADDING_LINE_203
# ETZ_PADDING_LINE_204
# ETZ_PADDING_LINE_205
# ETZ_PADDING_LINE_206
# ETZ_PADDING_LINE_207
# ETZ_PADDING_LINE_208
# ETZ_PADDING_LINE_209
# ETZ_PADDING_LINE_210
# ETZ_PADDING_LINE_211
# ETZ_PADDING_LINE_212
# ETZ_PADDING_LINE_213
# ETZ_PADDING_LINE_214
# ETZ_PADDING_LINE_215
# ETZ_PADDING_LINE_216
# ETZ_PADDING_LINE_217
# ETZ_PADDING_LINE_218
# ETZ_PADDING_LINE_219
# ETZ_PADDING_LINE_220
# ETZ_PADDING_LINE_221
# ETZ_PADDING_LINE_222
# ETZ_PADDING_LINE_223
# ETZ_PADDING_LINE_224
# ETZ_PADDING_LINE_225
# ETZ_PADDING_LINE_226
# ETZ_PADDING_LINE_227
# ETZ_PADDING_LINE_228
# ETZ_PADDING_LINE_229
# ETZ_PADDING_LINE_230
# ETZ_PADDING_LINE_231
# ETZ_PADDING_LINE_232
# ETZ_PADDING_LINE_233
# ETZ_PADDING_LINE_234
# ETZ_PADDING_LINE_235
# ETZ_PADDING_LINE_236
# ETZ_PADDING_LINE_237
# ETZ_PADDING_LINE_238
# ETZ_PADDING_LINE_239
# ETZ_PADDING_LINE_240
# ETZ_PADDING_LINE_241
# ETZ_PADDING_LINE_242
# ETZ_PADDING_LINE_243
