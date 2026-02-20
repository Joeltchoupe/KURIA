"""
Event System — Communication inter-agents.

Chaque agent PUBLIE des événements.
Le Router de l'orchestrateur décide QUI les reçoit et QUOI en faire.

V1 : stockage PostgreSQL (table events), polling toutes les minutes.
V2 : Redis pub/sub si besoin de temps réel.

Design decisions :
- Chaque événement a un type strict (enum, pas de string libre)
- Payload est un dict libre MAIS chaque EventType documente
  le payload attendu (contrat implicite, validation au routing)
- Priorité pour ordonner le traitement
- TTL pour ne pas accumuler les vieux événements
"""

from __future__ import annotations

from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


class EventType(str, Enum):
    """Types d'événements du système.

    Convention de nommage : {source}_{action}
    Chaque type documente son payload attendu.
    """

    # ── Revenue Velocity ──
    DEAL_STAGNANT_DETECTED = "deal_stagnant_detected"
    # Payload: {deal_id, deal_name, days_stagnant, value, owner, stage}

    DEAL_ZOMBIE_TAGGED = "deal_zombie_tagged"
    # Payload: {deal_id, deal_name, value, last_activity_date}

    LEAD_SCORED = "lead_scored"
    # Payload: {contact_id, score, action, source_channel}

    LEAD_HOT_DETECTED = "lead_hot_detected"
    # Payload: {contact_id, contact_name, score, reason}

    FORECAST_UPDATED = "forecast_updated"
    # Payload: {forecast_30d, forecast_60d, forecast_90d, confidence}

    PIPELINE_TRUTH_GENERATED = "pipeline_truth_generated"
    # Payload: {declared, realistic, gap_ratio, deals_active, deals_stagnant}

    # ── Process Clarity ──
    BOTTLENECK_DETECTED = "bottleneck_detected"
    # Payload: {process, stage, person, delay_vs_normal, estimated_cost}

    WASTE_DETECTED = "waste_detected"
    # Payload: {waste_type, description, hours_per_week, annual_cost}

    PROCESS_MAPPED = "process_mapped"
    # Payload: {process_name, stages_count, avg_cycle_days}

    # ── Cash Predictability ──
    CASH_ALERT_YELLOW = "cash_alert_yellow"
    # Payload: {days_until_threshold, scenario, threshold, current_cash}

    CASH_ALERT_RED = "cash_alert_red"
    # Payload: {days_until_zero, scenario, current_cash, actions}

    CASH_FORECAST_GENERATED = "cash_forecast_generated"
    # Payload: {base_30d, stress_30d, upside_30d, confidence}

    # ── Acquisition Efficiency ──
    CAC_ANOMALY_DETECTED = "cac_anomaly_detected"
    # Payload: {channel, cac_current, cac_previous, trend_pct}

    CHANNEL_SCORED = "channel_scored"
    # Payload: {channel, category, cac, ltv, recommendation}

    LEAD_QUALITY_UPDATED = "lead_quality_updated"
    # Payload: {channel, conversion_rate, avg_deal_size, quality_score}

    # ── Orchestrateur ──
    WEEKLY_REPORT_COMPILED = "weekly_report_compiled"
    # Payload: {company_id, clarity_score, sections_count}

    ADAPTATION_COMPLETED = "adaptation_completed"
    # Payload: {company_id, changes_count, changes_summary}

    SCAN_COMPLETED = "scan_completed"
    # Payload: {company_id, clarity_score, frictions_count}

    AGENT_CONFIG_UPDATED = "agent_config_updated"
    # Payload: {company_id, agent, changes}

    # ── System ──
    AGENT_ERROR = "agent_error"
    # Payload: {agent, error_type, error_message, recoverable}

    AGENT_HEALTH_CHECK = "agent_health_check"
    # Payload: {agent, status, last_run, next_run}


class EventPriority(str, Enum):
    """Priorité de traitement.
    Le Router traite les CRITICAL d'abord, puis HIGH, etc."""

    CRITICAL = "critical"   # Cash alert rouge, erreurs système
    HIGH = "high"           # Alertes stagnation, anomalies CAC
    NORMAL = "normal"       # Reports, scores, mises à jour
    LOW = "low"             # Health checks, logs


# ── Mapping type → priorité par défaut ──

_DEFAULT_PRIORITIES: dict[EventType, EventPriority] = {
    EventType.CASH_ALERT_RED: EventPriority.CRITICAL,
    EventType.AGENT_ERROR: EventPriority.CRITICAL,
    EventType.CASH_ALERT_YELLOW: EventPriority.HIGH,
    EventType.DEAL_STAGNANT_DETECTED: EventPriority.HIGH,
    EventType.CAC_ANOMALY_DETECTED: EventPriority.HIGH,
    EventType.LEAD_HOT_DETECTED: EventPriority.HIGH,
    EventType.BOTTLENECK_DETECTED: EventPriority.HIGH,
    EventType.AGENT_HEALTH_CHECK: EventPriority.LOW,
}


class Event(BaseModel):
    """Un événement dans le système Kuria.

    Cycle de vie :
    1. Un agent crée l'événement via publish_event()
    2. L'événement est stocké dans la table 'events'
    3. Le Router le lit, le route vers les handlers
    4. Le Router le marque comme processed
    """

    event_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="UUID unique de l'événement",
    )
    event_type: EventType
    priority: EventPriority = EventPriority.NORMAL

    # ── Source ──
    source_agent: str = Field(
        ...,
        description="Agent qui a émis l'événement",
    )
    company_id: str = Field(
        ...,
        min_length=1,
        description="Entreprise concernée",
    )

    # ── Données ──
    payload: dict[str, Any] = Field(
        default_factory=dict,
        description="Données de l'événement. Structure dépend du type.",
    )

    # ── État ──
    created_at: datetime = Field(default_factory=datetime.utcnow)
    processed: bool = False
    processed_at: Optional[datetime] = None
    processed_by: Optional[list[str]] = Field(
        default=None,
        description="Liste des handlers qui ont traité cet événement",
    )
    error: Optional[str] = None

    # ── TTL ──
    expires_at: datetime = Field(
        default_factory=lambda: datetime.utcnow() + timedelta(days=7),
        description="L'événement expire après 7 jours s'il n'est pas traité",
    )

    # ── Validators ──

    @field_validator("priority", mode="before")
    @classmethod
    def set_default_priority(cls, v, info):
        """Si la priorité n'est pas explicite,
        utiliser la priorité par défaut du type d'événement."""
        if v is None or v == EventPriority.NORMAL:
            event_type = info.data.get("event_type")
            if event_type and event_type in _DEFAULT_PRIORITIES:
                return _DEFAULT_PRIORITIES[event_type]
        return v

    # ── Méthodes ──

    def mark_processed(self, handler_name: str) -> None:
        """Marquer l'événement comme traité par un handler."""
        self.processed = True
        self.processed_at = datetime.utcnow()
        if self.processed_by is None:
            self.processed_by = []
        self.processed_by.append(handler_name)

    def mark_error(self, error_message: str) -> None:
        """Marquer l'événement en erreur."""
        self.error = error_message

    @property
    def is_expired(self) -> bool:
        return datetime.utcnow() > self.expires_at

    @property
    def age_seconds(self) -> float:
        return (datetime.utcnow() - self.created_at).total_seconds()

    def __repr__(self) -> str:
        return (
            f"Event(type={self.event_type.value}, "
            f"company={self.company_id}, "
            f"priority={self.priority.value}, "
            f"processed={self.processed})"
  )
