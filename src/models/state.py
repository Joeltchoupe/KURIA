"""
State — État compressé envoyé au LLM.

On ne recalcule JAMAIS à partir du raw en live.
Chaque agent travaille sur un state propre et structuré.

Le state est la MÉMOIRE DE TRAVAIL de l'agent.
Il est mis à jour par les events, lu par le LLM.
"""

from __future__ import annotations

from datetime import datetime, date
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class DealState(BaseModel):
    """État d'un deal dans le pipeline."""
    deal_id: str
    name: str = ""
    amount: float = Field(default=0, ge=0)
    stage: str = ""
    owner: str = ""
    created_at: datetime | None = None
    last_activity_at: datetime | None = None
    days_in_stage: int = Field(default=0, ge=0)
    days_no_activity: int = Field(default=0, ge=0)
    probability: float = Field(default=0.5, ge=0, le=1)
    source: str = ""
    contact_email: str = ""
    notes: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class TaskState(BaseModel):
    """État d'une tâche."""
    task_id: str
    title: str = ""
    assigned_to: str = ""
    status: str = ""
    deadline: datetime | None = None
    created_at: datetime | None = None
    completed_at: datetime | None = None
    days_overdue: int = Field(default=0, ge=0)
    process_name: str = ""
    tags: list[str] = Field(default_factory=list)


class CashState(BaseModel):
    """État de la trésorerie."""
    cash_balance: float = 0
    monthly_revenue: float = 0
    monthly_expenses: float = 0
    monthly_burn_rate: float = 0
    runway_months: float = 0
    outstanding_receivables: float = 0
    outstanding_payables: float = 0
    avg_payment_delay_days: float = 0
    overdue_invoices_count: int = Field(default=0, ge=0)
    overdue_invoices_amount: float = 0
    pipeline_weighted_value: float = 0
    as_of: date = Field(default_factory=date.today)


class MarketingState(BaseModel):
    """État marketing / acquisition."""
    channels: list[ChannelState] = Field(default_factory=list)
    total_leads_30d: int = Field(default=0, ge=0)
    total_spend_30d: float = Field(default=0, ge=0)
    blended_cac: float = Field(default=0, ge=0)
    avg_ltv: float = Field(default=0, ge=0)
    ltv_cac_ratio: float = Field(default=0, ge=0)
    as_of: date = Field(default_factory=date.today)


class ChannelState(BaseModel):
    """État d'un canal marketing."""
    channel: str
    spend_30d: float = Field(default=0, ge=0)
    leads_30d: int = Field(default=0, ge=0)
    clients_30d: int = Field(default=0, ge=0)
    cac: float = Field(default=0, ge=0)
    conversion_rate: float = Field(default=0, ge=0, le=1)
    trend_pct: float = 0  # % change vs previous period


# Fix forward reference
MarketingState.model_rebuild()


class StateSnapshot(BaseModel):
    """
    Snapshot complet envoyé au LLM.

    C'est le CONTEXTE que l'agent reçoit pour prendre une décision.
    Contient uniquement ce qui est pertinent pour l'agent.
    """
    id: str = Field(default_factory=lambda: str(uuid4()))
    company_id: str
    agent_type: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # States (chaque agent utilise ce dont il a besoin)
    deals: list[DealState] = Field(default_factory=list)
    tasks: list[TaskState] = Field(default_factory=list)
    cash: CashState | None = None
    marketing: MarketingState | None = None

    # Events récents pertinents
    recent_events: list[dict[str, Any]] = Field(default_factory=list)

    # Contexte additionnel
    company_context: dict[str, Any] = Field(
        default_factory=dict,
        description="Secteur, taille, historique, patterns connus",
    )
    previous_decisions: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Dernières décisions de cet agent (feedback loop)",
    )

    def to_llm_payload(self) -> str:
        """Sérialise le snapshot en JSON compact pour le LLM."""
        import json
        data = self.model_dump(exclude_none=True, exclude={"id"})
        # Convertir les dates en strings
        return json.dumps(data, default=str, ensure_ascii=False, indent=2)

    @property
    def summary(self) -> dict[str, Any]:
        """Résumé compact pour le logging."""
        return {
            "company_id": self.company_id,
            "agent": self.agent_type,
            "deals": len(self.deals),
            "tasks": len(self.tasks),
            "has_cash": self.cash is not None,
            "has_marketing": self.marketing is not None,
            "recent_events": len(self.recent_events),
  }
