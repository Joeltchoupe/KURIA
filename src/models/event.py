"""
Event — Tout ce qui se passe dans l'entreprise.

Chaque webhook, chaque changement CRM, chaque email,
chaque paiement devient un Event normalisé.

C'est l'INPUT du système. Tout part de là.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class EventType(str, Enum):
    """Types d'événements normalisés."""

    # CRM
    DEAL_CREATED = "deal.created"
    DEAL_UPDATED = "deal.updated"
    DEAL_STAGE_CHANGED = "deal.stage_changed"
    DEAL_WON = "deal.won"
    DEAL_LOST = "deal.lost"
    CONTACT_CREATED = "contact.created"
    CONTACT_UPDATED = "contact.updated"

    # Tasks
    TASK_CREATED = "task.created"
    TASK_ASSIGNED = "task.assigned"
    TASK_COMPLETED = "task.completed"
    TASK_OVERDUE = "task.overdue"

    # Finance
    INVOICE_CREATED = "invoice.created"
    INVOICE_PAID = "invoice.paid"
    INVOICE_OVERDUE = "invoice.overdue"
    PAYMENT_RECEIVED = "payment.received"
    EXPENSE_CREATED = "expense.created"

    # Email
    EMAIL_SENT = "email.sent"
    EMAIL_RECEIVED = "email.received"
    EMAIL_OPENED = "email.opened"

    # Marketing
    LEAD_CREATED = "lead.created"
    CAMPAIGN_UPDATED = "campaign.updated"
    AD_SPEND_UPDATED = "ad_spend.updated"

    # System
    AGENT_RUN = "agent.run"
    ALERT_TRIGGERED = "alert.triggered"
    USER_ACTION = "user.action"

    # Generic
    CUSTOM = "custom"


class Event(BaseModel):
    """
    Événement normalisé.

    Tout ce qui se passe dans les outils connectés
    devient un Event avec ce format.
    """
    id: str = Field(default_factory=lambda: str(uuid4()))
    event_type: EventType
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    company_id: str = ""
    source: str = Field(
        ..., description="Outil source : hubspot, stripe, gmail, slack, etc."
    )
    actor: str = Field(
        default="", description="Qui a déclenché : user email, system, agent"
    )
    payload: dict[str, Any] = Field(
        default_factory=dict,
        description="Données brutes de l'événement",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Métadonnées : IP, user_agent, etc.",
    )
    processed: bool = False
    processed_at: datetime | None = None
    processed_by: str | None = None

    def mark_processed(self, agent: str) -> None:
        """Marque l'événement comme traité par un agent."""
        self.processed = True
        self.processed_at = datetime.utcnow()
        self.processed_by = agent

    @property
    def age_seconds(self) -> float:
        """Âge de l'événement en secondes."""
        return (datetime.utcnow() - self.timestamp).total_seconds()

    def to_snapshot_dict(self) -> dict[str, Any]:
        """Format compact pour injection dans un state snapshot."""
        return {
            "type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "actor": self.actor,
            "payload": self.payload,
  }
