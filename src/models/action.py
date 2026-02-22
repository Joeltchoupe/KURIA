"""
Action — Exécution et traçabilité.

Chaque action demandée par le LLM est tracée :
  - Avant exécution (pending)
  - Pendant (executing)
  - Après (completed / failed)
  - Ou en attente d'approbation (pending_approval)

RIEN ne se passe sans trace.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class ActionStatus(str, Enum):
    """Statut d'une action."""
    PENDING = "pending"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"
    REJECTED = "rejected"
    BRIEFING_ONLY = "briefing_only"


class Action(BaseModel):
    """
    Action à exécuter par l'ActionExecutor.

    Créée à partir d'une ActionRequest (décision LLM),
    enrichie avec le tracking d'exécution.
    """
    id: str = Field(default_factory=lambda: str(uuid4()))
    decision_id: str = Field(..., description="ID de la Decision source")
    company_id: str = ""
    agent_type: str = ""

    # Quoi faire
    action: str = Field(..., description="Type d'action à exécuter")
    target: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)

    # Safety
    risk_level: str = "A"
    confidence: float = Field(default=0.5, ge=0, le=1)

    # Status
    status: ActionStatus = ActionStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime | None = None
    executed_at: datetime | None = None
    approved_by: str | None = None
    approved_at: datetime | None = None

    # Résultat
    result: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None

    def set_expiry(self, hours: int = 72) -> None:
        """Définit l'expiration pour les actions B."""
        self.expires_at = self.created_at + timedelta(hours=hours)

    @property
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at

    def approve(self, by: str) -> None:
        """Approuve une action pending."""
        self.status = ActionStatus.APPROVED
        self.approved_by = by
        self.approved_at = datetime.utcnow()

    def reject(self, by: str, reason: str = "") -> None:
        """Rejette une action pending."""
        self.status = ActionStatus.REJECTED
        self.approved_by = by
        self.error = reason

    def complete(self, result: dict[str, Any] | None = None) -> None:
        """Marque l'action comme complétée."""
        self.status = ActionStatus.COMPLETED
        self.executed_at = datetime.utcnow()
        if result:
            self.result = result

    def fail(self, error: str) -> None:
        """Marque l'action comme échouée."""
        self.status = ActionStatus.FAILED
        self.executed_at = datetime.utcnow()
        self.error = error


class ActionLog(BaseModel):
    """
    Log d'audit universel.

    CHAQUE action, CHAQUE décision, CHAQUE événement
    est loggé ici. C'est la source de vérité absolue.
    """
    id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    company_id: str = ""
    agent_type: str = ""

    # Quoi
    action_id: str = ""
    decision_id: str = ""
    event_type: str = ""
    description: str = ""

    # Contexte
    input_snapshot: dict[str, Any] = Field(default_factory=dict)
    output_decision: dict[str, Any] = Field(default_factory=dict)
    output_result: dict[str, Any] = Field(default_factory=dict)

    # Métriques
    llm_provider: str = ""
    llm_model: str = ""
    llm_tokens: int = Field(default=0, ge=0)
    llm_cost_usd: float = Field(default=0, ge=0)
    latency_ms: float = Field(default=0, ge=0)

    # Status
    success: bool = True
    error: str | None = None
