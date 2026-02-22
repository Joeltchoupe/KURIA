"""
Decision — Sortie structurée du LLM.

Le LLM ne parle JAMAIS directement à l'extérieur.
Il produit UNIQUEMENT un JSON structuré : la Decision.

Puis le système valide, logge, et exécute.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class DecisionType(str, Enum):
    """Types de décisions que le LLM peut prendre."""

    # Revenue Velocity
    CLASSIFY_DEAL = "classify_deal"
    SCORE_LEAD = "score_lead"
    FORECAST_REVENUE = "forecast_revenue"
    GENERATE_PROPOSAL = "generate_proposal"
    ANALYZE_WIN_LOSS = "analyze_win_loss"

    # Process Clarity
    GENERATE_SOP = "generate_sop"
    ROUTE_TASK = "route_task"
    DETECT_BOTTLENECK = "detect_bottleneck"
    GENERATE_CHECKIN = "generate_checkin"
    DETECT_DUPLICATE = "detect_duplicate"

    # Cash Predictability
    FORECAST_CASH = "forecast_cash"
    DETECT_PAYMENT_RISK = "detect_payment_risk"
    GENERATE_CASH_PLAN = "generate_cash_plan"

    # Acquisition Efficiency
    ANALYZE_CHANNELS = "analyze_channels"
    REALLOCATE_BUDGET = "reallocate_budget"
    SCORE_LEAD_QUALITY = "score_lead_quality"

    # Generic
    ALERT = "alert"
    RECOMMEND = "recommend"
    SUMMARIZE = "summarize"


class RiskLevel(str, Enum):
    """Niveau de risque d'une action.

    A → exécution directe (safe)
    B → queue pending_actions, validation 72h
    C → briefing généré, jamais exécuté auto
    """
    A = "A"  # Auto-execute
    B = "B"  # Needs approval
    C = "C"  # Briefing only


class Decision(BaseModel):
    """
    Décision structurée produite par le LLM.

    C'est le CONTRAT entre le LLM et le système.
    Le LLM remplit ce format. Le système l'exécute.
    """
    id: str = Field(default_factory=lambda: str(uuid4()))
    agent_type: str
    decision_type: DecisionType
    confidence: float = Field(..., ge=0, le=1)
    reasoning: str = Field(
        ..., description="Explication de la décision par le LLM"
    )
    actions: list[ActionRequest] = Field(
        default_factory=list,
        description="Actions à exécuter suite à cette décision",
    )
    risk_level: RiskLevel = RiskLevel.A
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    snapshot_id: str = Field(
        default="", description="ID du StateSnapshot utilisé"
    )

    # Résultat de la validation
    validated: bool = False
    validation_errors: list[str] = Field(default_factory=list)
    executed: bool = False
    executed_at: datetime | None = None

    @property
    def is_safe_to_execute(self) -> bool:
        """La décision peut être exécutée automatiquement ?"""
        return (
            self.validated
            and self.risk_level == RiskLevel.A
            and self.confidence >= 0.7
            and not self.validation_errors
        )

    @property
    def needs_approval(self) -> bool:
        return self.risk_level == RiskLevel.B

    @property
    def briefing_only(self) -> bool:
        return self.risk_level == RiskLevel.C


class ActionRequest(BaseModel):
    """
    Action demandée par le LLM.

    Le LLM ne fait que DEMANDER.
    L'ActionExecutor FAIT.
    """
    action: str = Field(
        ..., description="Ex: update_deal_stage, send_slack, create_task"
    )
    target: str = Field(
        default="", description="Cible : deal_id, user_email, channel"
    )
    parameters: dict[str, Any] = Field(default_factory=dict)
    risk_level: RiskLevel = RiskLevel.A
    priority: int = Field(default=5, ge=1, le=10)
