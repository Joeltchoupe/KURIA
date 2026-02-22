"""
Report — Rapport hebdomadaire unifié.

Chaque lundi 7h, le CEO reçoit UN email :
  - Score de Clarté (évolution)
  - 1 KPI par agent (vert/jaune/rouge)
  - Top 3 décisions de la semaine
  - Top 3 actions en attente d'approbation
  - 1 recommandation clé

2 minutes de lecture. Tout ce qui compte.

Le Reporter (orchestrator/reporter.py) PRODUIT ce modèle.
Ce fichier DÉFINIT la structure.
"""

from __future__ import annotations

from datetime import datetime, date
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class AttentionLevel(str, Enum):
    """Niveau d'attention requis."""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"
    SUCCESS = "success"


class KPIStatus(BaseModel):
    """Snapshot d'un KPI agent pour le rapport."""
    agent: str
    metric_name: str
    value: float
    unit: str = ""
    health: str = "healthy"  # healthy, warning, critical
    trend: str = "stable"  # improving, stable, degrading
    previous_value: float | None = None
    context: str = ""


class DecisionSummary(BaseModel):
    """Résumé d'une décision pour le rapport."""
    agent: str
    decision_type: str
    confidence: float = Field(ge=0, le=1)
    reasoning: str
    actions_count: int = Field(default=0, ge=0)
    risk_level: str = "A"
    executed: bool = False


class PendingApproval(BaseModel):
    """Action en attente d'approbation."""
    action_id: str
    agent: str
    action: str
    target: str = ""
    reasoning: str = ""
    confidence: float = Field(ge=0, le=1)
    expires_at: datetime | None = None


class AdaptationNote(BaseModel):
    """Ajustement de config par l'Adaptateur."""
    agent: str
    parameter: str
    old_value: Any
    new_value: Any
    reason: str


class WeeklyReport(BaseModel):
    """
    Rapport hebdomadaire complet.

    C'est le LIVRABLE que le CEO reçoit.
    Tout le système converge vers ce document.
    """
    id: str = Field(default_factory=lambda: str(uuid4()))
    company_id: str
    company_name: str = ""
    period_start: date
    period_end: date

    # Score de Clarté
    clarity_score: float | None = Field(default=None, ge=0, le=100)
    clarity_previous: float | None = Field(default=None, ge=0, le=100)
    clarity_trend: str = "stable"
    clarity_health: str = "healthy"

    # KPIs par agent
    kpis: list[KPIStatus] = Field(default_factory=list)

    # Top décisions de la semaine
    top_decisions: list[DecisionSummary] = Field(default_factory=list)

    # Actions en attente d'approbation
    pending_approvals: list[PendingApproval] = Field(default_factory=list)

    # Recommandation clé
    key_recommendation: str = ""
    attention_level: AttentionLevel = AttentionLevel.INFO

    # Adaptations auto
    adaptations: list[AdaptationNote] = Field(default_factory=list)

    # Métriques système
    total_decisions: int = Field(default=0, ge=0)
    total_actions_executed: int = Field(default=0, ge=0)
    total_actions_pending: int = Field(default=0, ge=0)
    total_llm_cost_usd: float = Field(default=0, ge=0)
    agents_active: list[str] = Field(default_factory=list)

    # Méta
    generated_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def has_critical(self) -> bool:
        """Y a-t-il quelque chose de critique ?"""
        return (
            self.attention_level == AttentionLevel.CRITICAL
            or any(k.health == "critical" for k in self.kpis)
            or (self.clarity_health == "critical")
        )

    @property
    def summary_one_line(self) -> str:
        """Résumé en une ligne pour Slack."""
        icons = {
            "critical": "🔴",
            "warning": "🟡",
            "info": "🔵",
            "success": "🟢",
        }
        icon = icons.get(self.attention_level.value, "⚪")
        return (
            f"{icon} Kuria S{self.period_end.isocalendar()[1]} — "
            f"Clarté {self.clarity_score or '?'}/100 — "
            f"{self.total_decisions} décisions — "
            f"{self.total_actions_pending} en attente"
        )

    def format_email(self) -> str:
        """
        Formate le rapport en texte email.

        2 minutes de lecture max.
        """
        lines = [
            "📊 KURIA — Rapport Hebdomadaire",
            "=" * 40,
            f"📅 {self.period_start} → {self.period_end}",
            f"🏢 {self.company_name or self.company_id}",
            "",
        ]

        # Clarté
        if self.clarity_score is not None:
            trend_icon = {
                "improving": "📈",
                "stable": "➡️",
                "degrading": "📉",
                "initial": "🆕",
            }
            icon = trend_icon.get(self.clarity_trend, "➡️")
            lines.append(
                f"🎯 Score de Clarté : {self.clarity_score:.0f}/100 {icon}"
            )
            if self.clarity_previous is not None:
                delta = self.clarity_score - self.clarity_previous
                lines.append(f"   (variation : {delta:+.1f})")
            lines.append("")

        # KPIs
        if self.kpis:
            lines.append("📊 KPIs")
            lines.append("-" * 30)
            health_icons = {
                "healthy": "🟢",
                "warning": "🟡",
                "critical": "🔴",
            }
            for kpi in self.kpis:
                icon = health_icons.get(kpi.health, "⚪")
                line = f"  {icon} {kpi.agent} : {kpi.value} {kpi.unit}"
                if kpi.context:
                    line += f" — {kpi.context}"
                lines.append(line)
            lines.append("")

        # Top décisions
        if self.top_decisions:
            lines.append("🧠 Décisions clés")
            lines.append("-" * 30)
            for d in self.top_decisions[:3]:
                exec_icon = "✅" if d.executed else "⏳"
                lines.append(
                    f"  {exec_icon} [{d.agent}] {d.reasoning[:100]}"
                )
            lines.append("")

        # Pending approvals
        if self.pending_approvals:
            lines.append(f"⏳ {len(self.pending_approvals)} action(s) en attente")
            lines.append("-" * 30)
            for p in self.pending_approvals:
                lines.append(
                    f"  • [{p.agent}] {p.action} → {p.target}"
                )
                if p.expires_at:
                    lines.append(
                        f"    Expire : {p.expires_at.strftime('%d/%m %Hh%M')}"
                    )
            lines.append("")

        # Recommandation
        if self.key_recommendation:
            lines.append("💡 Recommandation")
            lines.append("-" * 30)
            lines.append(f"  {self.key_recommendation}")
            lines.append("")

        # Adaptations
        if self.adaptations:
            lines.append("🔧 Ajustements auto")
            lines.append("-" * 30)
            for a in self.adaptations:
                lines.append(
                    f"  • {a.agent} — {a.parameter} : "
                    f"{a.old_value} → {a.new_value}"
                )
            lines.append("")

        # Footer
        lines.append("—")
        lines.append(
            f"💰 Coût IA cette semaine : ${self.total_llm_cost_usd:.2f}"
        )
        lines.append("Généré par Kuria 明晰")

        return "\n".join(lines)
