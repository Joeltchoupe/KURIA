"""
Report — Le rapport unifié du lundi matin.

UN email. 2 minutes de lecture. Tout ce qui compte.
L'orchestrateur compile les 4 rapports agents en UN SEUL.

Design decisions :
- Le rapport est un objet structuré (pas du texte brut)
- Chaque section = 1 agent
- Les AttentionItems sont triés par niveau (red > yellow > info)
- Le format final (email, dashboard) est généré PAR-DESSUS le modèle
- Le ROI cumulé est calculé et affiché (justification du pricing)
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from models.clarity_score import ClarityScore
from models.metrics import MetricTrend


class AttentionLevel(str, Enum):
    """Niveau d'attention requis."""

    RED = "red"          # Action immédiate requise
    YELLOW = "yellow"    # À surveiller cette semaine
    INFO = "info"        # Pour information


class AttentionItem(BaseModel):
    """Un point nécessitant l'attention du CEO."""

    level: AttentionLevel
    department: str
    title: str = Field(..., min_length=1, max_length=256)
    detail: str
    suggested_action: Optional[str] = None
    potential_value: Optional[float] = Field(
        default=None,
        ge=0,
        description="Valeur en € de l'action si elle est prise",
    )

    @property
    def priority_order(self) -> int:
        """Pour le tri : RED=0, YELLOW=1, INFO=2."""
        return {"red": 0, "yellow": 1, "info": 2}[self.level.value]


class ReportSection(BaseModel):
    """Section du rapport pour un agent."""

    agent_name: str
    agent_display_name: str
    kpi_name: str
    kpi_value: str = Field(
        ...,
        description="Valeur formatée du KPI (ex: '4,200€/jour')",
    )
    kpi_trend: Optional[MetricTrend] = None
    summary: str = Field(
        ...,
        min_length=1,
        description="Résumé en 1-2 phrases de l'activité de la semaine",
    )
    highlights: list[str] = Field(
        default_factory=list,
        description="Points clés en bullet points",
    )
    attention_items: list[AttentionItem] = Field(default_factory=list)
    actions_taken: int = Field(
        default=0,
        ge=0,
        description="Nombre d'actions prises par l'agent cette semaine",
    )
    confidence: float = Field(
        default=1.0,
        ge=0,
        le=1.0,
    )

    @property
    def has_red_items(self) -> bool:
        return any(
            item.level == AttentionLevel.RED for item in self.attention_items
        )


class WeeklyReport(BaseModel):
    """Le rapport unifié du lundi matin.

    Compilé par le WeeklyReportCompiler de l'orchestrateur.
    Envoyé par le NotificationService.
    Affiché dans le dashboard.
    """

    # ── Identité ──
    company_id: str = Field(..., min_length=1)
    company_name: str = Field(..., min_length=1)
    report_week: str = Field(
        ...,
        description="Semaine du rapport (ex: '2026-W03')",
    )
    generated_at: datetime = Field(default_factory=datetime.utcnow)

    # ── Score de Clarté ──
    clarity_score: ClarityScore

    # ── Sections agents ──
    sections: list[ReportSection] = Field(
        ...,
        min_length=1,
        description="Une section par agent actif",
    )

    # ── Items d'attention agrégés ──
    # (extraits des sections, triés par priorité)

    # ── Recommandation principale ──
    primary_recommendation: str = Field(
        ...,
        min_length=1,
        description=(
            "LA recommandation #1 de la semaine. "
            "Une seule. Claire. Actionnable."
        ),
    )
    primary_recommendation_value: Optional[float] = Field(
        default=None,
        ge=0,
        description="Valeur estimée si la recommandation est suivie",
    )

    # ── ROI ──
    roi_cumulated: float = Field(
        default=0,
        ge=0,
        description="ROI cumulé depuis le déploiement des agents",
    )

    # ── Propriétés calculées ──

    @property
    def all_attention_items(self) -> list[AttentionItem]:
        """Tous les items d'attention, triés par priorité."""
        items = []
        for section in self.sections:
            items.extend(section.attention_items)
        return sorted(items, key=lambda i: i.priority_order)

    @property
    def red_items(self) -> list[AttentionItem]:
        return [
            i for i in self.all_attention_items
            if i.level == AttentionLevel.RED
        ]

    @property
    def yellow_items(self) -> list[AttentionItem]:
        return [
            i for i in self.all_attention_items
            if i.level == AttentionLevel.YELLOW
        ]

    @property
    def has_critical_items(self) -> bool:
        return len(self.red_items) > 0

    @property
    def total_actions_taken(self) -> int:
        return sum(s.actions_taken for s in self.sections)

    @property
    def total_potential_value(self) -> float:
        """Valeur totale des actions suggérées."""
        return sum(
            item.potential_value
            for item in self.all_attention_items
            if item.potential_value is not None
        )

    # ── Formatage ──

    def format_email_subject(self) -> str:
        """Sujet de l'email du lundi."""
        score = self.clarity_score
        trend = score.trend_emoji
        score_str = f"{score.overall:.0f}/100"
        trend_str = ""
        if score.trend is not None:
            sign = "+" if score.trend >= 0 else ""
            trend_str = f" ({trend}{sign}{score.trend:.0f})"

        alert = " 🔴" if self.has_critical_items else ""
        return f"Kuria — {self.report_week} — Score {score_str}{trend_str}{alert}"

    def format_email_body(self) -> str:
        """Corps de l'email du lundi.
        2 minutes de lecture max. Tout ce qui compte."""

        lines = [
            f"Bonjour,\n",
            self.clarity_score.format_for_email(),
            "",
        ]

        # Ce qui va bien
        good_sections = [
            s for s in self.sections
            if not s.has_red_items and s.confidence >= 0.6
        ]
        if good_sections:
            lines.append("CE QUI VA BIEN :")
            for s in good_sections:
                kpi_line = s.kpi_value
                if s.kpi_trend and s.kpi_trend.delta_pct is not None:
                    sign = "+" if s.kpi_trend.delta_pct >= 0 else ""
                    kpi_line += f" ({s.kpi_trend.emoji}{sign}{s.kpi_trend.delta_pct:.0f}%)"
                lines.append(f"→ {s.agent_display_name} : {kpi_line}")
            lines.append("")

        # Ce qui nécessite attention
        if self.all_attention_items:
            lines.append("CE QUI NÉCESSITE ATTENTION :")
            for item in self.all_attention_items[:5]:  # Max 5
                prefix = "🔴" if item.level == AttentionLevel.RED else "🟡"
                lines.append(f"{prefix} {item.title}")
                if item.detail:
                    lines.append(f"   {item.detail}")
            lines.append("")

        # Recommandation principale
        lines.append(f"ACTION RECOMMANDÉE #1 :")
        lines.append(f"→ {self.primary_recommendation}")
        if self.primary_recommendation_value:
            lines.append(
                f"   Impact estimé : {self.primary_recommendation_value:,.0f}€"
            )
        lines.append("")

        # ROI
        if self.roi_cumulated > 0:
            lines.append(
                f"ROI CUMULÉ KURIA : {self.roi_cumulated:,.0f}€"
            )
            lines.append("")

        lines.append("[Ouvrir le dashboard →]")

        return "\n".join(lines)
