"""
WeeklyReporter — Compile les outputs des 4 agents en 1 rapport.

Chaque lundi à 7h, le CEO reçoit UN email avec :
  - Le Score de Clarté (évolution)
  - 1 KPI par agent (vert/jaune/rouge)
  - Les frictions top 3
  - 1 recommandation clé
  - Les ajustements de l'Adaptateur

2 minutes de lecture. Tout ce qui compte.

Design decisions :
  - Le rapport est un WeeklyReport (modèle Pydantic)
  - Le Reporter ne fait que COMPILER — il n'analyse pas
  - Chaque section est autonome (peut être affichée séparément)
  - Le format est prêt pour email (texte) et dashboard (JSON)
"""

from __future__ import annotations

from datetime import datetime, date
from typing import Any

from pydantic import BaseModel, Field

from models.agent_config import AgentType
from models.metrics import MetricTrend, HealthStatus


# ══════════════════════════════════════════════════════════════
# MODÈLES DU RAPPORT
# ══════════════════════════════════════════════════════════════


class KPISnapshot(BaseModel):
    """Snapshot d'un KPI pour le rapport."""
    agent_type: AgentType
    agent_name: str
    metric_name: str
    value: float
    unit: str
    health: str  # "healthy", "warning", "critical"
    trend: str = "stable"  # "improving", "stable", "degrading"
    previous_value: float | None = None
    context: str = ""


class FrictionHighlight(BaseModel):
    """Friction mise en avant dans le rapport."""
    rank: int = Field(..., ge=1, le=5)
    title: str
    source_agent: str
    severity: str
    estimated_cost_monthly: float = 0
    recommended_action: str = ""


class AdaptationNote(BaseModel):
    """Note d'adaptation pour le rapport."""
    agent: str
    parameter: str
    change: str
    reason: str


class WeeklyReportData(BaseModel):
    """Données complètes du rapport hebdomadaire."""
    id: str
    company_id: str
    company_name: str = ""
    period_start: date
    period_end: date

    # Score de Clarté
    clarity_score: float | None = None
    clarity_score_previous: float | None = None
    clarity_trend: str = "stable"

    # KPIs
    kpis: list[KPISnapshot] = Field(default_factory=list)

    # Frictions
    top_frictions: list[FrictionHighlight] = Field(default_factory=list)

    # Recommandation
    key_recommendation: str = ""

    # Adaptations
    adaptations: list[AdaptationNote] = Field(default_factory=list)

    # Méta
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    agents_active: list[str] = Field(default_factory=list)


# ══════════════════════════════════════════════════════════════
# REPORTER
# ══════════════════════════════════════════════════════════════


class WeeklyReporter:
    """
    Compile les résultats des agents en rapport hebdomadaire.

    Usage :
        reporter = WeeklyReporter()
        report = reporter.generate(
            run_result=weekly_run_result,
            company_id="acme",
            company_name="Acme Corp",
            adaptation_result=adaptation_result,
        )
        email_text = reporter.format_email(report)
    """

    def __init__(self) -> None:
        self._reports: list[WeeklyReportData] = []

    def generate(
        self,
        run_result: Any,
        company_id: str,
        company_name: str = "",
        adaptation_result: Any | None = None,
        previous_report: WeeklyReportData | None = None,
    ) -> WeeklyReportData:
        """
        Génère le rapport hebdomadaire.

        Args:
            run_result: RunResult du cycle weekly.
            company_id: ID de l'entreprise.
            company_name: Nom de l'entreprise.
            adaptation_result: Résultat de l'adaptation (optionnel).
            previous_report: Rapport précédent pour les trends.
        """
        from uuid import uuid4
        now = datetime.utcnow()

        report = WeeklyReportData(
            id=str(uuid4()),
            company_id=company_id,
            company_name=company_name,
            period_start=(now - __import__("datetime").timedelta(days=7)).date(),
            period_end=now.date(),
        )

        # KPIs depuis les agent runs
        if run_result:
            report.kpis = self._extract_kpis(run_result, previous_report)
            report.top_frictions = self._extract_frictions(run_result)
            report.key_recommendation = self._generate_recommendation(report)
            report.agents_active = [
                r.agent_type.value
                for r in getattr(run_result, "agent_runs", [])
                if getattr(r, "status", None)
                and r.status.value == "success"
            ]

        # Adaptations
        if adaptation_result:
            report.adaptations = self._extract_adaptations(adaptation_result)

        self._reports.append(report)
        return report

    def format_email(self, report: WeeklyReportData) -> str:
        """
        Formate le rapport en texte pour email.

        Format : 2 minutes de lecture max.
        """
        lines = [
            f"📊 KURIA — Rapport Hebdomadaire",
            f"{'=' * 40}",
            f"📅 {report.period_start} → {report.period_end}",
            f"🏢 {report.company_name or report.company_id}",
            "",
        ]

        # Score de Clarté
        if report.clarity_score is not None:
            trend_icon = {"improving": "📈", "stable": "➡️", "degrading": "📉"}
            icon = trend_icon.get(report.clarity_trend, "➡️")
            lines.append(f"🎯 Score de Clarté : {report.clarity_score}/100 {icon}")
            if report.clarity_score_previous is not None:
                delta = report.clarity_score - report.clarity_score_previous
                lines.append(f"   (vs semaine dernière : {delta:+.1f})")
            lines.append("")

        # KPIs
        if report.kpis:
            lines.append("📊 KPIs")
            lines.append("-" * 30)
            health_icons = {"healthy": "🟢", "warning": "🟡", "critical": "🔴"}
            for kpi in report.kpis:
                icon = health_icons.get(kpi.health, "⚪")
                lines.append(
                    f"  {icon} {kpi.agent_name} : "
                    f"{kpi.value} {kpi.unit}"
                )
                if kpi.context:
                    lines.append(f"     → {kpi.context}")
            lines.append("")

        # Top frictions
        if report.top_frictions:
            lines.append("⚠️ Frictions prioritaires")
            lines.append("-" * 30)
            for f in report.top_frictions:
                lines.append(
                    f"  {f.rank}. {f.title} "
                    f"({f.severity}, ~{f.estimated_cost_monthly:.0f}€/mois)"
                )
                if f.recommended_action:
                    lines.append(f"     → {f.recommended_action}")
            lines.append("")

        # Recommandation
        if report.key_recommendation:
            lines.append("💡 Recommandation clé")
            lines.append("-" * 30)
            lines.append(f"  {report.key_recommendation}")
            lines.append("")

        # Adaptations
        if report.adaptations:
            lines.append("🔧 Ajustements automatiques")
            lines.append("-" * 30)
            for a in report.adaptations:
                lines.append(f"  • {a.agent} — {a.parameter} : {a.change}")
                lines.append(f"    Raison : {a.reason}")
            lines.append("")

        lines.append("—")
        lines.append("Généré par Kuria 明晰")

        return "\n".join(lines)

    # ──────────────────────────────────────────────────────
    # EXTRACTION
    # ──────────────────────────────────────────────────────

    def _extract_kpis(
        self,
        run_result: Any,
        previous: WeeklyReportData | None,
    ) -> list[KPISnapshot]:
        """Extrait les KPIs depuis un RunResult."""
        kpis: list[KPISnapshot] = []
        agent_names = {
            AgentType.REVENUE_VELOCITY: "Revenue Velocity",
            AgentType.PROCESS_CLARITY: "Process Clarity",
            AgentType.CASH_PREDICTABILITY: "Cash Predictability",
            AgentType.ACQUISITION_EFFICIENCY: "Acquisition Efficiency",
        }
        metric_names = {
            AgentType.REVENUE_VELOCITY: ("revenue_velocity_eur_day", "€/jour"),
            AgentType.PROCESS_CLARITY: ("cycle_time_days", "jours"),
            AgentType.CASH_PREDICTABILITY: ("forecast_accuracy_30d", "%"),
            AgentType.ACQUISITION_EFFICIENCY: ("blended_cac", "€/client"),
        }

        for record in getattr(run_result, "agent_runs", []):
            agent_type = getattr(record, "agent_type", None)
            if agent_type is None or agent_type == AgentType.SCANNER:
                continue

            kpi_value = getattr(record, "kpi_value", None)
            if kpi_value is None:
                continue

            metric, unit = metric_names.get(agent_type, ("unknown", ""))
            name = agent_names.get(agent_type, agent_type.value)

            # Trend vs précédent
            prev_value = None
            trend = "stable"
            if previous:
                for prev_kpi in previous.kpis:
                    if prev_kpi.agent_type == agent_type:
                        prev_value = prev_kpi.value
                        if kpi_value > prev_value * 1.05:
                            trend = "improving"
                        elif kpi_value < prev_value * 0.95:
                            trend = "degrading"
                        break

            # Health
            frictions = getattr(record, "frictions_detected", 0)
            if frictions > 3:
                health = "critical"
            elif frictions > 0:
                health = "warning"
            else:
                health = "healthy"

            kpis.append(KPISnapshot(
                agent_type=agent_type,
                agent_name=name,
                metric_name=metric,
                value=round(kpi_value, 2),
                unit=unit,
                health=health,
                trend=trend,
                previous_value=prev_value,
            ))

        return kpis

    def _extract_frictions(self, run_result: Any) -> list[FrictionHighlight]:
        """Extrait les top 3 frictions depuis un RunResult."""
        all_frictions: list[dict[str, Any]] = []

        for record in getattr(run_result, "agent_runs", []):
            result = getattr(record, "result", {})
            if isinstance(result, dict):
                frictions_count = result.get("frictions_detected", 0)
                if frictions_count > 0:
                    all_frictions.append({
                        "agent": getattr(record, "agent_type", AgentType.SCANNER).value,
                        "count": frictions_count,
                        "recommendation": result.get("recommendation", ""),
                    })

        # Trier par count et prendre top 3
        all_frictions.sort(key=lambda f: f["count"], reverse=True)

        highlights: list[FrictionHighlight] = []
        for i, f in enumerate(all_frictions[:3]):
            highlights.append(FrictionHighlight(
                rank=i + 1,
                title=f"Frictions détectées par {f['agent']}",
                source_agent=f["agent"],
                severity="high" if f["count"] > 3 else "medium",
                recommended_action=f.get("recommendation", ""),
            ))

        return highlights

    def _extract_adaptations(
        self, adaptation_result: Any
    ) -> list[AdaptationNote]:
        """Extrait les adaptations depuis un AdaptationResult."""
        notes: list[AdaptationNote] = []

        adjustments = getattr(adaptation_result, "adjustments", [])
        for adj in adjustments:
            notes.append(AdaptationNote(
                agent=getattr(adj, "agent_type", "unknown").value
                if hasattr(getattr(adj, "agent_type", None), "value")
                else str(getattr(adj, "agent_type", "unknown")),
                parameter=getattr(adj, "parameter", ""),
                change=f"{getattr(adj, 'old_value', '?')} → {getattr(adj, 'new_value', '?')}",
                reason=getattr(adj, "reason", ""),
            ))

        return notes

    def _generate_recommendation(
        self, report: WeeklyReportData
    ) -> str:
        """Génère LA recommandation clé du rapport."""
        critical_kpis = [k for k in report.kpis if k.health == "critical"]
        if critical_kpis:
            kpi = critical_kpis[0]
            return (
                f"PRIORITÉ : {kpi.agent_name} en zone critique "
                f"({kpi.value} {kpi.unit}). Action immédiate requise."
            )

        degrading_kpis = [k for k in report.kpis if k.trend == "degrading"]
        if degrading_kpis:
            kpi = degrading_kpis[0]
            return (
                f"ATTENTION : {kpi.agent_name} en dégradation "
                f"({kpi.previous_value} → {kpi.value} {kpi.unit}). "
                f"Surveiller cette semaine."
            )

        if report.top_frictions:
            friction = report.top_frictions[0]
            return (
                f"Focus : {friction.title}. {friction.recommended_action}"
            )

        return "Opérations stables. Maintenir le cap."

    # ──────────────────────────────────────────────────────
    # HISTORY
    # ──────────────────────────────────────────────────────

    @property
    def last_report(self) -> WeeklyReportData | None:
        return self._reports[-1] if self._reports else None

    @property
    def report_count(self) -> int:
        return len(self._reports)

    def get_report(self, report_id: str) -> WeeklyReportData | None:
        """Récupère un rapport par ID."""
        for r in self._reports:
            if r.id == report_id:
                return r
        return None
