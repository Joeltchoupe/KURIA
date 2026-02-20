"""
ScannerAgent — Agent de diagnostic initial (Phase 1 du scan).

C'est l'agent qui tourne AVANT la session consulting.
Il aspire les données, détecte les anomalies, et génère
le Data Portrait qui servira de brief au consultant.

Ce n'est PAS un agent opérationnel (il ne tourne pas au quotidien).
Il tourne UNE FOIS lors du scan, puis ponctuellement pour les re-scans.

Output principal : DataPortrait (models/data_portrait.py)
Output secondaire : ClarityScore (models/clarity_score.py)

Design decisions :
- Mélange LLM + logique déterministe
  → Les calculs factuels (moyennes, ratios) sont déterministes
  → L'interprétation (anomalies, patterns) utilise le LLM
- Le Score de Clarté est CALCULÉ, pas estimé par le LLM
  → Formule explicite, reproductible, auditable
- Le Data Portrait est structuré via Pydantic, pas du texte libre
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from models.agent_config import BaseAgentConfig
from models.clarity_score import ClarityScore, DepartmentScores
from models.data_portrait import (
    Anomaly,
    AnomalyType,
    DataPortrait,
    FinanceAnalysis,
    MarketingAnalysis,
    OperationsAnalysis,
    PersonDependency,
    SalesAnalysis,
    TopExpense,
)
from models.events import EventType
from models.friction import Friction, FrictionMap, FrictionSeverity, PriorityQuadrant

from agents.base import AgentError, BaseAgent, InsufficientDataError


class ScannerAgent(BaseAgent):
    """Agent de diagnostic initial.

    Cycle :
    1. Reçoit les données de tous les connecteurs
    2. Analyse chaque département (sales, ops, finance, marketing)
    3. Calcule le Score de Clarté
    4. Identifie les frictions
    5. Génère le Data Portrait
    """

    AGENT_NAME = "scanner"

    # Seuils pour le calcul du Score de Clarté
    # Ces constantes sont les CRITÈRES de scoring.
    # Chaque critère note de 0-100 sa dimension.

    # Machine Readability weights
    MR_WEIGHT_DATA_COMPLETENESS = 0.30
    MR_WEIGHT_DATA_FRESHNESS = 0.25
    MR_WEIGHT_DATA_STRUCTURE = 0.25
    MR_WEIGHT_TOOL_COVERAGE = 0.20

    # Structural Compatibility weights
    SC_WEIGHT_PROCESS_EXPLICITNESS = 0.30
    SC_WEIGHT_ROLE_CLARITY = 0.25
    SC_WEIGHT_DATA_CENTRALIZATION = 0.25
    SC_WEIGHT_DECISION_TRACEABILITY = 0.20

    async def analyze(self, data: dict[str, Any]) -> dict[str, Any]:
        """Analyse complète : CRM + Email + Finance + Marketing.

        Args:
            data: {
                "crm": output de HubSpotConnector.extract(),
                "email": output de GmailConnector.extract() (optional),
                "finance": output de QuickBooksConnector.extract() (optional),
                "company_info": {
                    "employee_count": int,
                    "annual_revenue": float,
                    "sector": str,
                }
            }

        Returns:
            DataPortrait.model_dump()
        """
        company_info = data.get("company_info", {})
        crm_data = data.get("crm", {})
        email_data = data.get("email")
        finance_data = data.get("finance")

        # Analyser chaque département
        sales = await self._analyze_sales(crm_data, company_info)
        operations = await self._analyze_operations(email_data, crm_data)
        finance = await self._analyze_finance(finance_data, company_info)
        marketing = self._analyze_marketing(crm_data)

        # Calculer les tools connectés
        tools_connected = ["crm"]
        if email_data:
            tools_connected.append("email")
        if finance_data:
            tools_connected.append("finance")

        # Calculer le Score de Clarté
        clarity_score = self._calculate_clarity_score(
            sales=sales,
            operations=operations,
            finance=finance,
            marketing=marketing,
            tools_connected=tools_connected,
        )

        # Identifier les frictions
        frictions = await self._identify_frictions(
            sales=sales,
            operations=operations,
            finance=finance,
            marketing=marketing,
            company_info=company_info,
        )

        # Calculer le coût total
        total_cost = sum(f.estimated_annual_cost for f in frictions)

        # Assembler le Data Portrait
        portrait = DataPortrait(
            company_id=self.company_id,
            scan_date=datetime.now(tz=timezone.utc),
            tools_connected=tools_connected,
            sales=sales,
            operations=operations,
            finance=finance,
            marketing=marketing,
            clarity_score=clarity_score,
            estimated_total_implicit_cost=total_cost,
        )

        # Publier l'événement scan_completed
        self.publish_event(
            event_type=EventType.SCAN_COMPLETED,
            payload={
                "company_id": self.company_id,
                "clarity_score": clarity_score.overall,
                "frictions_count": len(frictions),
                "estimated_cost": total_cost,
            },
        )

        # Sauvegarder le portrait en DB
        await self.db.save_snapshot(
            company_id=self.company_id,
            snapshot_type="data_portrait",
            data=portrait.model_dump(),
        )

        return {
            "portrait": portrait.model_dump(),
            "frictions": [f.model_dump() for f in frictions],
            "friction_map": FrictionMap(
                company_id=self.company_id,
                frictions=frictions,
            ).model_dump(),
        }

    # ── Validation ──

    def _validate_input(self, data: dict[str, Any]) -> list[str]:
        warnings = []
        crm = data.get("crm", {})

        if not crm:
            raise InsufficientDataError(
                agent_name=self.AGENT_NAME,
                detail="No CRM data provided. At minimum, CRM data is required.",
            )

        deals = crm.get("deals", [])
        if len(deals) < 5:
            warnings.append(
                f"Only {len(deals)} deals found. Analysis may be unreliable."
            )

        if not data.get("email"):
            warnings.append("No email data. Operations analysis will be limited.")

        if not data.get("finance"):
            warnings.append("No finance data. Cash analysis will use estimates.")

        return warnings

    def _calculate_confidence(
        self,
        input_data: dict[str, Any],
        output_data: dict[str, Any],
    ) -> float:
        """Confiance basée sur la quantité et qualité des données."""
        score = 0.0
        max_score = 0.0

        # CRM data quality
        crm = input_data.get("crm", {})
        deals = crm.get("deals", [])
        contacts = crm.get("contacts", [])

        max_score += 40
        if len(deals) >= 30:
            score += 40
        elif len(deals) >= 10:
            score += 25
        elif len(deals) >= 5:
            score += 15
        else:
            score += 5

        max_score += 20
        if len(contacts) >= 50:
            score += 20
        elif len(contacts) >= 20:
            score += 12
        else:
            score += 5

        # Email data
        max_score += 20
        if input_data.get("email"):
            email_summary = input_data["email"].get("summary", {})
            if email_summary.get("total_messages", 0) >= 100:
                score += 20
            else:
                score += 10
        # pas d'email = 0 points

        # Finance data
        max_score += 20
        if input_data.get("finance"):
            finance_summary = input_data["finance"].get("summary", {})
            if finance_summary.get("cash_position") is not None:
                score += 20
            else:
                score += 10

        return round(score / max_score, 2) if max_score > 0 else 0.0

    # ──────────────────────────────────────────
    # SALES ANALYSIS
    # ──────────────────────────────────────────

    async def _analyze_sales(
        self,
        crm_data: dict[str, Any],
        company_info: dict[str, Any],
    ) -> SalesAnalysis:
        """Analyse complète du département commercial."""
        deals = crm_data.get("deals", [])
        contacts = crm_data.get("contacts", [])

        if not deals:
            raise InsufficientDataError(
                agent_name=self.AGENT_NAME,
                detail="No deals in CRM data.",
            )

        # ── Filtrer deals ouverts vs fermés ──
        open_deals = [d for d in deals if not d.get("is_closed", False)]
        closed_deals = [d for d in deals if d.get("is_closed", False)]
        won_deals = [d for d in deals if d.get("is_won", False)]
        lost_deals = [
            d for d in closed_deals if not d.get("is_won", False)
        ]

        # ── Pipeline ──
        pipeline_declared = sum(
            d.get("amount", 0) or 0 for d in open_deals
        )

        # ── Stagnation ──
        stagnation_threshold = 30  # Default, recalibré par l'orchestrateur
        if hasattr(self.config, "stagnation_threshold_days"):
            stagnation_threshold = self.config.stagnation_threshold_days

        stagnant_deals = [
            d for d in open_deals
            if (d.get("days_since_last_activity") or 0) > stagnation_threshold
        ]
        deals_without_next = [
            d for d in open_deals
            if not d.get("has_next_step", False)
        ]

        # ── Probabilités réalistes ──
        # Calculer via LLM la probabilité réelle de chaque deal ouvert
        realistic_probabilities = await self._estimate_deal_probabilities(
            open_deals, won_deals, lost_deals
        )

        pipeline_realistic = sum(
            d.get("amount", 0) * prob
            for d, prob in zip(open_deals, realistic_probabilities)
        )

        gap_ratio = self.safe_divide(pipeline_declared, pipeline_realistic, default=1.0)

        # ── Cycle de vente ──
        cycle_times = []
        for d in won_deals:
            create = d.get("create_date")
            close = d.get("close_date")
            if create and close:
                if isinstance(create, str):
                    from connectors.utils import normalize_date
                    create = normalize_date(create)
                    close = normalize_date(close)
                if create and close:
                    delta = (close - create).days
                    if 0 < delta < 365:
                        cycle_times.append(delta)

        avg_cycle = self.avg(cycle_times) if cycle_times else 0.0

        # ── Bottleneck stage ──
        stage_times: dict[str, list[float]] = {}
        for d in open_deals:
            stage = d.get("stage", "unknown")
            days = d.get("days_in_current_stage", 0) or 0
            stage_times.setdefault(stage, []).append(days)

        bottleneck_stage = None
        bottleneck_days = 0.0
        if stage_times:
            stage_avgs = {
                stage: self.avg(times) for stage, times in stage_times.items()
            }
            bottleneck_stage = max(stage_avgs, key=stage_avgs.get)
            bottleneck_days = stage_avgs[bottleneck_stage]

        # ── Win rate ──
        total_closed = len(won_deals) + len(lost_deals)
        win_rate = self.safe_divide(len(won_deals), total_closed, default=0.0)

        # ── Forecast confidence ──
        forecast_confidence = self._score_forecast_confidence(
            open_deals, stagnant_deals, gap_ratio
        )

        # ── Concentration risk ──
        concentration, key_people = self._detect_concentration(
            open_deals, field="owner_name"
        )

        # ── Anomalies (via LLM) ──
        anomalies = await self._detect_sales_anomalies(
            open_deals=open_deals,
            stagnant_deals=stagnant_deals,
            pipeline_declared=pipeline_declared,
            pipeline_realistic=pipeline_realistic,
            win_rate=win_rate,
            avg_cycle=avg_cycle,
            concentration=concentration,
        )

        return SalesAnalysis(
            pipeline_total_declared=round(pipeline_declared, 2),
            pipeline_total_realistic=round(pipeline_realistic, 2),
            pipeline_gap_ratio=round(gap_ratio, 2),
            deals_active=len(open_deals),
            deals_stagnant=len(stagnant_deals),
            deals_without_next_step=len(deals_without_next),
            stagnation_threshold_days=stagnation_threshold,
            avg_cycle_days=round(avg_cycle, 1),
            bottleneck_stage=bottleneck_stage,
            bottleneck_stage_avg_days=round(bottleneck_days, 1) if bottleneck_stage else None,
            win_rate_90d=round(win_rate, 3),
            forecast_confidence=round(forecast_confidence, 2),
            concentration_risk=concentration,
            key_people=key_people,
            anomalies=anomalies,
        )

    async def _estimate_deal_probabilities(
        self,
        open_deals: list[dict],
        won_deals: list[dict],
        lost_deals: list[dict],
    ) -> list[float]:
        """Estime la probabilité réelle de chaque deal ouvert.

        Combine logique déterministe + LLM pour les cas ambigus.
        """
        probabilities = []

        # Calculer les stats de base pour le scoring
        avg_won_activity = self.avg([
            d.get("activity_count_30d", 0) or 0 for d in won_deals
        ]) if won_deals else 5

        for deal in open_deals:
            days_stagnant = deal.get("days_since_last_activity", 0) or 0
            activity = deal.get("activity_count_30d", 0) or 0
            has_next = deal.get("has_next_step", False)
            amount = deal.get("amount", 0) or 0

            # Scoring déterministe par facteur
            prob = 0.5  # Base

            # Facteur activité
            if days_stagnant > 60:
                prob *= 0.1
            elif days_stagnant > 30:
                prob *= 0.3
            elif days_stagnant > 14:
                prob *= 0.6

            # Facteur engagement
            if activity > 0 and avg_won_activity > 0:
                engagement_ratio = min(activity / avg_won_activity, 2.0)
                prob *= (0.5 + engagement_ratio * 0.25)

            # Facteur next step
            if not has_next:
                prob *= 0.7

            # Cap
            prob = max(0.02, min(0.95, prob))
            probabilities.append(prob)

        return probabilities

    def _score_forecast_confidence(
        self,
        open_deals: list[dict],
        stagnant_deals: list[dict],
        gap_ratio: float,
    ) -> float:
        """Score de confiance dans le forecast CRM actuel."""
        if not open_deals:
            return 0.0

        score = 1.0

        # Pénalité pour stagnation
        stagnation_pct = len(stagnant_deals) / len(open_deals)
        score -= stagnation_pct * 0.4

        # Pénalité pour gap
        if gap_ratio > 3:
            score -= 0.3
        elif gap_ratio > 2:
            score -= 0.2
        elif gap_ratio > 1.5:
            score -= 0.1

        # Pénalité pour deals sans next step
        no_next = sum(1 for d in open_deals if not d.get("has_next_step"))
        no_next_pct = no_next / len(open_deals)
        score -= no_next_pct * 0.2

        return max(0.0, min(1.0, score))

    def _detect_concentration(
        self,
        items: list[dict],
        field: str,
    ) -> tuple[str, list[PersonDependency]]:
        """Détecte le risque de concentration sur quelques personnes."""
        if not items:
            return "low", []

        counts: dict[str, int] = {}
        for item in items:
            person = item.get(field, "Unknown") or "Unknown"
            counts[person] = counts.get(person, 0) + 1

        total = len(items)
        sorted_people = sorted(counts.items(), key=lambda x: x[1], reverse=True)

        key_people = []
        for name, count in sorted_people[:5]:
            pct = count / total
            key_people.append(
                PersonDependency(
                    name=name,
                    involvement_pct=round(pct, 3),
                )
            )

        # Top 2 personnes détiennent quel % ?
        top_2_pct = sum(p.involvement_pct for p in key_people[:2])

        if top_2_pct >= 0.7:
            return "high", key_people
        if top_2_pct >= 0.5:
            return "medium", key_people
        return "low", key_people

    async def _detect_sales_anomalies(self, **kwargs) -> list[Anomaly]:
        """Détecte les anomalies commerciales via LLM + règles."""
        anomalies = []

        pipeline_declared = kwargs.get("pipeline_declared", 0)
        pipeline_realistic = kwargs.get("pipeline_realistic", 0)
        stagnant_deals = kwargs.get("stagnant_deals", [])
        open_deals = kwargs.get("open_deals", [])
        win_rate = kwargs.get("win_rate", 0)
        concentration = kwargs.get("concentration", "low")

        # Règle 1 : Pipeline gonflé
        gap = kwargs.get("pipeline_declared", 0) / max(
            kwargs.get("pipeline_realistic", 1), 1
        )
        if gap > 2:
            anomalies.append(Anomaly(
                type=AnomalyType.INCONSISTENCY,
                department="sales",
                title=f"Pipeline gonflé de {gap:.1f}x",
                description=(
                    f"Pipeline déclaré : {self.format_currency(pipeline_declared)}. "
                    f"Pipeline réaliste : {self.format_currency(pipeline_realistic)}. "
                    f"L'écart suggère des deals morts encore comptés."
                ),
                evidence=(
                    f"{len(stagnant_deals)}/{len(open_deals)} deals "
                    f"sans activité depuis plus de 30 jours."
                ),
                severity=min(1.0, gap / 5),
                estimated_annual_cost=round((pipeline_declared - pipeline_realistic) * win_rate, 0),
                cost_confidence=0.5,
            ))

        # Règle 2 : Concentration critique
        if concentration == "high":
            anomalies.append(Anomaly(
                type=AnomalyType.CONCENTRATION,
                department="sales",
                title="Dépendance critique à 1-2 commerciaux",
                description=(
                    "Plus de 70% des deals sont gérés par 2 personnes ou moins. "
                    "Risque majeur en cas de départ."
                ),
                evidence=f"Distribution des deals par owner dans le CRM.",
                severity=0.8,
            ))

        # Règle 3 : Win rate très bas
        if win_rate < 0.15 and len(open_deals) > 10:
            anomalies.append(Anomaly(
                type=AnomalyType.PATTERN,
                department="sales",
                title=f"Taux de conversion critique : {win_rate:.0%}",
                description=(
                    f"Le taux de win sur 90 jours est de {win_rate:.0%}. "
                    f"Les benchmarks sectoriels sont généralement entre 20-35%."
                ),
                evidence=f"Basé sur les deals closés des 90 derniers jours.",
                severity=0.7,
            ))

        return anomalies

    # ──────────────────────────────────────────
    # OPERATIONS ANALYSIS
    # ──────────────────────────────────────────

    async def _analyze_operations(
        self,
        email_data: Optional[dict[str, Any]],
        crm_data: dict[str, Any],
    ) -> OperationsAnalysis:
        """Analyse des opérations basée sur les emails et le CRM."""
        anomalies = []
        key_dependencies: list[PersonDependency] = []

        # Données email
        response_internal = None
        response_external = None
        after_hours_pct = None
        tasks_overdue = None
        tasks_unassigned = None
        doc_score = "unknown"

        if email_data:
            summary = email_data.get("summary", {})
            response_avg = summary.get("avg_response_time_hours", 0)

            # Estimer interne vs externe (simplifi
