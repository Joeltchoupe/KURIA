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

            # Estimer interne vs externe (simplifié)
            internal_pct = summary.get("internal_pct", 0.5)
            response_internal = response_avg * 0.6 if internal_pct > 0.3 else None
            response_external = response_avg

            after_hours_pct = summary.get("after_hours_pct")

            # Détecter les key dependencies via email
            busiest = summary.get("busiest_recipients", [])
            total_msgs = summary.get("total_messages", 1)
            for person in busiest[:5]:
                pct = person.get("count", 0) / max(total_msgs, 1)
                if pct > 0.15:
                    key_dependencies.append(PersonDependency(
                        name=person.get("email", "Unknown"),
                        involvement_pct=round(pct, 3),
                    ))

            # Anomalies email
            if response_avg > 48:
                anomalies.append(Anomaly(
                    type=AnomalyType.PATTERN,
                    department="ops",
                    title=f"Temps de réponse email critique : {response_avg:.0f}h",
                    description=(
                        f"Le temps de réponse moyen est de {response_avg:.1f} heures. "
                        f"Au-dessus de 24h, la réactivité perçue chute drastiquement."
                    ),
                    evidence=f"Basé sur {summary.get('total_threads', 0)} threads analysés.",
                    severity=min(1.0, response_avg / 96),
                    estimated_annual_cost=None,
                    cost_confidence=None,
                ))

            if after_hours_pct and after_hours_pct > 0.30:
                anomalies.append(Anomaly(
                    type=AnomalyType.PATTERN,
                    department="ops",
                    title=f"{after_hours_pct:.0%} d'activité hors heures",
                    description=(
                        f"{after_hours_pct:.0%} de l'activité email se déroule "
                        f"en dehors des heures de bureau. Risque de burnout."
                    ),
                    evidence="Basé sur les timestamps des emails.",
                    severity=0.6,
                ))

        # Process documentation score (heuristique)
        deals = crm_data.get("deals", [])
        deals_with_next = sum(1 for d in deals if d.get("has_next_step"))
        if deals:
            next_step_ratio = deals_with_next / len(deals)
            if next_step_ratio > 0.7:
                doc_score = "high"
            elif next_step_ratio > 0.4:
                doc_score = "medium"
            else:
                doc_score = "low"

        return OperationsAnalysis(
            email_response_internal_avg_hours=(
                round(response_internal, 1) if response_internal else None
            ),
            email_response_external_avg_hours=(
                round(response_external, 1) if response_external else None
            ),
            after_hours_activity_pct=(
                round(after_hours_pct, 3) if after_hours_pct else None
            ),
            tasks_overdue=tasks_overdue,
            tasks_unassigned=tasks_unassigned,
            process_documentation_score=doc_score,
            key_person_dependencies=key_dependencies,
            anomalies=anomalies,
        )

    # ──────────────────────────────────────────
    # FINANCE ANALYSIS
    # ──────────────────────────────────────────

    async def _analyze_finance(
        self,
        finance_data: Optional[dict[str, Any]],
        company_info: dict[str, Any],
    ) -> FinanceAnalysis:
        """Analyse financière basée sur les données comptables."""
        anomalies = []

        if not finance_data:
            # Mode dégradé : estimation basée sur le CA déclaré
            annual_revenue = company_info.get("annual_revenue", 0)
            estimated_burn = annual_revenue / 12 * 0.9  # 90% du revenu

            return FinanceAnalysis(
                cash_position=0,
                monthly_burn=round(estimated_burn, 2),
                runway_months=99.0,
                avg_client_payment_days=45.0,
                avg_supplier_payment_days=30.0,
                cash_gap_days=15.0,
                invoices_overdue_count=0,
                invoices_overdue_value=0,
                revenue_recurring_pct=0.0,
                anomalies=[Anomaly(
                    type=AnomalyType.GAP,
                    department="finance",
                    title="Aucune donnée financière connectée",
                    description="Analyse basée sur des estimations. Connecter QuickBooks pour des données réelles.",
                    evidence="Aucun outil finance connecté.",
                    severity=0.5,
                )],
            )

        summary = finance_data.get("summary", {})

        cash_position = summary.get("cash_position", 0)
        monthly_burn = summary.get("monthly_burn_rate", 0)
        monthly_revenue = summary.get("monthly_revenue_avg", 0)

        # Runway
        net_monthly = monthly_revenue - monthly_burn
        runway = 99.0
        if net_monthly < 0 and monthly_burn > 0:
            runway = abs(cash_position / net_monthly) if net_monthly != 0 else 0

        avg_client_days = summary.get("avg_client_payment_days", 0)
        avg_supplier_days = summary.get("avg_supplier_payment_days", 0)
        cash_gap = avg_client_days - avg_supplier_days

        overdue_count = summary.get("invoices_overdue_count", 0)
        overdue_value = summary.get("invoices_overdue_value", 0)
        recurring_pct = summary.get("revenue_recurring_pct", 0)

        # Top expenses
        top_exp_raw = summary.get("top_expenses", [])
        top_expenses = []
        for exp in top_exp_raw[:10]:
            top_expenses.append(TopExpense(
                category=exp.get("category", "unknown"),
                monthly_amount=exp.get("monthly_avg", 0),
                annual_amount=exp.get("total", 0),
                pct_of_total=exp.get("pct_of_total", 0),
            ))

        # Anomalies finance
        if runway < 3.0:
            anomalies.append(Anomaly(
                type=AnomalyType.PATTERN,
                department="finance",
                title=f"Runway critique : {runway:.1f} mois",
                description=f"Au rythme actuel, la trésorerie tient {runway:.1f} mois.",
                evidence=f"Cash: {cash_position:,.0f}€, Burn net: {abs(net_monthly):,.0f}€/mois.",
                severity=0.95,
                estimated_annual_cost=None,
            ))

        if cash_gap > 20:
            gap_cost = abs(cash_gap) * (monthly_burn / 30) * 0.08
            anomalies.append(Anomaly(
                type=AnomalyType.PATTERN,
                department="finance",
                title=f"Gap de trésorerie : {cash_gap:.0f} jours",
                description=(
                    f"Les clients paient en {avg_client_days:.0f} jours, "
                    f"les fournisseurs sont payés en {avg_supplier_days:.0f} jours. "
                    f"Gap de {cash_gap:.0f} jours."
                ),
                evidence="Basé sur les délais de paiement des 90 derniers jours.",
                severity=min(1.0, cash_gap / 60),
                estimated_annual_cost=round(gap_cost, 0),
                cost_confidence=0.4,
            ))

        if overdue_value > 0:
            anomalies.append(Anomaly(
                type=AnomalyType.STAGNATION,
                department="finance",
                title=f"{overdue_count} factures en retard ({self.format_currency(overdue_value)})",
                description=f"{overdue_count} factures pour un total de {overdue_value:,.0f}€ sont en retard de paiement.",
                evidence="Basé sur les factures avec échéance dépassée.",
                severity=min(1.0, overdue_value / 100_000),
                estimated_annual_cost=round(overdue_value * 0.1, 0),
                cost_confid



    # ──────────────────────────────────────────
    # CLARITY SCORE CALCULATION
    # ──────────────────────────────────────────

    def _calculate_clarity_score(
        self,
        sales: SalesAnalysis,
        operations: OperationsAnalysis,
        finance: FinanceAnalysis,
        marketing: MarketingAnalysis,
        tools_connected: list[str],
    ) -> ClarityScore:
        """Calcule le Score de Clarté™.

        FORMULE EXPLICITE, REPRODUCTIBLE, AUDITABLE.
        Pas d'estimation LLM. Du calcul pur.
        """
        # ── Machine Readability ──
        mr_scores = []

        # 1. Data completeness (30%)
        completeness = 0
        if sales.deals_active > 20:
            completeness += 30
        elif sales.deals_active > 5:
            completeness += 15
        if finance.cash_position != 0:
            completeness += 25
        if operations.email_response_external_avg_hours is not None:
            completeness += 25
        if marketing.has_source_tracking:
            completeness += 20
        mr_scores.append(("completeness", min(100, completeness), self.MR_WEIGHT_DATA_COMPLETENESS))

        # 2. Data freshness (25%)
        freshness = 70  # Base score, réduit si données stale
        if sales.stagnation_rate > 0.5:
            freshness -= 30
        if finance.invoices_overdue_count > 10:
            freshness -= 15
        mr_scores.append(("freshness", max(0, freshness), self.MR_WEIGHT_DATA_FRESHNESS))

        # 3. Data structure (25%)
        structure = 0
        if sales.forecast_confidence > 0.5:
            structure += 30
        if sales.concentration_risk != "high":
            structure += 20
        if operations.process_documentation_score in ("high", "medium"):
            structure += 30
        if marketing.leads_with_source_pct > 0.5:
            structure += 20
        mr_scores.append(("structure", min(100, structure), self.MR_WEIGHT_DATA_STRUCTURE))

        # 4. Tool coverage (20%)
        tool_score = len(tools_connected) / 4 * 100  # 4 tools max en V1
        mr_scores.append(("tools", min(100, tool_score), self.MR_WEIGHT_TOOL_COVERAGE))

        machine_readability = sum(score * weight for _, score, weight in mr_scores)

        # ── Structural Compatibility ──
        sc_scores = []

        # 1. Process explicitness (30%)
        process_score = {"high": 80, "medium": 50, "low": 20, "unknown": 10}
        sc_scores.append((
            "process",
            process_score.get(operations.process_documentation_score, 10),
            self.SC_WEIGHT_PROCESS_EXPLICITNESS,
        ))

        # 2. Role clarity (25%)
        role_clarity = 70
        if sales.concentration_risk == "high":
            role_clarity = 25
        elif sales.concentration_risk == "medium":
            role_clarity = 50
        sc_scores.append(("roles", role_clarity, self.SC_WEIGHT_ROLE_CLARITY))

        # 3. Data centralization (25%)
        centralization = len(tools_connected) / 4 * 100
        sc_scores.append(("central", min(100, centralization), self.SC_WEIGHT_DATA_CENTRALIZATION))

        # 4. Decision traceability (20%)
        traceability = 30  # Base faible, amélioré si next steps présents
        if sales.deals_without_next_step < sales.deals_active * 0.3:
            traceability = 70
        sc_scores.append(("trace", traceability, self.SC_WEIGHT_DECISION_TRACEABILITY))

        structural_compatibility = sum(
            score * weight for _, score, weight in sc_scores
        )

        # ── Overall ──
        overall = ClarityScore.calculate_overall(
            machine_readability, structural_compatibility
        )

        # ── Department scores ──
        dept_sales = min(100, (
            (30 if sales.forecast_confidence > 0.5 else 10) +
            (25 if sales.stagnation_rate < 0.3 else 5) +
            (25 if sales.concentration_risk != "high" else 5) +
            (20 if sales.deals_without_next_step < sales.deals_active * 0.3 else 5)
        ))
        dept_ops = min(100, (
            (40 if operations.process_documentation_score == "high" else
             25 if operations.process_documentation_score == "medium" else 10) +
            (30 if operations.after_hours_activity_pct is None or
             operations.after_hours_activity_pct < 0.2 else 10) +
            (30 if not operations.key_person_dependencies or
             all(p.involvement_pct < 0.3 for p in operations.key_person_dependencies) else 10)
        ))
        dept_finance = min(100, (
            (30 if finance.runway_months > 6 else 10) +
            (25 if finance.cash_gap_days < 15 else 5) +
            (25 if finance.invoices_overdue_count < 3 else 5) +
            (20 if finance.revenue_recurring_pct > 0.5 else 5)
        ))
        dept_marketing = min(100, (
            (40 if marketing.has_source_tracking else 5) +
            (30 if marketing.leads_with_source_pct > 0.7 else
             15 if marketing.leads_with_source_pct > 0.3 else 5) +
            (30 if marketing.lead_to_client_conversion and
             marketing.lead_to_client_conversion > 0.1 else 10)
        ))

        return ClarityScore(
            machine_readability=round(machine_readability, 1),
            structural_compatibility=round(structural_compatibility, 1),
            overall=overall,
            by_department=DepartmentScores(
                sales=round(dept_sales, 1),
                ops=round(dept_ops, 1),
                finance=round(dept_finance, 1),
                marketing=round(dept_marketing, 1),
            ),
            confidence=self._calculate_confidence(
                {"crm": {"deals": [], "contacts": []}, "email": None, "finance": None},
                {},
            ),
            data_sources_used=len(tools_connected),
        )

    # ──────────────────────────────────────────
    # FRICTION IDENTIFICATION
    # ──────────────────────────────────────────

    async def _identify_frictions(
        self,
        sales: SalesAnalysis,
        operations: OperationsAnalysis,
        finance: FinanceAnalysis,
        marketing: MarketingAnalysis,
        company_info: dict[str, Any],
    ) -> list[Friction]:
        """Identifie et chiffre les frictions à partir des anomalies."""
        frictions = []
        friction_id = 0
        employee_count = company_info.get("employee_count", 30)
        hourly_rate = company_info.get("annual_revenue", 3_000_000) / employee_count / 1800

        all_analyses = [
            ("sales", sales),
            ("ops", operations),
            ("finance", finance),
            ("marketing", marketing),
        ]

        for dept, analysis in all_analyses:
            for anomaly in analysis.anomalies:
                friction_id += 1

                # Estimer le coût si pas déjà fait
                cost = anomaly.estimated_annual_cost or 0
                if cost == 0 and anomaly.severity > 0.5:
                    cost = anomaly.severity * employee_count * hourly_rate * 40
                cost_confidence = anomaly.cost_confidence or 0.3

                # Classifier impact et difficulté
                impact = anomaly.severity * 10
                difficulty = self._estimate_difficulty(anomaly, dept)

                priority = Friction.classify_priority(impact, difficulty)
                severity = Friction.classify_severity(cost)

                agent_map = {
                    "sales": "revenue_velocity",
                    "ops": "process_clarity",
                    "finance": "cash_predictability",
                    "marketing": "acquisition_efficiency",
                }

                frictions.append(Friction(
                    friction_id=f"F{friction_id}",
                    department=dept,
                    title=anomaly.title,
                    description=anomaly.description,
                    estimated_annual_cost=round(cost, 0),
                    cost_confidence=cost_confidence,
                    evidence=anomaly.evidence,
                    severity=severity,
                    impact_score=round(impact, 1),
                    difficulty_score=round(difficulty, 1),
                    priority=priority,
                    agent_recommended=agent_map.get(dept),
                ))

        # Trier par coût décroissant
        frictions.sort(key=lambda f: f.estimated_annual_cost, reverse=True)
        return frictions

    @staticmethod
    def _estimate_difficulty(anomaly: Anomaly, department: str) -> float:
        """Estime la difficulté de résolution d'une anomalie."""
        difficulty = 5.0  # Base

        if anomaly.type == AnomalyType.GAP:
            difficulty = 3.0  # Manque de données = connecter un outil
        elif anomaly.type == AnomalyType.CONCENTRATION:
            difficulty = 7.0  # Changer la dépendance = organisationnel
        elif anomaly.type == AnomalyType.STAGNATION:
            difficulty = 3.0  # Relancer = action directe
        elif anomaly.type == AnomalyType.INCONSISTENCY:
            difficulty = 4.0  # Nettoyer les données
        elif anomaly.type == AnomalyType.PATTERN:
            difficulty = 6.0  # Changer un comportement

        return difficulty








