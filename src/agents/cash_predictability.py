"""
Cash Predictability Agent — Cash forecast, scénarios, alertes préventives.

UN KPI : Cash Forecast Accuracy à 30 jours (|prédiction - réalité| / réalité)

4 fonctions :
1. Cash Position     → Où en est le cash NOW
2. Forecast Rolling  → Où sera le cash dans 30/60/90 jours
3. Alerte Préventive → Le problème arrive dans X jours
4. Weekly Report     → Résumé financier

Interconnexion CRITIQUE avec Revenue Velocity :
→ Le forecast cash INTÈGRE le forecast pipeline.
→ Si l'agent Revenue détecte des deals morts,
   l'agent Cash recalcule automatiquement.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from models.agent_config import CashPredictabilityConfig
from models.events import EventType, EventPriority
from models.metrics import CashPredictabilityMetrics, MetricTrend

from agents.base import BaseAgent, InsufficientDataError


class CashPredictabilityAgent(BaseAgent):
    """Agent Cash Predictability — le radar de trésorerie."""

    AGENT_NAME = "cash_predictability"

    def __init__(self, company_id: str, config: CashPredictabilityConfig, **kwargs):
        super().__init__(company_id=company_id, config=config, **kwargs)
        self.cp_config: CashPredictabilityConfig = config

    async def analyze(self, data: dict[str, Any]) -> dict[str, Any]:
        """Analyse complète cash.

        Args:
            data: {
                "finance": output de QuickBooksConnector.extract(),
                "pipeline_forecast": {  # depuis l'agent Revenue Velocity
                    "forecast_30d": float,
                    "forecast_60d": float,
                    "forecast_90d": float,
                    "confidence": float,
                } (optional)
            }
        """
        finance = data.get("finance", {})
        pipeline_forecast = data.get("pipeline_forecast")

        # 1. Cash Position
        position = self._get_cash_position(finance)

        # 2. Forecast Rolling
        forecast = self._forecast_rolling(
            position, finance, pipeline_forecast
        )

        # 3. Alertes
        alerts = self._check_alerts(position, forecast)

        # 4. Overdue invoices
        invoices_overdue = self._analyze_overdue(finance)

        # 5. Métriques
        previous = await self.get_previous_metrics()
        prev_accuracy = previous.get("forecast_accuracy_30d") if previous else None

        # Calculer l'accuracy si on a un forecast précédent à comparer
        accuracy = self._calculate_accuracy(previous, position)

        alert_level = None
        days_until = None
        if alerts:
            worst_alert = alerts[0]
            alert_level = worst_alert.get("level")
            days_until = worst_alert.get("days_until_threshold")

        metrics = CashPredictabilityMetrics(
            company_id=self.company_id,
            forecast_accuracy_30d=accuracy,
            cash_position=position["cash_current"],
            runway_months=position["runway_months"],
            monthly_burn=position["monthly_burn"],
            forecast_base_30d=forecast["base"]["day_30"],
            forecast_stress_30d=forecast["stress"]["day_30"],
            forecast_upside_30d=forecast["upside"]["day_30"],
            forecast_confidence=forecast["confidence"],
            alert_level=alert_level,
            days_until_threshold=days_until,
            invoices_overdue_count=invoices_overdue["count"],
            invoices_overdue_value=invoices_overdue["value"],
        )
        await self.save_metrics(metrics)

        # Publier forecast
        self.publish_event(
            event_type=EventType.CASH_FORECAST_GENERATED,
            payload={
                "base_30d": round(forecast["base"]["day_30"], 2),
                "stress_30d": round(forecast["stress"]["day_30"], 2),
                "upside_30d": round(forecast["upside"]["day_30"], 2),
                "confidence": round(forecast["confidence"], 2),
            },
        )

        # Publier alertes
        for alert in alerts:
            event_type = (
                EventType.CASH_ALERT_RED
                if alert["level"] == "red"
                else EventType.CASH_ALERT_YELLOW
            )
            self.publish_event(
                event_type=event_type,
                priority=(
                    EventPriority.CRITICAL
                    if alert["level"] == "red"
                    else EventPriority.HIGH
                ),
                payload={
                    "days_until_threshold": alert["days_until_threshold"],
                    "scenario": alert["scenario"],
                    "threshold": alert["threshold"],
                    "current_cash": position["cash_current"],
                    "actions": alert.get("actions", []),
                },
            )

        return {
            "position": position,
            "forecast": forecast,
            "alerts": alerts,
            "invoices_overdue": invoices_overdue,
            "metrics": metrics.model_dump(),
        }

    # ══════════════════════════════════════════
    # FONCTION 1 — CASH POSITION
    # ══════════════════════════════════════════

    def _get_cash_position(self, finance: dict[str, Any]) -> dict[str, Any]:
        """Position cash actuelle."""
        summary = finance.get("summary", {})

        cash_current = summary.get("cash_position", 0)
        monthly_burn = summary.get("monthly_burn_rate", 0)
        monthly_revenue = summary.get("monthly_revenue_avg", 0)
        total_receivable = summary.get("total_receivable", 0)
        total_payable = summary.get("total_payable", 0)

        net_monthly = monthly_revenue - monthly_burn
        runway = 99.0
        if net_monthly < 0:
            runway = self.safe_divide(cash_current, abs(net_monthly), default=0)

        return {
            "cash_current": round(cash_current, 2),
            "monthly_burn": round(monthly_burn, 2),
            "monthly_revenue": round(monthly_revenue, 2),
            "net_monthly": round(net_monthly, 2),
            "runway_months": round(min(runway, 99.0), 1),
            "total_receivable": round(total_receivable, 2),
            "total_payable": round(total_payable, 2),
        }

    # ══════════════════════════════════════════
    # FONCTION 2 — FORECAST ROLLING
    # ══════════════════════════════════════════

    def _forecast_rolling(
        self,
        position: dict[str, Any],
        finance: dict[str, Any],
        pipeline_forecast: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        """Forecast cash rolling 30/60/90 avec 3 scénarios."""
        cash = position["cash_current"]
        monthly_revenue = position["monthly_revenue"]
        monthly_burn = position["monthly_burn"]

        # Revenus pipeline (si disponibles via l'agent Revenue Velocity)
        pipeline_30 = 0
        pipeline_60 = 0
        pipeline_90 = 0
        if pipeline_forecast:
            pipeline_30 = pipeline_forecast.get("forecast_30d", 0)
            pipeline_60 = pipeline_forecast.get("forecast_60d", 0)
            pipeline_90 = pipeline_forecast.get("forecast_90d", 0)

        # Facteur de correction de l'Adaptateur
        correction = self.cp_config.forecast_optimism_correction

        # Facteurs de paiement (les clients ne paient pas immédiatement)
        summary = finance.get("summary", {})
        avg_payment_days = summary.get("avg_client_payment_days", 45)
        payment_delay_factor = min(1.0, 30 / max(avg_payment_days, 1))

        # ── SCÉNARIO BASE ──
        base = self._project_scenario(
            starting_cash=cash,
            monthly_recurring_revenue=monthly_revenue * correction,
            pipeline_revenue={30: pipeline_30, 60: pipeline_60, 90: pipeline_90},
            pipeline_realization=0.7,
            payment_factor=payment_delay_factor,
            monthly_expenses=monthly_burn,
            extra_delay_days=0,
        )

        # ── SCÉNARIO STRESS ──
        stress = self._project_scenario(
            starting_cash=cash,
            monthly_recurring_revenue=monthly_revenue * correction * 0.8,
            pipeline_revenue={30: pipeline_30, 60: pipeline_60, 90: pipeline_90},
            pipeline_realization=self.cp_config.stress_pipeline_factor,
            payment_factor=payment_delay_factor * 0.7,
            monthly_expenses=monthly_burn * 1.1,  # +10% surprises
            extra_delay_days=self.cp_config.stress_payment_delay_days,
        )

        # ── SCÉNARIO UPSIDE ──
        upside = self._project_scenario(
            starting_cash=cash,
            monthly_recurring_revenue=monthly_revenue * correction,
            pipeline_revenue={30: pipeline_30, 60: pipeline_60, 90: pipeline_90},
            pipeline_realization=self.cp_config.upside_pipeline_factor,
            payment_factor=min(1.0, payment_delay_factor * 1.2),
            monthly_expenses=monthly_burn * 0.95,
            extra_delay_days=-5,
        )

        # Confiance
        confidence = self._forecast_confidence(
            finance, pipeline_forecast
        )

        return {
            "base": base,
            "stress": stress,
            "upside": upside,
            "confidence": round(confidence, 2),
        }

    def _project_scenario(
        self,
        starting_cash: float,
        monthly_recurring_revenue: float,
        pipeline_revenue: dict[int, float],
        pipeline_realization: float,
        payment_factor: float,
        monthly_expenses: float,
        extra_delay_days: int,
    ) -> dict[str, Any]:
        """Projette le cash pour un scénario donné."""
        cash = starting_cash
        projections = {"day_0": round(cash, 2)}

        for month, days in [(1, 30), (2, 60), (3, 90)]:
            # Revenus récurrents
            revenue_in = monthly_recurring_revenue * payment_factor

            # Revenus pipeline
            pipeline_in = (
                pipeline_revenue.get(days, 0)
                * pipeline_realization
                * payment_factor
            )

            # Total entrant
            total_in = revenue_in + pipeline_in

            # Sortant
            total_out = monthly_expenses

            # Net
            cash += total_in - total_out
            projections[f"day_{days}"] = round(cash, 2)

        return projections

    def _forecast_confidence(
        self,
        finance: dict[str, Any],
        pipeline_forecast: Optional[dict[str, Any]],
    ) -> float:
        """Confiance dans le forecast."""
        score = 0.5

        summary = finance.get("summary", {})

        # Boost si données financières complètes
        if summary.get("cash_position") is not None:
            score += 0.15
        if summary.get("monthly_burn_rate", 0) > 0:
            score += 0.1
        if summary.get("revenue_recurring_pct", 0) > 0.5:
            score += 0.1  # Revenu prévisible

        # Boost si pipeline forecast dispo
        if pipeline_forecast and pipeline_forecast.get("confidence", 0) > 0.5:
            score += 0.15

        return min(1.0, score)

    # ══════════════════════════════════════════
    # FONCTION 3 — ALERTES PRÉVENTIVES
    # ══════════════════════════════════════════

    def _check_alerts(
        self,
        position: dict[str, Any],
        forecast: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Vérifie si des alertes doivent être déclenchées."""
        alerts = []
        monthly_burn = position["monthly_burn"]

        yellow_threshold = monthly_burn * self.cp_config.cash_threshold_yellow_months
        red_threshold = monthly_burn * self.cp_config.cash_threshold_red_months

        # ── Vérifier scénario BASE pour alerte JAUNE ──
        base = forecast["base"]
        for day_key in ["day_30", "day_60", "day_90"]:
            cash_at = base.get(day_key, 0)
            days = int(day_key.split("_")[1])

            if cash_at < yellow_threshold:
                alerts.append({
                    "level": "yellow",
                    "scenario": "base",
                    "days_until_threshold": days,
                    "threshold": round(yellow_threshold, 2),
                    "projected_cash": round(cash_at, 2),
                    "message": (
                        f"Dans le scénario base, votre cash passe sous "
                        f"{self.format_currency(yellow_threshold)} dans {days} jours."
                    ),
                    "actions": self._generate_actions(position, "yellow"),
                })
                break  # Une seule alerte jaune

        # ── Vérifier scénario STRESS pour alerte ROUGE ──
        stress = forecast["stress"]
        for day_key in ["day_30", "day_60", "day_90"]:
            cash_at = stress.get(day_key, 0)
            days = int(day_key.split("_")[1])

            if cash_at < red_threshold:
                alerts.append({
                    "level": "red",
                    "scenario": "stress",
                    "days_until_threshold": days,
                    "threshold": round(red_threshold, 2),
                    "projected_cash": round(cash_at, 2),
                    "message": (
                        f"ALERTE : dans le scénario stress, votre cash passe sous "
                        f"{self.format_currency(red_threshold)} dans {days} jours."
                    ),
                    "actions": self._generate_actions(position, "red"),
                })
                break

        # Trier : rouge d'abord
        alerts.sort(key=lambda a: 0 if a["level"] == "red" else 1)
        return alerts

    def _generate_actions(
        self,
        position: dict[str, Any],
        alert_level: str,
    ) -> list[dict[str, str]]:
        """Génère des actions concrètes pour résoudre l'alerte."""
        actions = []

        receivable = position.get("total_receivable", 0)
        if receivable > 0:
            actions.append({
                "action": f"Relancer les factures en attente ({self.format_currency(receivable)})",
                "impact": f"Récupérer jusqu'à {self.format_currency(receivable)}",
                "difficulty": "easy",
            })

        if alert_level == "red":
            actions.append({
                "action": "Accélérer les 3 deals les plus proches du closing",
                "impact": "Accélérer les entrées de cash",
                "difficulty": "medium",
            })
            actions.append({
                "action": "Identifier les dépenses reportables ce mois",
                "impact": f"Réduire le burn de 10-20%",
                "difficulty": "easy",
            })

        return actions

    # ══════════════════════════════════════════
    # ANALYSE FACTURES
    # ══════════════════════════════════════════

    def _analyze_overdue(self, finance: dict[str, Any]) -> dict[str, Any]:
        """Analyse les factures en retard."""
        invoices = finance.get("invoices", [])
        overdue = [i for i in invoices if i.get("is_overdue", False)]

        return {
            "count": len(overdue),
            "value": sum(i.get("amount_due", 0) for i in overdue),
            "details": [
                {
                    "invoice_id": i.get("invoice_id"),
                    "customer": i.get("customer_name"),
                    "amount_due": i.get("amount_due", 0),
                    "days_overdue": i.get("days_overdue", 0),
                }
                for i in sorted(
                    overdue,
                    key=lambda x: x.get("amount_due", 0),
                    reverse=True,
                )[:10]
            ],
        }

    def _calculate_accuracy(
        self,
        previous_metrics: Optional[dict],
        current_position: dict[str, Any],
    ) -> Optional[float]:
        """Calcule la précision du forecast précédent."""
        if not previous_metrics:
            return None

        predicted = previous_metrics.get("forecast_base_30d")
        actual = current_position.get("cash_current")

        if predicted is None or actual is None or actual == 0:
            return None

        error = abs(predicted - actual) / abs(actual)
        accuracy = max(0, 1 - error)
        return round(accuracy, 3)

    # ══════════════════════════════════════════
    # VALIDATION & CONFIDENCE
    # ══════════════════════════════════════════

    def _validate_input(self, data: dict[str, Any]) -> list[str]:
        warnings = []
        finance = data.get("finance", {})
        summary = finance.get("summary", {})

        if not finance:
            raise InsufficientDataError(
                agent_name=self.AGENT_NAME,
                detail="No finance data provided.",
            )

        if summary.get("cash_position") is None:
            warnings.append("No cash position available. Using 0.")

        if not data.get("pipeline_forecast"):
            warnings.append(
                "No pipeline forecast from Revenue Velocity. "
                "Cash forecast will not include pipeline revenues."
            )

        return warnings

    def _calculate_confidence(
        self,
        input_data: dict[str, Any],
        output_data: dict[str, Any],
    ) -> float:
        finance = input_data.get("finance", {})
        score = 0.3

        summary = finance.get("summary", {})
        if summary.get("cash_position") is not None:
            score += 0.25
        if summary.get("monthly_burn_rate", 0) > 0:
            score += 0.15
        if summary.get("revenue_recurring_pct", 0) > 0.3:
            score += 0.1
        if input_data.get("pipeline_forecast"):
            score += 0.2

        return min(1.0, score)
