"""
Cash Predictability Agent — Cash forecast, scénarios, alertes préventives.

KPI NORD : Cash Forecast Accuracy à 30 jours (|prédiction - réalité| / réalité)

4 fonctions :
1. Cash Position     → Où en est le cash NOW
2. Forecast Rolling  → Où sera le cash dans 30/60/90 jours (3 scénarios)
3. Alerte Préventive → Le problème arrive dans X jours + actions concrètes
4. Overdue Analysis  → Factures en retard = cash récupérable

Interconnexion CRITIQUE avec Revenue Velocity :
→ Le forecast cash INTÈGRE le forecast pipeline.
→ Si l'agent Revenue détecte des deals morts,
   l'agent Cash recalcule automatiquement.
"""

from __future__ import annotations

from typing import Any, Optional

from models.agent_config import CashPredictabilityConfig
from models.events import EventType, EventPriority
from models.metrics import CashPredictabilityMetrics, MetricTrend

from agents.base import BaseAgent, InsufficientDataError


# ══════════════════════════════════════════
# CONSTANTES
# ══════════════════════════════════════════

_HORIZONS: list[tuple[int, int]] = [(1, 30), (2, 60), (3, 90)]
_MAX_RUNWAY: float = 99.0
_TOP_OVERDUE_LIMIT: int = 10


# ══════════════════════════════════════════
# AGENT
# ══════════════════════════════════════════


class CashPredictabilityAgent(BaseAgent[CashPredictabilityConfig]):
    """Agent Cash Predictability — le radar de trésorerie."""

    AGENT_NAME = "cash_predictability"

    # ──────────────────────────────────────
    # CONTRAT BASE AGENT
    # ──────────────────────────────────────

    def _validate(self, data: dict[str, Any]) -> list[str]:
        """Valide les données financières d'entrée."""
        warnings: list[str] = []
        finance = data.get("finance", {})

        if not finance:
            raise InsufficientDataError(
                agent_name=self.AGENT_NAME,
                detail="Aucune donnée financière fournie.",
                available=0,
                required=1,
            )

        summary = finance.get("summary", {})

        if summary.get("cash_position") is None:
            warnings.append(
                "Pas de position cash disponible. Valeur 0 utilisée par défaut."
            )

        if summary.get("monthly_burn_rate") is None:
            warnings.append(
                "Pas de burn rate mensuel. Le forecast sera moins fiable."
            )

        if not finance.get("invoices"):
            warnings.append(
                "Aucune facture trouvée. L'analyse des retards ne sera pas disponible."
            )

        if not data.get("pipeline_forecast"):
            warnings.append(
                "Pas de forecast pipeline (Revenue Velocity). "
                "Le forecast cash n'intègrera pas les revenus pipeline."
            )

        return warnings

    async def _execute(self, data: dict[str, Any]) -> dict[str, Any]:
        """Logique métier : position → forecast → alertes → overdue."""
        finance = data.get("finance", {})
        pipeline_forecast = data.get("pipeline_forecast")

        # 1. Cash Position — état actuel
        position = self._compute_position(finance)

        # 2. Forecast Rolling — 3 scénarios × 3 horizons
        forecast = self._compute_forecast(position, finance, pipeline_forecast)

        # 3. Alertes préventives
        alerts = self._compute_alerts(position, forecast)

        # 4. Factures en retard
        overdue = self._compute_overdue(finance)

        # 5. Accuracy du forecast précédent
        previous = await self.get_previous_metrics()
        accuracy = self._compute_accuracy(previous, position)

        # 6. Métriques
        worst_alert = alerts[0] if alerts else None
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
            alert_level=worst_alert["level"] if worst_alert else None,
            days_until_threshold=worst_alert["days_until_threshold"] if worst_alert else None,
            invoices_overdue_count=overdue["count"],
            invoices_overdue_value=overdue["value"],
        )

        # Persist métriques agent
        if self.supabase:
            try:
                await self.supabase.insert("agent_metrics", {
                    "agent_name": self.AGENT_NAME,
                    "company_id": self.company_id,
                    "run_id": self.run_id,
                    "metrics": metrics.model_dump(mode="json"),
                })
            except Exception as e:
                self._log("error", f"Failed to save agent metrics: {e}")

        # 7. Événements
        self._emit_forecast(forecast)
        self._emit_alerts(alerts, position)

        # 8. Summary
        summary_text = self._build_summary(position, forecast, alerts, overdue)

        return {
            "position": position,
            "forecast": forecast,
            "alerts": alerts,
            "invoices_overdue": overdue,
            "accuracy": accuracy,
            "metrics": metrics.model_dump(),
            "summary": summary_text,
        }

    def _confidence(self, input_data: dict[str, Any], output_data: dict[str, Any]) -> float:
        """Confiance basée sur la complétude des données financières."""
        finance = input_data.get("finance", {})
        summary = finance.get("summary", {})
        score = 0.2  # Base

        # Position cash connue
        if summary.get("cash_position") is not None:
            score += 0.25

        # Burn rate calculable
        if summary.get("monthly_burn_rate", 0) > 0:
            score += 0.15

        # Revenu récurrent → forecast plus fiable
        if summary.get("revenue_recurring_pct", 0) > 0.3:
            score += 0.1

        # Historique factures
        if finance.get("invoices"):
            score += 0.1

        # Pipeline forecast disponible
        pipeline = input_data.get("pipeline_forecast")
        if pipeline and pipeline.get("confidence", 0) > 0.5:
            score += 0.2
        elif pipeline:
            score += 0.1

        return min(1.0, round(score, 2))

    async def _observation_mode(
        self, data: dict[str, Any], error: InsufficientDataError
    ) -> dict[str, Any]:
        """Mode dégradé : pas assez de données financières."""
        finance = data.get("finance", {})
        summary = finance.get("summary", {})

        return {
            "message": error.detail,
            "available_data": {
                "has_cash_position": summary.get("cash_position") is not None,
                "has_burn_rate": summary.get("monthly_burn_rate") is not None,
                "has_invoices": bool(finance.get("invoices")),
                "has_pipeline": bool(data.get("pipeline_forecast")),
            },
            "recommendation": (
                "Connectez QuickBooks ou importez un export CSV comptable "
                "pour activer le forecast de trésorerie."
            ),
        }

    # ══════════════════════════════════════════
    # FONCTION 1 — CASH POSITION
    # ══════════════════════════════════════════

    def _compute_position(self, finance: dict[str, Any]) -> dict[str, Any]:
        """Position cash actuelle — snapshot."""
        summary = finance.get("summary", {})

        cash = summary.get("cash_position", 0)
        burn = summary.get("monthly_burn_rate", 0)
        revenue = summary.get("monthly_revenue_avg", 0)
        receivable = summary.get("total_receivable", 0)
        payable = summary.get("total_payable", 0)

        net_monthly = revenue - burn

        # Runway : mois restants avant épuisement
        if net_monthly < 0:
            runway = self.safe_divide(cash, abs(net_monthly))
        else:
            runway = _MAX_RUNWAY  # Cash positif net → pas de date d'épuisement

        return {
            "cash_current": round(cash, 2),
            "monthly_burn": round(burn, 2),
            "monthly_revenue": round(revenue, 2),
            "net_monthly": round(net_monthly, 2),
            "runway_months": round(min(runway, _MAX_RUNWAY), 1),
            "total_receivable": round(receivable, 2),
            "total_payable": round(payable, 2),
        }

    # ══════════════════════════════════════════
    # FONCTION 2 — FORECAST ROLLING
    # ══════════════════════════════════════════

    def _compute_forecast(
        self,
        position: dict[str, Any],
        finance: dict[str, Any],
        pipeline_forecast: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        """Forecast 30/60/90 jours — 3 scénarios : base, stress, upside."""
        cash = position["cash_current"]
        revenue = position["monthly_revenue"]
        burn = position["monthly_burn"]

        # Revenus pipeline (si Revenue Velocity a tourné)
        p30, p60, p90 = self._extract_pipeline(pipeline_forecast)

        # Correction optimisme (config Adaptateur)
        correction = self.config.forecast_optimism_correction

        # Délai de paiement moyen → facteur d'encaissement
        summary = finance.get("summary", {})
        avg_days = summary.get("avg_client_payment_days", 45)
        payment_factor = min(1.0, 30 / max(avg_days, 1))

        # ── SCÉNARIO BASE ──
        base = self._project(
            starting_cash=cash,
            recurring_revenue=revenue * correction,
            pipeline={30: p30, 60: p60, 90: p90},
            pipeline_realization=0.7,
            payment_factor=payment_factor,
            expenses=burn,
        )

        # ── SCÉNARIO STRESS ──
        stress = self._project(
            starting_cash=cash,
            recurring_revenue=revenue * correction * 0.8,
            pipeline={30: p30, 60: p60, 90: p90},
            pipeline_realization=self.config.stress_pipeline_factor,
            payment_factor=payment_factor * 0.7,
            expenses=burn * 1.1,  # +10% imprévus
        )

        # ── SCÉNARIO UPSIDE ──
        upside = self._project(
            starting_cash=cash,
            recurring_revenue=revenue * correction,
            pipeline={30: p30, 60: p60, 90: p90},
            pipeline_realization=self.config.upside_pipeline_factor,
            payment_factor=min(1.0, payment_factor * 1.2),
            expenses=burn * 0.95,
        )

        # Confiance forecast
        confidence = self._forecast_confidence(finance, pipeline_forecast)

        return {
            "base": base,
            "stress": stress,
            "upside": upside,
            "confidence": round(confidence, 2),
        }

    def _project(
        self,
        starting_cash: float,
        recurring_revenue: float,
        pipeline: dict[int, float],
        pipeline_realization: float,
        payment_factor: float,
        expenses: float,
    ) -> dict[str, float]:
        """Projette le cash sur 3 horizons pour un scénario donné."""
        cash = starting_cash
        result: dict[str, float] = {"day_0": round(cash, 2)}

        for _month, days in _HORIZONS:
            revenue_in = recurring_revenue * payment_factor
            pipeline_in = pipeline.get(days, 0) * pipeline_realization * payment_factor
            total_in = revenue_in + pipeline_in
            total_out = expenses

            cash += total_in - total_out
            result[f"day_{days}"] = round(cash, 2)

        return result

    @staticmethod
    def _extract_pipeline(
        pipeline_forecast: Optional[dict[str, Any]],
    ) -> tuple[float, float, float]:
        """Extrait les forecasts pipeline 30/60/90."""
        if not pipeline_forecast:
            return 0.0, 0.0, 0.0
        return (
            pipeline_forecast.get("forecast_30d", 0),
            pipeline_forecast.get("forecast_60d", 0),
            pipeline_forecast.get("forecast_90d", 0),
        )

    def _forecast_confidence(
        self,
        finance: dict[str, Any],
        pipeline_forecast: Optional[dict[str, Any]],
    ) -> float:
        """Confiance dans le forecast — indépendante de _confidence globale."""
        score = 0.5
        summary = finance.get("summary", {})

        if summary.get("cash_position") is not None:
            score += 0.15
        if summary.get("monthly_burn_rate", 0) > 0:
            score += 0.1
        if summary.get("revenue_recurring_pct", 0) > 0.5:
            score += 0.1

        if pipeline_forecast and pipeline_forecast.get("confidence", 0) > 0.5:
            score += 0.15

        return min(1.0, score)

    # ══════════════════════════════════════════
    # FONCTION 3 — ALERTES PRÉVENTIVES
    # ══════════════════════════════════════════

    def _compute_alerts(
        self,
        position: dict[str, Any],
        forecast: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Détecte les alertes jaune (base) et rouge (stress)."""
        alerts: list[dict[str, Any]] = []
        burn = position["monthly_burn"]

        yellow_threshold = burn * self.config.cash_threshold_yellow_months
        red_threshold = burn * self.config.cash_threshold_red_months

        # ── Alerte JAUNE : scénario base passe sous le seuil ──
        yellow = self._find_breach(
            scenario=forecast["base"],
            threshold=yellow_threshold,
            level="yellow",
            scenario_name="base",
            position=position,
        )
        if yellow:
            alerts.append(yellow)

        # ── Alerte ROUGE : scénario stress passe sous le seuil ──
        red = self._find_breach(
            scenario=forecast["stress"],
            threshold=red_threshold,
            level="red",
            scenario_name="stress",
            position=position,
        )
        if red:
            alerts.append(red)

        # Rouge en premier
        alerts.sort(key=lambda a: 0 if a["level"] == "red" else 1)
        return alerts

    def _find_breach(
        self,
        scenario: dict[str, float],
        threshold: float,
        level: str,
        scenario_name: str,
        position: dict[str, Any],
    ) -> Optional[dict[str, Any]]:
        """Cherche le premier horizon où le cash passe sous le seuil."""
        for _month, days in _HORIZONS:
            cash_at = scenario.get(f"day_{days}", 0)
            if cash_at < threshold:
                return {
                    "level": level,
                    "scenario": scenario_name,
                    "days_until_threshold": days,
                    "threshold": round(threshold, 2),
                    "projected_cash": round(cash_at, 2),
                    "message": self._alert_message(level, threshold, days),
                    "actions": self._generate_actions(position, level),
                }
        return None

    def _alert_message(self, level: str, threshold: float, days: int) -> str:
        """Message d'alerte humain."""
        prefix = "ALERTE CRITIQUE" if level == "red" else "Attention"
        scenario_label = "stress" if level == "red" else "base"
        return (
            f"{prefix} : dans le scénario {scenario_label}, votre cash passe sous "
            f"{self.format_currency(threshold)} dans {days} jours."
        )

    def _generate_actions(
        self,
        position: dict[str, Any],
        alert_level: str,
    ) -> list[dict[str, str]]:
        """Actions concrètes graduées selon le niveau d'alerte."""
        actions: list[dict[str, str]] = []

        receivable = position.get("total_receivable", 0)
        if receivable > 0:
            actions.append({
                "action": (
                    f"Relancer les factures en attente "
                    f"({self.format_currency(receivable)})"
                ),
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
                "impact": "Réduire le burn de 10-20%",
                "difficulty": "easy",
            })
        elif alert_level == "yellow":
            actions.append({
                "action": "Vérifier les conditions de paiement des nouveaux contrats",
                "impact": "Raccourcir le cycle d'encaissement",
                "difficulty": "easy",
            })

        return actions

    # ══════════════════════════════════════════
    # FONCTION 4 — ANALYSE OVERDUE
    # ══════════════════════════════════════════

    def _compute_overdue(self, finance: dict[str, Any]) -> dict[str, Any]:
        """Analyse les factures en retard — cash récupérable."""
        invoices = finance.get("invoices", [])
        overdue = [i for i in invoices if i.get("is_overdue", False)]

        overdue_sorted = sorted(
            overdue,
            key=lambda x: x.get("amount_due", 0),
            reverse=True,
        )

        return {
            "count": len(overdue),
            "value": round(sum(i.get("amount_due", 0) for i in overdue), 2),
            "details": [
                {
                    "invoice_id": i.get("invoice_id"),
                    "customer": i.get("customer_name"),
                    "amount_due": round(i.get("amount_due", 0), 2),
                    "days_overdue": i.get("days_overdue", 0),
                }
                for i in overdue_sorted[:_TOP_OVERDUE_LIMIT]
            ],
        }

    # ══════════════════════════════════════════
    # ACCURACY — Feedback loop
    # ══════════════════════════════════════════

    @staticmethod
    def _compute_accuracy(
        previous_metrics: Optional[dict],
        current_position: dict[str, Any],
    ) -> Optional[float]:
        """Calcule la précision du forecast précédent vs la réalité.
        accuracy = 1 - |predicted - actual| / |actual|
        """
        if not previous_metrics:
            return None

        predicted = previous_metrics.get("forecast_base_30d")
        actual = current_position.get("cash_current")

        if predicted is None or actual is None or actual == 0:
            return None

        error = abs(predicted - actual) / abs(actual)
        return round(max(0.0, 1.0 - error), 3)

    # ══════════════════════════════════════════
    # ÉVÉNEMENTS
    # ══════════════════════════════════════════

    def _emit_forecast(self, forecast: dict[str, Any]) -> None:
        """Émet le forecast pour consommation inter-agents."""
        self._emit_event(
            event_type=EventType.CASH_FORECAST_GENERATED,
            priority=EventPriority.MEDIUM,
            payload={
                "base_30d": forecast["base"]["day_30"],
                "stress_30d": forecast["stress"]["day_30"],
                "upside_30d": forecast["upside"]["day_30"],
                "confidence": forecast["confidence"],
            },
        )

    def _emit_alerts(
        self,
        alerts: list[dict[str, Any]],
        position: dict[str, Any],
    ) -> None:
        """Émet les événements d'alerte cash."""
        for alert in alerts:
            if alert["level"] == "red":
                event_type = EventType.CASH_ALERT_RED
                priority = EventPriority.CRITICAL
            else:
                event_type = EventType.CASH_ALERT_YELLOW
                priority = EventPriority.HIGH

            self._emit_event(
                event_type=event_type,
                priority=priority,
                payload={
                    "days_until_threshold": alert["days_until_threshold"],
                    "scenario": alert["scenario"],
                    "threshold": alert["threshold"],
                    "projected_cash": alert["projected_cash"],
                    "current_cash": position["cash_current"],
                    "actions_count": len(alert.get("actions", [])),
                },
            )

    # ══════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════

    def _build_summary(
        self,
        position: dict[str, Any],
        forecast: dict[str, Any],
        alerts: list[dict[str, Any]],
        overdue: dict[str, Any],
    ) -> str:
        """Résumé texte pour dashboard et rapports."""
        parts: list[str] = []

        # Position
        parts.append(
            f"Cash actuel : {self.format_currency(position['cash_current'])}"
        )
        parts.append(
            f"Runway : {position['runway_months']} mois"
        )

        # Forecast base 30j
        base_30 = forecast["base"]["day_30"]
        parts.append(
            f"Forecast 30j (base) : {self.format_currency(base_30)}"
        )

        # Forecast stress 30j
        stress_30 = forecast["stress"]["day_30"]
        parts.append(
            f"Forecast 30j (stress) : {self.format_currency(stress_30)}"
        )

        # Alertes
        if alerts:
            worst = alerts[0]
            parts.append(
                f"⚠️ Alerte {worst['level'].upper()} : "
                f"seuil franchi dans {worst['days_until_threshold']}j "
                f"(scénario {worst['scenario']})"
            )
        else:
            parts.append("✅ Aucune alerte cash")

        # Overdue
        if overdue["count"] > 0:
            parts.append(
                f"Factures en retard : {overdue['count']} "
                f"({self.format_currency(overdue['value'])})"
            )

        return " | ".join(parts)
