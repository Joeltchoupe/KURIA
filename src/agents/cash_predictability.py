"""
Agent 3 : Cash Predictability — LLM-first.

KPI : Précision forecast cash à 30 jours.

Actions :
  3.2 — Payment Pattern Detection [A]
  3.5 — Cash Preventive Plan [B]
  3.x — Cash Forecast + 3 Scenarios [A]

Rigueur > Créativité.
Les CALCULS restent déterministes (Python).
Le LLM fait l'INTERPRÉTATION et les recommandations.

System prompt : prompts/system/cash_predictability.txt
User prompts : prompts/cash_predictability/*.txt
"""

from __future__ import annotations

import statistics as stats_lib
from typing import Any

from models.event import Event
from models.state import StateSnapshot, CashState
from models.decision import Decision, DecisionType, RiskLevel
from models.agent_config import AgentConfig, AgentType
from agents.base import BaseAgent


class CashPredictabilityAgent(BaseAgent):
    """
    Cash Predictability — Rigueur maximale.

    Ne laisse JAMAIS le LLM faire les calculs.
    Le LLM interprète, recommande, alerte.
    Les chiffres sont pré-calculés en Python.
    """

    @property
    def agent_type(self) -> AgentType:
        return AgentType.CASH_PREDICTABILITY

    def _custom_prompt_variables(self) -> dict[str, Any]:
        """Variables spécifiques injectées dans les prompts Cash."""
        params = self.config.parameters
        return {
            "cash_threshold_yellow_months": params.get(
                "cash_threshold_yellow_months", 3.0
            ),
            "cash_threshold_red_months": params.get(
                "cash_threshold_red_months", 1.5
            ),
            "stress_pipeline_factor": params.get(
                "stress_pipeline_factor", 0.5
            ),
            "stress_payment_delay_days": params.get(
                "stress_payment_delay_days", 15
            ),
            "upside_pipeline_factor": params.get(
                "upside_pipeline_factor", 1.2
            ),
            "forecast_optimism_correction": params.get(
                "forecast_optimism_correction", 1.0
            ),
        }

    async def build_snapshot(
        self,
        events: list[Event] | None = None,
        raw_data: dict[str, Any] | None = None,
    ) -> StateSnapshot:
        """
        Construit le snapshot cash.

        Les calculs déterministes (burn rate, runway) sont faits ICI,
        pas par le LLM.
        """
        cash = None

        if raw_data and "cash" in raw_data:
            c = raw_data["cash"]
            cash = CashState(
                cash_balance=c.get("cash_balance", 0),
                monthly_revenue=c.get("monthly_revenue", 0),
                monthly_expenses=c.get("monthly_expenses", 0),
                monthly_burn_rate=c.get("monthly_burn_rate", 0),
                runway_months=c.get("runway_months", 0),
                outstanding_receivables=c.get("outstanding_receivables", 0),
                outstanding_payables=c.get("outstanding_payables", 0),
                avg_payment_delay_days=c.get("avg_payment_delay_days", 0),
                overdue_invoices_count=c.get("overdue_invoices_count", 0),
                overdue_invoices_amount=c.get("overdue_invoices_amount", 0),
                pipeline_weighted_value=c.get("pipeline_weighted_value", 0),
            )

            # Enrichir avec calculs déterministes
            cash = self._compute_deterministic_metrics(cash, raw_data)

        recent = []
        if events:
            recent = [e.to_snapshot_dict() for e in events[:30]]

        # Payment history pour le contexte
        company_context: dict[str, Any] = {}
        if raw_data:
            if "payment_history" in raw_data:
                company_context["payment_history"] = raw_data[
                    "payment_history"
                ]
            if "invoices" in raw_data:
                company_context["invoices"] = raw_data["invoices"]

        return StateSnapshot(
            company_id=self.company_id,
            agent_type=self.agent_type.value,
            cash=cash,
            recent_events=recent,
            company_context=company_context,
            previous_decisions=[
                {
                    "type": d.decision_type.value,
                    "confidence": d.confidence,
                    "reasoning": d.reasoning[:200],
                }
                for d in self._decisions[-5:]
            ],
        )

    def _compute_deterministic_metrics(
        self, cash: CashState, raw_data: dict[str, Any]
    ) -> CashState:
        """
        Calculs DÉTERMINISTES — jamais délégués au LLM.

        Le LLM reçoit les RÉSULTATS, pas les inputs.
        Il interprète, il ne calcule pas.
        """
        # Burn rate
        if cash.monthly_expenses > 0 and cash.monthly_revenue >= 0:
            cash.monthly_burn_rate = cash.monthly_expenses - cash.monthly_revenue

        # Runway
        if cash.monthly_burn_rate > 0:
            cash.runway_months = round(
                cash.cash_balance / cash.monthly_burn_rate, 1
            )
        elif cash.monthly_burn_rate <= 0:
            cash.runway_months = 99.0  # Cash positive

        # Payment delay stats from history
        history = raw_data.get("payment_history", {})
        delays = history.get("delays", [])
        if delays and len(delays) >= 3:
            cash.avg_payment_delay_days = round(stats_lib.mean(delays), 1)

        return cash

    def validate_decision(self, decision: Decision) -> list[str]:
        """
        Valide les règles métier Cash.

        Interdictions :
          - Actions financières directes (paiements, virements)
          - Invoice reminders sans invoice_id
          - Montants inventés
        """
        errors: list[str] = []

        for action in decision.actions:
            # Pas de paiement automatique
            if action.action in (
                "make_payment",
                "transfer_funds",
                "wire_transfer",
                "refund",
            ):
                errors.append(
                    f"INTERDIT : action financière directe '{action.action}'"
                )

            # Les rappels de facture doivent avoir un invoice_id
            if action.action == "send_invoice_reminder":
                if not action.parameters.get("invoice_id"):
                    errors.append(
                        "send_invoice_reminder sans invoice_id"
                    )
                amount = action.parameters.get("amount")
                if amount is not None and amount < 0:
                    errors.append(
                        f"Montant négatif dans invoice_reminder : {amount}"
                    )

        return errors

    def _default_decision_type(self) -> DecisionType:
        return DecisionType.FORECAST_CASH

    # ──────────────────────────────────────────────────────
    # CONVENIENCE METHODS — Raccourcis par action
    # ──────────────────────────────────────────────────────

    async def forecast(
        self, cash_data: dict[str, Any]
    ) -> Decision:
        """
        Cash Forecast + 3 scénarios.

        Base, Stress, Upside — toujours les 3.
        Calculs déterministes dans build_snapshot,
        interprétation par le LLM.
        """
        return await self.run(
            raw_data={"cash": cash_data},
            prompt_name="cash_predictability/forecast",
        )

    async def detect_payment_risks(
        self,
        cash_data: dict[str, Any],
        payment_history: list[dict[str, Any]] | list[float],
    ) -> Decision:
        """
        Action 3.2 — Payment Pattern Detection [A].

        Analyse : mean delay, variance, 90-day trend slope.
        Statistiques en Python, interprétation par le LLM.
        """
        # Normaliser l'historique
        if payment_history and isinstance(payment_history[0], (int, float)):
            delays = payment_history
        else:
            delays = [
                p.get("delay_days", 0)
                for p in payment_history
                if isinstance(p, dict)
            ]

        return await self.run(
            raw_data={
                "cash": cash_data,
                "payment_history": {"delays": delays},
            },
            prompt_name="cash_predictability/payment_risk",
        )

    async def generate_preventive_plan(
        self, cash_data: dict[str, Any]
    ) -> Decision:
        """
        Action 3.5 — Cash Preventive Plan [B].

        Risk B : recommandations stratégiques → validation CEO.
        3 leviers quantifiés minimum.
        """
        return await self.run(
            raw_data={"cash": cash_data},
            prompt_name="cash_predictability/preventive_plan",
        )
