"""
Agent 3 : Cash Predictability — LLM-first.

KPI : Précision forecast cash à 30 jours.

Actions :
  3.2 — Payment Pattern Detection [A]
  3.5 — Cash Preventive Plan [B]
  3.x — Cash Forecast + Scenarios [A]

Rigueur > Créativité.
Les CALCULS restent déterministes.
Le LLM fait l'INTERPRÉTATION et les recommandations.
"""

from __future__ import annotations

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
    """

    @property
    def agent_type(self) -> AgentType:
        return AgentType.CASH_PREDICTABILITY

    @property
    def system_prompt(self) -> str:
        return """You are a world-class CFO AI assistant.

Your job is to predict cash flow, detect payment risks, and recommend preventive actions.

You receive a cash state snapshot with: balance, revenue, expenses, receivables, payables, pipeline.

IMPORTANT: All financial calculations are pre-computed and provided to you.
DO NOT recalculate numbers. Trust the data.
Your role is INTERPRETATION and RECOMMENDATIONS.

You MUST respond with a JSON object containing:
{
  "decision_type": "forecast_cash" | "detect_payment_risk" | "generate_cash_plan" | "recommend",
  "confidence": 0.0-1.0,
  "reasoning": "your analysis in 2-3 sentences",
  "risk_level": "A" | "B" | "C",
  "actions": [
    {
      "action": "send_alert" | "send_invoice_reminder" | "create_task" | "generate_report",
      "target": "...",
      "parameters": {...},
      "risk_level": "A",
      "priority": 1-10
    }
  ],
  "metadata": {
    "runway_months": 0.0,
    "cash_health": "healthy" | "warning" | "critical",
    "overdue_amount": 0.0,
    "scenario_base": {"description": "...", "cash_30d": 0.0},
    "scenario_stress": {"description": "...", "cash_30d": 0.0},
    "scenario_upside": {"description": "...", "cash_30d": 0.0},
    "top_risks": ["..."],
    "action_levers": [
      {
        "lever": "description",
        "amount": 0.0,
        "timeline": "X days",
        "probability": 0.0
      }
    ]
  }
}

RULES:
- NEVER invent numbers. Use only what's provided.
- Always produce 3 scenarios: base, stress, upside.
- Each action lever must be quantified: amount, timeline, probability.
- Alert if runway < 3 months (warning) or < 1.5 months (critical).
- Detect payment patterns: mean delay, variance, trend.
- Be conservative. Optimism kills companies.
"""

    async def build_snapshot(
        self,
        events: list[Event] | None = None,
        raw_data: dict[str, Any] | None = None,
    ) -> StateSnapshot:
        """Construit le snapshot cash."""
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

        # Pre-compute deterministic metrics
        if cash and raw_data:
            cash = self._enrich_cash_state(cash, raw_data)

        recent = []
        if events:
            recent = [e.to_snapshot_dict() for e in events[:30]]

        return StateSnapshot(
            company_id=self.company_id,
            agent_type=self.agent_type.value,
            cash=cash,
            recent_events=recent,
            company_context=raw_data.get("payment_history", {}) if raw_data else {},
            previous_decisions=[
                {
                    "type": d.decision_type.value,
                    "confidence": d.confidence,
                    "reasoning": d.reasoning[:200],
                }
                for d in self._decisions[-5:]
            ],
        )

    def _enrich_cash_state(
        self, cash: CashState, raw_data: dict[str, Any]
    ) -> CashState:
        """
        Calculs DÉTERMINISTES — jamais délégués au LLM.

        Le LLM reçoit les résultats, pas les inputs.
        """
        # Burn rate
        if cash.monthly_expenses > 0:
            cash.monthly_burn_rate = cash.monthly_expenses - cash.monthly_revenue

        # Runway
        if cash.monthly_burn_rate > 0:
            cash.runway_months = round(
                cash.cash_balance / cash.monthly_burn_rate, 1
            )
        elif cash.monthly_burn_rate <= 0:
            cash.runway_months = 99.0  # Cash positive

        # Payment stats from history
        history = raw_data.get("payment_history", {})
        if history.get("delays"):
            delays = history["delays"]
            import statistics as stats
            cash.avg_payment_delay_days = round(stats.mean(delays), 1)

        return cash

    def validate_decision(self, decision: Decision) -> list[str]:
        """Valide les règles métier Cash."""
        errors: list[str] = []

        for action in decision.actions:
            # Pas de paiement automatique
            if action.action in ("make_payment", "transfer_funds"):
                errors.append("INTERDIT : action financière directe")

            # Les rappels de facture sont OK mais doivent avoir un montant
            if action.action == "send_invoice_reminder":
                if not action.parameters.get("invoice_id"):
                    errors.append("send_invoice_reminder sans invoice_id")

        return errors

    def _default_decision_type(self) -> DecisionType:
        return DecisionType.FORECAST_CASH

    # ──────────────────────────────────────────────────────
    # CONVENIENCE
    # ──────────────────────────────────────────────────────

    async def forecast(self, cash_data: dict[str, Any]) -> Decision:
        """Forecast cash + 3 scénarios."""
        return await self.run(
            raw_data={"cash": cash_data},
            prompt_name="cash_predictability/forecast",
        )

    async def detect_payment_risks(
        self,
        cash_data: dict[str, Any],
        payment_history: list[dict[str, Any]],
    ) -> Decision:
        """Action 3.2 — Payment pattern detection."""
        return await self.run(
            raw_data={
                "cash": cash_data,
                "payment_history": {"delays": payment_history},
            },
            prompt_name="cash_predictability/payment_risk",
        )

    async def generate_preventive_plan(
        self, cash_data: dict[str, Any]
    ) -> Decision:
        """Action 3.5 — Cash preventive plan [B]."""
        return await self.run(
            raw_data={"cash": cash_data},
            prompt_name="cash_predictability/preventive_plan",
       )
