"""
Agent 1 : Revenue Velocity — LLM-first.

KPI : €/jour traversant le pipeline.

Actions :
  1.1 — Pipeline Cleaner [A]
  1.4 — Lead Scoring [A]
  1.8 — Proposal Generator [B]
  1.9 — Win/Loss Analyzer [A]

System prompt : prompts/system/revenue_velocity.txt
User prompts : prompts/revenue_velocity/*.txt
"""

from __future__ import annotations

from typing import Any

from models.event import Event
from models.state import StateSnapshot, DealState
from models.decision import Decision, DecisionType, RiskLevel
from models.agent_config import AgentConfig, AgentType
from agents.base import BaseAgent


class RevenueVelocityAgent(BaseAgent):
    """
    Revenue Velocity — Le LLM est votre meilleur sales ops.

    Il voit le pipeline, classifie les deals, score les leads,
    analyse les win/loss, et génère des propositions.
    """

    @property
    def agent_type(self) -> AgentType:
        return AgentType.REVENUE_VELOCITY

    def _custom_prompt_variables(self) -> dict[str, Any]:
        """Variables spécifiques injectées dans les prompts Revenue Velocity."""
        params = self.config.parameters
        weights = params.get("score_weights", {})
        return {
            "stagnation_threshold_days": params.get(
                "stagnation_threshold_days", 21
            ),
            "zombie_threshold_days": params.get("zombie_threshold_days", 45),
            "hot_lead_threshold": params.get("hot_lead_threshold", 75),
            "archive_threshold": params.get("archive_threshold", 20),
            "score_fit": weights.get("fit", 0.3),
            "score_activity": weights.get("activity", 0.4),
            "score_timing": weights.get("timing", 0.2),
            "score_source": weights.get("source", 0.1),
            "pipeline_realistic_factor": params.get(
                "pipeline_realistic_factor", 0.5
            ),
            "high_value_deal_threshold": params.get(
                "high_value_deal_threshold", 10000
            ),
            "forecast_correction_factor": params.get(
                "forecast_correction_factor", 1.0
            ),
        }

    async def build_snapshot(
        self,
        events: list[Event] | None = None,
        raw_data: dict[str, Any] | None = None,
    ) -> StateSnapshot:
        """Construit le snapshot pipeline + deals."""
        deals: list[DealState] = []

        if raw_data and "deals" in raw_data:
            for d in raw_data["deals"]:
                deals.append(DealState(
                    deal_id=d.get("id", ""),
                    name=d.get("name", ""),
                    amount=d.get("amount", 0),
                    stage=d.get("stage", ""),
                    owner=d.get("owner", ""),
                    created_at=d.get("created_at"),
                    last_activity_at=d.get("last_activity_at"),
                    days_in_stage=d.get("days_in_stage", 0),
                    days_no_activity=d.get("days_no_activity", 0),
                    probability=d.get("probability", 0.5),
                    source=d.get("source", ""),
                    contact_email=d.get("contact_email", ""),
                    notes=d.get("notes", []),
                    tags=d.get("tags", []),
                ))

        recent = []
        if events:
            recent = [e.to_snapshot_dict() for e in events[:50]]

        return StateSnapshot(
            company_id=self.company_id,
            agent_type=self.agent_type.value,
            deals=deals,
            recent_events=recent,
            previous_decisions=[
                {
                    "type": d.decision_type.value,
                    "confidence": d.confidence,
                    "reasoning": d.reasoning[:200],
                }
                for d in self._decisions[-5:]
            ],
        )

    def validate_decision(self, decision: Decision) -> list[str]:
        """
        Valide les règles métier Revenue Velocity.

        Interdictions :
          - Modifier les montants des deals
          - Supprimer des deals
          - Actions sans target
        """
        errors: list[str] = []

        for action in decision.actions:
            # Le LLM ne peut PAS changer les montants
            if action.action == "update_deal_amount":
                errors.append(
                    "INTERDIT : modification de montant de deal"
                )

            # Le LLM ne peut pas supprimer des deals
            if action.action == "delete_deal":
                errors.append("INTERDIT : suppression de deal")

            # Vérifier que les targets existent
            if action.action.startswith("update_deal") and not action.target:
                errors.append(
                    f"Action '{action.action}' sans target deal_id"
                )

            # Lead score doit être dans les bornes
            if action.action == "update_lead_score":
                score = action.parameters.get("score")
                if score is not None and (score < 0 or score > 100):
                    errors.append(
                        f"Lead score hors bornes : {score} (doit être 0-100)"
                    )

        return errors

    def _default_decision_type(self) -> DecisionType:
        return DecisionType.CLASSIFY_DEAL

    # ──────────────────────────────────────────────────────
    # CONVENIENCE METHODS — Raccourcis par action
    # ──────────────────────────────────────────────────────

    async def clean_pipeline(
        self, deals_data: list[dict[str, Any]]
    ) -> Decision:
        """
        Action 1.1 — Pipeline Cleaner [A].

        CRON quotidien 6h.
        Classifie chaque deal : active, stagnant, zombie,
        recoverable, misclassified.
        """
        return await self.run(
            raw_data={"deals": deals_data},
            prompt_name="revenue_velocity/pipeline_cleaner",
        )

    async def score_leads(
        self, leads_data: list[dict[str, Any]]
    ) -> Decision:
        """
        Action 1.4 — Lead Scoring [A].

        Score 0-100 basé sur fit, activity, timing, source.
        Hybride : scoring rule-based + LLM explainability.
        """
        return await self.run(
            raw_data={"deals": leads_data},
            prompt_name="revenue_velocity/lead_scoring",
        )

    async def analyze_win_loss(
        self,
        won_deals: list[dict[str, Any]],
        lost_deals: list[dict[str, Any]],
    ) -> Decision:
        """
        Action 1.9 — Win/Loss Analyzer [A].

        Détecte les patterns de victoire et d'échec.
        Produit des knowledge artifacts réutilisables.
        """
        return await self.run(
            raw_data={"deals": won_deals + lost_deals},
            prompt_name="revenue_velocity/win_loss_analysis",
            prompt_variables={
                "won_deals": won_deals,
                "lost_deals": lost_deals,
            },
        )

    async def generate_proposal(
        self, deal_data: dict[str, Any]
    ) -> Decision:
        """
        Action 1.8 — Proposal Generator [B].

        Génère une proposition commerciale structurée.
        Risk B : validation humaine obligatoire avant envoi.
        """
        return await self.run(
            raw_data={"deals": [deal_data]},
            prompt_name="revenue_velocity/proposal_generator",
)

                    
