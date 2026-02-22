"""
Agent 0 : Scanner — Diagnostic permanent.

PAS un one-shot. Tourne tous les 3 jours.

Cycle de vie :
  Jour 0      → Scan initial (diagnostic complet, Clarity Score baseline)
  Jour 3      → Delta scan (qu'est-ce qui a changé ?)
  Jour 6      → Delta scan + trend
  ...
  Chaque scan → Compare vs le précédent, mesure la progression,
                détecte les nouveaux problèmes, recalibre les agents

C'est du consulting permanent automatisé.
Le CEO voit son entreprise devenir plus lisible scan après scan.

3 modes :
  INITIAL   → Premier diagnostic complet (Jour 0)
  DELTA     → Scan différentiel (chaque 3 jours)
  DEEP      → Re-diagnostic complet (sur demande ou mensuel)

System prompt : prompts/system/scanner.txt
User prompts :
  - scanner/initial_scan.txt
  - scanner/delta_scan.txt
  - scanner/deep_scan.txt
  - scanner/data_quality.txt
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from models.event import Event
from models.state import (
    StateSnapshot,
    DealState,
    TaskState,
    CashState,
    MarketingState,
    ChannelState,
)
from models.decision import Decision, DecisionType, RiskLevel
from models.agent_config import AgentConfig, AgentType
from agents.base import BaseAgent


class ScanMode(str, Enum):
    """Mode de scan."""
    INITIAL = "initial"   # Jour 0 — diagnostic complet
    DELTA = "delta"       # Tous les 3 jours — différentiel
    DEEP = "deep"         # Sur demande — re-diagnostic complet


class ScannerAgent(BaseAgent):
    """
    Scanner — Consulting permanent automatisé.

    Tourne tous les 3 jours. Compare chaque scan au précédent.
    Mesure la progression du Score de Clarté.
    Détecte les nouvelles frictions et les améliorations.
    Recommande des ajustements de config agents.

    Le scanner est le SEUL agent qui voit TOUT.
    Les autres agents ne voient que leur domaine.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        # Historique des scans (pour le delta)
        self._scan_history: list[dict[str, Any]] = []
        self._current_clarity_score: float | None = None
        self._scan_count: int = 0

    @property
    def agent_type(self) -> AgentType:
        return AgentType.SCANNER

    def _custom_prompt_variables(self) -> dict[str, Any]:
        """Variables spécifiques au scanner."""
        return {
            "scan_count": self._scan_count,
            "previous_clarity_score": self._current_clarity_score,
            "has_previous_scan": len(self._scan_history) > 0,
            "min_deals_for_rv": 5,
            "min_leads_for_ae": 10,
            "min_channels_for_ae": 2,
            "scan_frequency_days": 3,
        }

    async def build_snapshot(
        self,
        events: list[Event] | None = None,
        raw_data: dict[str, Any] | None = None,
    ) -> StateSnapshot:
        """
        Construit le snapshot COMPLET de l'entreprise.

        Le scanner remplit TOUS les states :
        deals, tasks, cash, marketing.
        Les autres agents ne lisent que leur domaine.
        """
        deals: list[DealState] = []
        tasks: list[TaskState] = []
        cash: CashState | None = None
        marketing: MarketingState | None = None

        if raw_data:
            # ── CRM / Deals ──
            if "deals" in raw_data:
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

            # ── Tasks ──
            if "tasks" in raw_data:
                for t in raw_data["tasks"]:
                    tasks.append(TaskState(
                        task_id=t.get("id", ""),
                        title=t.get("title", ""),
                        assigned_to=t.get("assigned_to", ""),
                        status=t.get("status", ""),
                        deadline=t.get("deadline"),
                        created_at=t.get("created_at"),
                        completed_at=t.get("completed_at"),
                        days_overdue=t.get("days_overdue", 0),
                        process_name=t.get("process_name", ""),
                        tags=t.get("tags", []),
                    ))

            # ── Finance ──
            if "cash" in raw_data:
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

            # ── Marketing ──
            if "channels" in raw_data:
                channels: list[ChannelState] = []
                for ch in raw_data["channels"]:
                    channels.append(ChannelState(
                        channel=ch.get("channel", ""),
                        spend_30d=ch.get("spend_30d", 0),
                        leads_30d=ch.get("leads_30d", 0),
                        clients_30d=ch.get("clients_30d", 0),
                        cac=ch.get("cac", 0),
                        conversion_rate=ch.get("conversion_rate", 0),
                        trend_pct=ch.get("trend_pct", 0),
                    ))

                total_spend = sum(ch.spend_30d for ch in channels)
                total_leads = sum(ch.leads_30d for ch in channels)
                total_clients = sum(ch.clients_30d for ch in channels)
                blended_cac = (
                    round(total_spend / total_clients, 2)
                    if total_clients > 0
                    else 0
                )
                avg_ltv = raw_data.get("avg_ltv", 0)

                marketing = MarketingState(
                    channels=channels,
                    total_leads_30d=total_leads,
                    total_spend_30d=total_spend,
                    blended_cac=blended_cac,
                    avg_ltv=avg_ltv,
                    ltv_cac_ratio=(
                        round(avg_ltv / blended_cac, 2)
                        if blended_cac > 0
                        else 0
                    ),
                )

        # ── Méta + contexte ──
        company_context: dict[str, Any] = {}
        if raw_data:
            company_context = {
                "connected_tools": raw_data.get("connected_tools", []),
                "company_name": raw_data.get("company_name", ""),
                "company_size": raw_data.get("company_size", ""),
                "industry": raw_data.get("industry", ""),
                "data_stats": {
                    "total_deals": len(deals),
                    "total_tasks": len(tasks),
                    "has_finance": cash is not None,
                    "has_marketing": marketing is not None,
                    "marketing_channels": (
                        len(marketing.channels) if marketing else 0
                    ),
                },
            }

            if "email_stats" in raw_data:
                company_context["email_stats"] = raw_data["email_stats"]
            if "team" in raw_data:
                company_context["team"] = raw_data["team"]

        # ── Historique des scans précédents (pour delta) ──
        previous_decisions = []
        if self._scan_history:
            last_scan = self._scan_history[-1]
            previous_decisions.append({
                "type": "previous_scan",
                "scan_number": last_scan.get("scan_number", 0),
                "clarity_score": last_scan.get("clarity_score", 0),
                "timestamp": last_scan.get("timestamp", ""),
                "top_frictions": last_scan.get("top_frictions", []),
                "agent_recommendations": last_scan.get(
                    "agent_recommendations", {}
                ),
            })

        # Ajouter les 3 dernières décisions d'agents (feedback loop)
        previous_decisions.extend([
            {
                "type": d.decision_type.value,
                "confidence": d.confidence,
                "reasoning": d.reasoning[:200],
            }
            for d in self._decisions[-3:]
        ])

        recent = []
        if events:
            recent = [e.to_snapshot_dict() for e in events[:50]]

        return StateSnapshot(
            company_id=self.company_id,
            agent_type=self.agent_type.value,
            deals=deals,
            tasks=tasks,
            cash=cash,
            marketing=marketing,
            recent_events=recent,
            company_context=company_context,
            previous_decisions=previous_decisions,
        )

    def validate_decision(self, decision: Decision) -> list[str]:
        """
        Valide la décision du scanner.

        Le scanner est safe — il ne fait que lire et analyser.
        Vérifications :
          - Clarity score 0-100
          - Sous-scores 0-100
          - Agents recommandés valides
          - Waste estimate non négatif
        """
        errors: list[str] = []
        metadata = decision.metadata

        # Clarity score
        clarity = metadata.get("clarity_score", {})
        if isinstance(clarity, dict):
            score = clarity.get("score")
            if score is not None:
                if not (0 <= score <= 100):
                    errors.append(
                        f"Clarity score hors bornes : {score}"
                    )

            for sub in [
                "machine_readability",
                "structural_compatibility",
                "data_quality",
                "data_completeness",
                "process_explicitness",
                "tool_integration",
            ]:
                val = clarity.get(sub)
                if val is not None and not (0 <= val <= 100):
                    errors.append(f"Sous-score '{sub}' hors bornes : {val}")

        # Agent recommendations
        agent_recs = metadata.get("agent_recommendations", {})
        if isinstance(agent_recs, dict):
            valid_agents = {
                "revenue_velocity",
                "process_clarity",
                "cash_predictability",
                "acquisition_efficiency",
            }
            for name in agent_recs:
                if name not in valid_agents:
                    errors.append(f"Agent inconnu : {name}")

        # Waste estimate
        waste = metadata.get("estimated_annual_waste")
        if waste is not None and waste < 0:
            errors.append(f"Waste négatif : {waste}")

        # Delta-specific : progression doit être cohérente
        progression = metadata.get("progression", {})
        if isinstance(progression, dict):
            delta = progression.get("score_delta")
            if delta is not None and abs(delta) > 30:
                errors.append(
                    f"Score delta suspect : {delta} points en 3 jours"
                )

        return errors

    def _default_decision_type(self) -> DecisionType:
        return DecisionType.SUMMARIZE

    # ──────────────────────────────────────────────────────
    # POST-RUN HOOK — Track scan history
    # ──────────────────────────────────────────────────────

    def _record_scan(self, decision: Decision) -> None:
        """Enregistre le scan dans l'historique pour les deltas."""
        metadata = decision.metadata
        clarity = metadata.get("clarity_score", {})
        score = clarity.get("score") if isinstance(clarity, dict) else None

        if score is not None:
            self._current_clarity_score = score

        self._scan_count += 1

        self._scan_history.append({
            "scan_number": self._scan_count,
            "timestamp": datetime.utcnow().isoformat(),
            "clarity_score": score,
            "top_frictions": metadata.get("top_3_frictions", []),
            "agent_recommendations": metadata.get(
                "agent_recommendations", {}
            ),
            "estimated_annual_waste": metadata.get(
                "estimated_annual_waste", 0
            ),
            "decision_id": decision.id,
        })

    # ──────────────────────────────────────────────────────
    # SCAN MODES
    # ──────────────────────────────────────────────────────

    async def initial_scan(
        self, raw_data: dict[str, Any]
    ) -> Decision:
        """
        JOUR 0 — Scan initial complet.

        Premier diagnostic. Produit :
          - Clarity Score (baseline)
          - Data Portrait complet
          - Agent recommendations
          - Estimated annual waste
          - Top 3 frictions
          - Quick wins

        Après ce scan, les agents sont configurés et activés.
        """
        decision = await self.run(
            raw_data=raw_data,
            prompt_name="scanner/initial_scan",
            prompt_variables={"scan_mode": ScanMode.INITIAL.value},
        )

        if decision.validated:
            self._record_scan(decision)

        return decision

    async def delta_scan(
        self, raw_data: dict[str, Any]
    ) -> Decision:
        """
        TOUS LES 3 JOURS — Scan différentiel.

        Compare avec le scan précédent :
          - Score de Clarté : progression ? régression ?
          - Nouvelles frictions détectées
          - Frictions résolues
          - Quick wins appliqués ?
          - Agents à recalibrer ?

        C'est le consulting permanent.
        Le CEO voit son entreprise s'améliorer scan après scan.
        """
        if not self._scan_history:
            # Pas d'historique → force un initial scan
            return await self.initial_scan(raw_data)

        last_scan = self._scan_history[-1]

        decision = await self.run(
            raw_data=raw_data,
            prompt_name="scanner/delta_scan",
            prompt_variables={
                "scan_mode": ScanMode.DELTA.value,
                "previous_scan": last_scan,
                "previous_clarity_score": last_scan.get(
                    "clarity_score", 0
                ),
                "previous_frictions": last_scan.get("top_frictions", []),
                "previous_agent_recs": last_scan.get(
                    "agent_recommendations", {}
                ),
                "scan_number": self._scan_count + 1,
                "days_since_last_scan": 3,
            },
        )

        if decision.validated:
            self._record_scan(decision)

        return decision

    async def deep_scan(
        self, raw_data: dict[str, Any]
    ) -> Decision:
        """
        SUR DEMANDE ou MENSUEL — Re-diagnostic complet.

        Comme un initial scan mais avec tout l'historique.
        Produit un rapport de progression sur toute la période.
        Recommande des changements structurels si nécessaire.
        """
        decision = await self.run(
            raw_data=raw_data,
            prompt_name="scanner/deep_scan",
            prompt_variables={
                "scan_mode": ScanMode.DEEP.value,
                "scan_history": self._scan_history[-10:],
                "total_scans": self._scan_count,
                "initial_clarity_score": (
                    self._scan_history[0].get("clarity_score", 0)
                    if self._scan_history
                    else None
                ),
                "current_clarity_score": self._current_clarity_score,
            },
        )

        if decision.validated:
            self._record_scan(decision)

        return decision

    async def assess_data_quality(
        self, raw_data: dict[str, Any]
    ) -> Decision:
        """
        Évaluation ciblée de la qualité des données.

        Plus léger qu'un scan complet.
        Focus : fill rate, doublons, fraîcheur, complétude.
        """
        return await self.run(
            raw_data=raw_data,
            prompt_name="scanner/data_quality",
        )

    # ──────────────────────────────────────────────────────
    # ACCESSORS
    # ──────────────────────────────────────────────────────

    @property
    def scan_count(self) -> int:
        return self._scan_count

    @property
    def current_clarity_score(self) -> float | None:
        return self._current_clarity_score

    @property
    def clarity_trend(self) -> list[dict[str, Any]]:
        """Historique du Clarity Score pour le dashboard."""
        return [
            {
                "scan": s.get("scan_number", 0),
                "date": s.get("timestamp", ""),
                "score": s.get("clarity_score"),
            }
            for s in self._scan_history
            if s.get("clarity_score") is not None
        ]

    @property
    def scan_history(self) -> list[dict[str, Any]]:
        return list(self._scan_history)
