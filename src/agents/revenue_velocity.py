"""
Revenue Velocity Agent — Pipeline truth, lead scoring, deal acceleration.

KPI NORD : Revenue Velocity = Pipeline réaliste ÷ Cycle moyen = €/jour

10 actions :
1.1  Nettoyage automatique pipeline          [A]
1.2  Mise à jour automatique forecast        [A]
1.3  Réorganisation automatique pipeline     [A]
1.4  Scoring et routing automatique          [A]
1.5  Nurture séquences automatiques          [B]
1.6  Enrichissement automatique leads        [A]
1.7  Relances automatiques                   [B]
1.8  Génération de propositions              [B]
1.9  Analyse win/loss automatique            [A]
1.10 Reporting pipeline automatique          [A]

Regroupées en 4 fonctions d'analyse :
1. Pipeline Truth    → Vérité sur le pipeline (1.1, 1.2, 1.3)
2. Lead Scoring      → Qui appeler en premier (1.4, 1.5, 1.6)
3. Stagnation Alert  → Deals mourants + actions (1.7, 1.8)
4. Win/Loss + Report → Apprentissage continu (1.9, 1.10)

C'est l'agent le plus VISIBLE. Résultats immédiats dès J1.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Optional

from models.agent_config import RevenueVelocityConfig, ScoreWeights
from models.events import EventType, EventPriority
from models.metrics import MetricTrend, RevenueVelocityMetrics

from agents.base import BaseAgent, InsufficientDataError


# ══════════════════════════════════════════
# CONSTANTES
# ══════════════════════════════════════════

_STAGNANT_REVIEW_STATUS = "Stagnant — Review"
_TASK_DEADLINE_HOURS = 48
_ESCALATION_HOURS = 4
_HOT_RESPONSE_HOURS = 2
_FORECAST_ALERT_THRESHOLD_PCT = 0.15  # 15%
_MAX_CYCLE_DAYS = 365
_DEFAULT_CYCLE_DAYS = 30.0
_DEFAULT_AVG_ACTIVITY = 5.0
_DEFAULT_AVG_AMOUNT = 10_000.0

# Classification des deals
_DEAL_CLASS_ZOMBIE = "zombie"
_DEAL_CLASS_DYING = "dying"
_DEAL_CLASS_AT_RISK = "at_risk"
_DEAL_CLASS_ACTIVE = "active"
_DEAL_CLASS_HEALTHY = "healthy"

# Lead actions
_ACTION_CALL_NOW = "call_now"
_ACTION_NURTURE = "nurture"
_ACTION_WATCH = "watch"
_ACTION_ARCHIVE = "archive"


# ══════════════════════════════════════════
# AGENT
# ══════════════════════════════════════════


class RevenueVelocityAgent(BaseAgent[RevenueVelocityConfig]):
    """Agent Revenue Velocity — le gardien du pipeline."""

    AGENT_NAME = "revenue_velocity"

    # ──────────────────────────────────────
    # CONTRAT BASE AGENT
    # ──────────────────────────────────────

    def _validate(self, data: dict[str, Any]) -> list[str]:
        """Valide les données CRM d'entrée."""
        warnings: list[str] = []
        deals = data.get("deals", [])
        contacts = data.get("contacts", [])

        if not deals and not contacts:
            raise InsufficientDataError(
                agent_name=self.AGENT_NAME,
                detail="Aucun deal ni contact trouvé dans le CRM.",
                available=0,
                required=1,
            )

        open_deals = [d for d in deals if not d.get("is_closed", False)]
        if not open_deals:
            warnings.append(
                "Aucun deal ouvert dans le pipeline. "
                "L'analyse sera limitée aux leads et à l'historique."
            )

        won_deals = [d for d in deals if d.get("is_won", False)]
        if len(won_deals) < 3:
            warnings.append(
                f"Seulement {len(won_deals)} deals gagnés. "
                f"Les benchmarks de probabilité seront moins fiables."
            )

        # Vérifier la qualité des données deals
        deals_without_amount = [d for d in open_deals if not d.get("amount")]
        if deals_without_amount:
            warnings.append(
                f"{len(deals_without_amount)} deals sans montant estimé. "
                f"L'agent va estimer les montants (action 1.3)."
            )

        deals_without_contact = [d for d in open_deals if not d.get("contact_ids")]
        if deals_without_contact:
            warnings.append(
                f"{len(deals_without_contact)} deals sans contact associé."
            )

        # Vérifier les contacts
        contacts_without_source = [c for c in contacts if not c.get("source")]
        if len(contacts_without_source) > len(contacts) * 0.5:
            warnings.append(
                f"{len(contacts_without_source)}/{len(contacts)} contacts sans source. "
                f"Le scoring source sera limité."
            )

        return warnings

    async def _execute(self, data: dict[str, Any]) -> dict[str, Any]:
        """Logique métier complète — 4 fonctions d'analyse."""
        deals = data.get("deals", [])
        contacts = data.get("contacts", [])
        owners = data.get("owners", {})
        stages = data.get("stages", {})

        # Segmentation des deals
        open_deals = [d for d in deals if not d.get("is_closed", False)]
        closed_deals = [d for d in deals if d.get("is_closed", False)]
        won_deals = [d for d in closed_deals if d.get("is_won", False)]
        lost_deals = [d for d in closed_deals if not d.get("is_won", False)]

        # ── FONCTION 1 : Pipeline Truth (actions 1.1, 1.2, 1.3) ──
        pipeline = self._pipeline_truth(open_deals, won_deals, lost_deals)

        # ── FONCTION 2 : Lead Scoring (actions 1.4, 1.5, 1.6) ──
        scoring = self._score_all_leads(contacts, won_deals, lost_deals)

        # ── FONCTION 3 : Stagnation (actions 1.7, 1.8) ──
        stagnation = self._detect_stagnation(open_deals)

        # ── FONCTION 4 : Win/Loss Analysis (action 1.9) ──
        winloss = self._analyze_winloss(won_deals, lost_deals)

        # ── KPI NORD : Revenue Velocity ──
        pipeline_realistic = pipeline["pipeline_realistic"]
        avg_cycle = pipeline["avg_cycle_days"]
        velocity = self.safe_divide(pipeline_realistic, max(avg_cycle, 1))

        # ── Forecast 30/60/90 (action 1.2) ──
        forecast = self._generate_forecast(
            open_deals, pipeline["deal_probabilities"], avg_cycle
        )

        # ── CRM Actions à exécuter (actions 1.1, 1.3) ──
        crm_actions = self._generate_crm_actions(
            pipeline, stagnation, scoring, open_deals, won_deals
        )

        # ── Tendance ──
        previous = await self.get_previous_metrics()
        prev_velocity = previous.get("revenue_velocity_per_day") if previous else None
        velocity_trend = None
        if prev_velocity is not None:
            velocity_trend = MetricTrend.calculate(velocity, prev_velocity).model_dump()

        # ── Forecast change alert (action 1.2) ──
        prev_forecast = previous.get("forecast_30d") if previous else None
        self._check_forecast_change(forecast["forecast_30d"], prev_forecast)

        # ── Métriques ──
        metrics = RevenueVelocityMetrics(
            company_id=self.company_id,
            revenue_velocity_per_day=round(velocity, 2),
            pipeline_declared=pipeline["pipeline_declared"],
            pipeline_realistic=round(pipeline_realistic, 2),
            pipeline_gap_ratio=round(pipeline["gap_ratio"], 2),
            deals_active=len(open_deals),
            deals_stagnant=len(stagnation["stagnant_deals"]),
            deals_zombie=len(stagnation["zombie_deals"]),
            forecast_30d=round(forecast["forecast_30d"], 2),
            forecast_60d=round(forecast["forecast_60d"], 2),
            forecast_90d=round(forecast["forecast_90d"], 2),
            forecast_confidence=round(forecast["confidence"], 2),
            leads_scored=scoring["total_scored"],
            leads_hot=scoring["hot_count"],
            leads_archived=scoring["archive_count"],
            deals_won_this_period=len(won_deals),
            deals_lost_this_period=len(lost_deals),
            revenue_won_this_period=sum(d.get("amount", 0) or 0 for d in won_deals),
        )

        # Persist métriques
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

        # ── Événements ──
        self._emit_pipeline_truth(pipeline, stagnation, velocity)
        self._emit_forecast_update(forecast)

        return {
            "revenue_velocity": round(velocity, 2),
            "velocity_trend": velocity_trend,
            "pipeline": pipeline,
            "scoring": scoring,
            "stagnation": stagnation,
            "winloss": winloss,
            "forecast": forecast,
            "crm_actions": crm_actions,
            "metrics": metrics.model_dump(),
            "summary": self._build_summary(
                velocity, pipeline, stagnation, scoring, forecast
            ),
        }

    def _confidence(self, input_data: dict[str, Any], output_data: dict[str, Any]) -> float:
        """Confiance basée sur volume et qualité des données CRM."""
        deals = input_data.get("deals", [])
        contacts = input_data.get("contacts", [])
        score = 0.15  # Base

        open_deals = [d for d in deals if not d.get("is_closed", False)]
        won_deals = [d for d in deals if d.get("is_won", False)]

        # Volume de deals ouverts
        if len(open_deals) >= 20:
            score += 0.2
        elif len(open_deals) >= 10:
            score += 0.15
        elif len(open_deals) >= 3:
            score += 0.1

        # Historique won (pour le benchmark)
        if len(won_deals) >= 20:
            score += 0.25
        elif len(won_deals) >= 10:
            score += 0.2
        elif len(won_deals) >= 3:
            score += 0.1

        # Qualité des deals (montant renseigné)
        deals_with_amount = [d for d in open_deals if d.get("amount")]
        if open_deals:
            amount_coverage = len(deals_with_amount) / len(open_deals)
            score += amount_coverage * 0.15

        # Contacts avec source
        contacts_with_source = [c for c in contacts if c.get("source")]
        if contacts:
            source_coverage = len(contacts_with_source) / len(contacts)
            score += source_coverage * 0.1

        # Activity tracking
        deals_with_activity = [d for d in open_deals if d.get("activity_count_30d")]
        if open_deals:
            activity_coverage = len(deals_with_activity) / len(open_deals)
            score += activity_coverage * 0.15

        return min(1.0, round(score, 2))

    async def _observation_mode(
        self, data: dict[str, Any], error: InsufficientDataError
    ) -> dict[str, Any]:
        """Mode observation : pas assez de données CRM."""
        deals = data.get("deals", [])
        contacts = data.get("contacts", [])

        return {
            "message": error.detail,
            "data_snapshot": {
                "total_deals": len(deals),
                "open_deals": len([d for d in deals if not d.get("is_closed")]),
                "won_deals": len([d for d in deals if d.get("is_won")]),
                "total_contacts": len(contacts),
            },
            "recommendation": (
                "Connectez votre CRM (HubSpot) et assurez-vous d'avoir "
                "au minimum 3 deals ouverts pour activer l'analyse pipeline."
            ),
        }

    # ══════════════════════════════════════════
    # FONCTION 1 — PIPELINE TRUTH
    # Actions 1.1 (nettoyage), 1.2 (forecast), 1.3 (réorg)
    # ══════════════════════════════════════════

    def _pipeline_truth(
        self,
        open_deals: list[dict],
        won_deals: list[dict],
        lost_deals: list[dict],
    ) -> dict[str, Any]:
        """La vérité sur le pipeline — probabilités réelles basées sur l'activité."""
        pipeline_declared = sum(d.get("amount", 0) or 0 for d in open_deals)

        # Benchmark des deals gagnés
        won_stats = self._compute_won_stats(won_deals)

        # Probabilité et classification de chaque deal
        deal_probabilities: list[float] = []
        deal_classifications: list[dict[str, Any]] = []
        misclassified: list[dict[str, Any]] = []  # Action 1.3
        missing_amounts: list[dict[str, Any]] = []  # Action 1.3
        missing_contacts: list[dict[str, Any]] = []  # Action 1.3

        for deal in open_deals:
            prob = self._calculate_real_probability(deal, won_stats)
            classification = self._classify_deal(deal, prob)

            deal_probabilities.append(prob)
            deal_classifications.append({
                "deal_id": deal.get("deal_id"),
                "name": deal.get("name"),
                "amount": deal.get("amount", 0),
                "declared_stage": deal.get("stage"),
                "real_probability": round(prob, 3),
                "classification": classification,
                "days_stagnant": deal.get("days_since_last_activity", 0),
                "has_next_step": deal.get("has_next_step", False),
                "owner": deal.get("owner_name"),
            })

            # Action 1.3 — Détection des deals mal stagés
            suggested_stage = self._suggest_stage(deal)
            if suggested_stage and suggested_stage != deal.get("stage"):
                misclassified.append({
                    "deal_id": deal.get("deal_id"),
                    "name": deal.get("name"),
                    "current_stage": deal.get("stage"),
                    "suggested_stage": suggested_stage,
                    "reason": self._stage_mismatch_reason(deal, suggested_stage),
                })

            # Action 1.3 — Deals sans montant
            if not deal.get("amount"):
                estimated = self._estimate_amount(deal, won_stats)
                missing_amounts.append({
                    "deal_id": deal.get("deal_id"),
                    "name": deal.get("name"),
                    "estimated_amount": round(estimated, 2),
                    "basis": "Moyenne des deals gagnés similaires",
                })

            # Action 1.3 — Deals sans contact
            if not deal.get("contact_ids"):
                missing_contacts.append({
                    "deal_id": deal.get("deal_id"),
                    "name": deal.get("name"),
                })

        # Pipeline réaliste
        pipeline_realistic = sum(
            (d.get("amount", 0) or 0) * p
            for d, p in zip(open_deals, deal_probabilities)
        )
        pipeline_realistic *= self.config.forecast_correction_factor

        gap_ratio = self.safe_divide(
            pipeline_declared, max(pipeline_realistic, 1), default=1.0
        )

        # Cycle moyen
        cycle_times = self._compute_cycle_times(won_deals)
        avg_cycle = self.avg(cycle_times) if cycle_times else _DEFAULT_CYCLE_DAYS

        return {
            "pipeline_declared": round(pipeline_declared, 2),
            "pipeline_realistic": round(pipeline_realistic, 2),
            "gap_ratio": round(gap_ratio, 2),
            "avg_cycle_days": round(avg_cycle, 1),
            "deal_probabilities": deal_probabilities,
            "deal_classifications": deal_classifications,
            "won_stats": won_stats,
            # Action 1.3 outputs
            "misclassified_deals": misclassified,
            "missing_amounts": missing_amounts,
            "missing_contacts": missing_contacts,
        }

    def _compute_won_stats(self, won_deals: list[dict]) -> dict[str, float]:
        """Benchmark des deals gagnés — référence pour le scoring."""
        if not won_deals:
            return {
                "avg_activity": _DEFAULT_AVG_ACTIVITY,
                "avg_cycle_days": _DEFAULT_CYCLE_DAYS,
                "avg_amount": _DEFAULT_AVG_AMOUNT,
            }

        activities = [d.get("activity_count_30d", 0) or 0 for d in won_deals]
        amounts = [d.get("amount", 0) or 0 for d in won_deals]
        cycles = self._compute_cycle_times(won_deals)

        return {
            "avg_activity": self.avg(activities) or _DEFAULT_AVG_ACTIVITY,
            "avg_cycle_days": self.avg(cycles) if cycles else _DEFAULT_CYCLE_DAYS,
            "avg_amount": self.avg(amounts) or _DEFAULT_AVG_AMOUNT,
        }

    def _compute_cycle_times(self, deals: list[dict]) -> list[float]:
        """Temps de cycle des deals fermés."""
        cycles: list[float] = []
        for d in deals:
            create = _parse_date(d.get("create_date"))
            close = _parse_date(d.get("close_date"))
            if create and close:
                delta = (close - create).days
                if 0 < delta < _MAX_CYCLE_DAYS:
                    cycles.append(float(delta))
        return cycles

    def _calculate_real_probability(
        self,
        deal: dict,
        won_stats: dict[str, float],
    ) -> float:
        """Probabilité réelle basée sur 4 facteurs comportementaux.

        Facteur 1 : Stagnation (activité récente)
        Facteur 2 : Engagement vs benchmark gagnants
        Facteur 3 : Next step défini
        Facteur 4 : Taille du deal (gros deals closent moins)
        """
        prob = 0.5  # Neutre

        # ── F1 : Stagnation ──
        days_stagnant = deal.get("days_since_last_activity", 0) or 0
        stag_threshold = self.config.stagnation_threshold_days
        zombie_threshold = self.config.zombie_threshold_days

        if days_stagnant > zombie_threshold:
            prob *= 0.05
        elif days_stagnant > stag_threshold:
            decay = 1 - (
                (days_stagnant - stag_threshold)
                / max(zombie_threshold - stag_threshold, 1)
            )
            prob *= max(0.1, decay * 0.5)
        elif days_stagnant > stag_threshold * 0.5:
            prob *= 0.7

        # ── F2 : Engagement ──
        activity = deal.get("activity_count_30d", 0) or 0
        avg_won_activity = won_stats.get("avg_activity", _DEFAULT_AVG_ACTIVITY)
        if avg_won_activity > 0:
            ratio = min(activity / avg_won_activity, 2.0)
            prob *= (0.3 + ratio * 0.5)

        # ── F3 : Next step ──
        if not deal.get("has_next_step", False):
            prob *= 0.65

        # ── F4 : Montant ──
        amount = deal.get("amount", 0) or 0
        avg_amount = won_stats.get("avg_amount", _DEFAULT_AVG_AMOUNT)
        if avg_amount > 0 and amount > avg_amount * 3:
            prob *= 0.8

        return max(0.02, min(0.95, prob))

    def _classify_deal(self, deal: dict, probability: float) -> str:
        """Classifie un deal en catégorie actionnable."""
        days = deal.get("days_since_last_activity", 0) or 0

        if days > self.config.zombie_threshold_days:
            return _DEAL_CLASS_ZOMBIE
        if probability < 0.1:
            return _DEAL_CLASS_DYING
        if probability < 0.3:
            return _DEAL_CLASS_AT_RISK
        if probability > 0.6:
            return _DEAL_CLASS_HEALTHY
        return _DEAL_CLASS_ACTIVE

    def _suggest_stage(self, deal: dict) -> Optional[str]:
        """Action 1.3 — Suggère le bon stage basé sur les activités réelles."""
        has_demo = deal.get("has_demo", False)
        has_proposal = deal.get("has_proposal_sent", False)
        has_negotiation = deal.get("has_negotiation", False)
        current = (deal.get("stage") or "").lower()

        if has_negotiation and "negotiation" not in current:
            return "Negotiation"
        if has_proposal and "proposal" not in current and "negotiation" not in current:
            return "Proposal Sent"
        if has_demo and "demo" not in current and "proposal" not in current:
            return "Demo Completed"
        return None

    @staticmethod
    def _stage_mismatch_reason(deal: dict, suggested: str) -> str:
        """Explication de la correction de stage."""
        return (
            f"Deal marqué '{deal.get('stage')}' mais les activités "
            f"indiquent qu'il devrait être en '{suggested}'."
        )

    @staticmethod
    def _estimate_amount(deal: dict, won_stats: dict[str, float]) -> float:
        """Estime le montant d'un deal sans montant (action 1.3)."""
        return won_stats.get("avg_amount", _DEFAULT_AVG_AMOUNT)


    # ══════════════════════════════════════════
    # FONCTION 2 — LEAD SCORING
    # Actions 1.4 (scoring/routing), 1.5 (nurture), 1.6 (enrichissement)
    # ══════════════════════════════════════════

    def _score_all_leads(
        self,
        contacts: list[dict],
        won_deals: list[dict],
        lost_deals: list[dict],
    ) -> dict[str, Any]:
        """Score chaque lead et détermine l'action — scoring + routing."""
        weights = self.config.score_weights
        won_profiles = self._build_won_profiles(won_deals, contacts)

        scored_leads: list[dict[str, Any]] = []
        hot_count = 0
        warm_count = 0
        archive_count = 0

        for contact in contacts:
            score_breakdown = self._score_single_lead(contact, weights, won_profiles)
            total_score = score_breakdown["total"]
            action = self._determine_action(total_score)

            lead_data: dict[str, Any] = {
                "contact_id": contact.get("contact_id"),
                "email": contact.get("email"),
                "name": _contact_name(contact),
                "company": contact.get("company"),
                "score": total_score,
                "score_breakdown": score_breakdown,
                "action": action,
                "source": contact.get("source"),
            }

            # Action 1.6 — Enrichissement (champs manquants)
            missing_fields = self._detect_missing_fields(contact)
            if missing_fields:
                lead_data["enrichment_needed"] = missing_fields

            # Routing et événements selon le score
            if action == _ACTION_CALL_NOW:
                hot_count += 1
                lead_data["routing"] = self._route_hot_lead(contact, total_score)
                self._emit_event(
                    event_type=EventType.LEAD_HOT_DETECTED,
                    priority=EventPriority.HIGH,
                    payload={
                        "contact_id": contact.get("contact_id"),
                        "contact_name": lead_data["name"],
                        "company": contact.get("company"),
                        "score": total_score,
                        "source": contact.get("source"),
                        "action": f"Appeler dans les {_HOT_RESPONSE_HOURS}h",
                        "escalation": f"Si pas d'appel dans {_ESCALATION_HOURS}h → escalade manager",
                    },
                )
            elif action == _ACTION_NURTURE:
                warm_count += 1
                lead_data["nurture_sequence"] = self._plan_nurture(contact, total_score)
            elif action == _ACTION_ARCHIVE:
                archive_count += 1

            scored_leads.append(lead_data)

        # Trier par score décroissant
        scored_leads.sort(key=lambda x: x["score"], reverse=True)

        return {
            "scored_leads": scored_leads,
            "total_scored": len(scored_leads),
            "hot_count": hot_count,
            "warm_count": warm_count,
            "archive_count": archive_count,
            "avg_score": round(self.avg([s["score"] for s in scored_leads])),
            "top_5": scored_leads[:5],
            "won_profiles": won_profiles,
        }

    def _build_won_profiles(
        self,
        won_deals: list[dict],
        contacts: list[dict],
    ) -> dict[str, Any]:
        """Profil type du lead qui convertit — base du fit scoring."""
        won_contact_ids: set[str] = set()
        for d in won_deals:
            for cid in d.get("contact_ids", []):
                won_contact_ids.add(cid)

        won_contacts = [c for c in contacts if c.get("contact_id") in won_contact_ids]

        won_sources: dict[str, int] = defaultdict(int)
        won_sectors: dict[str, int] = defaultdict(int)
        for c in won_contacts:
            src = (c.get("source") or "unknown").lower()
            won_sources[src] += 1
            sector = (c.get("industry") or c.get("sector") or "unknown").lower()
            won_sectors[sector] += 1

        return {
            "count": len(won_contacts),
            "sources": dict(won_sources),
            "sectors": dict(won_sectors),
            "top_source": max(won_sources, key=won_sources.get) if won_sources else None,
            "top_sector": max(won_sectors, key=won_sectors.get) if won_sectors else None,
        }

    def _score_single_lead(
        self,
        contact: dict,
        weights: ScoreWeights,
        won_profiles: dict,
    ) -> dict[str, Any]:
        """Score un lead individuel — 4 dimensions pondérées."""

        # ── FIT (0-100) : le lead ressemble-t-il à un client gagné ? ──
        fit = 30  # Base
        if contact.get("company"):
            fit += 15
        if contact.get("job_title"):
            fit += 10
        if contact.get("lifecycle_stage") in ("opportunity", "salesqualifiedlead"):
            fit += 25
        # Match secteur
        sector = (contact.get("industry") or contact.get("sector") or "").lower()
        if sector and sector == (won_profiles.get("top_sector") or ""):
            fit += 20
        fit = min(100, fit)

        # ── ACTIVITY (0-100) : fraîcheur du lead ──
        activity = 20  # Base basse
        created = _parse_date(contact.get("create_date"))
        if created:
            days_since = (datetime.now(tz=timezone.utc) - created).days
            if days_since < 3:
                activity = 95
            elif days_since < 7:
                activity = 80
            elif days_since < 14:
                activity = 60
            elif days_since < 30:
                activity = 40

        # ── TIMING (0-100) : activité récente ──
        timing = 30
        last = _parse_date(contact.get("last_activity_date"))
        if last:
            days_since = (datetime.now(tz=timezone.utc) - last).days
            if days_since < 1:
                timing = 100
            elif days_since < 3:
                timing = 90
            elif days_since < 7:
                timing = 70
            elif days_since < 14:
                timing = 45
            else:
                timing = max(5, 40 - days_since)

        # ── SOURCE (0-100) : le canal d'acquisition ──
        source_score = 35  # Base
        contact_source = (contact.get("source") or "").lower()
        top_source = (won_profiles.get("top_source") or "").lower()
        won_sources = won_profiles.get("sources", {})

        if contact_source and contact_source == top_source:
            source_score = 90
        elif contact_source in [s.lower() for s in won_sources]:
            source_score = 70
        elif contact_source in ("referral", "partner"):
            source_score = 85
        elif contact_source in ("outbound", "cold_email"):
            source_score = 30

        # ── Weighted total ──
        total = (
            fit * weights.fit
            + activity * weights.activity
            + timing * weights.timing
            + source_score * weights.source
        )
        total = max(0, min(100, round(total)))

        return {
            "total": total,
            "fit": fit,
            "activity": activity,
            "timing": timing,
            "source": source_score,
        }

    def _determine_action(self, score: int) -> str:
        """Action recommandée — basée sur les seuils de config."""
        if score >= self.config.hot_lead_threshold:
            return _ACTION_CALL_NOW
        if score >= 50:
            return _ACTION_NURTURE
        if score >= self.config.archive_threshold:
            return _ACTION_WATCH
        return _ACTION_ARCHIVE

    def _route_hot_lead(self, contact: dict, score: int) -> dict[str, Any]:
        """Action 1.4 — Routing du lead HOT vers le meilleur closer."""
        return {
            "priority": "immediate",
            "task": f"Appeler dans les {_HOT_RESPONSE_HOURS}h",
            "escalation": f"Si pas d'appel dans {_ESCALATION_HOURS}h → escalade manager",
            "context": (
                f"Lead score {score}/100. "
                f"Source : {contact.get('source', 'inconnue')}. "
                f"Entreprise : {contact.get('company', 'non renseignée')}."
            ),
        }

    def _plan_nurture(self, contact: dict, score: int) -> dict[str, Any]:
        """Action 1.5 — Séquence nurture pour lead WARM."""
        return {
            "type": "warm_nurture",
            "steps": [
                {"day": 1, "action": "email_bienvenue", "channel": "email"},
                {"day": 3, "action": "contenu_valeur", "channel": "email"},
                {"day": 7, "action": "invitation_call", "channel": "email"},
                {"day": 14, "action": "re_score", "channel": "system"},
            ],
            "upgrade_trigger": "Interaction (ouverture, clic) → re-score → possible HOT",
            "current_score": score,
        }

    @staticmethod
    def _detect_missing_fields(contact: dict) -> list[str]:
        """Action 1.6 — Détecte les champs manquants pour enrichissement."""
        required = ["company", "job_title", "phone", "industry"]
        return [f for f in required if not contact.get(f)]

    # ══════════════════════════════════════════
    # FONCTION 3 — STAGNATION / DEAL ACCELERATION
    # Actions 1.7 (relances), 1.8 (propositions)
    # ══════════════════════════════════════════

    def _detect_stagnation(self, open_deals: list[dict]) -> dict[str, Any]:
        """Détecte les deals stagnants et zombies — actions de relance."""
        stagnant_deals: list[dict] = []
        zombie_deals: list[dict] = []
        relance_actions: list[dict[str, Any]] = []

        for deal in open_deals:
            days = deal.get("days_since_last_activity", 0) or 0
            amount = deal.get("amount", 0) or 0

            if days > self.config.zombie_threshold_days:
                zombie_deals.append(deal)
                self._emit_event(
                    event_type=EventType.DEAL_ZOMBIE_TAGGED,
                    priority=EventPriority.MEDIUM,
                    payload={
                        "deal_id": deal.get("deal_id"),
                        "deal_name": deal.get("name"),
                        "value": amount,
                        "days_stagnant": days,
                        "crm_action": (
                            f"Statut → '{_STAGNANT_REVIEW_STATUS}'. "
                            f"Tâche créée : décider archiver/relancer dans {_TASK_DEADLINE_HOURS}h."
                        ),
                    },
                )

            elif days > self.config.stagnation_threshold_days:
                stagnant_deals.append(deal)
                action = self._build_relance_action(deal, days)
                relance_actions.append(action)

                # Événement pour deals de valeur
                if amount >= self.config.high_value_deal_threshold:
                    self._emit_event(
                        event_type=EventType.DEAL_STAGNANT_DETECTED,
                        priority=EventPriority.HIGH,
                        payload={
                            "deal_id": deal.get("deal_id"),
                            "deal_name": deal.get("name"),
                            "value": amount,
                            "days_stagnant": days,
                            "stage": deal.get("stage"),
                            "recommended_action": action["recommended_action"],
                        },
                    )

        # Value at risk
        stagnant_value = sum(d.get("amount", 0) or 0 for d in stagnant_deals)
        zombie_value = sum(d.get("amount", 0) or 0 for d in zombie_deals)

        return {
            "stagnant_deals": stagnant_deals,
            "zombie_deals": zombie_deals,
            "relance_actions": relance_actions,
            "stagnant_count": len(stagnant_deals),
            "zombie_count": len(zombie_deals),
            "stagnant_value": round(stagnant_value, 2),
            "zombie_value": round(zombie_value, 2),
            "total_at_risk": round(stagnant_value + zombie_value, 2),
        }

    def _build_relance_action(self, deal: dict, days_stagnant: int) -> dict[str, Any]:
        """Action 1.7 — Construit l'action de relance pour un deal stagnant."""
        stage = (deal.get("stage") or "").lower()

        # Choisir le type de relance selon le stage
        if "proposal" in stage or "negotiation" in stage:
            relance_type = "follow_up_proposal"
            recommended = "Relancer sur la proposition — demander feedback"
        elif "demo" in stage:
            relance_type = "follow_up_demo"
            recommended = "Proposer un second call pour répondre aux questions"
        else:
            relance_type = "check_in"
            recommended = "Email de check-in — vérifier l'intérêt"

        return {
            "deal_id": deal.get("deal_id"),
            "deal_name": deal.get("name"),
            "amount": deal.get("amount", 0),
            "days_stagnant": days_stagnant,
            "stage": deal.get("stage"),
            "owner": deal.get("owner_name"),
            "relance_type": relance_type,
            "recommended_action": recommended,
            "draft_context": {
                "deal_name": deal.get("name"),
                "contact_name": deal.get("primary_contact_name"),
                "last_activity": deal.get("last_activity_date"),
                "stage": deal.get("stage"),
            },
            "channels": ["email", "linkedin"] if days_stagnant > 14 else ["email"],
        }

    # ══════════════════════════════════════════
    # FONCTION 4 — WIN/LOSS ANALYSIS
    # Action 1.9
    # ══════════════════════════════════════════

    def _analyze_winloss(
        self,
        won_deals: list[dict],
        lost_deals: list[dict],
    ) -> dict[str, Any]:
        """Action 1.9 — Analyse automatique win/loss pour apprentissage continu."""
        won_analysis = self._analyze_deal_group(won_deals, "won")
        lost_analysis = self._analyze_deal_group(lost_deals, "lost")

        # Patterns de perte
        loss_patterns: list[dict[str, Any]] = []
        if lost_deals:
            loss_patterns = self._detect_loss_patterns(lost_deals, won_analysis)

        # Insights croisés
        insights: list[str] = []
        if won_analysis and lost_analysis:
            won_cycle = won_analysis.get("avg_cycle_days", 0)
            lost_cycle = lost_analysis.get("avg_cycle_days", 0)
            if lost_cycle > won_cycle * 1.5 and lost_cycle > 0:
                insights.append(
                    f"Les deals perdus durent {lost_cycle:.0f}j en moyenne "
                    f"vs {won_cycle:.0f}j pour les gagnés. "
                    f"Raccourcir le cycle augmenterait le win rate."
                )

            won_activities = won_analysis.get("avg_activities", 0)
            lost_activities = lost_analysis.get("avg_activities", 0)
            if won_activities > lost_activities * 1.5 and lost_activities > 0:
                insights.append(
                    f"Les deals gagnés ont {won_activities:.0f} activités "
                    f"vs {lost_activities:.0f} pour les perdus. "
                    f"Plus d'engagement = plus de closing."
                )

        return {
            "won": won_analysis,
            "lost": lost_analysis,
            "loss_patterns": loss_patterns,
            "insights": insights,
            "win_rate": round(
                self.safe_divide(len(won_deals), len(won_deals) + len(lost_deals)), 3
            ),
        }

    def _analyze_deal_group(
        self, deals: list[dict], group: str
    ) -> dict[str, Any]:
        """Analyse un groupe de deals (won ou lost)."""
        if not deals:
            return {}

        amounts = [d.get("amount", 0) or 0 for d in deals]
        cycles = self._compute_cycle_times(deals)
        activities = [d.get("total_activities", 0) or 0 for d in deals]

        # Stage où les deals passent le plus de temps
        stage_times: dict[str, list[float]] = defaultdict(list)
        for d in deals:
            for stage_info in d.get("stage_history", []):
                stage_name = stage_info.get("stage", "unknown")
                duration = stage_info.get("duration_days", 0)
                stage_times[stage_name].append(duration)

        stage_avg = {
            stage: round(self.avg(times), 1)
            for stage, times in stage_times.items()
        }

        return {
            "count": len(deals),
            "total_value": round(sum(amounts), 2),
            "avg_amount": round(self.avg(amounts), 2),
            "avg_cycle_days": round(self.avg(cycles), 1) if cycles else None,
            "avg_activities": round(self.avg(activities), 1),
            "stage_durations": stage_avg,
        }

    def _detect_loss_patterns(
        self,
        lost_deals: list[dict],
        won_analysis: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Détecte les patterns récurrents dans les deals perdus."""
        patterns: list[dict[str, Any]] = []

        # Pattern : deals perdus au stage Proposal
        proposal_losses = [
            d for d in lost_deals
            if "proposal" in (d.get("last_stage") or d.get("stage") or "").lower()
        ]
        if len(proposal_losses) >= 2:
            patterns.append({
                "pattern": "proposal_dropout",
                "count": len(proposal_losses),
                "insight": (
                    f"{len(proposal_losses)} deals perdus après l'envoi de proposition. "
                    f"Hypothèse : pricing ou manque d'urgence."
                ),
                "recommendation": "Raccourcir le stage Proposal à 5j max + ajouter un incentive temporel.",
            })

        # Pattern : pas de réponse après relance
        no_response = [
            d for d in lost_deals
            if (d.get("days_since_last_activity", 0) or 0) > 30
        ]
        if len(no_response) >= 2:
            patterns.append({
                "pattern": "ghost",
                "count": len(no_response),
                "insight": (
                    f"{len(no_response)} deals perdus par ghosting (> 30j sans réponse)."
                ),
                "recommendation": "Automatiser les relances multi-canal (email + LinkedIn).",
            })

        return patterns

    # ══════════════════════════════════════════
    # FORECAST
    # Action 1.2
    # ══════════════════════════════════════════

    def _generate_forecast(
        self,
        open_deals: list[dict],
        probabilities: list[float],
        avg_cycle: float,
    ) -> dict[str, Any]:
        """Forecast 30/60/90 basé sur les probabilités réelles."""
        now = datetime.now(tz=timezone.utc)
        forecast_30 = 0.0
        forecast_60 = 0.0
        forecast_90 = 0.0

        for deal, prob in zip(open_deals, probabilities):
            amount = (deal.get("amount", 0) or 0) * prob
            close_date = _parse_date(deal.get("expected_close_date") or deal.get("close_date"))

            if close_date:
                days_to_close = (close_date - now).days
            else:
                # Estimer basé sur le cycle moyen et le temps déjà passé
                create = _parse_date(deal.get("create_date"))
                if create:
                    days_in = (now - create).days
                    days_to_close = max(1, avg_cycle - days_in)
                else:
                    days_to_close = avg_cycle

            if days_to_close <= 30:
                forecast_30 += amount
                forecast_60 += amount
                forecast_90 += amount
            elif days_to_close <= 60:
                forecast_60 += amount
                forecast_90 += amount
            elif days_to_close <= 90:
                forecast_90 += amount

        # Confidence basée sur la couverture des données
        deals_with_close_date = sum(
            1 for d in open_deals
            if d.get("expected_close_date") or d.get("close_date")
        )
        date_coverage = self.safe_divide(deals_with_close_date, len(open_deals))
        confidence = 0.3 + date_coverage * 0.4 + min(0.3, len(open_deals) * 0.02)

        return {
            "forecast_30d": round(forecast_30, 2),
            "forecast_60d": round(forecast_60, 2),
            "forecast_90d": round(forecast_90, 2),
            "confidence": round(min(1.0, confidence), 2),
            "deals_in_forecast": len(open_deals),
        }

    # ══════════════════════════════════════════
    # CRM ACTIONS — Actions à exécuter dans le CRM
    # Actions 1.1, 1.3
    # ══════════════════════════════════════════

    def _generate_crm_actions(
        self,
        pipeline: dict[str, Any],
        stagnation: dict[str, Any],
        scoring: dict[str, Any],
        open_deals: list[dict],
        won_deals: list[dict],
    ) -> dict[str, Any]:
        """Compile toutes les actions CRM à exécuter."""
        actions: list[dict[str, Any]] = []

        # Action 1.1 — Nettoyage : zombies → statut review
        for deal in stagnation["zombie_deals"]:
            actions.append({
                "action_type": "update_deal_status",
                "deal_id": deal.get("deal_id"),
                "deal_name": deal.get("name"),
                "changes": {
                    "status": _STAGNANT_REVIEW_STATUS,
                    "axio_note": (
                        f"⚡ Kuria : ce deal n'a aucune activité depuis "
                        f"{deal.get('days_since_last_activity', 0)} jours. "
                        f"Action recommandée : archiver ou relancer."
                    ),
                },
                "create_task": {
                    "title": f"Décider : archiver ou relancer [{deal.get('name')}]",
                    "deadline_hours": _TASK_DEADLINE_HOURS,
                    "assigned_to": deal.get("owner_id"),
                },
                "autonomous": True,  # [A]
                "source_action": "1.1",
            })

        # Action 1.2 — Écriture probabilité réelle dans le CRM
        for classification in pipeline.get("deal_classifications", []):
            actions.append({
                "action_type": "update_deal_field",
                "deal_id": classification["deal_id"],
                "changes": {
                    "axio_real_probability": classification["real_probability"],
                    "axio_forecast_value": round(
                        (classification.get("amount", 0) or 0)
                        * classification["real_probability"], 2
                    ),
                },
                "autonomous": True,  # [A]
                "source_action": "1.2",
            })

        # Action 1.3 — Corrections de stage
        for mis in pipeline.get("misclassified_deals", []):
            actions.append({
                "action_type": "update_deal_stage",
                "deal_id": mis["deal_id"],
                "deal_name": mis["name"],
                "changes": {
                    "stage": mis["suggested_stage"],
                    "axio_note": mis["reason"],
                },
                "autonomous": True,  # [A]
                "source_action": "1.3",
            })

        # Action 1.3 — Estimation de montant
        for missing in pipeline.get("missing_amounts", []):
            actions.append({
                "action_type": "update_deal_field",
                "deal_id": missing["deal_id"],
                "deal_name": missing["name"],
                "changes": {
                    "axio_estimated_amount": missing["estimated_amount"],
                },
                "autonomous": True,  # [A]
                "source_action": "1.3",
            })

        # Action 1.4 — Lead scoring CRM write
        for lead in scoring.get("scored_leads", []):
            actions.append({
                "action_type": "update_contact_field",
                "contact_id": lead["contact_id"],
                "changes": {
                    "axio_lead_score": lead["score"],
                    "axio_lead_action": lead["action"],
                },
                "autonomous": True,  # [A]
                "source_action": "1.4",
            })

        return {
            "total_actions": len(actions),
            "autonomous": [a for a in actions if a.get("autonomous")],
            "supervised": [a for a in actions if not a.get("autonomous")],
            "by_type": _group_actions(actions),
            "actions": actions,
        }

    # ══════════════════════════════════════════
    # ÉVÉNEMENTS
    # ══════════════════════════════════════════

    def _emit_pipeline_truth(
        self,
        pipeline: dict[str, Any],
        stagnation: dict[str, Any],
        velocity: float,
    ) -> None:
        """Émet l'événement pipeline truth."""
        self._emit_event(
            event_type=EventType.PIPELINE_TRUTH_GENERATED,
            priority=EventPriority.MEDIUM,
            payload={
                "declared": pipeline["pipeline_declared"],
                "realistic": pipeline["pipeline_realistic"],
                "gap_ratio": pipeline["gap_ratio"],
                "deals_active": len(pipeline.get("deal_classifications", [])),
                "deals_stagnant": stagnation["stagnant_count"],
                "deals_zombie": stagnation["zombie_count"],
                "velocity": round(velocity, 2),
                "value_at_risk": stagnation["total_at_risk"],
            },
        )

    def _emit_forecast_update(self, forecast: dict[str, Any]) -> None:
        """Émet l'événement forecast pour consommation par Cash agent."""
        self._emit_event(
            event_type=EventType.FORECAST_UPDATED,
            priority=EventPriority.MEDIUM,
            payload={
                "forecast_30d": forecast["forecast_30d"],
                "forecast_60d": forecast["forecast_60d"],
                "forecast_90d": forecast["forecast_90d"],
                "confidence": forecast["confidence"],
            },
        )

    def _check_forecast_change(
        self,
        current_30d: float,
        previous_30d: Optional[float],
    ) -> None:
        """Action 1.2 — Alerte si le forecast change de > 15%."""
        if previous_30d is None or previous_30d == 0:
            return

        change_pct = abs(current_30d - previous_30d) / previous_30d
        if change_pct > _FORECAST_ALERT_THRESHOLD_PCT:
            direction = "augmenté" if current_30d > previous_30d else "diminué"
            self._emit_event(
                event_type=EventType.FORECAST_UPDATED,
                priority=EventPriority.HIGH,
                payload={
                    "alert": True,
                    "message": (
                        f"Le forecast 30j a {direction} de {change_pct:.0%} : "
                        f"{self.format_currency(previous_30d)} → {self.format_currency(current_30d)}"
                    ),
                    "previous": round(previous_30d, 2),
                    "current": round(current_30d, 2),
                    "change_pct": round(change_pct * 100, 1),
                },
            )

    # ══════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════

    def _build_summary(
        self,
        velocity: float,
        pipeline: dict[str, Any],
        stagnation: dict[str, Any],
        scoring: dict[str, Any],
        forecast: dict[str, Any],
    ) -> str:
        """Résumé texte pour dashboard et rapport hebdo (action 1.10)."""
        parts: list[str] = []

        # KPI Nord
        parts.append(f"Revenue Velocity : {self.format_currency(velocity)}/jour")

        # Pipeline
        parts.append(
            f"Pipeline : {self.format_currency(pipeline['pipeline_declared'])} déclaré → "
            f"{self.format_currency(pipeline['pipeline_realistic'])} réaliste "
            f"(ratio {pipeline['gap_ratio']:.1f}x)"
        )

        # Stagnation
        if stagnation["zombie_count"] > 0:
            parts.append(
                f"🧟 {stagnation['zombie_count']} deals zombies "
                f"({self.format_currency(stagnation['zombie_value'])})"
            )
        if stagnation["stagnant_count"] > 0:
            parts.append(
                f"⚠️ {stagnation['stagnant_count']} deals stagnants "
                f"({self.format_currency(stagnation['stagnant_value'])})"
            )

        # Leads
        parts.append(
            f"Leads : {scoring['hot_count']} HOT, "
            f"{scoring['warm_count']} WARM, "
            f"{scoring['archive_count']} archivés"
        )

        # Forecast
        parts.append(
            f"Forecast 30j : {self.format_currency(forecast['forecast_30d'])} "
            f"(confiance {forecast['confidence']:.0%})"
        )

        return " | ".join(parts)


# ══════════════════════════════════════════
# HELPERS MODULE-LEVEL
# ══════════════════════════════════════════


def _parse_date(value: Any) -> Optional[datetime]:
    """Parse une date de manière sûre."""
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    if isinstance(value, str):
        try:
            from connectors.utils import normalize_date
            return normalize_date(value)
        except Exception:
            return None
    return None


def _contact_name(contact: dict) -> str:
    """Nom complet d'un contact."""
    first = contact.get("first_name", "")
    last = contact.get("last_name", "")
    return f"{first} {last}".strip() or contact.get("email", "Inconnu")


def _group_actions(actions: list[dict]) -> dict[str, int]:
    """Groupe les actions CRM par type."""
    groups: dict[str, int] = defaultdict(int)
    for a in actions:
        groups[a.get("action_type", "unknown")] += 1
    return dict(groups)
