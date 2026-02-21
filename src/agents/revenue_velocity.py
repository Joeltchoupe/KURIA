"""
Revenue Velocity Agent — Pipeline truth, lead scoring, stagnation alerts.

UN KPI : Revenue Velocity = Pipeline réaliste ÷ Cycle moyen en jours = €/jour

4 fonctions :
1. Pipeline Truth   → La vérité sur le pipeline
2. Lead Scoring     → Qui appeler en premier
3. Stagnation Alert → Quand un deal meurt
4. Weekly Report    → Tout est clair le lundi

C'est l'agent le plus VISIBLE pour le client.
Ses résultats sont immédiats (dès J1, le pipeline est nettoyé).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from models.agent_config import RevenueVelocityConfig, ScoreWeights
from models.events import EventType, EventPriority
from models.metrics import MetricTrend, RevenueVelocityMetrics

from agents.base import BaseAgent, InsufficientDataError


class RevenueVelocityAgent(BaseAgent):
    """Agent Revenue Velocity — le gardien du pipeline."""

    AGENT_NAME = "revenue_velocity"

    def __init__(self, company_id: str, config: RevenueVelocityConfig, **kwargs):
        super().__init__(company_id=company_id, config=config, **kwargs)
        self.rv_config: RevenueVelocityConfig = config

    # ══════════════════════════════════════════
    # POINT D'ENTRÉE
    # ══════════════════════════════════════════

    async def analyze(self, data: dict[str, Any]) -> dict[str, Any]:
        """Analyse complète : pipeline truth + scoring + stagnation.

        Args:
            data: {"deals": [...], "contacts": [...], "owners": {...}, "stages": {...}}
        """
        deals = data.get("deals", [])
        contacts = data.get("contacts", [])

        open_deals = [d for d in deals if not d.get("is_closed", False)]
        closed_deals = [d for d in deals if d.get("is_closed", False)]
        won_deals = [d for d in closed_deals if d.get("is_won", False)]
        lost_deals = [d for d in closed_deals if not d.get("is_won", False)]

        # 1. Pipeline Truth
        pipeline_result = await self._pipeline_truth(
            open_deals, won_deals, lost_deals
        )

        # 2. Lead Scoring
        scoring_result = await self._score_leads(
            contacts, won_deals, lost_deals
        )

        # 3. Stagnation Alerts
        stagnation_result = self._detect_stagnation(open_deals)

        # 4. Revenue Velocity KPI
        pipeline_realistic = pipeline_result["pipeline_realistic"]
        avg_cycle = pipeline_result["avg_cycle_days"]
        velocity = self.safe_divide(pipeline_realistic, max(avg_cycle, 1))

        # 5. Forecast
        forecast = self._generate_forecast(
            open_deals, pipeline_result["deal_probabilities"], avg_cycle
        )

        # 6. Métriques
        previous = await self.get_previous_metrics()
        prev_velocity = previous.get("revenue_velocity_per_day") if previous else None

        metrics = RevenueVelocityMetrics(
            company_id=self.company_id,
            revenue_velocity_per_day=round(velocity, 2),
            pipeline_declared=pipeline_result["pipeline_declared"],
            pipeline_realistic=round(pipeline_realistic, 2),
            pipeline_gap_ratio=round(pipeline_result["gap_ratio"], 2),
            deals_active=len(open_deals),
            deals_stagnant=len(stagnation_result["stagnant_deals"]),
            deals_zombie=len(stagnation_result["zombie_deals"]),
            forecast_30d=round(forecast["forecast_30d"], 2),
            forecast_60d=round(forecast["forecast_60d"], 2),
            forecast_90d=round(forecast["forecast_90d"], 2),
            forecast_confidence=round(forecast["confidence"], 2),
            leads_scored=scoring_result["total_scored"],
            leads_hot=scoring_result["hot_count"],
            leads_archived=scoring_result["archive_count"],
            deals_won_this_period=len(won_deals),
            deals_lost_this_period=len(lost_deals),
            revenue_won_this_period=sum(d.get("amount", 0) or 0 for d in won_deals),
        )

        await self.save_metrics(metrics)

        # 7. Publier pipeline truth event
        self.publish_event(
            event_type=EventType.PIPELINE_TRUTH_GENERATED,
            payload={
                "declared": pipeline_result["pipeline_declared"],
                "realistic": round(pipeline_realistic, 2),
                "gap_ratio": round(pipeline_result["gap_ratio"], 2),
                "deals_active": len(open_deals),
                "deals_stagnant": len(stagnation_result["stagnant_deals"]),
                "velocity": round(velocity, 2),
            },
        )

        # 8. Publier forecast event
        self.publish_event(
            event_type=EventType.FORECAST_UPDATED,
            payload={
                "forecast_30d": round(forecast["forecast_30d"], 2),
                "forecast_60d": round(forecast["forecast_60d"], 2),
                "forecast_90d": round(forecast["forecast_90d"], 2),
                "confidence": round(forecast["confidence"], 2),
            },
        )

        return {
            "revenue_velocity": round(velocity, 2),
            "velocity_trend": MetricTrend.calculate(
                velocity, prev_velocity
            ).model_dump() if prev_velocity else None,
            "pipeline": pipeline_result,
            "scoring": scoring_result,
            "stagnation": stagnation_result,
            "forecast": forecast,
            "metrics": metrics.model_dump(),
        }

    # ══════════════════════════════════════════
    # FONCTION 1 — PIPELINE TRUTH
    # ══════════════════════════════════════════

    async def _pipeline_truth(
        self,
        open_deals: list[dict],
        won_deals: list[dict],
        lost_deals: list[dict],
    ) -> dict[str, Any]:
        """La vérité sur le pipeline.

        Pour chaque deal, calcule la probabilité RÉELLE
        basée sur l'activité, pas sur le stage déclaré.
        """
        pipeline_declared = sum(d.get("amount", 0) or 0 for d in open_deals)

        # Statistiques des deals gagnés (pour le benchmark)
        won_stats = self._compute_won_stats(won_deals)

        # Probabilité réelle de chaque deal
        deal_probabilities = []
        deal_classifications = []

        for deal in open_deals:
            prob = self._calculate_real_probability(deal, won_stats)
            classification = self._classify_deal(deal, prob)

            deal_probabilities.append(prob)
            deal_classifications.append({
                "deal_id": deal.get("deal_id"),
                "name": deal.get("name"),
                "amount": deal.get("amount", 0),
                "probability": round(prob, 3),
                "classification": classification,
                "stage": deal.get("stage"),
                "days_stagnant": deal.get("days_since_last_activity", 0),
                "has_next_step": deal.get("has_next_step", False),
            })

        # Pipeline réaliste
        pipeline_realistic = sum(
            d.get("amount", 0) * p
            for d, p in zip(open_deals, deal_probabilities)
        )

        # Appliquer le facteur de correction de l'Adaptateur
        pipeline_realistic *= self.rv_config.forecast_correction_factor

        gap_ratio = self.safe_divide(pipeline_declared, pipeline_realistic, default=1.0)

        # Cycle moyen
        cycle_times = self._compute_cycle_times(won_deals)
        avg_cycle = self.avg(cycle_times) if cycle_times else 30.0

        return {
            "pipeline_declared": round(pipeline_declared, 2),
            "pipeline_realistic": round(pipeline_realistic, 2),
            "gap_ratio": round(gap_ratio, 2),
            "avg_cycle_days": round(avg_cycle, 1),
            "deal_probabilities": deal_probabilities,
            "deal_classifications": deal_classifications,
            "won_stats": won_stats,
        }

    def _compute_won_stats(self, won_deals: list[dict]) -> dict[str, float]:
        """Statistiques des deals gagnés — benchmark pour le scoring."""
        if not won_deals:
            return {
                "avg_activity": 5.0,
                "avg_cycle_days": 30.0,
                "avg_amount": 10000.0,
            }

        activities = [d.get("activity_count_30d", 0) or 0 for d in won_deals]
        amounts = [d.get("amount", 0) or 0 for d in won_deals]
        cycles = self._compute_cycle_times(won_deals)

        return {
            "avg_activity": self.avg(activities) or 5.0,
            "avg_cycle_days": self.avg(cycles) if cycles else 30.0,
            "avg_amount": self.avg(amounts) or 10000.0,
        }

    def _compute_cycle_times(self, deals: list[dict]) -> list[float]:
        """Calcule les temps de cycle des deals fermés."""
        from connectors.utils import normalize_date

        cycles = []
        for d in deals:
            create = normalize_date(d.get("create_date"))
            close = normalize_date(d.get("close_date"))
            if create and close:
                delta = (close - create).days
                if 0 < delta < 365:
                    cycles.append(float(delta))
        return cycles

    def _calculate_real_probability(
        self,
        deal: dict,
        won_stats: dict[str, float],
    ) -> float:
        """Calcule la probabilité réelle d'un deal.

        Combine 4 facteurs :
        - Activité récente vs benchmark des deals gagnés
        - Temps dans le stage actuel vs moyenne
        - Présence d'un next step
        - Trend d'engagement (croissant/décroissant)
        """
        prob = 0.5  # Point de départ neutre

        # ── Facteur 1 : Activité / Stagnation ──
        days_stagnant = deal.get("days_since_last_activity", 0) or 0
        stagnation_threshold = self.rv_config.stagnation_threshold_days
        zombie_threshold = self.rv_config.zombie_threshold_days

        if days_stagnant > zombie_threshold:
            prob *= 0.05  # Quasi-mort
        elif days_stagnant > stagnation_threshold:
            # Décroissance progressive entre stagnation et zombie
            decay = 1 - (
                (days_stagnant - stagnation_threshold)
                / (zombie_threshold - stagnation_threshold)
            )
            prob *= max(0.1, decay * 0.5)
        elif days_stagnant > stagnation_threshold * 0.5:
            prob *= 0.7

        # ── Facteur 2 : Engagement vs benchmark ──
        activity = deal.get("activity_count_30d", 0) or 0
        avg_won_activity = won_stats.get("avg_activity", 5)

        if avg_won_activity > 0:
            engagement_ratio = min(activity / avg_won_activity, 2.0)
            # 0 activité = 0.3x, activité moyenne = 1.0x, 2x activité = 1.3x
            prob *= (0.3 + engagement_ratio * 0.5)

        # ── Facteur 3 : Next step ──
        if not deal.get("has_next_step", False):
            prob *= 0.65

        # ── Facteur 4 : Montant (les gros deals closent moins) ──
        amount = deal.get("amount", 0) or 0
        avg_amount = won_stats.get("avg_amount", 10000)
        if avg_amount > 0 and amount > avg_amount * 3:
            prob *= 0.8  # Pénalité légère sur les gros deals

        # Borner
        return max(0.02, min(0.95, prob))

    def _classify_deal(self, deal: dict, probability: float) -> str:
        """Classifie un deal en catégorie actionnable."""
        days_stagnant = deal.get("days_since_last_activity", 0) or 0

        if days_stagnant > self.rv_config.zombie_threshold_days:
            return "zombie"
        if probability < 0.1:
            return "dying"
        if probability < 0.3:
            return "at_risk"
        if probability > 0.6:
            return "healthy"
        return "active"


    # ══════════════════════════════════════════
    # FONCTION 2 — LEAD SCORING
    # ══════════════════════════════════════════

    async def _score_leads(
        self,
        contacts: list[dict],
        won_deals: list[dict],
        lost_deals: list[dict],
    ) -> dict[str, Any]:
        """Score chaque contact/lead selon 4 dimensions."""
        weights = self.rv_config.score_weights
        scored_leads = []
        hot_count = 0
        archive_count = 0

        # Profils des clients gagnés (pour le fit scoring)
        won_profiles = self._build_won_profiles(won_deals, contacts)

        for contact in contacts:
            score = self._score_single_lead(contact, weights, won_profiles)
            action = self._determine_action(score)

            scored_leads.append({
                "contact_id": contact.get("contact_id"),
                "email": contact.get("email"),
                "name": f"{contact.get('first_name', '')} {contact.get('last_name', '')}".strip(),
                "score": score,
                "action": action,
                "source": contact.get("source"),
            })

            if action == "call_now":
                hot_count += 1
                # Publier événement lead chaud
                self.publish_event(
                    event_type=EventType.LEAD_HOT_DETECTED,
                    priority=EventPriority.HIGH,
                    payload={
                        "contact_id": contact.get("contact_id"),
                        "contact_name": scored_leads[-1]["name"],
                        "score": score,
                        "source": contact.get("source"),
                    },
                )
            elif action == "archive":
                archive_count += 1

            # Publier l'événement de scoring
            self.publish_event(
                event_type=EventType.LEAD_SCORED,
                payload={
                    "contact_id": contact.get("contact_id"),
                    "score": score,
                    "action": action,
                    "source_channel": contact.get("source"),
                },
            )

        # Trier par score décroissant
        scored_leads.sort(key=lambda x: x["score"], reverse=True)

        return {
            "scored_leads": scored_leads,
            "total_scored": len(scored_leads),
            "hot_count": hot_count,
            "archive_count": archive_count,
            "avg_score": self.avg([s["score"] for s in scored_leads]),
            "top_5": scored_leads[:5],
        }

    def _build_won_profiles(
        self,
        won_deals: list[dict],
        contacts: list[dict],
    ) -> dict[str, Any]:
        """Construit le profil type d'un lead qui convertit."""
        won_contact_ids = set()
        for d in won_deals:
            for cid in d.get("contact_ids", []):
                won_contact_ids.add(cid)

        won_contacts = [
            c for c in contacts if c.get("contact_id") in won_contact_ids
        ]

        # Sources des gagnants
        won_sources: dict[str, int] = {}
        for c in won_contacts:
            src = c.get("source", "unknown") or "unknown"
            won_sources[src] = won_sources.get(src, 0) + 1

        return {
            "count": len(won_contacts),
            "sources": won_sources,
            "top_source": max(won_sources, key=won_sources.get) if won_sources else None,
        }

    def _score_single_lead(
        self,
        contact: dict,
        weights: ScoreWeights,
        won_profiles: dict,
    ) -> int:
        """Score un lead individuel de 0 à 100."""
        # ── FIT (0-100) ──
        fit_score = 50  # Base neutre
        if contact.get("company"):
            fit_score += 15
        if contact.get("job_title"):
            fit_score += 15
        if contact.get("lifecycle_stage") in ("opportunity", "salesqualifiedlead"):
            fit_score += 20

        # ── ACTIVITY (0-100) ──
        activity_score = 30  # Base basse (pas d'activity tracking en V1 simple)
        create_date = contact.get("create_date")
        if create_date:
            from connectors.utils import normalize_date
            created = normalize_date(create_date)
            if created:
                days_since = (datetime.now(tz=timezone.utc) - created).days
                if days_since < 7:
                    activity_score = 90
                elif days_since < 14:
                    activity_score = 70
                elif days_since < 30:
                    activity_score = 50

        # ── TIMING (0-100) ──
        timing_score = 50
        last_activity = contact.get("last_activity_date")
        if last_activity:
            from connectors.utils import normalize_date
            last = normalize_date(last_activity)
            if last:
                days_since = (datetime.now(tz=timezone.utc) - last).days
                if days_since < 3:
                    timing_score = 95
                elif days_since < 7:
                    timing_score = 75
                elif days_since < 14:
                    timing_score = 50
                else:
                    timing_score = max(10, 50 - days_since)

        # ── SOURCE (0-100) ──
        source_score = 40  # Base
        contact_source = (contact.get("source") or "").lower()
        won_sources = won_profiles.get("sources", {})
        top_source = won_profiles.get("top_source", "")

        if contact_source and contact_source == (top_source or "").lower():
            source_score = 90
        elif contact_source and contact_source in [
            s.lower() for s in won_sources
        ]:
            source_score = 70
        elif contact_source in ("referral", "partner"):
            source_score = 80

        # ── Weighted total ──
        total = (
            fit_score * weights.fit
            + activity_score * weights.activity
            + timing_score * weights.timing
            + source_score * weights.source
        )

        return max(0, min(100, round(total)))

    def _determine_action(self, score: int) -> str:
        """Détermine l'action recommandée pour un lead."""
        if score >= self.rv_config.hot_lead_threshold:
            return "call_now"
        if score >= 50:
            return "nurture"
        if score >= self.rv_config.archive_threshold:
            return "watch"
        return "archive"

    # ══════════════════════════════════════════
    # FONCTION 3 — STAGNATION ALERT
    # ══════════════════════════════════════════

    def _detect_stagnation(
        self,
        open_deals: list[dict],
    ) -> dict[str, Any]:
        """Détecte les deals en stagnation et les zombies."""
        stagnant_deals = []
        zombie_deals = []
        alerts = []

        for deal in open_deals:
            days = deal.get("days_since_last_activity", 0) or 0
            amount = deal.get("amount", 0) or 0

            if days > self.rv_config.zombie_threshold_days:
                zombie_deals.append(deal)

                self.publish_event(
                    event_type=EventType.DEAL_ZOMBIE_TAGGED,
                    payload={
                        "deal_id": deal.get("deal_id"),
                        "deal_name": deal.get("name"),
                        "value": amount,
                        "last_activity_date": str(deal.get("last_activity_date")),
                    },
                )

            elif days > self.rv_config.stagnation_threshold_days:
                stagnant_deals.append(deal)

                action = self._recommend_stagnation_action(deal, days)

                alert = {
                    "deal_id": deal.get("deal_id"),
                    "deal_name": deal.get("name"),
                    "amount": amount,
                    "days_stagnant": days,
                    "stage": deal.get("stage"),
                    "owner": deal.get("owner_name"),
                    "recommended_action": action,
                }
                alerts.append(alert)

                # Publier événement si deal important
                if amount >= self.rv_config.high_value_deal_threshold:
                    self.publish_event(
                        event_type=EventType.DEAL_STAGNANT_DETECTED,
                        priority=EventPriority.HIGH,
                        payload={
                            "deal_id": deal.get("deal_id"),
                            "deal_name": deal.get("name"),
                            "days_stagnant": days,
                            "value": amount,
                            "owner": deal.get("owner_name"),
                            "stage": deal.get("stage"),
                        },
                    )

        return {
            "stagnant_deals": stagnant_deals,
            "zombie_deals": zombie_deals,
            "alerts": alerts,
            "total_stagnant_value": sum(
                d.get("amount", 0) or 0 for d in stagnant_deals
            ),
            "total_zombie_value": sum(
                d.get("amount", 0) or 0 for d in zombie_deals
            ),
        }

    def _recommend_stagnation_action(self, deal: dict, days: int) -> str:
        """Recommande une action pour un deal stagnant."""
        threshold = self.rv_config.stagnation_threshold_days
        zombie = self.rv_config.zombie_threshold_days

        ratio = days / threshold

        if ratio > 2.5:
            return "disqualify"
        if ratio > 1.5:
            return "escalate"
        if deal.get("has_next_step"):
            return "follow_up"
        return "requalify"





    # ══════════════════════════════════════════
    # FONCTION 4 — FORECAST
    # ══════════════════════════════════════════

    def _generate_forecast(
        self,
        open_deals: list[dict],
        probabilities: list[float],
        avg_cycle: float,
    ) -> dict[str, Any]:
        """Génère le forecast 30/60/90 jours."""
        now = datetime.now(tz=timezone.utc)
        forecast_30 = 0.0
        forecast_60 = 0.0
        forecast_90 = 0.0

        for deal, prob in zip(open_deals, probabilities):
            amount = (deal.get("amount", 0) or 0) * prob
            amount *= self.rv_config.forecast_correction_factor

            days_in_stage = deal.get("days_in_current_stage", 0) or 0
            remaining_cycle = max(1, avg_cycle - days_in_stage)

            if remaining_cycle <= 30:
                forecast_30 += amount
                forecast_60 += amount
                forecast_90 += amount
            elif remaining_cycle <= 60:
                forecast_60 += amount
                forecast_90 += amount
            elif remaining_cycle <= 90:
                forecast_90 += amount

        # Confiance
        confidence = self._forecast_confidence(
            open_deals, probabilities, avg_cycle
        )

        return {
            "forecast_30d": forecast_30,
            "forecast_60d": forecast_60,
            "forecast_90d": forecast_90,
            "confidence": confidence,
        }

    def _forecast_confidence(
        self,
        deals: list[dict],
        probabilities: list[float],
        avg_cycle: float,
    ) -> float:
        """Calcule la confiance dans le forecast."""
        if not deals:
            return 0.0

        confidence = 0.7  # Base

        # Pénalité si beaucoup de deals ont prob < 0.2
        low_prob_pct = sum(1 for p in probabilities if p < 0.2) / len(probabilities)
        confidence -= low_prob_pct * 0.3

        # Pénalité si peu de données historiques
        if len(deals) < 10:
            confidence -= 0.2
        elif len(deals) < 20:
            confidence -= 0.1

        return max(0.1, min(1.0, round(confidence, 2)))

    # ══════════════════════════════════════════
    # VALIDATION & CONFIDENCE
    # ══════════════════════════════════════════

    def _validate_input(self, data: dict[str, Any]) -> list[str]:
        warnings = []
        deals = data.get("deals", [])

        if not deals:
            raise InsufficientDataError(
                agent_name=self.AGENT_NAME,
                detail="No deals data provided.",
            )

        if len(deals) < 5:
            warnings.append(f"Only {len(deals)} deals. Analysis may be unreliable.")

        open_deals = [d for d in deals if not d.get("is_closed")]
        if not open_deals:
            warnings.append("No open deals in pipeline.")

        return warnings

    def _calculate_confidence(
        self,
        input_data: dict[str, Any],
        output_data: dict[str, Any],
    ) -> float:
        deals = input_data.get("deals", [])
        score = 0.5

        if len(deals) >= 30:
            score += 0.3
        elif len(deals) >= 10:
            score += 0.15

        won = [d for d in deals if d.get("is_won")]
        if len(won) >= 5:
            score += 0.2
        elif len(won) >= 2:
            score += 0.1

        return min(1.0, score)
