"""
Acquisition Efficiency Agent — CAC, channel scoring, lead quality.

UN KPI : Blended CAC (dépenses marketing totales ÷ nouveaux clients)

4 fonctions :
1. Source Tracking       → D'où viennent réellement les clients
2. Efficiency Scoring    → Quels canaux marchent, lesquels brûlent
3. Lead Quality Feedback → Les leads de quel canal convertissent
4. Weekly Report         → Résumé acquisition

Souvent le DERNIER agent activé car les PME trackent mal l'acquisition.
Le config a min_clients_for_analysis = 10 par défaut.
En dessous, l'agent se met en mode "observation" et collecte sans analyser.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Optional

from models.agent_config import AcquisitionEfficiencyConfig
from models.events import EventType, EventPriority
from models.metrics import AcquisitionEfficiencyMetrics, MetricTrend

from agents.base import BaseAgent, InsufficientDataError


class AcquisitionEfficiencyAgent(BaseAgent):
    """Agent Acquisition Efficiency — le traqueur de CAC."""

    AGENT_NAME = "acquisition_efficiency"

    def __init__(self, company_id: str, config: AcquisitionEfficiencyConfig, **kwargs):
        super().__init__(company_id=company_id, config=config, **kwargs)
        self.ae_config: AcquisitionEfficiencyConfig = config

    async def analyze(self, data: dict[str, Any]) -> dict[str, Any]:
        """Analyse complète acquisition.

        Args:
            data: {
                "crm": output de HubSpotConnector.extract(),
                "marketing_spend": {  # optionnel, fourni manuellement ou via API
                    "total": float,
                    "by_channel": {"google_ads": float, "linkedin": float, ...}
                }
            }
        """
        crm_data = data.get("crm", {})
        marketing_spend = data.get("marketing_spend", {})

        deals = crm_data.get("deals", [])
        contacts = crm_data.get("contacts", [])
        won_deals = [d for d in deals if d.get("is_won", False)]

        # Mode observation si pas assez de données
        if len(won_deals) < self.ae_config.min_clients_for_analysis:
            self.logger.info(
                f"Only {len(won_deals)} won deals. "
                f"Minimum {self.ae_config.min_clients_for_analysis} required. "
                f"Running in observation mode."
            )
            return await self._observation_mode(contacts, won_deals)

        # 1. Source Tracking
        source_data = self._track_sources(contacts, won_deals)

        # 2. Efficiency Scoring
        scored_channels = self._score_channels(
            source_data, marketing_spend
        )

        # 3. Lead Quality Feedback
        quality_data = self._analyze_lead_quality(
            contacts, deals, source_data
        )

        # 4. CAC Calculations
        blended_cac = None
        total_spend = marketing_spend.get("total", 0)
        if total_spend > 0 and len(won_deals) > 0:
            blended_cac = total_spend / len(won_deals)

        cac_by_channel = self._calculate_cac_by_channel(
            source_data, marketing_spend
        )

        # 5. Métriques
        previous = await self.get_previous_metrics()
        prev_cac = previous.get("blended_cac") if previous else None

        best_ch = None
        worst_ch = None
        performers = []
        gouffres = []

        if scored_channels:
            for ch in scored_channels:
                if ch["category"] == "performer":
                    performers.append(ch["channel"])
                elif ch["category"] == "gouffre":
                    gouffres.append(ch["channel"])

            if performers:
                best_ch = performers[0]
            if gouffres:
                worst_ch = gouffres[0]

        metrics = AcquisitionEfficiencyMetrics(
            company_id=self.company_id,
            blended_cac=round(blended_cac, 2) if blended_cac else None,
            cac_by_channel=cac_by_channel,
            best_channel=best_ch,
            worst_channel=worst_ch,
            total_leads=len(contacts),
            total_clients_acquired=len(won_deals),
            lead_to_client_conversion=(
                round(len(won_deals) / max(len(contacts), 1), 3)
            ),
            channels_performer=performers,
            channels_gouffre=gouffres,
        )
        await self.save_metrics(metrics)

        # Publier anomalies CAC
        if prev_cac and blended_cac:
            change_pct = abs(blended_cac - prev_cac) / prev_cac
            if change_pct > self.ae_config.cac_anomaly_threshold_pct:
                self.publish_event(
                    event_type=EventType.CAC_ANOMALY_DETECTED,
                    priority=EventPriority.HIGH,
                    payload={
                        "channel": "blended",
                        "cac_current": round(blended_cac, 2),
                        "cac_previous": round(prev_cac, 2),
                        "trend_pct": round(
                            ((blended_cac - prev_cac) / prev_cac) * 100, 1
                        ),
                    },
                )

        # Publier quality feedback pour le lead scoring
        if quality_data:
            self.publish_event(
                event_type=EventType.LEAD_QUALITY_UPDATED,
                payload={
                    "channels": {
                        ch: {
                            "conversion_rate": q.get("conversion_rate", 0),
                            "avg_deal_size": q.get("avg_deal_size", 0),
                            "quality_score": q.get("quality_score", 0),
                        }
                        for ch, q in quality_data.items()
                    }
                },
            )

        return {
            "blended_cac": round(blended_cac, 2) if blended_cac else None,
            "cac_trend": MetricTrend.calculate(
                blended_cac or 0, prev_cac
            ).model_dump() if prev_cac and blended_cac else None,
            "cac_by_channel": cac_by_channel,
            "source_data": source_data,
            "scored_channels": scored_channels,
            "quality_data": quality_data,
            "metrics": metrics.model_dump(),
        }

    # ══════════════════════════════════════════
    # MODE OBSERVATION
    # ══════════════════════════════════════════

    async def _observation_mode(
        self,
        contacts: list[dict],
        won_deals: list[dict],
    ) -> dict[str, Any]:
        """Mode dégradé quand pas assez de données.
        Collecte et prépare pour le futur."""
        sources: dict[str, int] = defaultdict(int)
        for c in contacts:
            src = (c.get("source") or "unknown").lower()
            sources[src] += 1

        return {
            "mode": "observation",
            "message": (
                f"Pas assez de données ({len(won_deals)} clients signés, "
                f"minimum {self.ae_config.min_clients_for_analysis} requis). "
                f"En mode observation."
            ),
            "contacts_by_source": dict(sources),
            "contacts_total": len(contacts),
            "won_deals_total": len(won_deals),
            "data_needed": self.ae_config.min_clients_for_analysis - len(won_deals),
        }

    # ══════════════════════════════════════════
    # FONCTION 1 — SOURCE TRACKING
    # ══════════════════════════════════════════

    def _track_sources(
        self,
        contacts: list[dict],
        won_deals: list[dict],
    ) -> dict[str, dict[str, Any]]:
        """Trace l'origine réelle des clients."""
        # Associer les deals gagnés à leurs contacts et sources
        won_contact_ids = set()
        for d in won_deals:
            for cid in d.get("contact_ids", []):
                won_contact_ids.add(cid)

        # Mapper source → contacts et deals
        source_stats: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "leads_count": 0,
                "clients_count": 0,
                "total_revenue": 0.0,
                "deal_sizes": [],
            }
        )

        contact_source_map: dict[str, str] = {}
        for c in contacts:
            source = self._normalize_source(c.get("source"))
            contact_source_map[c.get("contact_id", "")] = source
            source_stats[source]["leads_count"] += 1

        # Attribuer les deals gagnés aux sources
        for d in won_deals:
            # Trouver la source via le deal lui-même ou ses contacts
            source = self._normalize_source(d.get("source"))

            if source == "unknown":
                for cid in d.get("contact_ids", []):
                    if cid in contact_source_map:
                        source = contact_source_map[cid]
                        break

            amount = d.get("amount", 0) or 0
            source_stats[source]["clients_count"] += 1
            source_stats[source]["total_revenue"] += amount
            source_stats[source]["deal_sizes"].append(amount)

        # Calculer les métriques par source
        result = {}
        for source, stats in source_stats.items():
            leads = stats["leads_count"]
            clients = stats["clients_count"]
            revenue = stats["total_revenue"]
            sizes = stats["deal_sizes"]

            result[source] = {
                "leads_count": leads,
                "clients_count": clients,
                "total_revenue": round(revenue, 2),
                "conversion_rate": round(
                    self.safe_divide(clients, leads), 3
                ),
                "avg_deal_size": round(self.avg(sizes), 2) if sizes else 0,
            }

        return result

    def _normalize_source(self, source: Optional[str]) -> str:
        """Normalise les noms de sources."""
        if not source:
            return "unknown"

        source = source.lower().strip()

        # Mapping des variations courantes
        mappings = {
            "organic_search": ["organic", "seo", "google organic", "organic_search"],
            "paid_search": ["paid", "google_ads", "google ads", "adwords", "sem", "ppc"],
            "social_organic": ["social", "linkedin organic", "twitter", "social_media"],
            "social_paid": ["linkedin_ads", "linkedin ads", "meta_ads", "facebook_ads"],
            "referral": ["referral", "partner", "word_of_mouth", "bouche à oreille"],
            "direct": ["direct", "direct_traffic", "none"],
            "email": ["email", "newsletter", "email_marketing", "emailing"],
            "outbound": ["outbound", "cold_email", "cold_call", "prospection"],
            "event": ["event", "conference", "salon", "webinar", "webinaire"],
            "content": ["content", "blog", "ebook", "whitepaper", "livre_blanc"],
        }

        for normalized, variants in mappings.items():
            if source in variants:
                return normalized

        return source

    # ══════════════════════════════════════════
    # FONCTION 2 — EFFICIENCY SCORING
    # ══════════════════════════════════════════

    def _score_channels(
        self,
        source_data: dict[str, dict],
        marketing_spend: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Score chaque canal en 4 catégories."""
        spend_by_channel = marketing_spend.get("by_channel", {})
        scored = []

        for channel, stats in source_data.items():
            if channel == "unknown":
                continue

            clients = stats.get("clients_count", 0)
            leads = stats.get("leads_count", 0)
            revenue = stats.get("total_revenue", 0)
            conversion = stats.get("conversion_rate", 0)

            # CAC du canal
            channel_spend = spend_by_channel.get(channel, 0)
            cac = self.safe_divide(channel_spend, clients) if clients > 0 else None

            # Classifier
            category = self._classify_channel(
                conversion_rate=conversion,
                volume=leads,
                cac=cac,
                revenue=revenue,
            )

            recommendation = self._channel_recommendation(
                channel, category, stats, cac
            )

            scored.append({
                "channel": channel,
                "category": category,
                "leads": leads,
                "clients": clients,
                "conversion_rate": round(conversion, 3),
                "revenue": round(revenue, 2),
                "cac": round(cac, 2) if cac else None,
                "spend": channel_spend,
                "recommendation": recommendation,
            })

        # Trier : performers d'abord, gouffres en dernier
        category_order = {"performer": 0, "promising": 1, "inefficient": 2, "gouffre": 3}
        scored.sort(key=lambda x: category_order.get(x["category"], 4))

        return scored

    def _classify_channel(
        self,
        conversion_rate: float,
        volume: int,
        cac: Optional[float],
        revenue: float,
    ) -> str:
        """Classifie un canal en 4 catégories."""
        min_leads = self.ae_config.min_leads_per_channel

        if volume < min_leads:
            if conversion_rate > 0.15:
                return "promising"
            return "insufficient_data"

        # Si on a le CAC
        if cac is not None:
            avg_deal = self.safe_divide(revenue, max(1, volume * conversion_rate))
            roi = self.safe_divide(avg_deal, cac) if cac > 0 else 0

            if roi > 3 and conversion_rate > 0.1:
                return "performer"
            if roi > 1.5:
                return "promising"
            if roi < 0.5:
                return "gouffre"
            return "inefficient"

        # Sans CAC : se baser sur la conversion
        if conversion_rate > 0.2:
            return "performer"
        if conversion_rate > 0.1:
            return "promising"
        if conversion_rate < 0.03:
            return "gouffre"
        return "inefficient"

    def _channel_recommendation(
        self,
        channel: str,
        category: str,
        stats: dict,
        cac: Optional[float],
    ) -> str:
        """Génère une recommandation actionnable pour chaque canal."""
        if category == "performer":
            return f"DOUBLER le budget sur {channel}. ROI prouvé."
        if category == "promising":
            return f"TESTER {channel} avec plus de volume. Potentiel détecté."
        if category == "inefficient":
            cac_str = self.format_currency(cac) if cac else "inconnu"
            return f"RÉDUIRE le budget {channel}. CAC {cac_str} trop élevé."
        if category == "gouffre":
            return f"COUPER {channel}. Conversion < 3%, pas de ROI."
        return f"OBSERVER {channel}. Pas assez de données."

    # ══════════════════════════════════════════
    # FONCTION 3 — LEAD QUALITY FEEDBACK
    # ══════════════════════════════════════════

    def _analyze_lead_quality(
        self,
        contacts: list[dict],
        deals: list[dict],
        source_data: dict[str, dict],
    ) -> dict[str, dict[str, Any]]:
        """Analyse la qualité des leads par canal source."""
        quality = {}

        for channel, stats in source_data.items():
            if channel == "unknown":
                continue

            conversion = stats.get("conversion_rate", 0)
            avg_deal = stats.get("avg_deal_size", 0)

            quality_score = self._quality_score(conversion, avg_deal)

            quality[channel] = {
                "conversion_rate": round(conversion, 3),
                "avg_deal_size": round(avg_deal, 2),
                "quality_score": round(quality_score, 2),
                "verdict": self._quality_verdict(quality_score),
            }

        return quality

    def _quality_score(
        self,
        conversion_rate: float,
        avg_deal_size: float,
    ) -> float:
        """Score de qualité 0-100 basé sur conversion et valeur."""
        # 60% conversion, 40% deal size
        conversion_score = min(100, conversion_rate * 500)  # 20% conv = 100
        size_score = min(100, avg_deal_size / 100)  # 10K = 100

        return conversion_score * 0.6 + size_score * 0.4

    @staticmethod
    def _quality_verdict(score: float) -> str:
        if score >= 70:
            return "excellent"
        if score >= 40:
            return "acceptable"
        if score >= 20:
            return "mediocre"
        return "poor"

    # ══════════════════════════════════════════
    # CAC CALCULATION
    # ══════════════════════════════════════════

    def _calculate_cac_by_channel(
        self,
        source_data: dict[str, dict],
        marketing_spend: dict[str, Any],
    ) -> Optional[dict[str, float]]:
        """Calcule le CAC par canal."""
        spend_by_channel = marketing_spend.get("by_channel", {})
        if not spend_by_channel:
            return None

        cac_by_channel = {}
        for channel, spend in spend_by_channel.items():
            channel_norm = self._normalize_source(channel)
            clients = source_data.get(channel_norm, {}).get("clients_count", 0)
            if clients > 0:
                cac_by_channel[channel_norm] = round(spend / clients, 2)

        return cac_by_channel if cac_by_channel else None

    # ══════════════════════════════════════════
    # VALIDATION & CONFIDENCE
    # ══════════════════════════════════════════

    def _validate_input(self, data: dict[str, Any]) -> list[str]:
        warnings = []
        crm = data.get("crm", {})
        contacts = crm.get("contacts", [])
        deals = crm.get("deals", [])

        if not contacts:
            raise InsufficientDataError(
                agent_name=self.AGENT_NAME,
                detail="No contacts data.",
            )

        won = [d for d in deals if d.get("is_won")]
        if len(won) < self.ae_config.min_clients_for_analysis:
            warnings.append(
                f"Only {len(won)} won deals. "
                f"Minimum {self.ae_config.min_clients_for_analysis} for full analysis."
            )

        if not data.get("marketing_spend"):
            warnings.append(
                "No marketing spend data. CAC calculation will not be available."
            )

        contacts_with_source = [c for c in contacts if c.get("source")]
        source_pct = len(contacts_with_source) / max(len(contacts), 1)
        if source_pct < 0.3:
            warnings.append(
                f"Only {source_pct:.0%} of contacts have a source. "
                f"Attribution will be limited."
            )

        return warnings

    def _calculate_confidence(
        self,
        input_data: dict[str, Any],
        output_data: dict[str, Any],
    ) -> float:
        crm = input_data.get("crm", {})
        contacts = crm.get("contacts", [])
        deals = crm.get("deals", [])
        won = [d for d in deals if d.get("is_won")]

        score = 0.2

        if len(won) >= self.ae_config.min_clients_for_analysis:
            score += 0.3
        elif len(won) >= 5:
            score += 0.15

        contacts_with_source = [c for c in contacts if c.get("source")]
        source_pct = len(contacts_with_source) / max(len(contacts), 1)
        score += source_pct * 0.3

        if input_data.get("marketing_spend"):
            score += 0.2

        return min(1.0, score)
