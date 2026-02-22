"""
Acquisition Efficiency Agent — CAC, channel scoring, lead quality.

KPI NORD : Blended CAC (dépenses marketing totales ÷ nouveaux clients)

4 fonctions :
1. Source Tracking       → D'où viennent réellement les clients
2. Efficiency Scoring    → Quels canaux marchent, lesquels brûlent
3. Lead Quality Feedback → Les leads de quel canal convertissent
4. CAC Calculation       → Blended + par canal

Souvent le DERNIER agent activé : les PME trackent mal l'acquisition.
En dessous de min_clients_for_analysis, l'agent passe en mode observation.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Optional

from models.agent_config import AcquisitionEfficiencyConfig
from models.events import EventType, EventPriority
from models.metrics import AcquisitionEfficiencyMetrics, MetricTrend

from agents.base import BaseAgent, AgentResult, InsufficientDataError


# ══════════════════════════════════════════
# SOURCE NORMALIZATION — Mapping statique
# ══════════════════════════════════════════

_SOURCE_MAP: dict[str, list[str]] = {
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

# Lookup inversé pour O(1)
_SOURCE_LOOKUP: dict[str, str] = {}
for _norm, _variants in _SOURCE_MAP.items():
    for _v in _variants:
        _SOURCE_LOOKUP[_v] = _norm


def normalize_source(raw: Optional[str]) -> str:
    """Normalise un nom de source en catégorie canonique."""
    if not raw:
        return "unknown"
    key = raw.lower().strip()
    return _SOURCE_LOOKUP.get(key, key)


# ══════════════════════════════════════════
# AGENT
# ══════════════════════════════════════════


class AcquisitionEfficiencyAgent(BaseAgent[AcquisitionEfficiencyConfig]):
    """Agent Acquisition Efficiency — le traqueur de CAC."""

    AGENT_NAME = "acquisition_efficiency"

    # ──────────────────────────────────────
    # CONTRAT BASE AGENT
    # ──────────────────────────────────────

    def _validate(self, data: dict[str, Any]) -> list[str]:
        """Valide les données d'entrée CRM + marketing spend."""
        warnings: list[str] = []
        crm = data.get("crm", {})
        contacts = crm.get("contacts", [])
        deals = crm.get("deals", [])

        if not contacts:
            raise InsufficientDataError(
                agent_name=self.AGENT_NAME,
                detail="Aucun contact trouvé dans le CRM.",
                available=0,
                required=1,
            )

        won = [d for d in deals if d.get("is_won")]
        min_required = self.config.min_clients_for_analysis

        if len(won) < min_required:
            raise InsufficientDataError(
                agent_name=self.AGENT_NAME,
                detail=(
                    f"Seulement {len(won)} deals gagnés. "
                    f"Minimum {min_required} requis pour l'analyse complète."
                ),
                available=len(won),
                required=min_required,
            )

        if not data.get("marketing_spend"):
            warnings.append(
                "Aucune donnée de dépenses marketing. "
                "Le calcul du CAC ne sera pas disponible."
            )

        contacts_with_source = [c for c in contacts if c.get("source")]
        source_pct = len(contacts_with_source) / max(len(contacts), 1)
        if source_pct < 0.3:
            warnings.append(
                f"Seulement {source_pct:.0%} des contacts ont une source. "
                f"L'attribution sera limitée."
            )

        return warnings

    async def _execute(self, data: dict[str, Any]) -> dict[str, Any]:
        """Logique métier : source tracking → scoring → quality → CAC."""
        crm = data.get("crm", {})
        marketing_spend = data.get("marketing_spend", {})

        deals = crm.get("deals", [])
        contacts = crm.get("contacts", [])
        won_deals = [d for d in deals if d.get("is_won", False)]

        # 1. Source Tracking
        source_data = self._track_sources(contacts, won_deals)

        # 2. Channel Scoring
        scored_channels = self._score_channels(source_data, marketing_spend)

        # 3. Lead Quality
        quality_data = self._analyze_lead_quality(source_data)

        # 4. CAC
        blended_cac = self._blended_cac(marketing_spend, len(won_deals))
        cac_by_channel = self._cac_by_channel(source_data, marketing_spend)

        # 5. Tendance CAC
        previous = await self.get_previous_metrics()
        prev_cac = previous.get("blended_cac") if previous else None
        cac_trend = None
        if prev_cac is not None and blended_cac is not None:
            cac_trend = MetricTrend.calculate(blended_cac, prev_cac).model_dump()

        # 6. Classement canaux
        performers = [ch["channel"] for ch in scored_channels if ch["category"] == "performer"]
        gouffres = [ch["channel"] for ch in scored_channels if ch["category"] == "gouffre"]

        # 7. Métriques
        metrics = AcquisitionEfficiencyMetrics(
            company_id=self.company_id,
            blended_cac=_round_opt(blended_cac),
            cac_by_channel=cac_by_channel,
            best_channel=performers[0] if performers else None,
            worst_channel=gouffres[0] if gouffres else None,
            total_leads=len(contacts),
            total_clients_acquired=len(won_deals),
            lead_to_client_conversion=round(
                self.safe_divide(len(won_deals), len(contacts)), 3
            ),
            channels_performer=performers,
            channels_gouffre=gouffres,
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

        # 8. Événements
        self._emit_cac_anomaly(blended_cac, prev_cac)
        self._emit_quality_update(quality_data)

        return {
            "blended_cac": _round_opt(blended_cac),
            "cac_trend": cac_trend,
            "cac_by_channel": cac_by_channel,
            "source_data": source_data,
            "scored_channels": scored_channels,
            "quality_data": quality_data,
            "metrics": metrics.model_dump(),
            "summary": self._build_summary(
                blended_cac, performers, gouffres, len(won_deals), len(contacts)
            ),
        }

    def _confidence(self, input_data: dict[str, Any], output_data: dict[str, Any]) -> float:
        """Score de confiance basé sur la couverture des données."""
        crm = input_data.get("crm", {})
        contacts = crm.get("contacts", [])
        deals = crm.get("deals", [])
        won = [d for d in deals if d.get("is_won")]

        score = 0.2  # Base

        # Volume de won deals
        min_req = self.config.min_clients_for_analysis
        if len(won) >= min_req * 2:
            score += 0.3
        elif len(won) >= min_req:
            score += 0.2
        elif len(won) >= 5:
            score += 0.1

        # Couverture source
        contacts_with_source = [c for c in contacts if c.get("source")]
        source_pct = self.safe_divide(len(contacts_with_source), len(contacts))
        score += source_pct * 0.3

        # Données de spend
        if input_data.get("marketing_spend", {}).get("by_channel"):
            score += 0.2
        elif input_data.get("marketing_spend", {}).get("total"):
            score += 0.1

        return min(1.0, round(score, 2))

    async def _observation_mode(
        self, data: dict[str, Any], error: InsufficientDataError
    ) -> dict[str, Any]:
        """Mode observation : collecte sans analyser."""
        crm = data.get("crm", {})
        contacts = crm.get("contacts", [])
        deals = crm.get("deals", [])
        won = [d for d in deals if d.get("is_won")]

        sources: dict[str, int] = defaultdict(int)
        for c in contacts:
            src = normalize_source(c.get("source"))
            sources[src] += 1

        return {
            "message": (
                f"Pas assez de données ({len(won)} clients signés, "
                f"minimum {self.config.min_clients_for_analysis} requis). "
                f"En mode observation — collecte en cours."
            ),
            "contacts_by_source": dict(sources),
            "contacts_total": len(contacts),
            "won_deals_total": len(won),
            "data_needed": max(0, self.config.min_clients_for_analysis - len(won)),
            "top_sources": sorted(
                sources.items(), key=lambda x: x[1], reverse=True
            )[:5],
        }

    # ══════════════════════════════════════════
    # FONCTION 1 — SOURCE TRACKING
    # ══════════════════════════════════════════

    def _track_sources(
        self,
        contacts: list[dict],
        won_deals: list[dict],
    ) -> dict[str, dict[str, Any]]:
        """Trace l'origine des clients — mapping contact→source→deal."""
        # Index contacts par ID et source
        contact_source: dict[str, str] = {}
        source_stats: dict[str, _SourceAccumulator] = defaultdict(_SourceAccumulator)

        for c in contacts:
            cid = c.get("contact_id", "")
            source = normalize_source(c.get("source"))
            contact_source[cid] = source
            source_stats[source].leads += 1

        # Attribuer deals gagnés aux sources
        for d in won_deals:
            source = normalize_source(d.get("source"))

            # Fallback : chercher via les contacts du deal
            if source == "unknown":
                for cid in d.get("contact_ids", []):
                    if cid in contact_source and contact_source[cid] != "unknown":
                        source = contact_source[cid]
                        break

            amount = d.get("amount") or 0
            acc = source_stats[source]
            acc.clients += 1
            acc.revenue += amount
            acc.deal_sizes.append(amount)

        # Construire le résultat
        return {
            source: {
                "leads_count": acc.leads,
                "clients_count": acc.clients,
                "total_revenue": round(acc.revenue, 2),
                "conversion_rate": round(self.safe_divide(acc.clients, acc.leads), 3),
                "avg_deal_size": round(self.avg(acc.deal_sizes), 2) if acc.deal_sizes else 0,
            }
            for source, acc in source_stats.items()
        }

    # ══════════════════════════════════════════
    # FONCTION 2 — CHANNEL SCORING
    # ══════════════════════════════════════════

    def _score_channels(
        self,
        source_data: dict[str, dict],
        marketing_spend: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Score et classifie chaque canal d'acquisition."""
        spend_by_channel = marketing_spend.get("by_channel", {})
        scored: list[dict[str, Any]] = []

        for channel, stats in source_data.items():
            if channel == "unknown":
                continue

            leads = stats["leads_count"]
            clients = stats["clients_count"]
            revenue = stats["total_revenue"]
            conversion = stats["conversion_rate"]

            channel_spend = spend_by_channel.get(channel, 0)
            cac = self.safe_divide(channel_spend, clients) if clients > 0 else None

            category = self._classify_channel(conversion, leads, cac, revenue)
            recommendation = self._recommend(channel, category, cac)

            scored.append({
                "channel": channel,
                "category": category,
                "leads": leads,
                "clients": clients,
                "conversion_rate": round(conversion, 3),
                "revenue": round(revenue, 2),
                "cac": _round_opt(cac),
                "spend": channel_spend,
                "recommendation": recommendation,
            })

        # Tri : performers → promising → inefficient → gouffre
        _ORDER = {"performer": 0, "promising": 1, "inefficient": 2, "gouffre": 3, "insufficient_data": 4}
        scored.sort(key=lambda x: _ORDER.get(x["category"], 5))
        return scored

    def _classify_channel(
        self,
        conversion_rate: float,
        volume: int,
        cac: Optional[float],
        revenue: float,
    ) -> str:
        """Classifie un canal : performer | promising | inefficient | gouffre."""
        min_leads = self.config.min_leads_per_channel

        # Pas assez de données
        if volume < min_leads:
            return "promising" if conversion_rate > 0.15 else "insufficient_data"

        # Avec CAC : calcul du ROI
        if cac is not None and cac > 0:
            clients_est = max(1, volume * conversion_rate)
            avg_deal = self.safe_divide(revenue, clients_est)
            roi = self.safe_divide(avg_deal, cac)

            if roi > 3 and conversion_rate > 0.1:
                return "performer"
            if roi > 1.5:
                return "promising"
            if roi < 0.5:
                return "gouffre"
            return "inefficient"

        # Sans CAC : conversion seule
        if conversion_rate > 0.2:
            return "performer"
        if conversion_rate > 0.1:
            return "promising"
        if conversion_rate < 0.03:
            return "gouffre"
        return "inefficient"

    @staticmethod
    def _recommend(channel: str, category: str, cac: Optional[float]) -> str:
        """Recommandation actionnable par catégorie."""
        recs = {
            "performer": f"DOUBLER le budget sur {channel}. ROI prouvé.",
            "promising": f"TESTER {channel} avec plus de volume. Potentiel détecté.",
            "inefficient": f"RÉDUIRE le budget {channel}. CAC trop élevé ({_fmt_eur(cac)}).",
            "gouffre": f"COUPER {channel}. Conversion < 3%, pas de ROI.",
        }
        return recs.get(category, f"OBSERVER {channel}. Données insuffisantes.")

    # ══════════════════════════════════════════
    # FONCTION 3 — LEAD QUALITY FEEDBACK
    # ══════════════════════════════════════════

    def _analyze_lead_quality(
        self,
        source_data: dict[str, dict],
    ) -> dict[str, dict[str, Any]]:
        """Score qualité des leads par source.
        Score 0-100 = 60% conversion + 40% deal size."""
        quality: dict[str, dict[str, Any]] = {}

        for channel, stats in source_data.items():
            if channel == "unknown":
                continue

            conversion = stats["conversion_rate"]
            avg_deal = stats["avg_deal_size"]

            # Score : conversion→500x cap 100 (60%), deal_size→/100 cap 100 (40%)
            conv_score = min(100.0, conversion * 500)
            size_score = min(100.0, avg_deal / 100)
            score = conv_score * 0.6 + size_score * 0.4

            quality[channel] = {
                "conversion_rate": round(conversion, 3),
                "avg_deal_size": round(avg_deal, 2),
                "quality_score": round(score, 2),
                "verdict": _quality_verdict(score),
            }

        return quality

    # ══════════════════════════════════════════
    # FONCTION 4 — CAC CALCULATION
    # ══════════════════════════════════════════

    def _blended_cac(self, marketing_spend: dict[str, Any], won_count: int) -> Optional[float]:
        """CAC blended = total spend / won deals."""
        total = marketing_spend.get("total", 0)
        if total <= 0 or won_count <= 0:
            return None
        return total / won_count

    def _cac_by_channel(
        self,
        source_data: dict[str, dict],
        marketing_spend: dict[str, Any],
    ) -> Optional[dict[str, float]]:
        """CAC ventilé par canal."""
        spend_by = marketing_spend.get("by_channel", {})
        if not spend_by:
            return None

        result: dict[str, float] = {}
        for channel, spend in spend_by.items():
            norm = normalize_source(channel)
            clients = source_data.get(norm, {}).get("clients_count", 0)
            if clients > 0:
                result[norm] = round(spend / clients, 2)

        return result or None

    # ══════════════════════════════════════════
    # ÉVÉNEMENTS
    # ══════════════════════════════════════════

    def _emit_cac_anomaly(self, current_cac: Optional[float], prev_cac: Optional[float]) -> None:
        """Émet un événement si le CAC varie au-delà du seuil."""
        if current_cac is None or prev_cac is None or prev_cac == 0:
            return

        change_pct = abs(current_cac - prev_cac) / prev_cac
        if change_pct <= self.config.cac_anomaly_threshold_pct:
            return

        self._emit_event(
            event_type=EventType.CAC_ANOMALY_DETECTED,
            priority=EventPriority.HIGH,
            payload={
                "channel": "blended",
                "cac_current": round(current_cac, 2),
                "cac_previous": round(prev_cac, 2),
                "variation_pct": round(
                    ((current_cac - prev_cac) / prev_cac) * 100, 1
                ),
            },
        )

    def _emit_quality_update(self, quality_data: dict[str, dict]) -> None:
        """Émet le feedback qualité pour le lead scoring inter-agents."""
        if not quality_data:
            return

        self._emit_event(
            event_type=EventType.LEAD_QUALITY_UPDATED,
            priority=EventPriority.MEDIUM,
            payload={
                "channels": {
                    ch: {
                        "conversion_rate": q["conversion_rate"],
                        "avg_deal_size": q["avg_deal_size"],
                        "quality_score": q["quality_score"],
                    }
                    for ch, q in quality_data.items()
                }
            },
        )

    # ══════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════

    def _build_summary(
        self,
        blended_cac: Optional[float],
        performers: list[str],
        gouffres: list[str],
        won_count: int,
        leads_count: int,
    ) -> str:
        """Résumé texte pour le dashboard et les rapports."""
        parts: list[str] = []

        if blended_cac is not None:
            parts.append(f"CAC blended : {_fmt_eur(blended_cac)}")
        else:
            parts.append("CAC blended : non calculable (pas de données de dépenses)")

        conv = self.safe_divide(won_count, leads_count)
        parts.append(f"Conversion globale : {conv:.1%} ({won_count}/{leads_count})")

        if performers:
            parts.append(f"Canaux performants : {', '.join(performers)}")
        if gouffres:
            parts.append(f"Canaux à couper : {', '.join(gouffres)}")

        return " | ".join(parts)


# ══════════════════════════════════════════
# HELPERS PRIVÉS (module-level)
# ══════════════════════════════════════════


class _SourceAccumulator:
    """Accumulateur interne pour le tracking par source."""

    __slots__ = ("leads", "clients", "revenue", "deal_sizes")

    def __init__(self):
        self.leads: int = 0
        self.clients: int = 0
        self.revenue: float = 0.0
        self.deal_sizes: list[float] = []


   def _round_opt(value: Optional[float], decimals: int = 2) -> Optional[float]:
    """Round None-safe."""
    return round(value, decimals) if value is not None 
    else None
     
