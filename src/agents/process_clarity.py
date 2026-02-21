"""
Process Clarity Agent — Bottleneck detection, waste detection, process mapping.

UN KPI : Cycle Time des Process Critiques (jours)

4 fonctions :
1. Process Mapping     → Cartographier les flux réels
2. Bottleneck Detector → Trouver LE goulot (Goldratt)
3. Waste Detector      → Trouver le travail qui ne devrait pas exister
4. Weekly Report       → Résumé opérationnel

Inspiration : Toyota (muda), Goldratt (TOC), Basecamp (less is more)
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Optional

from models.agent_config import ProcessClarityConfig
from models.events import EventType, EventPriority
from models.metrics import ProcessClarityMetrics, MetricTrend

from agents.base import BaseAgent, InsufficientDataError


class ProcessClarityAgent(BaseAgent):
    """Agent Process Clarity — le chasseur de goulots."""

    AGENT_NAME = "process_clarity"

    def __init__(self, company_id: str, config: ProcessClarityConfig, **kwargs):
        super().__init__(company_id=company_id, config=config, **kwargs)
        self.pc_config: ProcessClarityConfig = config

    async def analyze(self, data: dict[str, Any]) -> dict[str, Any]:
        """Analyse complète des opérations."""
        crm_data = data.get("crm", {})
        email_data = data.get("email")

        # 1. Process Mapping
        processes = self._map_processes(crm_data)

        # 2. Bottleneck Detection
        bottleneck = self._detect_bottleneck(processes, crm_data)

        # 3. Waste Detection
        waste = self._detect_waste(crm_data, email_data)

        # 4. Cycle time KPI
        cycle_times = [p["avg_days"] for p in processes if p.get("avg_days")]
        avg_cycle = self.avg(cycle_times) if cycle_times else 0

        # 5. Person load analysis
        person_load = self._analyze_person_load(crm_data, email_data)

        # 6. Métriques
        previous = await self.get_previous_metrics()
        prev_cycle = previous.get("avg_cycle_time_days") if previous else None

        metrics = ProcessClarityMetrics(
            company_id=self.company_id,
            avg_cycle_time_days=round(avg_cycle, 1),
            bottleneck_stage=bottleneck.get("stage") if bottleneck else None,
            bottleneck_time_days=(
                round(bottleneck["avg_days"], 1) if bottleneck else None
            ),
            bottleneck_estimated_cost=(
                round(bottleneck.get("estimated_cost", 0), 0) if bottleneck else None
            ),
            tasks_overdue=waste.get("tasks_overdue_count", 0),
            tasks_overdue_trend="stable",
            waste_hours_per_week=waste.get("total_waste_hours_week"),
            waste_annual_cost=waste.get("total_waste_annual_cost"),
            person_max_load_name=(
                person_load["most_loaded"]["name"] if person_load.get("most_loaded") else None
            ),
            person_max_load_pct=(
                person_load["most_loaded"]["load_pct"] if person_load.get("most_loaded") else None
            ),
        )
        await self.save_metrics(metrics)

        # Publier bottleneck si trouvé
        if bottleneck:
            self.publish_event(
                event_type=EventType.BOTTLENECK_DETECTED,
                priority=EventPriority.HIGH,
                payload={
                    "process": bottleneck.get("process", "sales_pipeline"),
                    "stage": bottleneck.get("stage"),
                    "person": bottleneck.get("person"),
                    "delay_vs_normal": round(bottleneck.get("multiplier", 0), 1),
                    "estimated_cost": round(bottleneck.get("estimated_cost", 0), 0),
                },
            )

        # Publier waste si significatif
        if waste.get("total_waste_hours_week", 0) > 5:
            self.publish_event(
                event_type=EventType.WASTE_DETECTED,
                payload={
                    "waste_type": "aggregate",
                    "description": waste.get("summary", ""),
                    "hours_per_week": waste.get("total_waste_hours_week", 0),
                    "annual_cost": waste.get("total_waste_annual_cost", 0),
                },
            )

        return {
            "avg_cycle_time_days": round(avg_cycle, 1),
            "cycle_trend": MetricTrend.calculate(
                avg_cycle, prev_cycle
            ).model_dump() if prev_cycle else None,
            "processes": processes,
            "bottleneck": bottleneck,
            "waste": waste,
            "person_load": person_load,
            "metrics": metrics.model_dump(),
        }

    # ══════════════════════════════════════════
    # FONCTION 1 — PROCESS MAPPING
    # ══════════════════════════════════════════

    def _map_processes(self, crm_data: dict[str, Any]) -> list[dict[str, Any]]:
        """Cartographie le process de vente basé sur les données CRM.

        En V1, le process principal est le pipeline de vente.
        Les étapes sont les stages du CRM.
        Le temps par étape est calculé depuis les données réelles.
        """
        deals = crm_data.get("deals", [])
        stages_config = crm_data.get("stages", {})

        if not deals:
            return []

        # Calculer le temps moyen par stage
        stage_metrics: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"times": [], "deal_count": 0, "total_value": 0}
        )

        for deal in deals:
            stage = deal.get("stage", "unknown")
            days = deal.get("days_in_current_stage", 0) or 0
            amount = deal.get("amount", 0) or 0

            stage_metrics[stage]["times"].append(days)
            stage_metrics[stage]["deal_count"] += 1
            stage_metrics[stage]["total_value"] += amount

        processes = []
        for stage, metrics in stage_metrics.items():
            times = metrics["times"]
            avg_days = self.avg(times)
            median_days = self.median(times)

            stage_info = stages_config.get(stage, {})

            processes.append({
                "process": "sales_pipeline",
                "stage": stage,
                "stage_label": stage_info.get("label", stage),
                "display_order": stage_info.get("display_order", 0),
                "avg_days": round(avg_days, 1),
                "median_days": round(median_days, 1),
                "max_days": max(times) if times else 0,
                "deal_count": metrics["deal_count"],
                "total_value": round(metrics["total_value"], 2),
            })

        # Trier par order d'affichage
        processes.sort(key=lambda p: p.get("display_order", 0))

        return processes

    # ══════════════════════════════════════════
    # FONCTION 2 — BOTTLENECK DETECTOR
    # ══════════════════════════════════════════

    def _detect_bottleneck(
        self,
        processes: list[dict],
        crm_data: dict[str, Any],
    ) -> Optional[dict[str, Any]]:
        """Identifie LE bottleneck #1 (Goldratt).

        Un stage est un bottleneck si son temps moyen dépasse
        la moyenne des autres stages × le multiplier configuré.
        """
        if len(processes) < 2:
            return None

        # Calculer la moyenne de TOUS les stages
        all_times = [p["avg_days"] for p in processes if p["avg_days"] > 0]
        if not all_times:
            return None

        global_avg = self.avg(all_times)
        threshold = global_avg * self.pc_config.bottleneck_time_multiplier

        # Trouver les bottlenecks
        bottlenecks = []
        for process in processes:
            if process["avg_days"] > threshold and process["deal_count"] >= 2:
                multiplier = self.safe_divide(
                    process["avg_days"], global_avg, default=1.0
                )

                # Estimer le coût
                excess_days = process["avg_days"] - global_avg
                hourly_rate = self.pc_config.hourly_rate_estimate
                # Coût = excess_days × deals_par_an × heures_par_jour × hourly_rate
                deals_per_year = process["deal_count"] * 4  # Extrapolation trimestrielle
                estimated_cost = excess_days * deals_per_year * 2 * hourly_rate

                # Détecter la personne responsable
                person = self._detect_bottleneck_person(
                    crm_data.get("deals", []),
                    process["stage"],
                )

                bottlenecks.append({
                    "process": "sales_pipeline",
                    "stage": process["stage"],
                    "stage_label": process.get("stage_label", process["stage"]),
                    "avg_days": process["avg_days"],
                    "global_avg": round(global_avg, 1),
                    "multiplier": round(multiplier, 1),
                    "excess_days": round(excess_days, 1),
                    "deal_count": process["deal_count"],
                    "estimated_cost": round(estimated_cost, 0),
                    "person": person,
                })

        if not bottlenecks:
            return None

        # Retourner LE pire (Goldratt : focus sur UN SEUL goulot)
        bottlenecks.sort(key=lambda b: b["multiplier"], reverse=True)
        return bottlenecks[0]

    def _detect_bottleneck_person(
        self,
        deals: list[dict],
        stage: str,
    ) -> Optional[str]:
        """Identifie si une personne concentre le bottleneck."""
        stage_deals = [d for d in deals if d.get("stage") == stage]
        if not stage_deals:
            return None

        owner_counts: dict[str, int] = defaultdict(int)
        for d in stage_deals:
            owner = d.get("owner_name", "Unknown") or "Unknown"
            owner_counts[owner] += 1

        total = len(stage_deals)
        for owner, count in owner_counts.items():
            if count / total >= self.pc_config.concentration_threshold:
                return owner

        return None

    # ══════════════════════════════════════════
    # FONCTION 3 — WASTE DETECTOR
    # ══════════════════════════════════════════

    def _detect_waste(
        self,
        crm_data: dict[str, Any],
        email_data: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        """Détecte les gaspillages opérationnels."""
        wastes = []
        hourly_rate = self.pc_config.hourly_rate_estimate

        # ── 1. Deals zombies (travail sur des deals morts) ──
        deals = crm_data.get("deals", [])
        open_deals = [d for d in deals if not d.get("is_closed", False)]
        zombie_deals = [
            d for d in open_deals
            if (d.get("days_since_last_activity", 0) or 0) > 60
        ]

        if zombie_deals:
            # Temps estimé perdu : 30 min/semaine par deal zombie dans les réunions pipeline
            hours_per_week = len(zombie_deals) * 0.5
            annual_cost = hours_per_week * 50 * hourly_rate  # 50 semaines

            wastes.append({
                "type": "zombie_deals",
                "description": (
                    f"{len(zombie_deals)} deals sans activité depuis > 60 jours "
                    f"polluent le pipeline et les réunions commerciales."
                ),
                "items_count": len(zombie_deals),
                "hours_per_week": round(hours_per_week, 1),
                "annual_cost": round(annual_cost, 0),
                "fix_difficulty": "easy",
                "fix_action": "Archive these deals. Remove from pipeline.",
            })

        # ── 2. Contacts dupliqués ──
        if self.pc_config.duplicate_detection_enabled:
            contacts = crm_data.get("contacts", [])
            duplicates = self._find_duplicate_contacts(contacts)

            if duplicates:
                hours_per_week = len(duplicates) * 0.1
                annual_cost = hours_per_week * 50 * hourly_rate

                wastes.append({
                    "type": "duplicate_contacts",
                    "description": (
                        f"{len(duplicates)} contacts potentiellement dupliqués "
                        f"causent de la confusion et du travail en double."
                    ),
                    "items_count": len(duplicates),
                    "hours_per_week": round(hours_per_week, 1),
                    "annual_cost": round(annual_cost, 0),
                    "fix_difficulty": "easy",
                    "fix_action": "Merge duplicate contacts in CRM.",
                    "details": duplicates[:10],
                })

        # ── 3. Ping-pong email ──
        if email_data:
            threads = email_data.get("threads", [])
            ping_pong_threads = [
                t for t in threads
                if t.get("message_count", 0) >= self.pc_config.ping_pong_threshold
            ]

            if ping_pong_threads:
                hours_per_week = len(ping_pong_threads) * 0.3
                annual_cost = hours_per_week * 50 * hourly_rate

                wastes.append({
                    "type": "email_ping_pong",
                    "description": (
                        f"{len(ping_pong_threads)} conversations email avec "
                        f"{self.pc_config.ping_pong_threshold}+ messages. "
                        f"Ces sujets nécessitent un appel ou une réunion, pas un email."
                    ),
                    "items_count": len(ping_pong_threads),
                    "hours_per_week": round(hours_per_week, 1),
                    "annual_cost": round(annual_cost, 0),
                    "fix_difficulty": "medium",
                    "fix_action": "Implement a rule: >3 emails → switch to call.",
                })

        # ── 4. Deals sans next step ──
        no_next = [d for d in open_deals if not d.get("has_next_step", False)]
        if len(no_next) > len(open_deals) * 0.3:
            hours_per_week = len(no_next) * 0.2
            annual_cost = hours_per_week * 50 * hourly_rate

            wastes.append({
                "type": "no_next_step",
                "description": (
                    f"{len(no_next)}/{len(open_deals)} deals actifs n'ont pas "
                    f"de prochaine étape définie. Chaque deal sans next step "
                    f"est un deal qui stagne par défaut."
                ),
                "items_count": len(no_next),
                "hours_per_week": round(hours_per_week, 1),
                "annual_cost": round(annual_cost, 0),
                "fix_difficulty": "easy",
                "fix_action": "Require next step on every deal update.",
            })

        # Résumé
        total_hours = sum(w.get("hours_per_week", 0) for w in wastes)
        total_cost = sum(w.get("annual_cost", 0) for w in wastes)

        summary = ""
        if wastes:
            summary = (
                f"{len(wastes)} types de gaspillage détectés. "
                f"Total : {total_hours:.0f}h/semaine, "
                f"{self.format_currency(total_cost)}/an."
            )

        return {
            "wastes": wastes,
            "total_waste_hours_week": round(total_hours, 1),
            "total_waste_annual_cost": round(total_cost, 0),
            "tasks_overdue_count": 0,  # Pas de données de tâches en V1
            "summary": summary,
        }

    def _find_duplicate_contacts(
        self, contacts: list[dict]
    ) -> list[dict[str, Any]]:
        """Trouve les contacts potentiellement dupliqués."""
        email_map: dict[str, list[dict]] = defaultdict(list)
        name_map: dict[str, list[dict]] = defaultdict(list)

        for c in contacts:
            email = (c.get("email") or "").lower().strip()
            if email:
                email_map[email].append(c)

            name = (
                f"{c.get('first_name', '')} {c.get('last_name', '')}".lower().strip()
            )
            if len(name) > 3:
                name_map[name].append(c)

        duplicates = []

        # Emails identiques
        for email, group in email_map.items():
            if len(group) > 1:
                duplicates.append({
                    "match_type": "email",
                    "match_value": email,
                    "contact_ids": [c.get("contact_id") for c in group],
                    "count": len(group),
                })

        # Noms identiques (plus de faux positifs, mais utile)
        for name, group in name_map.items():
            if len(group) > 1 and name not in ("", " "):
                duplicates.append({
                    "match_type": "name",
                    "match_value": name,
                    "contact_ids": [c.get("contact_id") for c in group],
                    "count": len(group),
                })

        return duplicates

    # ══════════════════════════════════════════
    # ANALYSE DE CHARGE
    # ══════════════════════════════════════════

    def _analyze_person_load(
        self,
        crm_data: dict[str, Any],
        email_data: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        """Analyse la distribution de charge par personne."""
        deals = crm_data.get("deals", [])
        open_deals = [d for d in deals if not d.get("is_closed", False)]

        person_deals: dict[str, int] = defaultdict(int)
        person_value: dict[str, float] = defaultdict(float)

        for d in open_deals:
            owner = d.get("owner_name", "Unknown") or "Unknown"
            person_deals[owner] += 1
            person_value[owner] += d.get("amount", 0) or 0

        total_deals = len(open_deals)
        if total_deals == 0:
            return {"persons": [], "most_loaded": None, "concentration_risk": "low"}

        persons = []
        for name, count in person_deals.items():
            pct = count / total_deals
            persons.append({
                "name": name,
                "deals_count": count,
                "deals_value": round(person_value[name], 2),
                "load_pct": round(pct, 3),
            })

        persons.sort(key=lambda p: p["load_pct"], reverse=True)

        most_loaded = persons[0] if persons else None

        # Concentration risk
        top_2_pct = sum(p["load_pct"] for p in persons[:2])
        if top_2_pct >= 0.7:
            risk = "high"
        elif top_2_pct >= 0.5:
            risk = "medium"
        else:
            risk = "low"

        return {
            "persons": persons,
            "most_loaded": most_loaded,
            "concentration_risk": risk,
        }

    # ══════════════════════════════════════════
    # VALIDATION & CONFIDENCE
    # ══════════════════════════════════════════

    def _validate_input(self, data: dict[str, Any]) -> list[str]:
        warnings = []
        crm = data.get("crm", {})
        deals = crm.get("deals", [])

        if not deals:
            raise InsufficientDataError(
                agent_name=self.AGENT_NAME,
                detail="No deals data for process analysis.",
            )
        if len(deals) < 10:
            warnings.append(
                f"Only {len(deals)} deals. Process patterns may not be reliable."
            )
        if not data.get("email"):
            warnings.append("No email data. Waste detection will be limited.")
        return warnings

    def _calculate_confidence(
        self,
        input_data: dict[str, Any],
        output_data: dict[str, Any],
    ) -> float:
        crm = input_data.get("crm", {})
        deals = crm.get("deals", [])
        score = 0.4

        if len(deals) >= 30:
            score += 0.3
        elif len(deals) >= 15:
            score += 0.15

        if input_data.get("email"):
            score += 0.2

        stages = crm.get("stages", {})
        if len(stages) >= 3:
            score += 0.1

        return min(1.0, score)
