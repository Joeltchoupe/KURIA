"""
Agent 2 : Process Clarity — LLM-first.

KPI : Cycle time des process critiques (jours).

Actions :
  2.1 — SOP Generator [A]
  2.3 — Task Routing [A] (hybride : algo + LLM si ambiguïté)
  2.7 — Check-ins Automatiques [A]
  2.8 — Reporting Ops [A]

80% automation logic, 20% IA.
Le LLM intervient pour : SOP génération, synthèse, ambiguïté.

System prompt : prompts/system/process_clarity.txt
User prompts : prompts/process_clarity/*.txt
"""

from __future__ import annotations

from typing import Any

from models.event import Event
from models.state import StateSnapshot, TaskState
from models.decision import Decision, DecisionType, ActionRequest, RiskLevel
from models.agent_config import AgentConfig, AgentType
from agents.base import BaseAgent


class ProcessClarityAgent(BaseAgent):
    """
    Process Clarity — Le meilleur COO du monde.

    Son système produit le reporting.
    Il intervient uniquement sur les anomalies.
    """

    @property
    def agent_type(self) -> AgentType:
        return AgentType.PROCESS_CLARITY

    def _custom_prompt_variables(self) -> dict[str, Any]:
        """Variables spécifiques injectées dans les prompts Process Clarity."""
        params = self.config.parameters
        return {
            "sop_min_occurrences": params.get("sop_min_occurrences", 5),
            "max_tasks_per_person_warning": params.get(
                "max_tasks_per_person_warning", 10
            ),
            "hourly_rate_estimate": params.get("hourly_rate_estimate", 50),
            "deadline_warning_hours": params.get(
                "deadline_warning_hours", 48
            ),
            "deadline_escalation_days": params.get(
                "deadline_escalation_days", 3
            ),
            "checkin_time": params.get("checkin_time", "09:00"),
            "duplicate_auto_merge_threshold": params.get(
                "duplicate_auto_merge_threshold", 0.95
            ),
            "duplicate_confirm_threshold": params.get(
                "duplicate_confirm_threshold", 0.70
            ),
            "ping_pong_threshold": params.get("ping_pong_threshold", 5),
            "bottleneck_time_multiplier": params.get(
                "bottleneck_time_multiplier", 2.0
            ),
            "concentration_threshold": params.get(
                "concentration_threshold", 0.6
            ),
        }

    async def build_snapshot(
        self,
        events: list[Event] | None = None,
        raw_data: dict[str, Any] | None = None,
    ) -> StateSnapshot:
        """Construit le snapshot tâches + process + team."""
        tasks: list[TaskState] = []

        if raw_data and "tasks" in raw_data:
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

        recent = []
        if events:
            recent = [e.to_snapshot_dict() for e in events[:50]]

        # Contexte team + action sequences
        company_context: dict[str, Any] = {}
        if raw_data:
            if "team" in raw_data:
                company_context["team"] = raw_data["team"]
            if "action_sequences" in raw_data:
                company_context["action_sequences"] = raw_data[
                    "action_sequences"
                ]

        return StateSnapshot(
            company_id=self.company_id,
            agent_type=self.agent_type.value,
            tasks=tasks,
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

    def validate_decision(self, decision: Decision) -> list[str]:
        """
        Valide les règles métier Process Clarity.

        Vérifications :
          - assign_task doit avoir un target
          - create_sop doit avoir du contenu
          - merge_duplicate doit spécifier quoi garder
        """
        errors: list[str] = []

        for action in decision.actions:
            if action.action == "assign_task":
                if not action.target:
                    errors.append(
                        "assign_task sans target (personne à assigner)"
                    )
                task_id = action.parameters.get("task_id")
                if not task_id:
                    errors.append("assign_task sans task_id dans parameters")

            if action.action == "create_sop":
                if not action.parameters.get("content") and not action.parameters.get("steps"):
                    errors.append("create_sop sans contenu ni steps")
                if not action.parameters.get("title"):
                    errors.append("create_sop sans titre")

            if action.action == "merge_duplicate":
                if not action.parameters.get("keep"):
                    errors.append(
                        "merge_duplicate sans spécifier quel item garder"
                    )

            if action.action == "escalate":
                if not action.parameters.get("reason"):
                    errors.append("escalate sans raison")

        return errors

    def _default_decision_type(self) -> DecisionType:
        return DecisionType.DETECT_BOTTLENECK

    # ──────────────────────────────────────────────────────
    # TASK ROUTING — Hybride algo + LLM
    # ──────────────────────────────────────────────────────

    async def route_task(
        self,
        task: dict[str, Any],
        team: list[dict[str, Any]],
    ) -> Decision:
        """
        Action 2.3 — Task Routing [A].

        Hybride :
          1. Scoring algorithmique (skill + dispo + perf)
          2. Si ambiguïté (top 2 dans les 10%) → LLM tranche
          3. Si aucun candidat viable → LLM propose une alternative
        """
        # Step 1: Scoring algo
        scored = self._score_candidates(task, team)

        # Aucun candidat → LLM
        if not scored:
            return await self.run(
                raw_data={"tasks": [task], "team": team},
                prompt_name="process_clarity/task_routing_ambiguous",
            )

        top = scored[0]
        runner_up = scored[1] if len(scored) > 1 else None

        # Clair → pas besoin du LLM
        if runner_up is None or (top[0] - runner_up[0]) > 0.1:
            return Decision(
                agent_type=self.agent_type.value,
                decision_type=DecisionType.ROUTE_TASK,
                confidence=min(top[0] + 0.2, 1.0),
                reasoning=(
                    f"Assigné à {top[1]['name']} "
                    f"(score: {top[0]:.2f}, "
                    f"{top[1].get('active_tasks', 0)} tâches actives). "
                    f"Routing algorithmique — candidat clairement meilleur."
                ),
                actions=[ActionRequest(
                    action="assign_task",
                    target=top[1].get("id", top[1]["name"]),
                    parameters={
                        "task_id": task.get("id", ""),
                        "task_title": task.get("title", ""),
                        "assignee_name": top[1]["name"],
                        "reason": "algo_routing — best score",
                        "score": top[0],
                    },
                    risk_level=RiskLevel.A,
                    priority=3,
                )],
                risk_level=RiskLevel.A,
                validated=True,
            )

        # Ambiguïté → LLM tranche
        return await self.run(
            raw_data={
                "tasks": [task],
                "team": team,
            },
            prompt_name="process_clarity/task_routing",
            prompt_variables={
                "top_candidates": [
                    {
                        "score": round(s, 3),
                        "name": m.get("name", ""),
                        "id": m.get("id", ""),
                        "active_tasks": m.get("active_tasks", 0),
                        "skills": m.get("skills", []),
                        "on_time_rate": m.get("on_time_rate", 0.8),
                    }
                    for s, m in scored[:3]
                ],
            },
        )

    def _score_candidates(
        self,
        task: dict[str, Any],
        team: list[dict[str, Any]],
    ) -> list[tuple[float, dict[str, Any]]]:
        """
        Scoring algorithmique pour le task routing.

        Score = skill_match × 0.4 + availability × 0.3 + perf × 0.3

        Pur Python, pas de LLM.
        """
        task_tags = set(task.get("tags", []))
        max_tasks = self.config.parameters.get(
            "max_tasks_per_person_warning", 10
        )

        scored: list[tuple[float, dict[str, Any]]] = []

        for member in team:
            # Filtrer les indisponibles et surchargés
            if not member.get("is_available", True):
                continue
            if member.get("active_tasks", 0) >= max_tasks:
                continue

            # Skill match
            skills = set(member.get("skills", []))
            if task_tags:
                skill_match = len(task_tags & skills) / len(task_tags)
            else:
                skill_match = 0.3  # Score neutre si pas de tags

            # Availability
            active = member.get("active_tasks", 0)
            availability = 1.0 - (active / max(max_tasks, 1))

            # Performance
            perf = member.get("on_time_rate", 0.8)

            score = skill_match * 0.4 + availability * 0.3 + perf * 0.3
            scored.append((round(score, 3), member))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored

    # ──────────────────────────────────────────────────────
    # CONVENIENCE METHODS — Raccourcis par action
    # ──────────────────────────────────────────────────────

    async def generate_sops(
        self, action_sequences: list[dict[str, Any]]
    ) -> Decision:
        """
        Action 2.1 — SOP Generator [A].

        Détecte les séquences récurrentes et génère des SOPs.
        Publie automatiquement dans Notion.
        """
        return await self.run(
            raw_data={"action_sequences": action_sequences},
            prompt_name="process_clarity/sop_generator",
        )

    async def generate_checkins(
        self,
        tasks_data: list[dict[str, Any]],
        team_data: list[dict[str, Any]],
    ) -> Decision:
        """
        Action 2.7 — Check-ins automatiques [A].

        Style Basecamp : court, humain, actionable.
        Remplace les stand-up meetings quotidiens.
        """
        return await self.run(
            raw_data={"tasks": tasks_data, "team": team_data},
            prompt_name="process_clarity/daily_checkin",
        )

    async def generate_ops_report(
        self,
        tasks_data: list[dict[str, Any]],
        team_data: list[dict[str, Any]],
    ) -> Decision:
        """
        Action 2.8 — Reporting ops automatique [A].

        Rapport hebdomadaire complet : cycle times, bottleneck,
        workload, doublons, SOPs, recommandation clé.
        """
        return await self.run(
            raw_data={"tasks": tasks_data, "team": team_data},
            prompt_name="process_clarity/ops_report",
        )

    async def detect_duplicates(
        self, tasks_data: list[dict[str, Any]]
    ) -> Decision:
        """
        Action 2.6 — Détection doublons [A].

        Scanne les tâches pour trouver le travail en double.
        """
        return await self.run(
            raw_data={"tasks": tasks_data},
            prompt_name="process_clarity/ops_report",
            prompt_variables={"focus": "duplicates"},
      )
