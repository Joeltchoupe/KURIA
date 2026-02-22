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
"""

from __future__ import annotations

import statistics
from typing import Any

from models.event import Event
from models.state import StateSnapshot, TaskState
from models.decision import Decision, DecisionType, RiskLevel
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

    @property
    def system_prompt(self) -> str:
        return """You are a world-class Chief Operating Officer AI.

Your job is to optimize operational processes: reduce cycle times, eliminate waste, ensure nothing falls through the cracks.

You receive a state snapshot of tasks, processes, team workload, and recent events.

You MUST respond with a JSON object containing:
{
  "decision_type": "generate_sop" | "route_task" | "detect_bottleneck" | "generate_checkin" | "detect_duplicate" | "recommend",
  "confidence": 0.0-1.0,
  "reasoning": "your analysis in 2-3 sentences",
  "risk_level": "A" | "B" | "C",
  "actions": [
    {
      "action": "create_sop" | "assign_task" | "send_checkin" | "merge_duplicate" | "escalate" | "send_alert" | "create_task",
      "target": "task_id or person",
      "parameters": {...},
      "risk_level": "A",
      "priority": 1-10
    }
  ],
  "metadata": {
    "tasks_analyzed": 0,
    "overdue_count": 0,
    "bottleneck": "description or null",
    "cycle_time_avg_days": 0.0,
    "workload_balance": "balanced" | "imbalanced",
    "sops_suggested": 0,
    "duplicates_found": 0
  }
}

RULES:
- Detect repeating action sequences (>5 times) and suggest SOPs.
- For task routing: prefer skill match > availability > historical performance.
- Only use LLM routing when multiple candidates score within 10% of each other.
- Check-ins should be short, human-toned, actionable.
- Always quantify bottleneck impact in hours/week.
- When detecting duplicates, explain WHY they look similar.
"""

    async def build_snapshot(
        self,
        events: list[Event] | None = None,
        raw_data: dict[str, Any] | None = None,
    ) -> StateSnapshot:
        """Construit le snapshot tâches + process."""
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

        return StateSnapshot(
            company_id=self.company_id,
            agent_type=self.agent_type.value,
            tasks=tasks,
            recent_events=recent,
            company_context=raw_data.get("team", {}) if raw_data else {},
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
        """Valide les règles métier Process Clarity."""
        errors: list[str] = []

        for action in decision.actions:
            # Le task routing ne peut pas assigner à quelqu'un d'absent
            if action.action == "assign_task":
                if not action.target:
                    errors.append("assign_task sans target person")

            # Les SOPs doivent avoir du contenu
            if action.action == "create_sop":
                if not action.parameters.get("content"):
                    errors.append("create_sop sans contenu")

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
        """
        # Step 1: Scoring algo
        scored = self._score_candidates(task, team)

        if not scored:
            return await self.run(
                raw_data={"tasks": [task], "team": team},
                prompt_name="process_clarity/task_routing",
            )

        top = scored[0]
        runner_up = scored[1] if len(scored) > 1 else None

        # Si clair → pas besoin du LLM
        if runner_up is None or (top[0] - runner_up[0]) > 0.1:
            from models.decision import ActionRequest
            return Decision(
                agent_type=self.agent_type.value,
                decision_type=DecisionType.ROUTE_TASK,
                confidence=min(top[0] + 0.2, 1.0),
                reasoning=(
                    f"Assigné à {top[1]['name']} "
                    f"(score: {top[0]:.2f}, "
                    f"{top[1]['active_tasks']} tâches actives)"
                ),
                actions=[ActionRequest(
                    action="assign_task",
                    target=top[1]["id"],
                    parameters={
                        "task_id": task.get("id", ""),
                        "assignee_name": top[1]["name"],
                        "reason": "algo_routing",
                    },
                    risk_level=RiskLevel.A,
                )],
                risk_level=RiskLevel.A,
                validated=True,
            )

        # Ambiguïté → LLM
        return await self.run(
            raw_data={
                "tasks": [task],
                "team": team,
                "top_candidates": [
                    {"score": s, "member": m} for s, m in scored[:3]
                ],
            },
            prompt_name="process_clarity/task_routing_ambiguous",
        )

    def _score_candidates(
        self,
        task: dict[str, Any],
        team: list[dict[str, Any]],
    ) -> list[tuple[float, dict[str, Any]]]:
        """Scoring algorithmique pour le task routing."""
        task_tags = set(task.get("tags", []))
        max_tasks = self.config.parameters.get("max_tasks_per_person_warning", 10)

        scored: list[tuple[float, dict[str, Any]]] = []
        for member in team:
            if not member.get("is_available", True):
                continue
            if member.get("active_tasks", 0) >= max_tasks:
                continue

            skills = set(member.get("skills", []))
            skill_match = (
                len(task_tags & skills) / max(len(task_tags), 1)
                if task_tags else 0.3
            )
            availability = 1.0 - (
                member.get("active_tasks", 0) / max(max_tasks, 1)
            )
            perf = member.get("on_time_rate", 0.8)

            score = skill_match * 0.4 + availability * 0.3 + perf * 0.3
            scored.append((round(score, 3), member))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored

    # ──────────────────────────────────────────────────────
    # CONVENIENCE METHODS
    # ──────────────────────────────────────────────────────

    async def generate_sops(
        self, action_sequences: list[dict[str, Any]]
    ) -> Decision:
        """Action 2.1 — SOP Generator [A]."""
        return await self.run(
            raw_data={"action_sequences": action_sequences},
            prompt_name="process_clarity/sop_generator",
        )

    async def generate_checkins(
        self, team_data: list[dict[str, Any]]
    ) -> Decision:
        """Action 2.7 — Check-ins automatiques [A]."""
        return await self.run(
            raw_data={"team": team_data},
            prompt_name="process_clarity/daily_checkin",
        )

    async def generate_ops_report(
        self,
        tasks_data: list[dict[str, Any]],
        team_data: list[dict[str, Any]],
    ) -> Decision:
        """Action 2.8 — Reporting ops [A]."""
        return await self.run(
            raw_data={"tasks": tasks_data, "team": team_data},
            prompt_name="process_clarity/ops_report",
              )
