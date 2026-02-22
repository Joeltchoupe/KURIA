"""
AGENT 2 : PROCESS CLARITY
KPI : Cycle time des process critiques (jours)

Philosophie : Le meilleur COO du monde.
Il ne demande pas de reporting — son système LE PRODUIT.
Il intervient uniquement sur les anomalies.

8 actions, toutes autonomes :
  2.1 — Génération automatique de SOPs
  2.2 — Documentation des décisions
  2.3 — Routing intelligent des tâches
  2.4 — Suivi automatique des deadlines
  2.5 — Automatisation des handoffs
  2.6 — Suppression du travail en double
  2.7 — Check-ins automatiques
  2.8 — Reporting ops automatique

Inspirations :
  - Toyota Production System (détection + correction auto)
  - Basecamp (check-ins async, zéro réunion)
  - Zapier/Make (handoffs automatiques)
"""

from __future__ import annotations

import hashlib
import statistics
from datetime import datetime, timedelta, date
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from models.agent_config import AgentConfig, AgentType, Threshold
from models.friction import (
    Friction,
    FrictionType,
    FrictionSeverity,
    FrictionSource,
    FrictionStatus,
)
from models.metrics import (
    MetricResult,
    MetricTrend,
    HealthStatus,
    TrendDirection,
)


# ══════════════════════════════════════════════════════════════
# ENUMS
# ══════════════════════════════════════════════════════════════


class ProcessStatus(str, Enum):
    """Statut d'un process détecté."""
    DETECTED = "detected"
    DOCUMENTED = "documented"
    ACTIVE = "active"
    STALE = "stale"
    ARCHIVED = "archived"


class TaskStatus(str, Enum):
    """Statut d'une tâche dans le système."""
    UNASSIGNED = "unassigned"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    OVERDUE = "overdue"
    ESCALATED = "escalated"


class DecisionType(str, Enum):
    """Type de décision détectée."""
    STAGE_CHANGE = "stage_change"
    TASK_ASSIGNMENT = "task_assignment"
    INVOICE_VALIDATION = "invoice_validation"
    EMAIL_SENT = "email_sent"
    APPROVAL = "approval"
    ESCALATION = "escalation"
    OTHER = "other"


class HandoffStatus(str, Enum):
    """Statut d'un handoff entre personnes."""
    PENDING = "pending"
    TRANSFERRED = "transferred"
    ACKNOWLEDGED = "acknowledged"
    COMPLETED = "completed"
    LOST = "lost"


class DuplicateConfidence(str, Enum):
    """Niveau de confiance pour la détection de doublons."""
    HIGH = "high"          # > 95% → merge auto
    MEDIUM = "medium"      # 70-95% → demande confirmation
    LOW = "low"            # < 70% → ignore


class SOPStatus(str, Enum):
    """Statut d'une SOP."""
    DRAFT = "draft"
    PUBLISHED = "published"
    UPDATED = "updated"
    DEPRECATED = "deprecated"


# ══════════════════════════════════════════════════════════════
# MODÈLES DE DONNÉES
# ══════════════════════════════════════════════════════════════


class ProcessStep(BaseModel):
    """Une étape dans un process détecté."""
    step_number: int = Field(..., ge=1)
    name: str
    owner: Optional[str] = None
    tool_used: Optional[str] = None
    avg_duration_hours: Optional[float] = Field(None, ge=0)
    expected_output: Optional[str] = None


class DetectedProcess(BaseModel):
    """Process détecté par analyse des séquences récurrentes."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    trigger: str = Field(..., description="Ce qui déclenche ce process")
    steps: list[ProcessStep] = Field(default_factory=list)
    tools_involved: list[str] = Field(default_factory=list)
    occurrence_count: int = Field(..., ge=1)
    avg_cycle_time_hours: Optional[float] = Field(None, ge=0)
    status: ProcessStatus = ProcessStatus.DETECTED
    first_seen: datetime = Field(default_factory=datetime.utcnow)
    last_seen: datetime = Field(default_factory=datetime.utcnow)
    owner: Optional[str] = None


class SOP(BaseModel):
    """Standard Operating Procedure générée automatiquement."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    process_id: str
    process_name: str
    trigger_description: str
    steps: list[ProcessStep]
    tools_used: list[str]
    avg_total_time_hours: float = Field(..., ge=0)
    expected_output: str
    status: SOPStatus = SOPStatus.DRAFT
    version: int = Field(default=1, ge=1)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    published_to: Optional[str] = None


class DecisionLog(BaseModel):
    """Entrée dans le journal des décisions."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    decision_type: DecisionType
    who: str
    what: str
    when: datetime = Field(default_factory=datetime.utcnow)
    based_on: Optional[str] = Field(
        None, description="Informations sur lesquelles la décision est basée"
    )
    impact: Optional[str] = None
    source_tool: Optional[str] = None
    source_event_id: Optional[str] = None


class TaskRecord(BaseModel):
    """Tâche suivie par l'agent."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    title: str
    description: Optional[str] = None
    assigned_to: Optional[str] = None
    assigned_by: Optional[str] = None
    assignment_reason: Optional[str] = None
    status: TaskStatus = TaskStatus.UNASSIGNED
    deadline: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    process_id: Optional[str] = None
    next_task_id: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    source_tool: Optional[str] = None


class TeamMember(BaseModel):
    """Membre de l'équipe avec sa charge et ses compétences."""
    id: str
    name: str
    email: Optional[str] = None
    active_tasks: int = Field(default=0, ge=0)
    completed_tasks_30d: int = Field(default=0, ge=0)
    avg_completion_time_hours: Optional[float] = Field(None, ge=0)
    skills: list[str] = Field(default_factory=list)
    on_time_rate: float = Field(default=1.0, ge=0, le=1)
    is_available: bool = True


class DuplicateDetection(BaseModel):
    """Doublon détecté entre deux éléments."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    item_a_id: str
    item_a_title: str
    item_a_owner: Optional[str] = None
    item_b_id: str
    item_b_title: str
    item_b_owner: Optional[str] = None
    confidence: DuplicateConfidence
    confidence_score: float = Field(..., ge=0, le=1)
    item_type: str = "task"
    detected_at: datetime = Field(default_factory=datetime.utcnow)
    resolved: bool = False
    resolution: Optional[str] = None


class Handoff(BaseModel):
    """Transfert de contexte entre deux personnes."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    from_task_id: str
    to_task_id: str
    from_person: str
    to_person: str
    context_summary: str
    attachments: list[str] = Field(default_factory=list)
    status: HandoffStatus = HandoffStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.utcnow)
    acknowledged_at: Optional[datetime] = None


class CheckInMessage(BaseModel):
    """Message de check-in envoyé à un membre."""
    member_id: str
    member_name: str
    tasks_today: list[TaskRecord] = Field(default_factory=list)
    tasks_overdue: list[TaskRecord] = Field(default_factory=list)
    tasks_completed_yesterday: list[TaskRecord] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class BlockageReport(BaseModel):
    """Rapport de blocage signalé par un membre."""
    member_id: str
    member_name: str
    task_id: str
    task_title: str
    reported_at: datetime = Field(default_factory=datetime.utcnow)
    manager_notified: bool = False
    unblock_task_created: bool = False


class OpsReport(BaseModel):
    """Rapport opérationnel hebdomadaire/mensuel."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    period_start: date
    period_end: date
    cycle_times: dict[str, float] = Field(
        default_factory=dict,
        description="Process name → avg cycle time in hours",
    )
    on_time_rate: float = Field(..., ge=0, le=1)
    on_time_rate_by_person: dict[str, float] = Field(default_factory=dict)
    bottleneck: Optional[str] = None
    bottleneck_cost_estimate: Optional[float] = None
    tasks_completed: int = Field(default=0, ge=0)
    tasks_created: int = Field(default=0, ge=0)
    backlog_trend: TrendDirection = TrendDirection.STABLE
    workload_by_person: dict[str, int] = Field(default_factory=dict)
    duplicates_found: int = Field(default=0, ge=0)
    sops_created: int = Field(default=0, ge=0)
    sops_updated: int = Field(default=0, ge=0)
    decisions_logged: int = Field(default=0, ge=0)
    key_recommendation: Optional[str] = None
    generated_at: datetime = Field(default_factory=datetime.utcnow)


# ══════════════════════════════════════════════════════════════
# CONFIGURATION PAR DÉFAUT
# ══════════════════════════════════════════════════════════════


def default_process_clarity_config() -> AgentConfig:
    """Configuration par défaut de l'agent Process Clarity."""
    return AgentConfig(
        agent_type=AgentType.PROCESS_CLARITY,
        enabled=True,
        thresholds=[
            Threshold(
                metric_name="cycle_time_days",
                warning_value=7.0,
                critical_value=14.0,
                direction="above",
            ),
            Threshold(
                metric_name="on_time_delivery_rate",
                warning_value=0.80,
                critical_value=0.60,
                direction="below",
            ),
            Threshold(
                metric_name="overdue_tasks_count",
                warning_value=5.0,
                critical_value=15.0,
                direction="above",
            ),
            Threshold(
                metric_name="duplicate_rate",
                warning_value=0.05,
                critical_value=0.15,
                direction="above",
            ),
            Threshold(
                metric_name="sop_occurrence_threshold",
                warning_value=5.0,
                critical_value=5.0,
                direction="above",
            ),
            Threshold(
                metric_name="workload_imbalance_ratio",
                warning_value=2.0,
                critical_value=3.0,
                direction="above",
            ),
        ],
        schedule_cron="0 6 * * 1",
        data_sources=["hubspot", "gmail", "project_management"],
        parameters={
            "sop_min_occurrences": 5,
            "deadline_warning_hours": 48,
            "deadline_escalation_days": 3,
            "checkin_time": "09:00",
            "duplicate_auto_merge_threshold": 0.95,
            "duplicate_confirm_threshold": 0.70,
            "max_tasks_per_person_warning": 10,
            "handoff_context_max_words": 200,
        },
    )


# ══════════════════════════════════════════════════════════════
# AGENT PROCESS CLARITY
# ══════════════════════════════════════════════════════════════


class ProcessClarityAgent:
    """
    Agent 2 : Process Clarity.

    Le meilleur COO du monde — son système produit le reporting,
    il intervient uniquement sur les anomalies.

    KPI principal : cycle time des process critiques (jours).
    """

    def __init__(
        self,
        config: AgentConfig | None = None,
        company_id: str | None = None,
    ) -> None:
        self.config = config or default_process_clarity_config()
        self.company_id = company_id

        # State stores
        self._processes: dict[str, DetectedProcess] = {}
        self._sops: dict[str, SOP] = {}
        self._decisions: list[DecisionLog] = []
        self._tasks: dict[str, TaskRecord] = {}
        self._team: dict[str, TeamMember] = {}
        self._handoffs: list[Handoff] = []
        self._duplicates: list[DuplicateDetection] = []
        self._blockages: list[BlockageReport] = []
        self._checkins: list[CheckInMessage] = []
        self._reports: list[OpsReport] = []

        # Sequence detection buffer
        self._action_sequences: list[dict[str, Any]] = []

    @property
    def _params(self) -> dict[str, Any]:
        """Raccourci vers les paramètres de configuration."""
        return self.config.parameters

    # ──────────────────────────────────────────────────────
    # ACTION 2.1 — GÉNÉRATION AUTOMATIQUE DE SOPs
    # Trigger : séquence d'actions détectée > 5 occurrences
    # Fréquence : hebdomadaire
    # ──────────────────────────────────────────────────────

    def ingest_action(self, action: dict[str, Any]) -> None:
        """
        Ingère une action brute depuis les outils connectés.

        Chaque action doit contenir au minimum :
          - action_type: str
          - tool: str
          - timestamp: str (ISO)
          - actor: str (optionnel)
          - metadata: dict (optionnel)
        """
        required = {"action_type", "tool", "timestamp"}
        if not required.issubset(action.keys()):
            raise ValueError(
                f"Action manque les champs requis : {required - action.keys()}"
            )
        self._action_sequences.append(action)

    def detect_processes(self) -> list[DetectedProcess]:
        """
        ACTION 2.1 — Détecte les séquences récurrentes.

        Analyse le buffer d'actions pour trouver des patterns
        qui se répètent > sop_min_occurrences fois.

        Returns:
            Liste des nouveaux process détectés.
        """
        min_occ = self._params.get("sop_min_occurrences", 5)
        sequences = self._extract_sequences()
        new_processes: list[DetectedProcess] = []

        for seq_hash, seq_data in sequences.items():
            if seq_data["count"] < min_occ:
                continue

            if seq_hash in self._processes:
                existing = self._processes[seq_hash]
                existing.occurrence_count = seq_data["count"]
                existing.last_seen = datetime.utcnow()
                existing.avg_cycle_time_hours = seq_data.get("avg_duration")
                continue

            steps = [
                ProcessStep(
                    step_number=i + 1,
                    name=step["action_type"],
                    tool_used=step.get("tool"),
                    owner=step.get("actor"),
                    avg_duration_hours=step.get("avg_duration"),
                )
                for i, step in enumerate(seq_data["steps"])
            ]

            tools = list({s.tool_used for s in steps if s.tool_used})

            process = DetectedProcess(
                id=seq_hash,
                name=self._generate_process_name(seq_data["steps"]),
                trigger=f"Déclenché par : {seq_data['steps'][0]['action_type']}",
                steps=steps,
                tools_involved=tools,
                occurrence_count=seq_data["count"],
                avg_cycle_time_hours=seq_data.get("avg_duration"),
                status=ProcessStatus.DETECTED,
            )

            self._processes[seq_hash] = process
            new_processes.append(process)

        return new_processes

    def generate_sops(self) -> list[SOP]:
        """
        ACTION 2.1 — Génère des SOPs pour les process détectés.

        Pour chaque process détecté et non encore documenté,
        génère une SOP structurée.

        Returns:
            Liste des SOPs nouvellement créées.
        """
        new_sops: list[SOP] = []

        for proc_id, process in self._processes.items():
            if process.status != ProcessStatus.DETECTED:
                continue

            if proc_id in self._sops:
                existing_sop = self._sops[proc_id]
                if self._process_changed(process, existing_sop):
                    updated = self._update_sop(existing_sop, process)
                    new_sops.append(updated)
                continue

            sop = SOP(
                process_id=proc_id,
                process_name=process.name,
                trigger_description=process.trigger,
                steps=process.steps,
                tools_used=process.tools_involved,
                avg_total_time_hours=process.avg_cycle_time_hours or 0.0,
                expected_output=self._infer_expected_output(process),
                status=SOPStatus.DRAFT,
            )

            self._sops[proc_id] = sop
            process.status = ProcessStatus.DOCUMENTED
            new_sops.append(sop)

        return new_sops

    def publish_sop(self, sop_id: str, target: str = "notion") -> SOP:
        """Publie une SOP vers un outil externe."""
        sop = self._find_sop(sop_id)
        sop.status = SOPStatus.PUBLISHED
        sop.published_to = target
        sop.updated_at = datetime.utcnow()
        return sop

    def _extract_sequences(self) -> dict[str, dict[str, Any]]:
        """
        Extrait les séquences récurrentes depuis le buffer d'actions.

        Sliding window de 3-7 actions, groupé par actor.
        """
        sequences: dict[str, dict[str, Any]] = {}

        if len(self._action_sequences) < 3:
            return sequences

        by_actor: dict[str, list[dict]] = {}
        for action in self._action_sequences:
            actor = action.get("actor", "unknown")
            by_actor.setdefault(actor, []).append(action)

        for actor, actions in by_actor.items():
            sorted_actions = sorted(actions, key=lambda a: a["timestamp"])

            for window_size in range(3, min(8, len(sorted_actions) + 1)):
                for i in range(len(sorted_actions) - window_size + 1):
                    window = sorted_actions[i : i + window_size]
                    sig = self._sequence_signature(window)

                    if sig not in sequences:
                        sequences[sig] = {
                            "steps": window,
                            "count": 0,
                            "durations": [],
                        }

                    sequences[sig]["count"] += 1

                    try:
                        t_start = datetime.fromisoformat(window[0]["timestamp"])
                        t_end = datetime.fromisoformat(window[-1]["timestamp"])
                        duration = (t_end - t_start).total_seconds() / 3600
                        sequences[sig]["durations"].append(duration)
                    except (ValueError, KeyError):
                        pass

        for sig, data in sequences.items():
            if data["durations"]:
                data["avg_duration"] = statistics.mean(data["durations"])

        return sequences

    def _sequence_signature(self, actions: list[dict]) -> str:
        """Hash déterministe d'une séquence d'actions."""
        sig_parts = [
            f"{a['action_type']}:{a.get('tool', '?')}" for a in actions
        ]
        sig_str = "→".join(sig_parts)
        return hashlib.sha256(sig_str.encode()).hexdigest()[:16]

    def _generate_process_name(self, steps: list[dict]) -> str:
        """Génère un nom lisible pour un process détecté."""
        if not steps:
            return "Process inconnu"
        first = steps[0].get("action_type", "start")
        last = steps[-1].get("action_type", "end")
        return f"{first} → {last}"

    def _infer_expected_output(self, process: DetectedProcess) -> str:
        """Infère l'output attendu d'un process depuis ses étapes."""
        if not process.steps:
            return "Non déterminé"
        last_step = process.steps[-1]
        return f"Complétion de : {last_step.name}"

    def _process_changed(self, process: DetectedProcess, sop: SOP) -> bool:
        """Vérifie si un process a changé depuis la dernière SOP."""
        if len(process.steps) != len(sop.steps):
            return True
        for p_step, s_step in zip(process.steps, sop.steps):
            if p_step.name != s_step.name:
                return True
            if p_step.tool_used != s_step.tool_used:
                return True
        return False

    def _update_sop(self, sop: SOP, process: DetectedProcess) -> SOP:
        """Met à jour une SOP existante quand le process change."""
        sop.steps = process.steps
        sop.tools_used = process.tools_involved
        sop.avg_total_time_hours = process.avg_cycle_time_hours or 0.0
        sop.version += 1
        sop.status = SOPStatus.UPDATED
        sop.updated_at = datetime.utcnow()
        return sop

    def _find_sop(self, sop_id: str) -> SOP:
        """Trouve une SOP par ID ou lève une erreur."""
        for sop in self._sops.values():
            if sop.id == sop_id:
                return sop
        raise ValueError(f"SOP non trouvée : {sop_id}")

    # ──────────────────────────────────────────────────────
    # ACTION 2.2 — DOCUMENTATION DES DÉCISIONS
    # Trigger : chaque décision détectée
    # Fréquence : continu
    # ──────────────────────────────────────────────────────

    def log_decision(
        self,
        decision_type: DecisionType,
        who: str,
        what: str,
        based_on: str | None = None,
        impact: str | None = None,
        source_tool: str | None = None,
        source_event_id: str | None = None,
    ) -> DecisionLog:
        """
        ACTION 2.2 — Logge une décision détectée.

        Appelé automatiquement quand :
          - Deal avancé à un nouveau stage
          - Tâche assignée
          - Facture validée
          - Email important envoyé
        """
        entry = DecisionLog(
            decision_type=decision_type,
            who=who,
            what=what,
            based_on=based_on,
            impact=impact,
            source_tool=source_tool,
            source_event_id=source_event_id,
        )
        self._decisions.append(entry)
        return entry

    def detect_decisions_from_events(
        self, events: list[dict[str, Any]]
    ) -> list[DecisionLog]:
        """
        ACTION 2.2 — Détecte et logge les décisions depuis des événements bruts.

        Analyse les événements des connecteurs et crée des DecisionLog
        pour chaque décision identifiée.
        """
        logged: list[DecisionLog] = []

        for event in events:
            decision_type = self._classify_decision(event)
            if decision_type is None:
                continue

            entry = self.log_decision(
                decision_type=decision_type,
                who=event.get("actor", "unknown"),
                what=self._describe_decision(event),
                based_on=event.get("context"),
                source_tool=event.get("source", "unknown"),
                source_event_id=event.get("event_id"),
            )
            logged.append(entry)

        return logged

    def get_decision_summary(
        self,
        period_days: int = 30,
    ) -> dict[str, Any]:
        """
        Agrégation mensuelle des décisions.

        Returns:
            Résumé : total, par personne, par type.
        """
        cutoff = datetime.utcnow() - timedelta(days=period_days)
        recent = [d for d in self._decisions if d.when >= cutoff]

        by_person: dict[str, int] = {}
        by_type: dict[str, int] = {}

        for d in recent:
            by_person[d.who] = by_person.get(d.who, 0) + 1
            by_type[d.decision_type.value] = (
                by_type.get(d.decision_type.value, 0) + 1
            )

        total = len(recent)
        person_pcts = {
            person: round(count / total * 100, 1) if total > 0 else 0
            for person, count in by_person.items()
        }

        return {
            "period_days": period_days,
            "total_decisions": total,
            "by_person": by_person,
            "by_person_pct": person_pcts,
            "by_type": by_type,
            "top_decision_maker": (
                max(by_person, key=by_person.get) if by_person else None
            ),
        }

    def _classify_decision(self, event: dict[str, Any]) -> DecisionType | None:
        """Classifie un événement brut en type de décision."""
        event_type = event.get("type", "").lower()
        action = event.get("action", "").lower()

        mapping = {
            "stage_change": DecisionType.STAGE_CHANGE,
            "deal_moved": DecisionType.STAGE_CHANGE,
            "pipeline_change": DecisionType.STAGE_CHANGE,
            "task_assigned": DecisionType.TASK_ASSIGNMENT,
            "assignment": DecisionType.TASK_ASSIGNMENT,
            "invoice_approved": DecisionType.INVOICE_VALIDATION,
            "invoice_validated": DecisionType.INVOICE_VALIDATION,
            "payment_approved": DecisionType.INVOICE_VALIDATION,
            "email_sent": DecisionType.EMAIL_SENT,
            "approved": DecisionType.APPROVAL,
            "approval": DecisionType.APPROVAL,
            "escalated": DecisionType.ESCALATION,
            "escalation": DecisionType.ESCALATION,
        }

        for key, dtype in mapping.items():
            if key in event_type or key in action:
                return dtype

        return None

    def _describe_decision(self, event: dict[str, Any]) -> str:
        """Génère une description lisible d'une décision."""
        parts = []
        if event.get("action"):
            parts.append(event["action"])
        if event.get("subject"):
            parts.append(f"sur '{event['subject']}'")
        if event.get("target"):
            parts.append(f"→ {event['target']}")
        return " ".join(parts) if parts else "Décision détectée"

    # ──────────────────────────────────────────────────────
    # ACTION 2.3 — ROUTING INTELLIGENT DES TÂCHES
    # Trigger : tâche créée sans assignation ou mal assignée
    # Fréquence : continu
    # ──────────────────────────────────────────────────────

    def register_team_member(self, member: TeamMember) -> None:
        """Enregistre ou met à jour un membre de l'équipe."""
        self._team[member.id] = member

    def register_task(self, task: TaskRecord) -> TaskRecord:
        """
        Enregistre une tâche et déclenche le routing si nécessaire.

        Si non assignée → routing automatique.
        Si assignée à quelqu'un de surchargé → tag warning.
        """
        self._tasks[task.id] = task

        if task.status == TaskStatus.UNASSIGNED or task.assigned_to is None:
            return self.route_task(task.id)

        overload = self._check_overload(task.assigned_to)
        if overload is not None:
            task.tags.append("overload_warning")

        return task

    def route_task(self, task_id: str) -> TaskRecord:
        """
        ACTION 2.3 — Route intelligemment une tâche.

        Analyse :
          1. Le type de tâche (tags)
          2. La charge actuelle de chaque membre
          3. Les compétences (historique)

        Assigne à la meilleure personne disponible.
        """
        task = self._find_task(task_id)

        if not self._team:
            raise ValueError("Aucun membre d'équipe enregistré pour le routing")

        best = self._find_best_assignee(task)

        if best is None:
            task.tags.append("no_available_assignee")
            return task

        task.assigned_to = best.id
        task.assignment_reason = (
            f"Assigné par Kuria. "
            f"Raison : compétences [{', '.join(best.skills[:3])}] "
            f"+ disponibilité ({best.active_tasks} tâches actives)"
        )
        task.status = TaskStatus.ASSIGNED

        best.active_tasks += 1

        self.log_decision(
            decision_type=DecisionType.TASK_ASSIGNMENT,
            who="kuria_process_clarity",
            what=f"Tâche '{task.title}' assignée à {best.name}",
            based_on=task.assignment_reason,
            source_tool="kuria",
        )

        return task

    def suggest_reassignment(
        self, task_id: str
    ) -> dict[str, Any] | None:
        """
        Suggère une réassignation si la personne est surchargée.

        Returns:
            Suggestion avec alternative, ou None si pas nécessaire.
        """
        task = self._find_task(task_id)

        if task.assigned_to is None:
            return None

        overload = self._check_overload(task.assigned_to)
        if overload is None:
            return None

        member = self._team.get(task.assigned_to)
        alternative = self._find_best_assignee(task, exclude={task.assigned_to})

        if alternative is None:
            return None

        avg_load = self._avg_team_workload()

        return {
            "task_id": task.id,
            "task_title": task.title,
            "current_assignee": member.name if member else task.assigned_to,
            "current_load": member.active_tasks if member else 0,
            "team_avg_load": round(avg_load, 1),
            "suggested_assignee": alternative.name,
            "suggested_load": alternative.active_tasks,
            "reason": (
                f"{member.name if member else 'Assigné'} a "
                f"{member.active_tasks if member else '?'} tâches actives "
                f"(moyenne équipe : {avg_load:.1f}). "
                f"Réassigner à {alternative.name} "
                f"({alternative.active_tasks} tâches) ?"
            ),
        }

    def _find_best_assignee(
        self,
        task: TaskRecord,
        exclude: set[str] | None = None,
    ) -> TeamMember | None:
        """
        Trouve le meilleur assigné pour une tâche.

        Score = skill_match × 0.4 + availability × 0.3 + on_time_rate × 0.3
        """
        exclude = exclude or set()
        max_tasks = self._params.get("max_tasks_per_person_warning", 10)

        candidates = [
            m
            for m in self._team.values()
            if m.is_available
            and m.id not in exclude
            and m.active_tasks < max_tasks
        ]

        if not candidates:
            return None

        task_tags = set(task.tags)

        scored: list[tuple[float, TeamMember]] = []
        for member in candidates:
            skill_match = len(task_tags & set(member.skills)) / max(
                len(task_tags), 1
            )
            availability = 1.0 - (member.active_tasks / max(max_tasks, 1))
            on_time = member.on_time_rate

            score = skill_match * 0.4 + availability * 0.3 + on_time * 0.3
            scored.append((score, member))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]

    def _check_overload(self, member_id: str) -> dict[str, Any] | None:
        """Vérifie si un membre est surchargé."""
        member = self._team.get(member_id)
        if member is None:
            return None

        max_tasks = self._params.get("max_tasks_per_person_warning", 10)
        avg_load = self._avg_team_workload()

        if member.active_tasks <= max_tasks and member.active_tasks <= avg_load * 2:
            return None

        return {
            "member_id": member.id,
            "member_name": member.name,
            "active_tasks": member.active_tasks,
            "max_recommended": max_tasks,
            "team_avg": round(avg_load, 1),
        }

    def _avg_team_workload(self) -> float:
        """Charge moyenne de l'équipe."""
        if not self._team:
            return 0.0
        loads = [m.active_tasks for m in self._team.values() if m.is_available]
        return statistics.mean(loads) if loads else 0.0

    def _find_task(self, task_id: str) -> TaskRecord:
        """Trouve une tâche par ID ou lève une erreur."""
        task = self._tasks.get(task_id)
        if task is None:
            raise ValueError(f"Tâche non trouvée : {task_id}")
            return task

    # ──────────────────────────────────────────────────────
    # ACTION 2.4 — SUIVI AUTOMATIQUE DES DEADLINES
    # Trigger : tâche approchant sa deadline ou en retard
    # Fréquence : quotidien
    # ──────────────────────────────────────────────────────

    def check_deadlines(self) -> dict[str, Any]:
        """
        ACTION 2.4 — Vérifie toutes les tâches vs leurs deadlines.

        3 niveaux :
          - 48h avant → notification au responsable
          - Jour J → alerte + notification manager
          - 3 jours après → escalade automatique

        Returns:
            Résumé : warnings, due_today, overdue, escalated.
        """
        now = datetime.utcnow()
        warning_hours = self._params.get("deadline_warning_hours", 48)
        escalation_days = self._params.get("deadline_escalation_days", 3)

        warnings: list[dict[str, Any]] = []
        due_today: list[dict[str, Any]] = []
        overdue: list[dict[str, Any]] = []
        escalated: list[dict[str, Any]] = []

        for task in self._tasks.values():
            if task.status == TaskStatus.COMPLETED:
                continue
            if task.deadline is None:
                continue

            time_to_deadline = task.deadline - now
            hours_left = time_to_deadline.total_seconds() / 3600
            days_overdue = -time_to_deadline.days

            # ── 48h avant la deadline ──
            if 0 < hours_left <= warning_hours:
                inactive = self._task_inactive_days(task)
                entry = {
                    "task_id": task.id,
                    "task_title": task.title,
                    "assigned_to": task.assigned_to,
                    "hours_left": round(hours_left, 1),
                    "inactive_days": inactive,
                    "message": (
                        f"La tâche '{task.title}' est due dans "
                        f"{round(hours_left)}h. Statut actuel : {task.status.value}."
                    ),
                }
                if inactive >= 3:
                    entry["message"] += (
                        f" Cette tâche n'a pas eu d'activité depuis "
                        f"{inactive} jours. Besoin d'aide ? Bloqué quelque part ?"
                    )
                warnings.append(entry)

            # ── Jour J ──
            elif 0 >= hours_left > -24:
                if task.status != TaskStatus.COMPLETED:
                    task.status = TaskStatus.OVERDUE
                    entry = {
                        "task_id": task.id,
                        "task_title": task.title,
                        "assigned_to": task.assigned_to,
                        "message": (
                            f"La tâche '{task.title}' arrive à échéance "
                            f"aujourd'hui et n'est pas complétée."
                        ),
                        "options": [
                            "Repousser de 2 jours",
                            "Réassigner",
                            "Escalader",
                        ],
                    }
                    due_today.append(entry)

            # ── En retard ──
            elif hours_left <= -24:
                task.status = TaskStatus.OVERDUE

                entry = {
                    "task_id": task.id,
                    "task_title": task.title,
                    "assigned_to": task.assigned_to,
                    "days_overdue": days_overdue,
                    "message": (
                        f"Cette tâche est en retard de {days_overdue} jour(s)."
                    ),
                }

                # ── 3+ jours → escalade ──
                if days_overdue >= escalation_days:
                    task.status = TaskStatus.ESCALATED
                    impact = self._estimate_overdue_impact(task)
                    entry["escalated"] = True
                    entry["impact"] = impact
                    entry["message"] = (
                        f"Cette tâche est en retard de {days_overdue} jours. "
                        f"Impact potentiel : {impact}"
                    )
                    escalated.append(entry)
                else:
                    overdue.append(entry)

        return {
            "checked_at": now.isoformat(),
            "warnings": warnings,
            "due_today": due_today,
            "overdue": overdue,
            "escalated": escalated,
            "summary": {
                "total_warnings": len(warnings),
                "total_due_today": len(due_today),
                "total_overdue": len(overdue),
                "total_escalated": len(escalated),
            },
        }

    def extend_deadline(
        self, task_id: str, days: int = 2, reason: str | None = None
    ) -> TaskRecord:
        """
        Repousse la deadline d'une tâche.

        Args:
            task_id: ID de la tâche.
            days: Nombre de jours à ajouter.
            reason: Raison du report.

        Returns:
            Tâche mise à jour.
        """
        task = self._find_task(task_id)
        if task.deadline is None:
            raise ValueError(f"Tâche '{task_id}' n'a pas de deadline")

        old_deadline = task.deadline
        task.deadline = old_deadline + timedelta(days=days)
        task.status = TaskStatus.ASSIGNED

        self.log_decision(
            decision_type=DecisionType.APPROVAL,
            who="kuria_process_clarity",
            what=(
                f"Deadline de '{task.title}' repoussée de {days}j "
                f"({old_deadline.date()} → {task.deadline.date()})"
            ),
            based_on=reason or "Demande de report",
            source_tool="kuria",
        )

        return task

    def get_delivery_metrics(self) -> dict[str, Any]:
        """
        Métriques de livraison pour le dashboard.

        Returns:
            Taux on-time global, par personne, par type de tâche, trend.
        """
        completed = [
            t for t in self._tasks.values()
            if t.status == TaskStatus.COMPLETED and t.deadline is not None
        ]

        if not completed:
            return {
                "on_time_rate": 0.0,
                "by_person": {},
                "total_completed_with_deadline": 0,
                "overdue_count": len([
                    t for t in self._tasks.values()
                    if t.status in (TaskStatus.OVERDUE, TaskStatus.ESCALATED)
                ]),
            }

        on_time = [
            t for t in completed
            if t.completed_at is not None and t.completed_at <= t.deadline
        ]

        by_person: dict[str, dict[str, int]] = {}
        for t in completed:
            person = t.assigned_to or "unassigned"
            if person not in by_person:
                by_person[person] = {"on_time": 0, "total": 0}
            by_person[person]["total"] += 1
            if t.completed_at is not None and t.completed_at <= t.deadline:
                by_person[person]["on_time"] += 1

        person_rates = {
            person: round(data["on_time"] / data["total"], 3)
            for person, data in by_person.items()
            if data["total"] > 0
        }

        return {
            "on_time_rate": round(len(on_time) / len(completed), 3),
            "by_person": person_rates,
            "total_completed_with_deadline": len(completed),
            "total_on_time": len(on_time),
            "overdue_count": len([
                t for t in self._tasks.values()
                if t.status in (TaskStatus.OVERDUE, TaskStatus.ESCALATED)
            ]),
        }

    def _task_inactive_days(self, task: TaskRecord) -> int:
        """Nombre de jours sans activité sur une tâche."""
        # En production : vérifier les événements liés à la tâche
        # Ici : estimation basée sur created_at
        last_activity = task.completed_at or task.created_at
        return (datetime.utcnow() - last_activity).days

    def _estimate_overdue_impact(self, task: TaskRecord) -> str:
        """Estime l'impact d'une tâche en retard."""
        if task.process_id and task.process_id in self._processes:
            process = self._processes[task.process_id]
            return (
                f"Blocage du process '{process.name}' — "
                f"cycle time impacté (+{(datetime.utcnow() - (task.deadline or datetime.utcnow())).days}j)"
            )
        if task.next_task_id:
            return f"Tâche suivante '{task.next_task_id}' bloquée en attente"
        return "Impact non estimé — tâche isolée"

    # ──────────────────────────────────────────────────────
    # ACTION 2.5 — AUTOMATISATION DES HANDOFFS
    # Trigger : tâche terminée qui nécessite un handoff
    # Fréquence : continu
    # ──────────────────────────────────────────────────────

    def complete_task(self, task_id: str) -> dict[str, Any]:
        """
        ACTION 2.5 — Complète une tâche et déclenche le handoff.

        Quand une tâche est complétée :
          1. Si tâche suivante existe → crée le handoff automatique
          2. Si pas de suite → vérifie si le process est terminé

        Returns:
            Résultat : task complétée + handoff créé (ou process terminé).
        """
        task = self._find_task(task_id)
        task.status = TaskStatus.COMPLETED
        task.completed_at = datetime.utcnow()

        # Mettre à jour la charge du membre
        if task.assigned_to and task.assigned_to in self._team:
            member = self._team[task.assigned_to]
            member.active_tasks = max(0, member.active_tasks - 1)
            member.completed_tasks_30d += 1

        result: dict[str, Any] = {
            "task_id": task.id,
            "task_title": task.title,
            "completed_by": task.assigned_to,
            "completed_at": task.completed_at.isoformat(),
        }

        # Logger la décision
        self.log_decision(
            decision_type=DecisionType.APPROVAL,
            who=task.assigned_to or "unknown",
            what=f"Tâche '{task.title}' complétée",
            source_tool=task.source_tool or "kuria",
        )

        # Handoff vers la tâche suivante
        if task.next_task_id:
            handoff = self._create_handoff(task)
            result["handoff"] = {
                "handoff_id": handoff.id,
                "next_task_id": handoff.to_task_id,
                "next_person": handoff.to_person,
                "context_sent": True,
            }
        else:
            # Vérifier si le process est terminé
            if task.process_id:
                process_complete = self._check_process_complete(task.process_id)
                result["process_complete"] = process_complete
                if process_complete:
                    result["message"] = (
                        f"Process '{self._processes[task.process_id].name}' terminé"
                    )

        return result

    def _create_handoff(self, completed_task: TaskRecord) -> Handoff:
        """
        Crée un handoff automatique entre la tâche complétée
        et la tâche suivante.

        Transfère le contexte : résumé, fichiers, notes.
        """
        next_task = self._find_task(completed_task.next_task_id)
        max_words = self._params.get("handoff_context_max_words", 200)

        context = self._build_handoff_context(completed_task, max_words)

        # Assigner la tâche suivante si pas déjà fait
        if next_task.assigned_to is None and self._team:
            self.route_task(next_task.id)

        next_task.status = TaskStatus.ASSIGNED

        handoff = Handoff(
            from_task_id=completed_task.id,
            to_task_id=next_task.id,
            from_person=completed_task.assigned_to or "unknown",
            to_person=next_task.assigned_to or "unassigned",
            context_summary=context,
            status=HandoffStatus.TRANSFERRED,
        )

        self._handoffs.append(handoff)

        # Logger
        self.log_decision(
            decision_type=DecisionType.TASK_ASSIGNMENT,
            who="kuria_process_clarity",
            what=(
                f"Handoff : '{completed_task.title}' → '{next_task.title}' "
                f"({handoff.from_person} → {handoff.to_person})"
            ),
            based_on="Workflow automatique — tâche précédente complétée",
            source_tool="kuria",
        )

        return handoff

    def _build_handoff_context(
        self, task: TaskRecord, max_words: int
    ) -> str:
        """Construit le résumé de contexte pour un handoff."""
        parts = [
            f"Tâche complétée : {task.title}",
            f"Par : {task.assigned_to or 'inconnu'}",
            f"Le : {task.completed_at.isoformat() if task.completed_at else 'N/A'}",
        ]
        if task.description:
            desc = task.description
            words = desc.split()
            if len(words) > max_words:
                desc = " ".join(words[:max_words]) + "..."
            parts.append(f"Description : {desc}")
        if task.tags:
            parts.append(f"Tags : {', '.join(task.tags)}")

        return "\n".join(parts)

    def acknowledge_handoff(self, handoff_id: str) -> Handoff:
        """Confirme la réception d'un handoff."""
        for h in self._handoffs:
            if h.id == handoff_id:
                h.status = HandoffStatus.ACKNOWLEDGED
                h.acknowledged_at = datetime.utcnow()
                return h
        raise ValueError(f"Handoff non trouvé : {handoff_id}")

    def get_lost_handoffs(self, timeout_hours: int = 24) -> list[Handoff]:
        """
        Détecte les handoffs non acknowledg és après timeout.

        Returns:
            Liste des handoffs potentiellement perdus.
        """
        cutoff = datetime.utcnow() - timedelta(hours=timeout_hours)
        lost = []
        for h in self._handoffs:
            if (
                h.status == HandoffStatus.TRANSFERRED
                and h.created_at < cutoff
            ):
                h.status = HandoffStatus.LOST
                lost.append(h)
        return lost

    def _check_process_complete(self, process_id: str) -> bool:
        """Vérifie si toutes les tâches d'un process sont complétées."""
        process_tasks = [
            t for t in self._tasks.values()
            if t.process_id == process_id
        ]
        if not process_tasks:
            return False
        return all(t.status == TaskStatus.COMPLETED for t in process_tasks)

    # ──────────────────────────────────────────────────────
    # ACTION 2.6 — SUPPRESSION DU TRAVAIL EN DOUBLE
    # Trigger : détection de tâches similaires
    # Fréquence : hebdomadaire
    # ──────────────────────────────────────────────────────

    def scan_duplicates(self) -> list[DuplicateDetection]:
        """
        ACTION 2.6 — Scanne les tâches et contacts pour détecter les doublons.

        3 niveaux de confiance :
          - HIGH (>95%)  → merge automatique
          - MEDIUM (70-95%) → demande confirmation
          - LOW (<70%)   → ignoré

        Returns:
            Liste des doublons détectés.
        """
        auto_threshold = self._params.get("duplicate_auto_merge_threshold", 0.95)
        confirm_threshold = self._params.get("duplicate_confirm_threshold", 0.70)

        new_duplicates: list[DuplicateDetection] = []
        task_list = list(self._tasks.values())

        for i in range(len(task_list)):
            for j in range(i + 1, len(task_list)):
                task_a = task_list[i]
                task_b = task_list[j]

                # Skip si déjà complétées
                if (
                    task_a.status == TaskStatus.COMPLETED
                    and task_b.status == TaskStatus.COMPLETED
                ):
                    continue

                score = self._similarity_score(task_a, task_b)

                if score < confirm_threshold:
                    continue

                if score >= auto_threshold:
                    confidence = DuplicateConfidence.HIGH
                elif score >= confirm_threshold:
                    confidence = DuplicateConfidence.MEDIUM
                else:
                    confidence = DuplicateConfidence.LOW

                dup = DuplicateDetection(
                    item_a_id=task_a.id,
                    item_a_title=task_a.title,
                    item_a_owner=task_a.assigned_to,
                    item_b_id=task_b.id,
                    item_b_title=task_b.title,
                    item_b_owner=task_b.assigned_to,
                    confidence=confidence,
                    confidence_score=round(score, 3),
                    item_type="task",
                )

                self._duplicates.append(dup)
                new_duplicates.append(dup)

        # Auto-merge les HIGH
        for dup in new_duplicates:
            if dup.confidence == DuplicateConfidence.HIGH:
                self._auto_merge(dup)

        return new_duplicates

    def merge_duplicate(
        self,
        duplicate_id: str,
        keep: str = "a",
    ) -> dict[str, Any]:
        """
        Fusionne un doublon manuellement.

        Args:
            duplicate_id: ID du DuplicateDetection.
            keep: "a" pour garder item A, "b" pour garder item B.

        Returns:
            Résultat de la fusion.
        """
        dup = self._find_duplicate(duplicate_id)

        if keep == "a":
            keep_id = dup.item_a_id
            remove_id = dup.item_b_id
        else:
            keep_id = dup.item_b_id
            remove_id = dup.item_a_id

        kept_task = self._tasks.get(keep_id)
        removed_task = self._tasks.get(remove_id)

        if kept_task and removed_task:
            # Consolider les tags
            kept_task.tags = list(set(kept_task.tags + removed_task.tags))

            # Consolider la description
            if removed_task.description and not kept_task.description:
                kept_task.description = removed_task.description
            elif removed_task.description and kept_task.description:
                kept_task.description += f"\n\n[Fusionné] {removed_task.description}"

            # Fermer le doublon
            removed_task.status = TaskStatus.COMPLETED
            removed_task.completed_at = datetime.utcnow()
            removed_task.tags.append("merged_duplicate")

            # Mettre à jour la charge
            if removed_task.assigned_to and removed_task.assigned_to in self._team:
                self._team[removed_task.assigned_to].active_tasks = max(
                    0, self._team[removed_task.assigned_to].active_tasks - 1
                )

        dup.resolved = True
        dup.resolution = f"merged_{keep}"

        self.log_decision(
            decision_type=DecisionType.APPROVAL,
            who="kuria_process_clarity",
            what=(
                f"Doublon fusionné : '{dup.item_a_title}' / '{dup.item_b_title}' "
                f"→ gardé '{kept_task.title if kept_task else keep_id}'"
            ),
            source_tool="kuria",
        )

        return {
            "duplicate_id": duplicate_id,
            "kept": keep_id,
            "removed": remove_id,
            "resolution": f"merged_{keep}",
        }

    def _similarity_score(
        self, task_a: TaskRecord, task_b: TaskRecord
    ) -> float:
        """
        Calcule un score de similarité entre deux tâches.

        Combine :
          - Similarité du titre (Jaccard sur les mots)
          - Même process
          - Tags communs
        """
        # Jaccard sur les mots du titre
        words_a = set(task_a.title.lower().split())
        words_b = set(task_b.title.lower().split())
        if words_a or words_b:
            jaccard = len(words_a & words_b) / len(words_a | words_b)
        else:
            jaccard = 0.0

        # Bonus si même process
        process_bonus = 0.1 if (
            task_a.process_id
            and task_a.process_id == task_b.process_id
        ) else 0.0

        # Bonus tags communs
        tags_a = set(task_a.tags)
        tags_b = set(task_b.tags)
        if tags_a or tags_b:
            tag_sim = len(tags_a & tags_b) / len(tags_a | tags_b) * 0.2
        else:
            tag_sim = 0.0

        # Description similarity
        desc_sim = 0.0
        if task_a.description and task_b.description:
            d_words_a = set(task_a.description.lower().split())
            d_words_b = set(task_b.description.lower().split())
            if d_words_a or d_words_b:
                desc_sim = len(d_words_a & d_words_b) / len(d_words_a | d_words_b) * 0.2

        score = jaccard * 0.5 + process_bonus + tag_sim + desc_sim
        return min(score, 1.0)

    def _auto_merge(self, dup: DuplicateDetection) -> None:
        """Merge automatique pour les doublons HIGH confidence."""
        # Garder la tâche la plus ancienne (ou celle déjà assignée)
        task_a = self._tasks.get(dup.item_a_id)
        task_b = self._tasks.get(dup.item_b_id)

        if task_a and task_b:
            if task_a.assigned_to and not task_b.assigned_to:
                keep = "a"
            elif task_b.assigned_to and not task_a.assigned_to:
                keep = "b"
            elif task_a.created_at <= task_b.created_at:
                keep = "a"
            else:
                keep = "b"

            self.merge_duplicate(dup.id, keep=keep)

    def _find_duplicate(self, duplicate_id: str) -> DuplicateDetection:
        """Trouve un doublon par ID."""
        for d in self._duplicates:
            if d.id == duplicate_id:
                return d
        raise ValueError(f"Doublon non trouvé : {duplicate_id}")

    def get_pending_duplicates(self) -> list[DuplicateDetection]:
        """Retourne les doublons en attente de confirmation (MEDIUM)."""
        return [
            d for d in self._duplicates
            if not d.resolved and d.confidence == DuplicateConfidence.MEDIUM
        ]

    # ──────────────────────────────────────────────────────
    # ACTION 2.7 — CHECK-INS AUTOMATIQUES (Basecamp-style)
    # Trigger : quotidien 9h00
    # Fréquence : quotidien
    # ──────────────────────────────────────────────────────

    def generate_checkins(self) -> list[CheckInMessage]:
        """
        ACTION 2.7 — Génère les check-ins pour chaque membre.

        Chaque membre reçoit :
          - Ses tâches du jour (triées par deadline)
          - Ses tâches en retard
          - Ce qu'il a complété hier
          - Option de signaler un blocage

        Returns:
            Liste des messages de check-in générés.
        """
        now = datetime.utcnow()
        yesterday = now - timedelta(days=1)
        today_end = now.replace(hour=23, minute=59, second=59)

        checkins: list[CheckInMessage] = []

        for member in self._team.values():
            if not member.is_available:
                continue

            member_tasks = [
                t for t in self._tasks.values()
                if t.assigned_to == member.id
            ]

            # Tâches du jour : deadline aujourd'hui ou demain
            tasks_today = sorted(
                [
                    t for t in member_tasks
                    if t.deadline is not None
                    and t.status not in (TaskStatus.COMPLETED,)
                    and t.deadline <= today_end + timedelta(days=1)
                    and t.deadline > now - timedelta(days=1)
                ],
                key=lambda t: t.deadline,
            )

            # Tâches en retard
            tasks_overdue = [
                t for t in member_tasks
                if t.status in (TaskStatus.OVERDUE, TaskStatus.ESCALATED)
            ]

            # Complétées hier
            tasks_completed_yesterday = [
                t for t in member_tasks
                if t.status == TaskStatus.COMPLETED
                and t.completed_at is not None
                and t.completed_at >= yesterday
                and t.completed_at < now
            ]

            checkin = CheckInMessage(
                member_id=member.id,
                member_name=member.name,
                tasks_today=tasks_today,
                tasks_overdue=tasks_overdue,
                tasks_completed_yesterday=tasks_completed_yesterday,
            )

            self._checkins.append(checkin)
            checkins.append(checkin)

        return checkins

    def format_checkin_message(self, checkin: CheckInMessage) -> str:
        """
        Formate un check-in en message lisible (Slack/email).

        Format Basecamp-style :
          Bonjour [Prénom].
          Vos tâches pour aujourd'hui : ...
          Hier vous avez complété : ...
          Bloqué ? Répondez 'bloqué [tâche]'.
        """
        lines = [f"Bonjour {checkin.member_name}. 👋"]
        lines.append("")

        # Tâches du jour
        if checkin.tasks_today:
            lines.append("📋 **Vos tâches pour aujourd'hui :**")
            for task in checkin.tasks_today:
                deadline_str = (
                    task.deadline.strftime("%d/%m %Hh%M")
                    if task.deadline
                    else "pas de deadline"
                )
                lines.append(f"  → {task.title} (due {deadline_str})")
        else:
            lines.append("📋 Aucune tâche urgente pour aujourd'hui.")

        lines.append("")

        # Tâches en retard
        if checkin.tasks_overdue:
            lines.append("⚠️ **En retard :**")
            for task in checkin.tasks_overdue:
                days = (
                    (datetime.utcnow() - task.deadline).days
                    if task.deadline
                    else 0
                )
                lines.append(
                    f"  → {task.title} (en retard de {days} jour(s))"
                )
            lines.append("")

        # Complétées hier
        if checkin.tasks_completed_yesterday:
            names = [t.title for t in checkin.tasks_completed_yesterday]
            lines.append(
                f"✅ Hier vous avez complété : {', '.join(names)}."
            )
            lines.append("")

        # Blocage
        lines.append(
            "Bloqué quelque part ? Répondez `bloqué [nom de la tâche]` "
            "et votre manager sera notifié."
        )

        return "\n".join(lines)

    def handle_blockage(
        self,
        member_id: str,
        task_id: str,
    ) -> BlockageReport:
        """
        Traite un signalement de blocage par un membre.

        Quand quelqu'un répond 'bloqué [tâche]' :
          1. Notifie le manager avec le contexte
          2. Crée une tâche urgente pour débloquer
          3. Propose une réassignation si pertinent

        Returns:
            Le rapport de blocage créé.
        """
        member = self._team.get(member_id)
        task = self._find_task(task_id)

        if member is None:
            raise ValueError(f"Membre non trouvé : {member_id}")

        task.status = TaskStatus.BLOCKED

        report = BlockageReport(
            member_id=member.id,
            member_name=member.name,
            task_id=task.id,
            task_title=task.title,
            manager_notified=True,
            unblock_task_created=True,
        )

        # Créer une tâche urgente pour débloquer
        unblock_task = TaskRecord(
            title=f"[URGENT] Débloquer : {task.title}",
            description=(
                f"{member.name} est bloqué(e) sur '{task.title}'. "
                f"Contexte : {task.description or 'Non précisé'}. "
                f"Action requise : débloquer ou réassigner."
            ),
            status=TaskStatus.UNASSIGNED,
            deadline=datetime.utcnow() + timedelta(hours=4),
            process_id=task.process_id,
            tags=["urgent", "blocage", "auto_generated"],
            source_tool="kuria",
        )

        # Router la tâche de déblocage (pas au même membre)
        self._tasks[unblock_task.id] = unblock_task
        if self._team:
            try:
                self.route_task(unblock_task.id)
            except ValueError:
                pass

        self._blockages.append(report)

        self.log_decision(
            decision_type=DecisionType.ESCALATION,
            who=member.name,
            what=f"Blocage signalé sur '{task.title}'",
            based_on="Check-in automatique — réponse 'bloqué'",
            source_tool="kuria",
        )

        return report

    def get_daily_summary(self) -> dict[str, Any]:
        """
        Résumé de fin de journée pour le manager.

        Agrège tous les check-ins en un résumé :
        "[X] tâches complétées, [Y] en retard, [Z] bloquées"
        """
        now = datetime.utcnow()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

        completed_today = [
            t for t in self._tasks.values()
            if t.status == TaskStatus.COMPLETED
            and t.completed_at is not None
            and t.completed_at >= today_start
        ]

        overdue = [
            t for t in self._tasks.values()
            if t.status in (TaskStatus.OVERDUE, TaskStatus.ESCALATED)
        ]

        blocked = [
            t for t in self._tasks.values()
            if t.status == TaskStatus.BLOCKED
        ]

        return {
            "date": now.date().isoformat(),
            "completed": len(completed_today),
            "overdue": len(overdue),
            "blocked": len(blocked),
            "summary": (
                f"{len(completed_today)} tâches complétées, "
                f"{len(overdue)} en retard, "
                f"{len(blocked)} bloquées"
            ),
            "blockages": [
                {
                    "member": b.member_name,
                    "task": b.task_title,
                    "reported_at": b.reported_at.isoformat(),
                }
                for b in self._blockages
                if b.reported_at >= today_start
            ],
            }

    # ──────────────────────────────────────────────────────
    # ACTION 2.8 — REPORTING OPS AUTOMATIQUE
    # Trigger : hebdomadaire + mensuel
    # Fréquence : lundi 6h00
    # ──────────────────────────────────────────────────────

    def generate_ops_report(
        self,
        period_days: int = 7,
    ) -> OpsReport:
        """
        ACTION 2.8 — Génère le rapport opérationnel automatique.

        Compile :
          - Cycle time par process (trend)
          - Taux de livraison à temps (global + par personne)
          - Bottleneck actuel (#1 + coût estimé)
          - Tâches complétées vs créées (backlog trend)
          - Charge par personne
          - Gaspillages détectés
          - SOPs créées/mises à jour
          - Décisions loggées
          - 1 recommandation clé

        Returns:
            OpsReport complet.
        """
        now = datetime.utcnow()
        cutoff = now - timedelta(days=period_days)
        period_start = (now - timedelta(days=period_days)).date()
        period_end = now.date()

        # ── Cycle times par process ──
        cycle_times: dict[str, float] = {}
        for proc_id, process in self._processes.items():
            if process.avg_cycle_time_hours is not None:
                cycle_times[process.name] = round(
                    process.avg_cycle_time_hours / 24, 2
                )

        # ── Taux de livraison ──
        delivery = self.get_delivery_metrics()
        on_time_rate = delivery.get("on_time_rate", 0.0)
        on_time_by_person = delivery.get("by_person", {})

        # ── Bottleneck ──
        bottleneck, bottleneck_cost = self._detect_bottleneck()

        # ── Backlog trend ──
        tasks_completed = len([
            t for t in self._tasks.values()
            if t.status == TaskStatus.COMPLETED
            and t.completed_at is not None
            and t.completed_at >= cutoff
        ])
        tasks_created = len([
            t for t in self._tasks.values()
            if t.created_at >= cutoff
        ])

        if tasks_created > tasks_completed * 1.2:
            backlog_trend = TrendDirection.INCREASING
        elif tasks_completed > tasks_created * 1.2:
            backlog_trend = TrendDirection.DECREASING
        else:
            backlog_trend = TrendDirection.STABLE

        # ── Charge par personne ──
        workload: dict[str, int] = {
            m.name: m.active_tasks for m in self._team.values()
        }

        # ── Doublons ──
        recent_dups = len([
            d for d in self._duplicates
            if d.detected_at >= cutoff
        ])

        # ── SOPs ──
        sops_created = len([
            s for s in self._sops.values()
            if s.created_at >= cutoff and s.version == 1
        ])
        sops_updated = len([
            s for s in self._sops.values()
            if s.updated_at >= cutoff and s.version > 1
        ])

        # ── Décisions ──
        decisions_count = len([
            d for d in self._decisions if d.when >= cutoff
        ])

        # ── Recommandation ──
        recommendation = self._generate_recommendation(
            on_time_rate=on_time_rate,
            bottleneck=bottleneck,
            backlog_trend=backlog_trend,
            recent_dups=recent_dups,
            workload=workload,
        )

        report = OpsReport(
            period_start=period_start,
            period_end=period_end,
            cycle_times=cycle_times,
            on_time_rate=on_time_rate,
            on_time_rate_by_person=on_time_by_person,
            bottleneck=bottleneck,
            bottleneck_cost_estimate=bottleneck_cost,
            tasks_completed=tasks_completed,
            tasks_created=tasks_created,
            backlog_trend=backlog_trend,
            workload_by_person=workload,
            duplicates_found=recent_dups,
            sops_created=sops_created,
            sops_updated=sops_updated,
            decisions_logged=decisions_count,
            key_recommendation=recommendation,
        )

        self._reports.append(report)
        return report

    def _detect_bottleneck(self) -> tuple[str | None, float | None]:
        """
        Détecte le bottleneck #1 (Goldratt).

        Le bottleneck = l'étape avec le plus grand cycle time
        dans le process le plus lent.

        Returns:
            (description du bottleneck, coût estimé en €/mois)
        """
        if not self._processes:
            return None, None

        slowest_process = max(
            self._processes.values(),
            key=lambda p: p.avg_cycle_time_hours or 0,
        )

        if not slowest_process.steps or not slowest_process.avg_cycle_time_hours:
            return None, None

        slowest_step = max(
            slowest_process.steps,
            key=lambda s: s.avg_duration_hours or 0,
        )

        if not slowest_step.avg_duration_hours:
            return None, None

        step_pct = (
            slowest_step.avg_duration_hours
            / slowest_process.avg_cycle_time_hours
            * 100
        )

        description = (
            f"Process '{slowest_process.name}' — "
            f"étape '{slowest_step.name}' "
            f"({slowest_step.avg_duration_hours:.1f}h, "
            f"{step_pct:.0f}% du cycle time total)"
        )

        # Estimation coût : heures perdues × coût horaire estimé
        excess_hours = max(0, slowest_step.avg_duration_hours - 2)
        monthly_occurrences = slowest_process.occurrence_count * 4
        cost_per_hour = 50  # €/h moyen
        cost_estimate = excess_hours * monthly_occurrences * cost_per_hour

        return description, round(cost_estimate, 2) if cost_estimate > 0 else None

    def _generate_recommendation(
        self,
        on_time_rate: float,
        bottleneck: str | None,
        backlog_trend: TrendDirection,
        recent_dups: int,
        workload: dict[str, int],
    ) -> str:
        """Génère LA recommandation clé de la semaine."""
        if on_time_rate < 0.6:
            return (
                f"CRITIQUE : Taux de livraison à temps à {on_time_rate:.0%}. "
                f"Revoir la capacité de l'équipe et les deadlines."
            )

        if bottleneck:
            return f"BOTTLENECK : {bottleneck}. Priorité #1 cette semaine."

        if backlog_trend == TrendDirection.INCREASING:
            return (
                "Le backlog grandit. Plus de tâches créées que complétées. "
                "Réduire le WIP ou renforcer l'équipe."
            )

        if recent_dups > 5:
            return (
                f"{recent_dups} doublons détectés cette semaine. "
                f"Revoir les process de création de tâches."
            )

        if workload:
            max_load = max(workload.values())
            min_load = min(workload.values())
            if max_load > 0 and max_load / max(min_load, 1) > 3:
                overloaded = max(workload, key=workload.get)
                return (
                    f"Déséquilibre de charge : {overloaded} a {max_load} tâches "
                    f"vs minimum {min_load}. Rééquilibrer."
                )

        return "Opérations stables. Maintenir le cap."

    # ──────────────────────────────────────────────────────
    # KPI PRINCIPAL : CYCLE TIME
    # ──────────────────────────────────────────────────────

    def compute_kpi(self) -> MetricResult:
        """
        Calcule le KPI principal : cycle time moyen des process (jours).

        Returns:
            MetricResult avec le cycle time et le health status.
        """
        cycle_times = [
            p.avg_cycle_time_hours / 24
            for p in self._processes.values()
            if p.avg_cycle_time_hours is not None and p.avg_cycle_time_hours > 0
        ]

        if not cycle_times:
            return MetricResult(
                metric_name="cycle_time_days",
                value=0.0,
                unit="days",
                health=HealthStatus.UNKNOWN,
                details={"message": "Aucun process avec cycle time mesuré"},
            )

        avg_ct = statistics.mean(cycle_times)

        # Évaluer la santé via les thresholds
        ct_threshold = next(
            (t for t in self.config.thresholds if t.metric_name == "cycle_time_days"),
            None,
        )

        if ct_threshold:
            if avg_ct >= ct_threshold.critical_value:
                health = HealthStatus.CRITICAL
            elif avg_ct >= ct_threshold.warning_value:
                health = HealthStatus.WARNING
            else:
                health = HealthStatus.HEALTHY
        else:
            health = HealthStatus.UNKNOWN

        return MetricResult(
            metric_name="cycle_time_days",
            value=round(avg_ct, 2),
            unit="days",
            health=health,
            details={
                "process_count": len(cycle_times),
                "min_cycle_time": round(min(cycle_times), 2),
                "max_cycle_time": round(max(cycle_times), 2),
                "median_cycle_time": round(statistics.median(cycle_times), 2),
            },
        )

    # ──────────────────────────────────────────────────────
    # DÉTECTION DE FRICTIONS
    # ──────────────────────────────────────────────────────

    def detect_frictions(self) -> list[Friction]:
        """
        Détecte les frictions opérationnelles.

        Sources de friction :
          - Tâches en retard chronique
          - Handoffs perdus
          - Doublons récurrents
          - Surcharge d'un membre
          - Cycle time excessif

        Returns:
            Liste des frictions détectées.
        """
        frictions: list[Friction] = []

        # ── Tâches en retard chronique ──
        overdue_tasks = [
            t for t in self._tasks.values()
            if t.status in (TaskStatus.OVERDUE, TaskStatus.ESCALATED)
        ]
        if len(overdue_tasks) >= 5:
            frictions.append(Friction(
                type=FrictionType.PROCESS,
                severity=FrictionSeverity.HIGH,
                source=FrictionSource.DETECTED,
                title="Retards chroniques sur les tâches",
                description=(
                    f"{len(overdue_tasks)} tâches en retard. "
                    f"Le système de deadlines n'est pas respecté."
                ),
                estimated_cost_monthly=len(overdue_tasks) * 500,
                recommended_action=(
                    "Revoir la charge de l'équipe et la réalisme des deadlines. "
                    "Activer les escalades automatiques."
                ),
            ))

        # ── Handoffs perdus ──
        lost = [h for h in self._handoffs if h.status == HandoffStatus.LOST]
        if len(lost) >= 3:
            frictions.append(Friction(
                type=FrictionType.PROCESS,
                severity=FrictionSeverity.MEDIUM,
                source=FrictionSource.DETECTED,
                title="Handoffs perdus entre équipiers",
                description=(
                    f"{len(lost)} handoffs non acknowledgés. "
                    f"Le contexte se perd entre les étapes."
                ),
                estimated_cost_monthly=len(lost) * 300,
                recommended_action=(
                    "Imposer l'acknowledgement des handoffs. "
                    "Réduire le timeout de détection."
                ),
            ))

        # ── Doublons récurrents ──
        unresolved = [d for d in self._duplicates if not d.resolved]
        if len(unresolved) >= 5:
            frictions.append(Friction(
                type=FrictionType.DATA_QUALITY,
                severity=FrictionSeverity.MEDIUM,
                source=FrictionSource.DETECTED,
                title="Travail en double récurrent",
                description=(
                    f"{len(unresolved)} doublons non résolus. "
                    f"Du travail est fait deux fois."
                ),
                estimated_cost_monthly=len(unresolved) * 200,
                recommended_action=(
                    "Activer le merge automatique pour les doublons HIGH. "
                    "Former l'équipe à vérifier avant de créer une tâche."
                ),
            ))

        # ── Surcharge d'un membre ──
        for member in self._team.values():
            overload = self._check_overload(member.id)
            if overload is not None:
                frictions.append(Friction(
                    type=FrictionType.PROCESS,
                    severity=FrictionSeverity.MEDIUM,
                    source=FrictionSource.DETECTED,
                    title=f"Surcharge : {member.name}",
                    description=(
                        f"{member.name} a {member.active_tasks} tâches actives "
                        f"(moyenne équipe : {self._avg_team_workload():.1f})"
                    ),
                    estimated_cost_monthly=1000,
                    recommended_action=(
                        f"Rééquilibrer la charge de {member.name}. "
                        f"Réassigner les tâches non urgentes."
                    ),
                ))

        # ── Cycle time excessif ──
        kpi = self.compute_kpi()
        if kpi.health == HealthStatus.CRITICAL:
            frictions.append(Friction(
                type=FrictionType.PROCESS,
                severity=FrictionSeverity.CRITICAL,
                source=FrictionSource.DETECTED,
                title="Cycle time critique",
                description=(
                    f"Cycle time moyen : {kpi.value} jours. "
                    f"Les process sont trop lents."
                ),
                estimated_cost_monthly=kpi.value * 1000,
                recommended_action=(
                    "Identifier le bottleneck #1 (voir rapport ops) "
                    "et concentrer les efforts dessus."
                ),
            ))

        return frictions

    # ──────────────────────────────────────────────────────
    # RUN COMPLET
    # ──────────────────────────────────────────────────────

    def run_daily(self) -> dict[str, Any]:
        """
        Exécution quotidienne de l'agent.

        Séquence :
          1. Check deadlines (2.4)
          2. Detect lost handoffs (2.5)
          3. Generate check-ins (2.7)

        Returns:
            Résumé de l'exécution quotidienne.
        """
        deadlines = self.check_deadlines()
        lost_handoffs = self.get_lost_handoffs()
        checkins = self.generate_checkins()
        daily = self.get_daily_summary()

        return {
            "type": "daily",
            "timestamp": datetime.utcnow().isoformat(),
            "deadlines": deadlines["summary"],
            "lost_handoffs": len(lost_handoffs),
            "checkins_sent": len(checkins),
            "daily_summary": daily["summary"],
        }

    def run_weekly(self) -> dict[str, Any]:
        """
        Exécution hebdomadaire de l'agent.

        Séquence :
          1. Detect processes (2.1)
          2. Generate SOPs (2.1)
          3. Scan duplicates (2.6)
          4. Generate ops report (2.8)
          5. Detect frictions
          6. Compute KPI

        Returns:
            Résumé de l'exécution hebdomadaire.
        """
        new_processes = self.detect_processes()
        new_sops = self.generate_sops()
        duplicates = self.scan_duplicates()
        report = self.generate_ops_report(period_days=7)
        frictions = self.detect_frictions()
        kpi = self.compute_kpi()

        return {
            "type": "weekly",
            "timestamp": datetime.utcnow().isoformat(),
            "new_processes_detected": len(new_processes),
            "new_sops_generated": len(new_sops),
            "duplicates_found": len(duplicates),
            "ops_report_id": report.id,
            "frictions_detected": len(frictions),
            "kpi": {
                "cycle_time_days": kpi.value,
                "health": kpi.health.value,
            },
            "recommendation": report.key_recommendation,
        }
