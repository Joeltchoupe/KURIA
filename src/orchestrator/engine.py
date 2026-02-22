"""
OrchestrationEngine — Le cœur du système.

Responsabilités :
  1. Enregistrer les agents disponibles
  2. Décider quoi tourne quand (scheduling basé sur les configs)
  3. Dispatcher l'exécution (daily, weekly, on-demand)
  4. Gérer le state global (qui a tourné, quand, résultat)
  5. Propager les signaux inter-agents via le Coordinator

Design decisions :
  - L'engine ne connaît PAS l'implémentation des agents
  - Il connaît leurs configs (AgentConfig) et leurs interfaces
  - Il délègue la coordination au Coordinator
  - Il délègue l'adaptation à l'Adapter
  - Il délègue le reporting au Reporter
  - Chaque run produit un RunResult traçable
"""

from __future__ import annotations

from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Protocol
from uuid import uuid4

from pydantic import BaseModel, Field

from models.agent_config import AgentConfig, AgentConfigSet, AgentType


# ══════════════════════════════════════════════════════════════
# PROTOCOLS — Interface que chaque agent doit respecter
# ══════════════════════════════════════════════════════════════


class AgentProtocol(Protocol):
    """
    Interface minimale que chaque agent doit exposer.

    L'engine ne connaît pas les classes concrètes.
    Il connaît ce contrat.
    """

    config: AgentConfig

    def run_daily(self) -> dict[str, Any]: ...
    def run_weekly(self) -> dict[str, Any]: ...
    def compute_kpi(self) -> Any: ...
    def detect_frictions(self) -> list[Any]: ...


# ══════════════════════════════════════════════════════════════
# ENUMS + MODÈLES
# ══════════════════════════════════════════════════════════════


class AgentRunStatus(str, Enum):
    """Statut d'exécution d'un agent."""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    SKIPPED = "skipped"
    DISABLED = "disabled"


class RunType(str, Enum):
    """Type d'exécution."""
    DAILY = "daily"
    WEEKLY = "weekly"
    ON_DEMAND = "on_demand"
    INITIAL_SCAN = "initial_scan"


class AgentRunRecord(BaseModel):
    """Trace d'une exécution d'agent."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    agent_type: AgentType
    run_type: RunType
    status: AgentRunStatus
    started_at: datetime
    completed_at: datetime = Field(default_factory=datetime.utcnow)
    duration_seconds: float = Field(default=0.0, ge=0)
    result: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    kpi_value: float | None = None
    frictions_detected: int = Field(default=0, ge=0)


class RunResult(BaseModel):
    """Résultat d'un cycle d'orchestration complet."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    company_id: str
    run_type: RunType
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None
    agent_runs: list[AgentRunRecord] = Field(default_factory=list)
    signals_emitted: int = Field(default=0, ge=0)
    total_frictions: int = Field(default=0, ge=0)

    @property
    def duration_seconds(self) -> float:
        if self.completed_at is None:
            return 0.0
        return (self.completed_at - self.started_at).total_seconds()

    @property
    def all_success(self) -> bool:
        return all(
            r.status in (AgentRunStatus.SUCCESS, AgentRunStatus.SKIPPED, AgentRunStatus.DISABLED)
            for r in self.agent_runs
        )

    @property
    def summary(self) -> dict[str, Any]:
        return {
            "run_id": self.id,
            "run_type": self.run_type.value,
            "duration_seconds": round(self.duration_seconds, 2),
            "agents_run": len(self.agent_runs),
            "all_success": self.all_success,
            "signals_emitted": self.signals_emitted,
            "total_frictions": self.total_frictions,
            "per_agent": {
                r.agent_type.value: {
                    "status": r.status.value,
                    "kpi": r.kpi_value,
                    "frictions": r.frictions_detected,
                    "duration": round(r.duration_seconds, 2),
                }
                for r in self.agent_runs
            },
        }


# ══════════════════════════════════════════════════════════════
# SCHEDULING
# ══════════════════════════════════════════════════════════════


class ScheduleEntry(BaseModel):
    """Entrée de planification pour un agent."""
    agent_type: AgentType
    run_type: RunType
    last_run: datetime | None = None
    next_run: datetime | None = None
    interval_hours: float = Field(default=24.0, gt=0)
    enabled: bool = True


def frequency_to_hours(frequency: str) -> float:
    """Convertit une fréquence en intervalle d'heures."""
    mapping = {
        "realtime": 0.0833,  # ~5 min
        "hourly": 1.0,
        "daily": 24.0,
        "weekly": 168.0,
        "monthly": 720.0,
    }
    return mapping.get(frequency, 24.0)


# ══════════════════════════════════════════════════════════════
# ENGINE
# ══════════════════════════════════════════════════════════════


class OrchestrationEngine:
    """
    Le cœur du système Kuria.

    Enregistre les agents, décide quoi tourne quand,
    dispatche l'exécution, trace les résultats.

    Usage :
        engine = OrchestrationEngine(company_id="acme", config_set=configs)
        engine.register_agent(AgentType.REVENUE_VELOCITY, rv_agent)
        engine.register_agent(AgentType.PROCESS_CLARITY, pc_agent)

        # Run quotidien
        result = engine.run_daily()

        # Run hebdomadaire
        result = engine.run_weekly()

        # Run un agent spécifique
        result = engine.run_agent(AgentType.PROCESS_CLARITY, RunType.ON_DEMAND)
    """

    def __init__(
        self,
        company_id: str,
        config_set: AgentConfigSet | None = None,
        coordinator: Any | None = None,
        adapter: Any | None = None,
        reporter: Any | None = None,
    ) -> None:
        self.company_id = company_id
        self.config_set = config_set

        # Collaborateurs (injectés ou None)
        self._coordinator = coordinator
        self._adapter = adapter
        self._reporter = reporter

        # Agents enregistrés
        self._agents: dict[AgentType, AgentProtocol] = {}

        # Schedule
        self._schedule: dict[AgentType, ScheduleEntry] = {}

        # Historique
        self._run_history: list[RunResult] = []

        # Callbacks (notifications, webhooks, etc.)
        self._on_run_complete: list[Callable[[RunResult], None]] = []
        self._on_agent_error: list[Callable[[AgentType, Exception], None]] = []

    # ──────────────────────────────────────────────────────
    # REGISTRATION
    # ──────────────────────────────────────────────────────

    def register_agent(
        self,
        agent_type: AgentType,
        agent: AgentProtocol,
    ) -> None:
        """
        Enregistre un agent dans l'engine.

        L'agent doit respecter AgentProtocol :
          - .config: AgentConfig
          - .run_daily() -> dict
          - .run_weekly() -> dict
          - .compute_kpi() -> MetricResult
          - .detect_frictions() -> list[Friction]
        """
        self._agents[agent_type] = agent

        # Créer l'entrée de schedule basée sur la config
        config = agent.config
        cron = getattr(config, "schedule_cron", "0 6 * * *")
        frequency = self._cron_to_frequency(cron)

        self._schedule[agent_type] = ScheduleEntry(
            agent_type=agent_type,
            run_type=RunType.DAILY if frequency <= 24 else RunType.WEEKLY,
            interval_hours=frequency,
            enabled=config.enabled,
        )

    def unregister_agent(self, agent_type: AgentType) -> None:
        """Retire un agent de l'engine."""
        self._agents.pop(agent_type, None)
        self._schedule.pop(agent_type, None)

    @property
    def registered_agents(self) -> list[AgentType]:
        """Liste des agents enregistrés."""
        return list(self._agents.keys())

    @property
    def enabled_agents(self) -> list[AgentType]:
        """Liste des agents enregistrés ET activés."""
        return [
            at for at, entry in self._schedule.items()
            if entry.enabled
        ]

    # ──────────────────────────────────────────────────────
    # SCHEDULING
    # ──────────────────────────────────────────────────────

    def get_schedule(self) -> dict[str, Any]:
        """Retourne le planning de tous les agents."""
        return {
            agent_type.value: {
                "run_type": entry.run_type.value,
                "interval_hours": entry.interval_hours,
                "last_run": entry.last_run.isoformat() if entry.last_run else None,
                "next_run": entry.next_run.isoformat() if entry.next_run else None,
                "enabled": entry.enabled,
            }
            for agent_type, entry in self._schedule.items()
        }

    def get_due_agents(self) -> list[AgentType]:
        """
        Retourne les agents dont l'exécution est due.

        Un agent est "due" si :
          - Il est enabled
          - Il n'a jamais tourné, OU
          - Son intervalle est dépassé depuis le dernier run
        """
        now = datetime.utcnow()
        due: list[AgentType] = []

        for agent_type, entry in self._schedule.items():
            if not entry.enabled:
                continue

            if entry.last_run is None:
                due.append(agent_type)
                continue

            elapsed = (now - entry.last_run).total_seconds() / 3600
            if elapsed >= entry.interval_hours:
                due.append(agent_type)

        return due

    # ──────────────────────────────────────────────────────
    # EXECUTION — RUN SINGLE AGENT
    # ──────────────────────────────────────────────────────

    def run_agent(
        self,
        agent_type: AgentType,
        run_type: RunType = RunType.ON_DEMAND,
    ) -> AgentRunRecord:
        """
        Exécute un agent spécifique.

        Séquence :
          1. Vérifier que l'agent est enregistré et enabled
          2. Exécuter run_daily() ou run_weekly() selon le run_type
          3. Récupérer le KPI et les frictions
          4. Tracer le résultat
          5. Propager les signaux au Coordinator

        Returns:
            AgentRunRecord avec le résultat complet.
        """
        # Vérifications
        if agent_type not in self._agents:
            return AgentRunRecord(
                agent_type=agent_type,
                run_type=run_type,
                status=AgentRunStatus.SKIPPED,
                started_at=datetime.utcnow(),
                error=f"Agent {agent_type.value} non enregistré",
            )

        entry = self._schedule.get(agent_type)
        if entry and not entry.enabled:
            return AgentRunRecord(
                agent_type=agent_type,
                run_type=run_type,
                status=AgentRunStatus.DISABLED,
                started_at=datetime.utcnow(),
            )

        agent = self._agents[agent_type]
        started_at = datetime.utcnow()

        try:
            # Dispatch selon le type de run
            if run_type in (RunType.DAILY, RunType.ON_DEMAND):
                result = agent.run_daily()
            elif run_type == RunType.WEEKLY:
                result = agent.run_weekly()
            else:
                result = agent.run_daily()

            # KPI
            kpi = agent.compute_kpi()
            kpi_value = getattr(kpi, "value", None)

            # Frictions
            frictions = agent.detect_frictions()
            frictions_count = len(frictions) if frictions else 0

            completed_at = datetime.utcnow()
            duration = (completed_at - started_at).total_seconds()

            record = AgentRunRecord(
                agent_type=agent_type,
                run_type=run_type,
                status=AgentRunStatus.SUCCESS,
                started_at=started_at,
                completed_at=completed_at,
                duration_seconds=round(duration, 3),
                result=result if isinstance(result, dict) else {"raw": str(result)},
                kpi_value=kpi_value,
                frictions_detected=frictions_count,
            )

            # Mettre à jour le schedule
            if entry:
                entry.last_run = completed_at
                entry.next_run = completed_at + timedelta(
                    hours=entry.interval_hours
                )

            # Propager les signaux
            if self._coordinator and frictions:
                self._coordinator.process_frictions(
                    source_agent=agent_type,
                    frictions=frictions,
                )

            return record

        except Exception as e:
            completed_at = datetime.utcnow()
            duration = (completed_at - started_at).total_seconds()

            # Notifier les callbacks d'erreur
            for callback in self._on_agent_error:
                try:
                    callback(agent_type, e)
                except Exception:
                    pass

            return AgentRunRecord(
                agent_type=agent_type,
                run_type=run_type,
                status=AgentRunStatus.FAILED,
                started_at=started_at,
                completed_at=completed_at,
                duration_seconds=round(duration, 3),
                error=f"{type(e).__name__}: {str(e)}",
            )

    # ──────────────────────────────────────────────────────
    # EXECUTION — RUN CYCLES
    # ──────────────────────────────────────────────────────

    def run_daily(self) -> RunResult:
        """
        Exécution quotidienne de tous les agents éligibles.

        Séquence :
          1. Identifier les agents daily due
          2. Exécuter chacun
          3. Propager les signaux croisés
          4. Compiler le RunResult

        Returns:
            RunResult avec tous les AgentRunRecords.
        """
        return self._run_cycle(RunType.DAILY)

    def run_weekly(self) -> RunResult:
        """
        Exécution hebdomadaire complète.

        Séquence :
          1. Exécuter tous les agents en mode weekly
          2. Propager les signaux croisés
          3. Lancer l'adaptation (recalibration des configs)
          4. Générer le rapport hebdo
          5. Compiler le RunResult

        Returns:
            RunResult avec tous les AgentRunRecords.
        """
        result = self._run_cycle(RunType.WEEKLY)

        # Adaptation
        if self._adapter and self.config_set:
            try:
                self._adapter.adapt(
                    config_set=self.config_set,
                    run_result=result,
                )
            except Exception:
                pass  # L'adaptation ne doit pas casser le cycle

        # Rapport
        if self._reporter:
            try:
                self._reporter.generate(
                    run_result=result,
                    company_id=self.company_id,
                )
            except Exception:
                pass

        return result

    def run_due(self) -> RunResult:
        """
        Exécute uniquement les agents dont l'intervalle est dépassé.

        C'est cette méthode que le cron/scheduler externe appelle.
        """
        due_agents = self.get_due_agents()

        run_result = RunResult(
            company_id=self.company_id,
            run_type=RunType.ON_DEMAND,
        )

        for agent_type in due_agents:
            entry = self._schedule.get(agent_type)
            rt = entry.run_type if entry else RunType.DAILY
            record = self.run_agent(agent_type, rt)
            run_result.agent_runs.append(record)
            run_result.total_frictions += record.frictions_detected

        run_result.completed_at = datetime.utcnow()

        # Coordination
        if self._coordinator:
            signals = self._coordinator.flush_signals()
            run_result.signals_emitted = len(signals)

        self._run_history.append(run_result)
        self._notify_completion(run_result)

        return run_result

    def _run_cycle(self, run_type: RunType) -> RunResult:
        """Exécute un cycle complet (daily ou weekly)."""
        run_result = RunResult(
            company_id=self.company_id,
            run_type=run_type,
        )

        # Déterminer l'ordre d'exécution
        order = self._execution_order()

        for agent_type in order:
            entry = self._schedule.get(agent_type)
            if entry and not entry.enabled:
                run_result.agent_runs.append(AgentRunRecord(
                    agent_type=agent_type,
                    run_type=run_type,
                    status=AgentRunStatus.DISABLED,
                    started_at=datetime.utcnow(),
                ))
                continue

            record = self.run_agent(agent_type, run_type)
            run_result.agent_runs.append(record)
            run_result.total_frictions += record.frictions_detected

        run_result.completed_at = datetime.utcnow()

        # Coordination inter-agents
        if self._coordinator:
            signals = self._coordinator.flush_signals()
            run_result.signals_emitted = len(signals)

        self._run_history.append(run_result)
        self._notify_completion(run_result)

        return run_result

    def _execution_order(self) -> list[AgentType]:
        """
        Ordre d'exécution des agents.

        Ordre logique :
          1. Revenue Velocity (données pipeline)
          2. Process Clarity (détection process)
          3. Cash Predictability (données finance)
          4. Acquisition Efficiency (données marketing)

        Les agents non enregistrés sont ignorés.
        """
        preferred_order = [
            AgentType.REVENUE_VELOCITY,
            AgentType.PROCESS_CLARITY,
            AgentType.CASH_PREDICTABILITY,
            AgentType.ACQUISITION_EFFICIENCY,
        ]
        return [at for at in preferred_order if at in self._agents]

    # ──────────────────────────────────────────────────────
    # CALLBACKS
    # ──────────────────────────────────────────────────────

    def on_run_complete(self, callback: Callable[[RunResult], None]) -> None:
        """Enregistre un callback appelé après chaque cycle complet."""
        self._on_run_complete.append(callback)

    def on_agent_error(
        self, callback: Callable[[AgentType, Exception], None]
    ) -> None:
        """Enregistre un callback appelé quand un agent échoue."""
        self._on_agent_error.append(callback)

    def _notify_completion(self, result: RunResult) -> None:
        """Notifie les callbacks de complétion."""
        for callback in self._on_run_complete:
            try:
                callback(result)
            except Exception:
                pass

    # ──────────────────────────────────────────────────────
    # HISTORIQUE
    # ──────────────────────────────────────────────────────

    @property
    def last_run(self) -> RunResult | None:
        """Dernier RunResult."""
        return self._run_history[-1] if self._run_history else None

    @property
    def run_count(self) -> int:
        """Nombre total de cycles exécutés."""
        return len(self._run_history)

    def get_history(
        self,
        limit: int = 10,
        agent_type: AgentType | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retourne l'historique des exécutions.

        Args:
            limit: Nombre max de résultats.
            agent_type: Filtrer par agent (optionnel).
        """
        history = self._run_history[-limit:]

        if agent_type is None:
            return [r.summary for r in reversed(history)]

        results = []
        for run in reversed(history):
            for record in run.agent_runs:
                if record.agent_type == agent_type:
                    results.append({
                        "run_id": run.id,
                        "run_type": run.run_type.value,
                        "agent": record.agent_type.value,
                        "status": record.status.value,
                        "kpi": record.kpi_value,
                        "frictions": record.frictions_detected,
                        "duration": round(record.duration_seconds, 2),
                        "started_at": record.started_at.isoformat(),
                    })
            if len(results) >= limit:
                break

        return results

    # ──────────────────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────────────────

    @staticmethod
    def _cron_to_frequency(cron: str) -> float:
        """
        Convertit une cron expression en intervalle d'heures.

        Simplifié — ne parse pas le cron complet, juste les patterns courants.
        """
        if cron.startswith("* "):
            return 0.0833  # ~5 min
        if cron.startswith("0 * "):
            return 1.0
        if "* * 1" in cron or "* * MON" in cron.upper():
            return 168.0  # weekly
        if cron.startswith("0 6 1 "):
            return 720.0  # monthly

        # Default: daily
        return 24.0
  
  
