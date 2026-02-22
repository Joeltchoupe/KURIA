"""
BaseAgent — Pattern universel LLM-first.

CHAQUE agent suit le même cycle :

  Trigger
    → Build State Snapshot
    → Send to LLM (avec system prompt spécifique)
    → Parse Decision (JSON structuré)
    → Validate (règles métier, safety)
    → Execute Actions (via ActionExecutor)
    → Log Everything

L'agent ne FAIT rien lui-même.
Il ORCHESTRE : state → LLM → decision → action.

Le code est le plombier.
Le LLM est le cerveau.
Les prompts sont la logique métier.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from models.event import Event
from models.state import StateSnapshot
from models.decision import Decision, ActionRequest, RiskLevel
from models.action import Action, ActionStatus, ActionLog
from models.agent_config import AgentConfig, AgentType
from services.config import get_settings, LLMProvider
from services.llm.client import LLMClient, LLMResponse
from services.llm.prompts import PromptManager
from services.llm.parser import ResponseParser, ParseError


class BaseAgent(ABC):
    """
    Agent de base — LLM-first.

    Chaque agent concret implémente :
      - agent_type      → son type
      - system_prompt    → son identité
      - build_snapshot() → quelles données il envoie au LLM
      - validate()       → règles métier spécifiques
      - action_map       → quelles actions il peut demander

    Le cycle run() est IDENTIQUE pour tous les agents.
    """

    def __init__(
        self,
        config: AgentConfig,
        llm: LLMClient | None = None,
        prompts: PromptManager | None = None,
        parser: ResponseParser | None = None,
        executor: Any | None = None,
    ) -> None:
        self.config = config
        self.company_id = config.company_id

        # Services (injectés ou créés)
        self._llm = llm or LLMClient()
        self._prompts = prompts or PromptManager()
        self._parser = parser or ResponseParser()
        self._executor = executor

        # State
        self._logs: list[ActionLog] = []
        self._decisions: list[Decision] = []
        self._pending_actions: list[Action] = []

    # ──────────────────────────────────────────────────────
    # ABSTRACT — Chaque agent implémente
    # ──────────────────────────────────────────────────────

    @property
    @abstractmethod
    def agent_type(self) -> AgentType:
        """Type de l'agent."""
        ...

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """System prompt de l'agent."""
        ...

    @abstractmethod
    async def build_snapshot(
        self,
        events: list[Event] | None = None,
        raw_data: dict[str, Any] | None = None,
    ) -> StateSnapshot:
        """
        Construit le state snapshot à envoyer au LLM.

        Chaque agent décide quelles données sont pertinentes.
        """
        ...

    @abstractmethod
    def validate_decision(self, decision: Decision) -> list[str]:
        """
        Valide une décision selon les règles métier.

        Returns:
            Liste d'erreurs. Vide = valide.
        """
        ...

    # ──────────────────────────────────────────────────────
    # RUN CYCLE — Identique pour tous les agents
    # ──────────────────────────────────────────────────────

    async def run(
        self,
        events: list[Event] | None = None,
        raw_data: dict[str, Any] | None = None,
        prompt_name: str | None = None,
        prompt_variables: dict[str, Any] | None = None,
    ) -> Decision:
        """
        Cycle complet : Snapshot → LLM → Decision → Validate → Execute.

        Args:
            events: Événements déclencheurs.
            raw_data: Données brutes additionnelles.
            prompt_name: Template de prompt à utiliser (optionnel).
            prompt_variables: Variables additionnelles pour le prompt.

        Returns:
            Decision produite par le LLM.
        """
        started_at = datetime.utcnow()
        log = ActionLog(
            company_id=self.company_id,
            agent_type=self.agent_type.value,
            event_type="agent_run",
        )

        try:
            # 1. Build snapshot
            snapshot = await self.build_snapshot(events, raw_data)
            log.input_snapshot = snapshot.summary

            # 2. Build prompt
            user_prompt = self._build_prompt(
                snapshot, prompt_name, prompt_variables
            )

            # 3. Call LLM
            llm_response = await self._call_llm(user_prompt)
            log.llm_provider = llm_response.provider.value
            log.llm_model = llm_response.model
            log.llm_tokens = llm_response.total_tokens
            log.llm_cost_usd = llm_response.cost_usd
            log.latency_ms = llm_response.latency_ms

            if not llm_response.success:
                raise RuntimeError(f"LLM call failed: {llm_response.error}")

            # 4. Parse decision
            decision = self._parse_decision(llm_response, snapshot.id)
            log.output_decision = decision.model_dump(
                include={"decision_type", "confidence", "risk_level", "reasoning"}
            )

            # 5. Validate
            errors = self.validate_decision(decision)
            decision.validation_errors = errors
            decision.validated = len(errors) == 0

            # 6. Execute (if safe)
            if decision.is_safe_to_execute and self._executor:
                await self._execute_actions(decision)
            elif decision.needs_approval:
                self._queue_for_approval(decision)

            self._decisions.append(decision)
            log.success = True
            log.description = (
                f"Decision: {decision.decision_type.value} "
                f"(confidence: {decision.confidence:.0%}, "
                f"risk: {decision.risk_level.value})"
            )

            return decision

        except Exception as e:
            log.success = False
            log.error = f"{type(e).__name__}: {str(e)}"

            # Return a failed decision
            return Decision(
                agent_type=self.agent_type.value,
                decision_type=self._default_decision_type(),
                confidence=0.0,
                reasoning=f"Agent error: {str(e)}",
                risk_level=RiskLevel.C,
                validated=False,
                validation_errors=[str(e)],
            )

        finally:
            elapsed = (datetime.utcnow() - started_at).total_seconds() * 1000
            log.latency_ms = max(log.latency_ms, elapsed)
            self._logs.append(log)

    # ──────────────────────────────────────────────────────
    # PROMPT BUILDING
    # ──────────────────────────────────────────────────────

    def _build_prompt(
        self,
        snapshot: StateSnapshot,
        prompt_name: str | None = None,
        extra_vars: dict[str, Any] | None = None,
    ) -> str:
        """Construit le user prompt à envoyer au LLM."""
        variables = {
            "state": snapshot.to_llm_payload(),
            "company_id": self.company_id,
            "timestamp": datetime.utcnow().isoformat(),
            **(extra_vars or {}),
        }

        if prompt_name and self._prompts.exists(prompt_name):
            return self._prompts.render(prompt_name, **variables)

        # Default : envoyer le state brut
        return (
            f"Voici l'état actuel de l'entreprise :\n\n"
            f"{snapshot.to_llm_payload()}\n\n"
            f"Analyse et produis ta décision au format JSON."
        )

    # ──────────────────────────────────────────────────────
    # LLM CALL
    # ──────────────────────────────────────────────────────

    async def _call_llm(self, user_prompt: str) -> LLMResponse:
        """Appelle le LLM avec le system prompt de l'agent."""
        provider = None
        model = self.config.llm_model

        # Déterminer le provider depuis le nom du modèle
        if "claude" in model.lower():
            provider = LLMProvider.CLAUDE
        elif "gpt" in model.lower():
            provider = LLMProvider.OPENAI
        elif "gemini" in model.lower():
            provider = LLMProvider.GEMINI

        return await self._llm.complete_structured(
            prompt=user_prompt,
            system=self.system_prompt,
            provider=provider,
            model=model,
        )

    # ──────────────────────────────────────────────────────
    # DECISION PARSING
    # ──────────────────────────────────────────────────────

    def _parse_decision(
        self, response: LLMResponse, snapshot_id: str
    ) -> Decision:
        """Parse la réponse LLM en Decision structurée."""
        try:
            data = self._parser.extract_json(response.content)
        except ParseError:
            # Le LLM n'a pas retourné du JSON valide
            return Decision(
                agent_type=self.agent_type.value,
                decision_type=self._default_decision_type(),
                confidence=0.3,
                reasoning=response.content[:500],
                risk_level=RiskLevel.C,
                snapshot_id=snapshot_id,
            )

        if not isinstance(data, dict):
            data = {"reasoning": str(data)}

        # Construire la decision
        actions = []
        for a in data.get("actions", []):
            if isinstance(a, dict):
                actions.append(ActionRequest(
                    action=a.get("action", "unknown"),
                    target=a.get("target", ""),
                    parameters=a.get("parameters", {}),
                    risk_level=RiskLevel(a.get("risk_level", "A")),
                    priority=a.get("priority", 5),
                ))

        return Decision(
            agent_type=self.agent_type.value,
            decision_type=self._map_decision_type(
                data.get("decision_type", data.get("action", "recommend"))
            ),
            confidence=float(data.get("confidence", 0.5)),
            reasoning=data.get("reasoning", "No reasoning provided"),
            actions=actions,
            risk_level=RiskLevel(data.get("risk_level", "A")),
            metadata=data.get("metadata", {}),
            snapshot_id=snapshot_id,
        )

    def _map_decision_type(self, raw: str) -> "DecisionType":
        """Mappe une string brute vers un DecisionType."""
        from models.decision import DecisionType
        try:
            return DecisionType(raw)
        except ValueError:
            return self._default_decision_type()

    @abstractmethod
    def _default_decision_type(self) -> "DecisionType":
        """DecisionType par défaut pour cet agent."""
        ...

    # ──────────────────────────────────────────────────────
    # EXECUTION
    # ──────────────────────────────────────────────────────

    async def _execute_actions(self, decision: Decision) -> None:
        """Exécute les actions d'une decision validée."""
        for action_req in decision.actions:
            action = Action(
                decision_id=decision.id,
                company_id=self.company_id,
                agent_type=self.agent_type.value,
                action=action_req.action,
                target=action_req.target,
                parameters=action_req.parameters,
                risk_level=action_req.risk_level.value,
                confidence=decision.confidence,
                status=ActionStatus.EXECUTING,
            )

            try:
                if self._executor:
                    result = await self._executor.execute(action)
                    action.complete(result)
                else:
                    action.status = ActionStatus.PENDING
            except Exception as e:
                action.fail(str(e))

        decision.executed = True
        decision.executed_at = datetime.utcnow()

    def _queue_for_approval(self, decision: Decision) -> None:
        """Met les actions B en attente d'approbation."""
        for action_req in decision.actions:
            action = Action(
                decision_id=decision.id,
                company_id=self.company_id,
                agent_type=self.agent_type.value,
                action=action_req.action,
                target=action_req.target,
                parameters=action_req.parameters,
                risk_level=action_req.risk_level.value,
                confidence=decision.confidence,
                status=ActionStatus.PENDING_APPROVAL,
            )
            action.set_expiry(hours=72)
            self._pending_actions.append(action)

    # ──────────────────────────────────────────────────────
    # ACCESSORS
    # ──────────────────────────────────────────────────────

    @property
    def last_decision(self) -> Decision | None:
        return self._decisions[-1] if self._decisions else None

    @property
    def pending_actions(self) -> list[Action]:
        return [
            a for a in self._pending_actions
            if a.status == ActionStatus.PENDING_APPROVAL and not a.is_expired
        ]

    @property
    def logs(self) -> list[ActionLog]:
        return list(self._logs)

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "agent": self.agent_type.value,
            "total_decisions": len(self._decisions),
            "total_logs": len(self._logs),
            "pending_actions": len(self.pending_actions),
            "avg_confidence": (
                sum(d.confidence for d in self._decisions) / len(self._decisions)
                if self._decisions else 0
            ),
      }
