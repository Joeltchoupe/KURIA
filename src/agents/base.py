"""
BaseAgent — Pattern universel LLM-first.

CHAQUE agent suit le même cycle :

  Trigger
    → Build State Snapshot
    → Load prompt from file (system + user)
    → Send to LLM
    → Parse Decision (JSON structuré)
    → Validate (règles métier, safety)
    → Execute Actions (via ActionExecutor)
    → Log Everything

Le code est le plombier.
Les prompts (fichiers .txt) sont la logique métier.
Le LLM est le cerveau.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from models.event import Event
from models.state import StateSnapshot
from models.decision import Decision, DecisionType, ActionRequest, RiskLevel
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
      - agent_type             → son type
      - build_snapshot()       → quelles données il envoie au LLM
      - validate_decision()    → règles métier spécifiques
      - _default_decision_type → fallback
      - _custom_prompt_variables → variables pour les templates

    Le system_prompt est chargé AUTOMATIQUEMENT depuis
    prompts/system/{agent_type}.txt

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

        # Services
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
        ...

    @abstractmethod
    async def build_snapshot(
        self,
        events: list[Event] | None = None,
        raw_data: dict[str, Any] | None = None,
    ) -> StateSnapshot:
        ...

    @abstractmethod
    def validate_decision(self, decision: Decision) -> list[str]:
        ...

    @abstractmethod
    def _default_decision_type(self) -> DecisionType:
        ...

    # ──────────────────────────────────────────────────────
    # PROMPT LOADING (from files, not hardcoded)
    # ──────────────────────────────────────────────────────

    @property
    def system_prompt(self) -> str:
        """
        Charge le system prompt depuis prompts/system/{agent_type}.txt
        et injecte les variables de config.
        """
        template_name = f"system/{self.agent_type.value}"

        if self._prompts.exists(template_name):
            return self._prompts.render(
                template_name,
                **self._prompt_variables(),
            )

        return (
            f"You are Kuria {self.agent_type.value} AI agent. "
            f"Analyze the data provided and respond with valid JSON only."
        )

    def _prompt_variables(self) -> dict[str, Any]:
        """
        Variables injectées dans TOUS les prompts.

        Combine :
          - Paramètres de config (thresholds, seuils)
          - Variables custom de l'agent
          - Métadonnées (company_id, timestamp)
        """
        base_vars: dict[str, Any] = {
            "company_id": self.company_id,
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Config parameters
        if hasattr(self.config, "parameters") and self.config.parameters:
            base_vars.update(self.config.parameters)

        # Agent-specific variables
        base_vars.update(self._custom_prompt_variables())

        return base_vars

    def _custom_prompt_variables(self) -> dict[str, Any]:
        """
        Override par chaque agent pour ses variables spécifiques.

        Ces variables sont injectées dans les {{placeholders}}
        des templates de prompts.
        """
        return {}

    # ──────────────────────────────────────────────────────
    # RUN CYCLE
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

            # 2. Build prompt (from file)
            user_prompt = self._build_user_prompt(
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

    def _build_user_prompt(
        self,
        snapshot: StateSnapshot,
        prompt_name: str | None = None,
        extra_vars: dict[str, Any] | None = None,
    ) -> str:
        """Construit le user prompt depuis un fichier template."""
        variables = {
            **self._prompt_variables(),
            "state": snapshot.to_llm_payload(),
            **(extra_vars or {}),
        }

        if prompt_name and self._prompts.exists(prompt_name):
            return self._prompts.render(prompt_name, **variables)

        # Fallback : state brut
        return (
            f"Voici l'état actuel :\n\n"
            f"{snapshot.to_llm_payload()}\n\n"
            f"Analyse et produis ta décision au format JSON."
        )

    # ──────────────────────────────────────────────────────
    # LLM CALL
    # ──────────────────────────────────────────────────────

    async def _call_llm(self, user_prompt: str) -> LLMResponse:
        """Appelle le LLM avec le system prompt chargé depuis le fichier."""
        provider = None
        model = self.config.llm_model

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

        actions = []
        for a in data.get("actions", []):
            if isinstance(a, dict):
                try:
                    risk = RiskLevel(a.get("risk_level", "A"))
                except ValueError:
                    risk = RiskLevel.A
                actions.append(ActionRequest(
                    action=a.get("action", "unknown"),
                    target=a.get("target", ""),
                    parameters=a.get("parameters", {}),
                    risk_level=risk,
                    priority=a.get("priority", 5),
                ))

        try:
            decision_risk = RiskLevel(data.get("risk_level", "A"))
        except ValueError:
            decision_risk = RiskLevel.A

        return Decision(
            agent_type=self.agent_type.value,
            decision_type=self._map_decision_type(
                data.get("decision_type", data.get("action", "recommend"))
            ),
            confidence=float(data.get("confidence", 0.5)),
            reasoning=data.get("reasoning", "No reasoning provided"),
            actions=actions,
            risk_level=decision_risk,
            metadata=data.get("metadata", {}),
            snapshot_id=snapshot_id,
        )

    def _map_decision_type(self, raw: str) -> DecisionType:
        """Mappe une string brute vers un DecisionType."""
        try:
            return DecisionType(raw)
        except ValueError:
            return self._default_decision_type()

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
