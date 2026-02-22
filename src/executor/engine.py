"""
ActionExecutor — Routeur principal.

Reçoit une Action, la route vers le bon adapter,
exécute, logge le résultat.

Design decisions :
  - UN point d'entrée pour toutes les actions
  - Chaque action est mappée à un handler
  - Safety layer vérifié AVANT exécution
  - Dry-run mode pour les tests
  - Logging complet (avant + après)
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Awaitable
from uuid import uuid4

from models.action import Action, ActionStatus, ActionLog
from models.decision import RiskLevel
from services.config import get_settings

from executor.adapters.crm import CRMAdapter
from executor.adapters.messaging import MessagingAdapter
from executor.adapters.docs import DocsAdapter
from executor.adapters.ads import AdsAdapter


# Type pour les handlers
ActionHandler = Callable[[Action], Awaitable[dict[str, Any]]]


class ActionExecutor:
    """
    Routeur principal des actions.

    Usage :
        executor = ActionExecutor()
        result = await executor.execute(action)

    Dry-run :
        executor = ActionExecutor(dry_run=True)
        result = await executor.execute(action)  # log sans exécuter
    """

    def __init__(
        self,
        dry_run: bool = False,
        crm: CRMAdapter | None = None,
        messaging: MessagingAdapter | None = None,
        docs: DocsAdapter | None = None,
        ads: AdsAdapter | None = None,
    ) -> None:
        self.dry_run = dry_run

        # Adapters
        self._crm = crm or CRMAdapter()
        self._messaging = messaging or MessagingAdapter()
        self._docs = docs or DocsAdapter()
        self._ads = ads or AdsAdapter()

        # Logs
        self._logs: list[ActionLog] = []
        self._executed_count: int = 0
        self._failed_count: int = 0

        # Handler registry
        self._handlers: dict[str, ActionHandler] = self._build_registry()

    def _build_registry(self) -> dict[str, ActionHandler]:
        """Construit le mapping action → handler."""
        return {
            # CRM
            "update_deal_stage": self._crm.update_deal_stage,
            "add_deal_note": self._crm.add_deal_note,
            "archive_deal": self._crm.archive_deal,
            "create_deal_task": self._crm.create_deal_task,
            "update_contact": self._crm.update_contact,
            "update_lead_score": self._crm.update_lead_score,

            # Tasks
            "create_task": self._crm.create_task,
            "assign_task": self._crm.assign_task,
            "complete_task": self._crm.complete_task,
            "escalate_task": self._crm.escalate_task,

            # Messaging
            "send_slack": self._messaging.send_slack,
            "send_email": self._messaging.send_email,
            "send_alert": self._messaging.send_alert,
            "send_invoice_reminder": self._messaging.send_invoice_reminder,
            "send_checkin": self._messaging.send_checkin,

            # Docs
            "create_sop": self._docs.create_sop,
            "update_sop": self._docs.update_sop,
            "create_report": self._docs.create_report,
            "generate_proposal": self._docs.generate_proposal,

            # Ads
            "pause_channel": self._ads.pause_channel,
            "scale_channel": self._ads.scale_channel,
            "adjust_budget": self._ads.adjust_budget,
        }

    @property
    def supported_actions(self) -> list[str]:
        """Liste des actions supportées."""
        return sorted(self._handlers.keys())

    # ──────────────────────────────────────────────────────
    # EXECUTE
    # ──────────────────────────────────────────────────────

    async def execute(self, action: Action) -> dict[str, Any]:
        """
        Exécute une action.

        Séquence :
          1. Vérifier safety (A/B/C)
          2. Vérifier expiration
          3. Trouver le handler
          4. Exécuter (ou dry-run)
          5. Logger
          6. Retourner le résultat

        Returns:
            Résultat de l'exécution.
        """
        log = ActionLog(
            company_id=action.company_id,
            agent_type=action.agent_type,
            action_id=action.id,
            decision_id=action.decision_id,
            event_type=f"action.{action.action}",
        )

        try:
            # 1. Safety check
            safety_error = self._check_safety(action)
            if safety_error:
                action.status = ActionStatus.BRIEFING_ONLY
                log.description = f"Safety blocked: {safety_error}"
                log.success = False
                log.error = safety_error
                return {"blocked": True, "reason": safety_error}

            # 2. Expiration check
            if action.is_expired:
                action.status = ActionStatus.EXPIRED
                log.description = "Action expired"
                log.success = False
                log.error = "Action expirée"
                return {"expired": True}

            # 3. Find handler
            handler = self._handlers.get(action.action)
            if handler is None:
                action.fail(f"Action inconnue : {action.action}")
                log.description = f"Unknown action: {action.action}"
                log.success = False
                log.error = f"Action inconnue : {action.action}"
                return {"error": f"Action inconnue : {action.action}"}

            # 4. Execute or dry-run
            action.status = ActionStatus.EXECUTING

            if self.dry_run:
                result = {
                    "dry_run": True,
                    "action": action.action,
                    "target": action.target,
                    "parameters": action.parameters,
                    "would_execute": True,
                }
                action.complete(result)
                log.description = f"Dry-run: {action.action} → {action.target}"
            else:
                result = await handler(action)
                action.complete(result)
                log.description = f"Executed: {action.action} → {action.target}"

            log.output_result = result
            log.success = True
            self._executed_count += 1

            return result

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            action.fail(error_msg)
            log.success = False
            log.error = error_msg
            log.description = f"Failed: {action.action} — {error_msg}"
            self._failed_count += 1

            return {"error": error_msg}

        finally:
            elapsed = (datetime.utcnow() - log.timestamp).total_seconds() * 1000
            log.latency_ms = round(elapsed, 1)
            self._logs.append(log)

    async def execute_batch(
        self, actions: list[Action]
    ) -> list[dict[str, Any]]:
        """Exécute une liste d'actions dans l'ordre de priorité."""
        # Trier par priorité (1 = plus urgent)
        sorted_actions = sorted(
            actions,
            key=lambda a: a.parameters.get("priority", 5),
        )

        results = []
        for action in sorted_actions:
            result = await self.execute(action)
            results.append({
                "action_id": action.id,
                "action": action.action,
                "status": action.status.value,
                "result": result,
            })

        return results

    # ──────────────────────────────────────────────────────
    # SAFETY
    # ──────────────────────────────────────────────────────

    def _check_safety(self, action: Action) -> str | None:
        """
        Vérifie la safety d'une action.

        Returns:
            Message d'erreur si bloqué, None si OK.
        """
        risk = action.risk_level

        # C = jamais exécuté
        if risk == "C":
            return "Risk level C — briefing only, exécution interdite"

        # B = doit être approuvé
        if risk == "B":
            if action.status != ActionStatus.APPROVED:
                if action.status != ActionStatus.PENDING_APPROVAL:
                    action.status = ActionStatus.PENDING_APPROVAL
                    action.set_expiry(hours=72)
                return "Risk level B — en attente d'approbation"

        # A = go
        return None

    # ──────────────────────────────────────────────────────
    # APPROVAL MANAGEMENT
    # ──────────────────────────────────────────────────────

    async def approve_and_execute(
        self, action: Action, approved_by: str
    ) -> dict[str, Any]:
        """Approuve une action B et l'exécute."""
        action.approve(approved_by)
        return await self.execute(action)

    async def reject_action(
        self, action: Action, rejected_by: str, reason: str = ""
    ) -> None:
        """Rejette une action B."""
        action.reject(rejected_by, reason)

        log = ActionLog(
            company_id=action.company_id,
            agent_type=action.agent_type,
            action_id=action.id,
            decision_id=action.decision_id,
            event_type="action.rejected",
            description=f"Rejected by {rejected_by}: {reason}",
            success=True,
        )
        self._logs.append(log)

    # ──────────────────────────────────────────────────────
    # STATS
    # ──────────────────────────────────────────────────────

    @property
    def stats(self) -> dict[str, Any]:
        """Statistiques d'exécution."""
        return {
            "total_executed": self._executed_count,
            "total_failed": self._failed_count,
            "total_logs": len(self._logs),
            "dry_run": self.dry_run,
            "supported_actions": len(self._handlers),
        }

    def get_logs(self, limit: int = 20) -> list[dict[str, Any]]:
        """Retourne les derniers logs."""
        return [
            {
                "action_id": log.action_id,
                "event_type": log.event_type,
                "description": log.description,
                "success": log.success,
                "error": log.error,
                "latency_ms": log.latency_ms,
                "timestamp": log.timestamp.isoformat(),
            }
            for log in self._logs[-limit:]
]
