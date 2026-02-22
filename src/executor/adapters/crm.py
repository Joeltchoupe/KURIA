"""
CRM Adapter — HubSpot, Salesforce, Pipedrive.

Toutes les actions CRM passent par ici.
En V1 : HubSpot uniquement.
L'adapter abstrait le provider.
"""

from __future__ import annotations

from typing import Any

from models.action import Action
from services.config import get_settings


class CRMAdapter:
    """
    Adapter CRM.

    Encapsule les appels API vers HubSpot (V1).
    Chaque méthode reçoit une Action et retourne un dict résultat.
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._client: Any | None = None

        if self._settings.has_hubspot:
            self._init_client()

    def _init_client(self) -> None:
        """Initialise le client HubSpot."""
        try:
            import httpx
            self._client = httpx.AsyncClient(
                base_url="https://api.hubapi.com",
                headers={
                    "Authorization": f"Bearer {self._settings.hubspot_api_key}",
                    "Content-Type": "application/json",
                },
                timeout=30.0,
            )
        except ImportError:
            self._client = None

    @property
    def is_connected(self) -> bool:
        return self._client is not None

    # ──────────────────────────────────────────────────────
    # DEALS
    # ──────────────────────────────────────────────────────

    async def update_deal_stage(self, action: Action) -> dict[str, Any]:
        """Met à jour le stage d'un deal."""
        deal_id = action.target
        new_stage = action.parameters.get("stage", "")

        if not self.is_connected:
            return self._offline_result("update_deal_stage", action)

        response = await self._client.patch(
            f"/crm/v3/objects/deals/{deal_id}",
            json={"properties": {"dealstage": new_stage}},
        )

        return {
            "action": "update_deal_stage",
            "deal_id": deal_id,
            "new_stage": new_stage,
            "status_code": response.status_code,
            "success": response.status_code == 200,
        }

    async def add_deal_note(self, action: Action) -> dict[str, Any]:
        """Ajoute une note à un deal."""
        deal_id = action.target
        note = action.parameters.get("note", "")
        author = action.parameters.get("author", "Kuria AI")

        if not self.is_connected:
            return self._offline_result("add_deal_note", action)

        # HubSpot : créer une note (engagement)
        response = await self._client.post(
            "/crm/v3/objects/notes",
            json={
                "properties": {
                    "hs_note_body": f"[{author}] {note}",
                    "hs_timestamp": action.created_at.isoformat(),
                },
                "associations": [
                    {
                        "to": {"id": deal_id},
                        "types": [
                            {
                                "associationCategory": "HUBSPOT_DEFINED",
                                "associationTypeId": 214,
                            }
                        ],
                    }
                ],
            },
        )

        return {
            "action": "add_deal_note",
            "deal_id": deal_id,
            "note_preview": note[:100],
            "status_code": response.status_code,
            "success": response.status_code in (200, 201),
        }

    async def archive_deal(self, action: Action) -> dict[str, Any]:
        """Archive un deal (stage → perdu/archivé)."""
        deal_id = action.target
        reason = action.parameters.get("reason", "Archivé par Kuria")

        # D'abord ajouter une note avec la raison
        note_action = Action(
            id=action.id,
            decision_id=action.decision_id,
            company_id=action.company_id,
            agent_type=action.agent_type,
            action="add_deal_note",
            target=deal_id,
            parameters={"note": f"Deal archivé. Raison : {reason}"},
            created_at=action.created_at,
        )
        await self.add_deal_note(note_action)

        # Puis changer le stage
        archive_action = Action(
            id=action.id,
            decision_id=action.decision_id,
            company_id=action.company_id,
            agent_type=action.agent_type,
            action="update_deal_stage",
            target=deal_id,
            parameters={"stage": "closedlost"},
            created_at=action.created_at,
        )
        return await self.update_deal_stage(archive_action)

    async def create_deal_task(self, action: Action) -> dict[str, Any]:
        """Crée une tâche liée à un deal."""
        deal_id = action.target
        title = action.parameters.get("title", "Tâche Kuria")
        due_date = action.parameters.get("due_date")

        if not self.is_connected:
            return self._offline_result("create_deal_task", action)

        properties: dict[str, Any] = {
            "hs_task_subject": title,
            "hs_task_body": action.parameters.get("description", ""),
            "hs_task_status": "NOT_STARTED",
            "hs_task_priority": action.parameters.get("priority", "MEDIUM"),
        }
        if due_date:
            properties["hs_timestamp"] = due_date

        response = await self._client.post(
            "/crm/v3/objects/tasks",
            json={
                "properties": properties,
                "associations": [
                    {
                        "to": {"id": deal_id},
                        "types": [
                            {
                                "associationCategory": "HUBSPOT_DEFINED",
                                "associationTypeId": 216,
                            }
                        ],
                    }
                ],
            },
        )

        return {
            "action": "create_deal_task",
            "deal_id": deal_id,
            "title": title,
            "status_code": response.status_code,
            "success": response.status_code in (200, 201),
        }

    # ──────────────────────────────────────────────────────
    # CONTACTS
    # ──────────────────────────────────────────────────────

    async def update_contact(self, action: Action) -> dict[str, Any]:
        """Met à jour un contact."""
        contact_id = action.target
        properties = action.parameters.get("properties", {})

        if not self.is_connected:
            return self._offline_result("update_contact", action)

        response = await self._client.patch(
            f"/crm/v3/objects/contacts/{contact_id}",
            json={"properties": properties},
        )

        return {
            "action": "update_contact",
            "contact_id": contact_id,
            "properties_updated": list(properties.keys()),
            "status_code": response.status_code,
            "success": response.status_code == 200,
        }

    async def update_lead_score(self, action: Action) -> dict[str, Any]:
        """Met à jour le score d'un lead."""
        contact_id = action.target
        score = action.parameters.get("score", 0)
        reasoning = action.parameters.get("reasoning", "")

        if not self.is_connected:
            return self._offline_result("update_lead_score", action)

        response = await self._client.patch(
            f"/crm/v3/objects/contacts/{contact_id}",
            json={
                "properties": {
                    "hs_lead_status": self._score_to_status(score),
                    "kuria_lead_score": str(score),
                    "kuria_score_reasoning": reasoning[:500],
                }
            },
        )

        return {
            "action": "update_lead_score",
            "contact_id": contact_id,
            "score": score,
            "status_code": response.status_code,
            "success": response.status_code == 200,
        }

    # ──────────────────────────────────────────────────────
    # TASKS (generic, not deal-linked)
    # ──────────────────────────────────────────────────────

    async def create_task(self, action: Action) -> dict[str, Any]:
        """Crée une tâche générique."""
        title = action.parameters.get("title", "Tâche Kuria")

        if not self.is_connected:
            return self._offline_result("create_task", action)

        response = await self._client.post(
            "/crm/v3/objects/tasks",
            json={
                "properties": {
                    "hs_task_subject": title,
                    "hs_task_body": action.parameters.get("description", ""),
                    "hs_task_status": "NOT_STARTED",
                    "hs_task_priority": action.parameters.get("priority", "MEDIUM"),
                }
            },
        )

        return {
            "action": "create_task",
            "title": title,
            "status_code": response.status_code,
            "success": response.status_code in (200, 201),
        }

    async def assign_task(self, action: Action) -> dict[str, Any]:
        """Assigne une tâche à quelqu'un."""
        task_id = action.parameters.get("task_id", "")
        assignee = action.target

        if not self.is_connected:
            return self._offline_result("assign_task", action)

        response = await self._client.patch(
            f"/crm/v3/objects/tasks/{task_id}",
            json={
                "properties": {
                    "hubspot_owner_id": assignee,
                }
            },
        )

        return {
            "action": "assign_task",
            "task_id": task_id,
            "assignee": assignee,
            "status_code": response.status_code,
            "success": response.status_code == 200,
        }

    async def complete_task(self, action: Action) -> dict[str, Any]:
        """Marque une tâche comme complétée."""
        task_id = action.target

        if not self.is_connected:
            return self._offline_result("complete_task", action)

        response = await self._client.patch(
            f"/crm/v3/objects/tasks/{task_id}",
            json={
                "properties": {
                    "hs_task_status": "COMPLETED",
                }
            },
        )

        return {
            "action": "complete_task",
            "task_id": task_id,
            "status_code": response.status_code,
            "success": response.status_code == 200,
        }

    async def escalate_task(self, action: Action) -> dict[str, Any]:
        """Escalade une tâche (priorité haute + note)."""
        task_id = action.target
        reason = action.parameters.get("reason", "Escaladé par Kuria")

        if not self.is_connected:
            return self._offline_result("escalate_task", action)

        response = await self._client.patch(
            f"/crm/v3/objects/tasks/{task_id}",
            json={
                "properties": {
                    "hs_task_priority": "HIGH",
                    "hs_task_body": f"[ESCALADE] {reason}",
                }
            },
        )

        return {
            "action": "escalate_task",
            "task_id": task_id,
            "reason": reason,
            "status_code": response.status_code,
            "success": response.status_code == 200,
        }

    # ──────────────────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────────────────

    @staticmethod
    def _score_to_status(score: float) -> str:
        """Convertit un score en statut HubSpot."""
        if score >= 75:
            return "QUALIFIED"
        if score >= 50:
            return "OPEN"
        if score >= 25:
            return "IN_PROGRESS"
        return "UNQUALIFIED"

    @staticmethod
    def _offline_result(action_name: str, action: Action) -> dict[str, Any]:
        """Résultat quand le CRM n'est pas connecté."""
        return {
            "action": action_name,
            "offline": True,
            "target": action.target,
            "parameters": action.parameters,
            "message": "CRM non connecté — action loggée mais non exécutée",
  }
