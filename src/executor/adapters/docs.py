"""
Docs Adapter — Notion, Google Drive.

Création de SOPs, rapports, propositions commerciales.
En V1 : Notion API.
"""

from __future__ import annotations

from typing import Any

from models.action import Action
from services.config import get_settings


class DocsAdapter:
    """
    Adapter pour la documentation.

    V1 : Notion API pour les SOPs.
    Fallback : retourne le contenu sans publier.
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._client: Any | None = None

    @property
    def is_connected(self) -> bool:
        return self._client is not None

    # ──────────────────────────────────────────────────────
    # SOPs
    # ──────────────────────────────────────────────────────

    async def create_sop(self, action: Action) -> dict[str, Any]:
        """Crée une SOP dans Notion."""
        title = action.parameters.get("title", "SOP sans titre")
        content = action.parameters.get("content", "")
        process_name = action.parameters.get("process_name", "")
        steps = action.parameters.get("steps", [])

        if not self.is_connected:
            return {
                "action": "create_sop",
                "offline": True,
                "title": title,
                "content_length": len(content),
                "steps_count": len(steps),
                "message": "Notion non connecté — SOP générée mais non publiée",
                "sop": {
                    "title": title,
                    "process_name": process_name,
                    "content": content,
                    "steps": steps,
                },
            }

        # TODO: Notion API call
        return {
            "action": "create_sop",
            "title": title,
            "published_to": "notion",
            "success": True,
        }

    async def update_sop(self, action: Action) -> dict[str, Any]:
        """Met à jour une SOP existante."""
        sop_id = action.target
        content = action.parameters.get("content", "")
        version = action.parameters.get("version", 2)

        if not self.is_connected:
            return {
                "action": "update_sop",
                "offline": True,
                "sop_id": sop_id,
                "version": version,
                "message": "Notion non connecté",
            }

        # TODO: Notion API call
        return {
            "action": "update_sop",
            "sop_id": sop_id,
            "version": version,
            "success": True,
        }

    # ──────────────────────────────────────────────────────
    # REPORTS
    # ──────────────────────────────────────────────────────

    async def create_report(self, action: Action) -> dict[str, Any]:
        """Génère un rapport."""
        title = action.parameters.get("title", "Rapport Kuria")
        content = action.parameters.get("content", "")
        report_type = action.parameters.get("type", "ops")

        return {
            "action": "create_report",
            "title": title,
            "type": report_type,
            "content_length": len(content),
            "content": content,
            "success": True,
        }

    # ──────────────────────────────────────────────────────
    # PROPOSALS
    # ──────────────────────────────────────────────────────

    async def generate_proposal(self, action: Action) -> dict[str, Any]:
        """Génère une proposition commerciale."""
        deal_name = action.parameters.get("deal_name", "")
        sections = action.parameters.get("sections", [])
        template = action.parameters.get("template", "default")

        # Le contenu est généré par le LLM.
        # Ici on ne fait que le formatter / stocker.
        proposal_md = self._format_proposal_markdown(deal_name, sections)

        return {
            "action": "generate_proposal",
            "deal_name": deal_name,
            "sections_count": len(sections),
            "format": "markdown",
            "content": proposal_md,
            "success": True,
        }

    @staticmethod
    def _format_proposal_markdown(
        deal_name: str, sections: list[dict[str, Any]]
    ) -> str:
        """Formate les sections en markdown."""
        lines = [
            f"# Proposition — {deal_name}",
            "",
            f"*Générée par Kuria 明晰*",
            "",
        ]

        for section in sections:
            title = section.get("title", "Section")
            content = section.get("content", "")
            lines.append(f"## {title}")
            lines.append("")
            lines.append(content)
            lines.append("")

        return "\n".join(lines)
