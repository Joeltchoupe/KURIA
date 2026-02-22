"""
Messaging Adapter — Slack + Email (Resend).

Toutes les communications sortantes passent par ici.
"""

from __future__ import annotations

from typing import Any

from models.action import Action
from services.notifications import NotificationService


class MessagingAdapter:
    """
    Adapter de messaging unifié.

    Délègue à NotificationService pour l'envoi réel.
    Ajoute le formatage spécifique par type d'action.
    """

    def __init__(
        self,
        notification_service: NotificationService | None = None,
    ) -> None:
        self._notif = notification_service or NotificationService()

    # ──────────────────────────────────────────────────────
    # SLACK
    # ──────────────────────────────────────────────────────

    async def send_slack(self, action: Action) -> dict[str, Any]:
        """Envoie un message Slack."""
        channel = action.target or None
        message = action.parameters.get("message", "")

        record = await self._notif.send_slack(
            message=message,
            channel=channel,
        )

        return {
            "action": "send_slack",
            "channel": channel,
            "message_preview": message[:100],
            "status": record.status.value,
            "error": record.error,
        }

    # ──────────────────────────────────────────────────────
    # EMAIL
    # ──────────────────────────────────────────────────────

    async def send_email(self, action: Action) -> dict[str, Any]:
        """Envoie un email."""
        to = action.target
        subject = action.parameters.get("subject", "Notification Kuria")
        body = action.parameters.get("body", "")
        html = action.parameters.get("html")

        record = await self._notif.send_email(
            to=to,
            subject=subject,
            body=body,
            html=html,
        )

        return {
            "action": "send_email",
            "to": to,
            "subject": subject,
            "status": record.status.value,
            "error": record.error,
        }

    # ──────────────────────────────────────────────────────
    # ALERT (email + slack)
    # ──────────────────────────────────────────────────────

    async def send_alert(self, action: Action) -> dict[str, Any]:
        """Envoie une alerte (email + slack)."""
        message = action.parameters.get("message", "")
        severity = action.parameters.get("severity", "warning")
        recipients = action.parameters.get("recipients", [])
        channel = action.parameters.get("channel", "both")

        records = await self._notif.send_alert(
            message=message,
            severity=severity,
            recipients=recipients if recipients else None,
            channel=channel,
        )

        return {
            "action": "send_alert",
            "severity": severity,
            "channels_used": len(records),
            "statuses": [r.status.value for r in records],
        }

    # ──────────────────────────────────────────────────────
    # INVOICE REMINDER
    # ──────────────────────────────────────────────────────

    async def send_invoice_reminder(self, action: Action) -> dict[str, Any]:
        """Envoie un rappel de facture."""
        to = action.target
        invoice_id = action.parameters.get("invoice_id", "")
        amount = action.parameters.get("amount", 0)
        days_overdue = action.parameters.get("days_overdue", 0)
        company_name = action.parameters.get("company_name", "")

        subject = f"Rappel — Facture {invoice_id} en attente"
        body = (
            f"Bonjour,\n\n"
            f"La facture {invoice_id} d'un montant de {amount}€ "
            f"est en attente depuis {days_overdue} jours.\n\n"
            f"Merci de procéder au règlement.\n\n"
            f"Cordialement,\n{company_name}"
        )

        record = await self._notif.send_email(
            to=to,
            subject=subject,
            body=body,
        )

        return {
            "action": "send_invoice_reminder",
            "to": to,
            "invoice_id": invoice_id,
            "amount": amount,
            "days_overdue": days_overdue,
            "status": record.status.value,
            "error": record.error,
        }

    # ──────────────────────────────────────────────────────
    # CHECK-IN
    # ──────────────────────────────────────────────────────

    async def send_checkin(self, action: Action) -> dict[str, Any]:
        """Envoie un check-in à un membre de l'équipe."""
        to = action.target
        message = action.parameters.get("message", "")
        channel = action.parameters.get("channel", "slack")

        if channel == "slack":
            record = await self._notif.send_slack(message=message)
        else:
            record = await self._notif.send_email(
                to=to,
                subject="📋 Check-in Kuria",
                body=message,
            )

        return {
            "action": "send_checkin",
            "to": to,
            "channel": channel,
            "message_preview": message[:100],
            "status": record.status.value,
            "error": record.error,
      }
