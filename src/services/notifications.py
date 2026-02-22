"""
NotificationService — Resend (email) + Slack (webhook).

DEUX canaux, UNE interface.

Usage :
    from services.notifications import NotificationService

    notif = NotificationService()

    # Email
    await notif.send_email(
        to="ceo@acme.com",
        subject="Rapport Hebdo Kuria",
        body=report_text,
    )

    # Slack
    await notif.send_slack(
        message="🔴 Alerte : runway < 2 mois",
        channel="#kuria-alerts",
    )

    # Auto (utilise le canal configuré pour le client)
    await notif.notify(
        recipients=["ceo@acme.com"],
        channel="both",
        subject="Alerte Kuria",
        message="...",
    )
"""

from __future__ import annotations

import json
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

import httpx
from pydantic import BaseModel, Field

from services.config import get_settings


# ══════════════════════════════════════════════════════════════
# MODÈLES
# ══════════════════════════════════════════════════════════════


class NotificationChannel(str, Enum):
    EMAIL = "email"
    SLACK = "slack"
    BOTH = "both"


class NotificationStatus(str, Enum):
    SENT = "sent"
    FAILED = "failed"
    SKIPPED = "skipped"


class NotificationRecord(BaseModel):
    """Trace d'une notification envoyée."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    channel: NotificationChannel
    status: NotificationStatus
    recipient: str = ""
    subject: str = ""
    message_preview: str = ""
    sent_at: datetime = Field(default_factory=datetime.utcnow)
    error: str | None = None


# ══════════════════════════════════════════════════════════════
# SERVICE
# ══════════════════════════════════════════════════════════════


class NotificationService:
    """
    Service de notifications unifié.

    Gère Resend (email) et Slack (webhook).
    Trace chaque envoi.
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._history: list[NotificationRecord] = []

    @property
    def can_email(self) -> bool:
        return self._settings.has_resend

    @property
    def can_slack(self) -> bool:
        return self._settings.has_slack

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "total_sent": len([n for n in self._history if n.status == NotificationStatus.SENT]),
            "total_failed": len([n for n in self._history if n.status == NotificationStatus.FAILED]),
            "can_email": self.can_email,
            "can_slack": self.can_slack,
        }

    # ──────────────────────────────────────────────────────
    # EMAIL (Resend)
    # ──────────────────────────────────────────────────────

    async def send_email(
        self,
        to: str | list[str],
        subject: str,
        body: str,
        html: str | None = None,
    ) -> NotificationRecord:
        """
        Envoie un email via Resend.

        Args:
            to: Destinataire(s).
            subject: Sujet.
            body: Corps texte.
            html: Corps HTML (optionnel, prioritaire).
        """
        if not self.can_email:
            record = NotificationRecord(
                channel=NotificationChannel.EMAIL,
                status=NotificationStatus.SKIPPED,
                recipient=str(to),
                subject=subject,
                error="Resend non configuré",
            )
            self._history.append(record)
            return record

        recipients = [to] if isinstance(to, str) else to

        try:
            async with httpx.AsyncClient() as client:
                payload: dict[str, Any] = {
                    "from": self._settings.resend_from_email,
                    "to": recipients,
                    "subject": subject,
                }
                if html:
                    payload["html"] = html
                else:
                    payload["text"] = body

                if self._settings.resend_reply_to:
                    payload["reply_to"] = self._settings.resend_reply_to

                response = await client.post(
                    "https://api.resend.com/emails",
                    headers={
                        "Authorization": f"Bearer {self._settings.resend_api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=30.0,
                )

                if response.status_code in (200, 201):
                    record = NotificationRecord(
                        channel=NotificationChannel.EMAIL,
                        status=NotificationStatus.SENT,
                        recipient=", ".join(recipients),
                        subject=subject,
                        message_preview=body[:100],
                    )
                else:
                    record = NotificationRecord(
                        channel=NotificationChannel.EMAIL,
                        status=NotificationStatus.FAILED,
                        recipient=", ".join(recipients),
                        subject=subject,
                        error=f"HTTP {response.status_code}: {response.text[:200]}",
                    )

        except Exception as e:
            record = NotificationRecord(
                channel=NotificationChannel.EMAIL,
                status=NotificationStatus.FAILED,
                recipient=str(to),
                subject=subject,
                error=f"{type(e).__name__}: {str(e)}",
            )

        self._history.append(record)
        return record

    # ──────────────────────────────────────────────────────
    # SLACK (Webhook)
    # ──────────────────────────────────────────────────────

    async def send_slack(
        self,
        message: str,
        channel: str | None = None,
        blocks: list[dict[str, Any]] | None = None,
    ) -> NotificationRecord:
        """
        Envoie un message Slack via webhook.

        Args:
            message: Texte du message.
            channel: Channel override (optionnel).
            blocks: Blocs Slack riches (optionnel).
        """
        if not self.can_slack:
            record = NotificationRecord(
                channel=NotificationChannel.SLACK,
                status=NotificationStatus.SKIPPED,
                recipient=channel or self._settings.slack_default_channel,
                message_preview=message[:100],
                error="Slack webhook non configuré",
            )
            self._history.append(record)
            return record

        target_channel = channel or self._settings.slack_default_channel

        try:
            payload: dict[str, Any] = {"text": message}
            if channel:
                payload["channel"] = channel
            if blocks:
                payload["blocks"] = blocks

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self._settings.slack_webhook_url,
                    json=payload,
                    timeout=15.0,
                )

                if response.status_code == 200:
                    record = NotificationRecord(
                        channel=NotificationChannel.SLACK,
                        status=NotificationStatus.SENT,
                        recipient=target_channel,
                        message_preview=message[:100],
                    )
                else:
                    record = NotificationRecord(
                        channel=NotificationChannel.SLACK,
                        status=NotificationStatus.FAILED,
                        recipient=target_channel,
                        error=f"HTTP {response.status_code}: {response.text[:200]}",
                    )

        except Exception as e:
            record = NotificationRecord(
                channel=NotificationChannel.SLACK,
                status=NotificationStatus.FAILED,
                recipient=target_channel,
                message_preview=message[:100],
                error=f"{type(e).__name__}: {str(e)}",
            )

        self._history.append(record)
        return record

    # ──────────────────────────────────────────────────────
    # UNIFIED
    # ──────────────────────────────────────────────────────

    async def notify(
        self,
        message: str,
        subject: str = "Notification Kuria",
        recipients: list[str] | None = None,
        channel: str = "email",
    ) -> list[NotificationRecord]:
        """
        Envoie une notification via le canal configuré.

        Args:
            message: Contenu du message.
            subject: Sujet (email uniquement).
            recipients: Destinataires email (optionnel).
            channel: "email", "slack", ou "both".
        """
        records: list[NotificationRecord] = []

        if channel in ("email", "both") and recipients:
            record = await self.send_email(
                to=recipients,
                subject=subject,
                body=message,
            )
            records.append(record)

        if channel in ("slack", "both"):
            record = await self.send_slack(message=message)
            records.append(record)

        return records

    # ──────────────────────────────────────────────────────
    # TEMPLATES
    # ──────────────────────────────────────────────────────

    async def send_weekly_report(
        self,
        to: str | list[str],
        report_text: str,
        company_name: str = "",
    ) -> NotificationRecord:
        """Envoie le rapport hebdomadaire par email."""
        subject = f"📊 Kuria — Rapport Hebdomadaire{' — ' + company_name if company_name else ''}"
        return await self.send_email(to=to, subject=subject, body=report_text)

    async def send_alert(
        self,
        message: str,
        severity: str = "warning",
        recipients: list[str] | None = None,
        channel: str = "both",
    ) -> list[NotificationRecord]:
        """Envoie une alerte (email + slack)."""
        icons = {
            "critical": "🔴",
            "warning": "🟡",
            "info": "🔵",
        }
        icon = icons.get(severity, "⚪")
        formatted = f"{icon} ALERTE KURIA : {message}"

        return await self.notify(
            message=formatted,
            subject=f"{icon} Alerte Kuria — {severity.upper()}",
            recipients=recipients,
            channel=channel,
        )

    # ──────────────────────────────────────────────────────
    # HISTORY
    # ──────────────────────────────────────────────────────

    def get_history(self, limit: int = 20) -> list[dict[str, Any]]:
        """Retourne l'historique des notifications."""
        return [
            {
                "id": n.id,
                "channel": n.channel.value,
                "status": n.status.value,
                "recipient": n.recipient,
                "subject": n.subject,
                "sent_at": n.sent_at.isoformat(),
                "error": n.error,
            }
            for n in self._history[-limit:]
]
