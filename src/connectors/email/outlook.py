"""
Outlook / Microsoft 365 Connector — Métadonnées email via Microsoft Graph API.

Même philosophie que Gmail : métadonnées ONLY, jamais le contenu.
Utilise Microsoft Graph API v1.0.
OAuth 2.0 avec scopes : Mail.Read (lecture seule).

Rate limit : 10,000 req/10min (Microsoft Graph).

Output normalisé IDENTIQUE à Gmail :
même EmailMetadata, même ThreadAnalysis, même summary.
Les agents ne font AUCUNE différence.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import httpx

from connectors.base import (
    AuthenticationError,
    BaseConnector,
    ConnectorError,
    RateLimitError,
)
from connectors.utils import (
    chunk_list,
    clean_email,
    extract_domain,
    is_business_hours,
    normalize_date,
    retry_with_backoff,
)


logger = logging.getLogger("kuria.connectors.email.outlook")


class EmailMetadata:
    """Structure identique à Gmail — interchangeable."""

    __slots__ = (
        "message_id", "thread_id", "sender", "sender_domain",
        "recipients", "date", "subject_length", "is_reply",
        "is_internal", "is_business_hours", "labels", "has_attachment",
    )

    def __init__(self, **kwargs: Any):
        for slot in self.__slots__:
            setattr(self, slot, kwargs.get(slot))

    def to_dict(self) -> dict[str, Any]:
        return {slot: getattr(self, slot) for slot in self.__slots__}


class ThreadAnalysis:
    """Structure identique à Gmail — interchangeable."""

    __slots__ = (
        "thread_id", "message_count", "participants", "started_at",
        "last_message_at", "response_times_hours", "avg_response_time_hours",
        "is_resolved", "duration_hours",
    )

    def __init__(self, **kwargs: Any):
        for slot in self.__slots__:
            setattr(self, slot, kwargs.get(slot))

    def to_dict(self) -> dict[str, Any]:
        return {slot: getattr(self, slot) for slot in self.__slots__}


class OutlookConnector(BaseConnector):
    """Connecteur Outlook / Microsoft 365 — métadonnées uniquement."""

    CONNECTOR_NAME = "outlook"
    CONNECTOR_CATEGORY = "email"
    DATA_TYPES = ["messages", "threads", "summary"]

    BASE_URL = "https://graph.microsoft.com/v1.0"
    MAX_RESULTS_PER_PAGE = 100
    MAX_MESSAGES_TOTAL = 5000

    def __init__(self, company_id: str):
        super().__init__(company_id=company_id)
        self._client: Optional[httpx.AsyncClient] = None
        self._user_email: Optional[str] = None
        self._company_domain: Optional[str] = None

    async def authenticate(
        self,
        access_token: str,
        user_email: str = "me",
        company_domain: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Authentifie via OAuth2 Bearer token Microsoft Graph.

        Args:
            access_token: Bearer token Azure AD.
            user_email: Email de l'utilisateur ou "me".
            company_domain: Domaine de l'entreprise pour classifier interne/externe.
        """
        self._user_email = user_email
        self._company_domain = company_domain

        user_path = "me" if user_email == "me" else f"users/{user_email}"

        self._client = httpx.AsyncClient(
            base_url=f"{self.BASE_URL}/{user_path}",
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
                "Prefer": 'outlook.body-content-type="text"',
            },
            timeout=httpx.Timeout(30.0, connect=10.0),
        )

        try:
            response = await self._client.get("/mailFolders/inbox")
            if response.status_code == 401:
                raise AuthenticationError(
                    self.CONNECTOR_NAME, "Invalid or expired Microsoft token."
                )
            response.raise_for_status()

            # Récupérer le domaine si pas fourni
            if not self._company_domain:
                profile = await self._client.get("")
                if profile.status_code == 200:
                    email = profile.json().get("mail", "")
                    self._company_domain = extract_domain(email)

            self._authenticated = True
            self.logger.info(f"Authenticated for {user_email}")

        except httpx.HTTPStatusError as e:
            raise AuthenticationError(
                self.CONNECTOR_NAME,
                f"Microsoft Graph auth failed: {e.response.status_code}",
                raw_error=e,
            )

    async def disconnect(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
        await super().disconnect()

    async def health_check(self) -> bool:
        if not self._authenticated or not self._client:
            return False
        try:
            r = await self._client.get("/mailFolders/inbox")
            return r.status_code == 200
        except Exception:
            return False

    async def extract(self, days_back: int = 90) -> dict[str, Any]:
        """Extrait les métadonnées email depuis Microsoft Graph.

        Output identique à GmailConnector.extract().
        """
        self._require_auth()
        days_back = self._validate_days_back(days_back)
        metrics = self._start_metrics()

        since = datetime.now(tz=timezone.utc) - timedelta(days=days_back)
        since_iso = since.strftime("%Y-%m-%dT%H:%M:%SZ")

        try:
            # 1. Récupérer les messages
            raw_messages = await self._fetch_messages(since_iso)
            self.logger.info(f"Found {len(raw_messages)} messages")

            # 2. Parser les métadonnées
            messages = []
            for raw in raw_messages:
                meta = self._parse_message(raw)
                if meta:
                    messages.append(meta)

            # 3. Analyser les threads
            threads = self._analyze_threads(messages)

            # 4. Calculer le résumé
            summary = self._calculate_summary(messages, threads)

            metrics.complete(records=len(messages))

            return {
                "messages": [m.to_dict() for m in messages],
                "threads": [t.to_dict() for t in threads],
                "summary": summary,
                "extraction_metrics": metrics.model_dump(),
            }

        except (RateLimitError, AuthenticationError):
            raise
        except Exception as e:
            metrics.fail(str(e))
            raise ConnectorError(
                self.CONNECTOR_NAME, f"Outlook extraction failed: {e}", raw_error=e,
            )

    # ── Fetch ──

    @retry_with_backoff(max_retries=3, retryable_exceptions=(RateLimitError, httpx.RequestError))
    async def _api_get(self, endpoint: str, params: Optional[dict] = None) -> dict:
        assert self._client
        response = await self._client.get(endpoint, params=params)

        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", "60"))
            raise RateLimitError(self.CONNECTOR_NAME, retry_after)
        if response.status_code == 401:
            raise AuthenticationError(self.CONNECTOR_NAME, "Token expired.")

        response.raise_for_status()
        return response.json()

    async def _fetch_messages(self, since_iso: str) -> list[dict]:
        """Récupère les messages via Microsoft Graph avec pagination."""
        all_messages: list[dict] = []

        # Microsoft Graph utilise $filter et $top pour la pagination
        # $select pour ne récupérer que les champs nécessaires
        select_fields = (
            "id,conversationId,from,toRecipients,ccRecipients,"
            "receivedDateTime,subject,hasAttachments,categories,"
            "isRead,isDraft"
        )

        url = (
            f"/messages?"
            f"$filter=receivedDateTime ge {since_iso}&"
            f"$select={select_fields}&"
            f"$top={self.MAX_RESULTS_PER_PAGE}&"
            f"$orderby=receivedDateTime desc"
        )

        while url and len(all_messages) < self.MAX_MESSAGES_TOTAL:
            data = await self._api_get(url)

            messages = data.get("value", [])
            all_messages.extend(messages)

            # Pagination via @odata.nextLink
            url = data.get("@odata.nextLink")
            if url:
                # nextLink est une URL absolue, on doit extraire le path relatif
                if self.BASE_URL in url:
                    # Extraire tout après /me/ ou /users/{id}/
                    parts = url.split("/me/")
                    if len(parts) > 1:
                        url = "/" + parts[1]
                    else:
                        url = None

            if not messages:
                break

        return all_messages[:self.MAX_MESSAGES_TOTAL]


    # ── Parsing ──

    def _parse_message(self, raw: dict) -> Optional[EmailMetadata]:
        """Parse un message Microsoft Graph en EmailMetadata."""
        # Sender
        from_field = raw.get("from", {})
        sender_addr = from_field.get("emailAddress", {}).get("address", "")
        sender = clean_email(sender_addr)
        sender_domain = extract_domain(sender)

        # Recipients
        recipients = []
        for r in raw.get("toRecipients", []):
            addr = r.get("emailAddress", {}).get("address", "")
            cleaned = clean_email(addr)
            if cleaned:
                recipients.append(cleaned)
        for r in raw.get("ccRecipients", []):
            addr = r.get("emailAddress", {}).get("address", "")
            cleaned = clean_email(addr)
            if cleaned:
                recipients.append(cleaned)

        # Date
        date = normalize_date(raw.get("receivedDateTime"))

        # Internal
        is_internal = False
        if sender_domain and self._company_domain:
            is_internal = sender_domain == self._company_domain

        # Subject
        subject = raw.get("subject", "") or ""
        is_reply = subject.lower().startswith(("re:", "re :", "aw:", "sv:"))

        # Business hours
        biz_hours = True
        if date:
            biz_hours = is_business_hours(date)

        # Thread ID — Microsoft utilise conversationId
        thread_id = raw.get("conversationId", "")

        return EmailMetadata(
            message_id=raw.get("id", ""),
            thread_id=thread_id,
            sender=sender,
            sender_domain=sender_domain,
            recipients=recipients,
            date=date,
            subject_length=len(subject),
            is_reply=is_reply,
            is_internal=is_internal,
            is_business_hours=biz_hours,
            labels=raw.get("categories", []),
            has_attachment=raw.get("hasAttachments", False),
        )

    # ── Thread Analysis ──

    def _analyze_threads(self, messages: list[EmailMetadata]) -> list[ThreadAnalysis]:
        """Analyse identique à Gmail — même algorithme."""
        threads_map: dict[str, list[EmailMetadata]] = defaultdict(list)
        for msg in messages:
            if msg.thread_id:
                threads_map[msg.thread_id].append(msg)

        analyses: list[ThreadAnalysis] = []

        for thread_id, thread_messages in threads_map.items():
            if len(thread_messages) < 2:
                continue

            sorted_msgs = sorted(
                thread_messages,
                key=lambda m: m.date or datetime.min.replace(tzinfo=timezone.utc),
            )

            response_times: list[float] = []
            participants: set[str] = set()

            for i, msg in enumerate(sorted_msgs):
                if msg.sender:
                    participants.add(msg.sender)
                if i > 0 and sorted_msgs[i - 1].date and msg.date:
                    delta = (msg.date - sorted_msgs[i - 1].date).total_seconds()
                    hours = delta / 3600
                    if 0 < hours < 720:
                        response_times.append(hours)

            started = sorted_msgs[0].date
            ended = sorted_msgs[-1].date
            duration = 0.0
            if started and ended:
                duration = (ended - started).total_seconds() / 3600

            avg_response = 0.0
            if response_times:
                avg_response = sum(response_times) / len(response_times)

            analyses.append(ThreadAnalysis(
                thread_id=thread_id,
                message_count=len(thread_messages),
                participants=list(participants),
                started_at=started,
                last_message_at=ended,
                response_times_hours=response_times,
                avg_response_time_hours=round(avg_response, 2),
                is_resolved=True,
                duration_hours=round(duration, 2),
            ))

        return analyses

    # ── Summary ──

    def _calculate_summary(
        self,
        messages: list[EmailMetadata],
        threads: list[ThreadAnalysis],
    ) -> dict[str, Any]:
        """Résumé identique à Gmail — même format de sortie."""
        total = len(messages)
        if total == 0:
            return {
                "total_messages": 0,
                "total_threads": len(threads),
                "avg_response_time_hours": 0,
                "after_hours_pct": 0,
                "internal_pct": 0,
                "unanswered_count": 0,
                "busiest_senders": [],
                "busiest_recipients": [],
            }

        after_hours = sum(1 for m in messages if not m.is_business_hours)
        internal = sum(1 for m in messages if m.is_internal)

        all_response_times: list[float] = []
        for t in threads:
            if t.response_times_hours:
                all_response_times.extend(t.response_times_hours)
        avg_response = 0.0
        if all_response_times:
            avg_response = sum(all_response_times) / len(all_response_times)

        sender_counts: dict[str, int] = defaultdict(int)
        recipient_counts: dict[str, int] = defaultdict(int)
        for msg in messages:
            if msg.sender:
                sender_counts[msg.sender] += 1
            if msg.recipients:
                for r in msg.recipients:
                    recipient_counts[r] += 1

        busiest_senders = sorted(
            sender_counts.items(), key=lambda x: x[1], reverse=True
        )[:10]
        busiest_recipients = sorted(
            recipient_counts.items(), key=lambda x: x[1], reverse=True
        )[:10]

        # Unanswered
        thread_groups: dict[str, list[EmailMetadata]] = defaultdict(list)
        for msg in messages:
            if msg.thread_id:
                thread_groups[msg.thread_id].append(msg)

        unanswered = sum(
            1
            for tid, msgs in thread_groups.items()
            if len(msgs) == 1 and msgs[0].sender_domain == self._company_domain
        )

        return {
            "total_messages": total,
            "total_threads": len(threads),
            "avg_response_time_hours": round(avg_response, 2),
            "after_hours_pct": round(after_hours / total, 3) if total else 0,
            "internal_pct": round(internal / total, 3) if total else 0,
            "unanswered_count": unanswered,
            "busiest_senders": [
                {"email": email, "count": count} for email, count in busiest_senders
            ],
            "busiest_recipients": [
                {"email": email, "count": count} for email, count in busiest_recipients
            ],
              }
