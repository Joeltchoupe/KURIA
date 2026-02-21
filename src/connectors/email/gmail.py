"""
Gmail Connector — Extraction de métadonnées email.

IMPORTANT : Ce connecteur ne lit PAS le contenu des emails.
Il lit uniquement les MÉTADONNÉES :
- Expéditeur, destinataire, date, sujet
- Thread IDs (pour calculer les temps de réponse)
- Labels (pour catégoriser)

C'est un choix DÉLIBÉRÉ de privacy.
Le client connecte son Gmail, on ne lit que les enveloppes.

Design decisions :
- Google API client (google-api-python-client) pour OAuth
- Batch requests pour les performances (Gmail API les supporte)
- Calcul des temps de réponse basé sur les threads
- Détection after-hours basée sur les timestamps
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from connectors.base import (
    AuthenticationError,
    BaseConnector,
    ConnectorError,
    RateLimitError,
)
from connectors.utils import (
    clean_email,
    extract_domain,
    is_business_hours,
    normalize_date,
)


logger = logging.getLogger("kuria.connectors.gmail")


# ──────────────────────────────────────────────
# NORMALIZED STRUCTURES
# ──────────────────────────────────────────────


class EmailMetadata:
    """Métadonnées d'un email (pas le contenu)."""

    __slots__ = (
        "message_id",
        "thread_id",
        "sender",
        "sender_domain",
        "recipients",
        "date",
        "subject_length",
        "is_reply",
        "is_internal",
        "is_business_hours",
        "labels",
        "has_attachment",
    )

    def __init__(self, **kwargs: Any):
        for slot in self.__slots__:
            setattr(self, slot, kwargs.get(slot))

    def to_dict(self) -> dict[str, Any]:
        return {slot: getattr(self, slot) for slot in self.__slots__}


class ThreadAnalysis:
    """Analyse d'un thread email (conversation)."""

    __slots__ = (
        "thread_id",
        "message_count",
        "participants",
        "started_at",
        "last_message_at",
        "response_times_hours",
        "avg_response_time_hours",
        "is_resolved",
        "duration_hours",
    )

    def __init__(self, **kwargs: Any):
        for slot in self.__slots__:
            setattr(self, slot, kwargs.get(slot))

    def to_dict(self) -> dict[str, Any]:
        return {slot: getattr(self, slot) for slot in self.__slots__}


# ──────────────────────────────────────────────
# GMAIL CONNECTOR
# ──────────────────────────────────────────────


class GmailConnector(BaseConnector):
    """Connecteur Gmail — métadonnées uniquement.

    Authentification : OAuth2 credentials (Google API).
    Scope requis : gmail.readonly (lecture seule, métadonnées).
    """

    CONNECTOR_NAME = "gmail"
    MAX_RESULTS_PER_PAGE = 100
    MAX_MESSAGES_TOTAL = 5000  # Cap pour éviter les extractions trop longues

    def __init__(self, company_id: str):
        super().__init__(company_id=company_id)
        self._service: Any = None
        self._user_email: Optional[str] = None
        self._company_domain: Optional[str] = None

    async def authenticate(
        self,
        credentials_json: dict[str, Any],
        user_email: str = "me",
        company_domain: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Authentifie avec les credentials OAuth Google.

        Args:
            credentials_json: Le JSON des credentials OAuth.
            user_email: L'email de l'utilisateur (défaut: "me").
            company_domain: Le domaine de l'entreprise pour classifier
                           interne/externe.
        """
        try:
            from google.oauth2.credentials import Credentials
            from googleapiclient.discovery import build

            creds = Credentials.from_authorized_user_info(credentials_json)

            # Build est synchrone, on le wrappons
            loop = asyncio.get_event_loop()
            self._service = await loop.run_in_executor(
                None,
                lambda: build("gmail", "v1", credentials=creds),
            )

            self._user_email = user_email
            self._company_domain = company_domain

            # Test de connexion
            profile = await loop.run_in_executor(
                None,
                lambda: self._service.users()
                .getProfile(userId=user_email)
                .execute(),
            )

            if not self._company_domain:
                self._company_domain = extract_domain(
                    profile.get("emailAddress", "")
                )

            self._authenticated = True
            self.logger.info(
                f"Authenticated for {profile.get('emailAddress', user_email)}"
            )

        except ImportError:
            raise ConnectorError(
                connector_name=self.CONNECTOR_NAME,
                message=(
                    "google-api-python-client not installed. "
                    "Run: pip install google-api-python-client google-auth"
                ),
                recoverable=False,
            )
        except Exception as e:
            raise AuthenticationError(
                connector_name=self.CONNECTOR_NAME,
                message=f"Gmail authentication failed: {e}",
                raw_error=e,
            )

    async def health_check(self) -> bool:
        if not self._authenticated or not self._service:
            return False
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self._service.users()
                .getProfile(userId=self._user_email or "me")
                .execute(),
            )
            return True
        except Exception:
            return False

    async def extract(self, days_back: int = 90) -> dict[str, Any]:
        """Extrait les métadonnées email.

        Returns:
            {
                "messages": [EmailMetadata.to_dict(), ...],
                "threads": [ThreadAnalysis.to_dict(), ...],
                "summary": {
                    "total_messages": int,
                    "total_threads": int,
                    "avg_response_time_hours": float,
                    "after_hours_pct": float,
                    "internal_pct": float,
                    "unanswered_count": int,
                    "busiest_senders": [...],
                    "busiest_recipients": [...],
                },
                "extraction_metrics": {...}
            }
        """
        self._require_auth()
        days_back = self._validate_days_back(days_back)
        metrics = self._start_metrics()

        since = datetime.now(tz=timezone.utc) - timedelta(days=days_back)
        since_str = since.strftime("%Y/%m/%d")
        query = f"after:{since_str}"

        try:
            # 1. Récupérer la liste des message IDs
            message_ids = await self._list_message_ids(query)
            self.logger.info(f"Found {len(message_ids)} messages")

            # 2. Récupérer les métadonnées de chaque message
            messages = await self._fetch_messages_metadata(message_ids)
            self.logger.info(f"Fetched metadata for {len(messages)} messages")

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

        except Exception as e:
            metrics.fail(str(e))
            raise ConnectorError(
                connector_name=self.CONNECTOR_NAME,
                message=f"Gmail extraction failed: {e}",
                raw_error=e,
            )

    # ── Internal methods ──

    async def _list_message_ids(self, query: str) -> list[str]:
        """Récupère tous les message IDs correspondant à la query."""
        loop = asyncio.get_event_loop()
        message_ids: list[str] = []
        page_token: Optional[str] = None
        user = self._user_email or "me"

        while len(message_ids) < self.MAX_MESSAGES_TOTAL:
            def _list_page(pt=page_token):
                params = {
                    "userId": user,
                    "q": query,
                    "maxResults": self.MAX_RESULTS_PER_PAGE,
                }
                if pt:
                    params["pageToken"] = pt
                return self._service.users().messages().list(**params).execute()

            result = await loop.run_in_executor(None, _list_page)

            for msg in result.get("messages", []):
                message_ids.append(msg["id"])

            page_token = result.get("nextPageToken")
            if not page_token:
                break

        return message_ids[: self.MAX_MESSAGES_TOTAL]

    async def _fetch_messages_metadata(
        self, message_ids: list[str]
    ) -> list[EmailMetadata]:
        """Récupère les métadonnées de chaque message.

        Utilise des batches pour limiter les appels API.
        """
        loop = asyncio.get_event_loop()
        messages: list[EmailMetadata] = []
        user = self._user_email or "me"

        # Traiter par batch de 50 pour éviter les timeouts
        from connectors.utils import chunk_list

        for batch in chunk_list(message_ids, 50):
            tasks = []
            for msg_id in batch:
                tasks.append(
                    loop.run_in_executor(
                        None,
                        lambda mid=msg_id: self._service.users()
                        .messages()
                        .get(
                            userId=user,
                            id=mid,
                            format="metadata",
                            metadataHeaders=["From", "To", "Cc", "Date", "Subject"],
                        )
                        .execute(),
                    )
                )

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    self.logger.debug(f"Failed to fetch message: {result}")
                    continue

                metadata = self._parse_message_metadata(result)
                if metadata:
                    messages.append(metadata)

        return messages

    def _parse_message_metadata(
        self, raw: dict[str, Any]
    ) -> Optional[EmailMetadata]:
        """Parse les métadonnées brutes d'un message Gmail."""
        headers = {}
        for header in raw.get("payload", {}).get("headers", []):
            name = header.get("name", "").lower()
            value = header.get("value", "")
            headers[name] = value

        sender_raw = headers.get("from", "")
        sender = self._extract_email_from_header(sender_raw)
        sender_domain = extract_domain(sender)

        recipients_raw = headers.get("to", "")
        cc_raw = headers.get("cc", "")
        all_recipients = self._extract_emails_from_header(
            f"{recipients_raw}, {cc_raw}"
        )

        date = normalize_date(headers.get("date"))
        if date is None:
            date = normalize_date(raw.get("internalDate"))

        is_internal = False
        if sender_domain and self._company_domain:
            is_internal = sender_domain == self._company_domain

        subject = headers.get("subject", "")
        is_reply = subject.lower().startswith(("re:", "re :", "aw:"))

        biz_hours = True
        if date:
            biz_hours = is_business_hours(date)

        has_attachment = False
        parts = raw.get("payload", {}).get("parts", [])
        for part in parts:
            if part.get("filename"):
                has_attachment = True
                break

        return EmailMetadata(
            message_id=raw.get("id", ""),
            thread_id=raw.get("threadId", ""),
            sender=sender,
            sender_domain=sender_domain,
            recipients=all_recipients,
            date=date,
            subject_length=len(subject),
            is_reply=is_reply,
            is_internal=is_internal,
            is_business_hours=biz_hours,
            labels=raw.get("labelIds", []),
            has_attachment=has_attachment,
        )

    def _analyze_threads(
        self, messages: list[EmailMetadata]
    ) -> list[ThreadAnalysis]:
        """Analyse les threads pour calculer les temps de réponse."""
        # Grouper les messages par thread
        threads_map: dict[str, list[EmailMetadata]] = defaultdict(list)
        for msg in messages:
            if msg.thread_id:
                threads_map[msg.thread_id].append(msg)

        analyses: list[ThreadAnalysis] = []

        for thread_id, thread_messages in threads_map.items():
            if len(thread_messages) < 2:
                continue  # Un seul message = pas de conversation

            # Trier par date
            sorted_msgs = sorted(
                thread_messages,
                key=lambda m: m.date or datetime.min.replace(tzinfo=timezone.utc),
            )

            # Calculer les temps de réponse entre messages consécutifs
            response_times: list[float] = []
            participants: set[str] = set()

            for i, msg in enumerate(sorted_msgs):
                if msg.sender:
                    participants.add(msg.sender)

                if i > 0 and sorted_msgs[i - 1].date and msg.date:
                    delta = (msg.date - sorted_msgs[i - 1].date).total_seconds()
                    response_time_hours = delta / 3600
                    if 0 < response_time_hours < 720:  # Cap à 30 jours
                        response_times.append(response_time_hours)

            started = sorted_msgs[0].date
            ended = sorted_msgs[-1].date
            duration = 0.0
            if started and ended:
                duration = (ended - started).total_seconds() / 3600

            avg_response = 0.0
            if response_times:
                avg_response = sum(response_times) / len(response_times)

            analyses.append(
                ThreadAnalysis(
                    thread_id=thread_id,
                    message_count=len(thread_messages),
                    participants=list(participants),
                    started_at=started,
                    last_message_at=ended,
                    response_times_hours=response_times,
                    avg_response_time_hours=round(avg_response, 2),
                    is_resolved=True,  # Heuristique simple en V1
                    duration_hours=round(duration, 2),
                )
            )

        return analyses

    def _calculate_summary(
        self,
        messages: list[EmailMetadata],
        threads: list[ThreadAnalysis],
    ) -> dict[str, Any]:
        """Calcule le résumé agrégé des données email."""
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

        # After hours
        after_hours = sum(
            1 for m in messages if not m.is_business_hours
        )

        # Internal vs external
        internal = sum(1 for m in messages if m.is_internal)

        # Response times
        all_response_times: list[float] = []
        for t in threads:
            if t.response_times_hours:
                all_response_times.extend(t.response_times_hours)
        avg_response = 0.0
        if all_response_times:
            avg_response = sum(all_response_times) / len(all_response_times)

        # Busiest senders
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

        # Unanswered (threads avec 1 seul message envoyé par l'entreprise)
        unanswered = sum(
            1
            for t_id, t_msgs in self._group_by_thread(messages).items()
            if len(t_msgs) == 1
            and t_msgs[0].sender_domain == self._company_domain
        )

        return {
            "total_messages": total,
            "total_threads": len(threads),
            "avg_response_time_hours": round(avg_response, 2),
            "after_hours_pct": round(after_hours / total, 3) if total else 0,
            "internal_pct": round(internal / total, 3) if total else 0,
            "unanswered_count": unanswered,
            "busiest_senders": [
                {"email": email, "count": count}
                for email, count in busiest_senders
            ],
            "busiest_recipients": [
                {"email": email, "count": count}
                for email, count in busiest_recipients
            ],
        }

    def _group_by_thread(
        self, messages: list[EmailMetadata]
    ) -> dict[str, list[EmailMetadata]]:
        result: dict[str, list[EmailMetadata]] = defaultdict(list)
        for msg in messages:
            if msg.thread_id:
                result[msg.thread_id].append(msg)
        return result

    @staticmethod
    def _extract_email_from_header(header_value: str) -> Optional[str]:
        """Extrait l'email d'un header From/To.
        "John Doe <john@example.com>" → "john@example.com"
        "john@example.com" → "john@example.com"
        """
        if "<" in header_value and ">" in header_value:
            start = header_value.index("<") + 1
            end = header_value.index(">")
            return clean_email(header_value[start:end])
        return clean_email(header_value)

    @staticmethod
    def _extract_emails_from_header(header_value: str) -> list[str]:
        """Extrait toutes les emails d'un header To/Cc."""
        emails = []
        parts = header_value.split(",")
        for part in parts:
            email = GmailConnector._extract_email_from_header(part.strip())
            if email:
                emails.append(email)
        return emails
