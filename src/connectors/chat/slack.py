"""
Slack Connector — Messages, canaux, activité.

API Web v2. Bearer token (Bot).
Rate limit : tier 3 = 50/min pour la plupart.

Données pour Process Clarity :
- Canaux les plus actifs (où se passent les décisions ?)
- Messages par personne (qui est surchargé ?)
- Threads longs (sujets complexes)
- Activité after-hours

On ne lit PAS le contenu des messages. Métadonnées only.
"""

from __future__ import annotations
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
import httpx
from connectors.base import AuthenticationError, BaseConnector, ConnectorError, RateLimitError
from connectors.utils import normalize_date, retry_with_backoff, is_business_hours
import asyncio


class SlackConnector(BaseConnector):

    CONNECTOR_NAME = "slack"
    CONNECTOR_CATEGORY = "chat"
    DATA_TYPES = ["channels", "activity", "summary"]
    BASE_URL = "https://slack.com/api"

    def __init__(self, company_id: str):
        super().__init__(company_id=company_id)
        self._client: Optional[httpx.AsyncClient] = None

    async def authenticate(self, bot_token: str, **kwargs) -> None:
        self._client = httpx.AsyncClient(
            base_url=self.BASE_URL,
            headers={"Authorization": f"Bearer {bot_token}"},
            timeout=httpx.Timeout(30.0),
        )
        try:
            r = await self._client.get("/auth.test")
            data = r.json()
            if not data.get("ok"):
                raise AuthenticationError(self.CONNECTOR_NAME, data.get("error", "Auth failed"))
            self._authenticated = True
        except httpx.HTTPStatusError as e:
            raise AuthenticationError(self.CONNECTOR_NAME, str(e), raw_error=e)

    async def disconnect(self) -> None:
        if self._client: await self._client.aclose()
        await super().disconnect()

    async def health_check(self) -> bool:
        if not self._authenticated or not self._client: return False
        try:
            r = await self._client.get("/auth.test")
            return r.json().get("ok", False)
        except Exception: return False

    async def extract(self, days_back: int = 90) -> dict[str, Any]:
        self._require_auth()
        metrics = self._start_metrics()
        since_ts = str(int((datetime.now(tz=timezone.utc) - timedelta(days=days_back)).timestamp()))

        try:
            channels = await self._fetch_channels()
            channel_activity = []
            total_messages = 0
            after_hours_count = 0
            user_message_counts: dict[str, int] = defaultdict(int)

            for channel in channels[:30]:
                history = await self._fetch_channel_history(channel["id"], since_ts)
                msg_count = len(history)
                total_messages += msg_count

                for msg in history:
                    user = msg.get("user", "unknown")
                    user_message_counts[user] += 1
                    ts = float(msg.get("ts", "0"))
                    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                    if not is_business_hours(dt):
                        after_hours_count += 1

                channel_activity.append({
                    "channel_id": channel["id"],
                    "channel_name": channel.get("name", ""),
                    "member_count": channel.get("num_members", 0),
                    "message_count": msg_count,
                    "is_private": channel.get("is_private", False),
                })
                await asyncio.sleep(1.2)  # Rate limit

            channel_activity.sort(key=lambda c: c["message_count"], reverse=True)
            busiest_users = sorted(user_message_counts.items(), key=lambda x: x[1], reverse=True)[:10]

            summary = {
                "total_channels": len(channels),
                "active_channels": sum(1 for c in channel_activity if c["message_count"] > 0),
                "total_messages": total_messages,
                "after_hours_pct": round(after_hours_count / max(total_messages, 1), 3),
                "busiest_channels": channel_activity[:10],
                "busiest_users": [{"user_id": u, "messages": c} for u, c in busiest_users],
                "avg_messages_per_channel": round(total_messages / max(len(channels), 1), 1),
            }

            metrics.complete(records=total_messages)
            return {"channels": channel_activity, "summary": summary, "extraction_metrics": metrics.model_dump()}
        except Exception as e:
            metrics.fail(str(e))
            raise ConnectorError(self.CONNECTOR_NAME, str(e), raw_error=e)

    @retry_with_backoff(max_retries=3, retryable_exceptions=(RateLimitError, httpx.RequestError))
    async def _api_get(self, endpoint: str, params: Optional[dict] = None) -> dict:
        assert self._client
        r = await self._client.get(endpoint, params=params)
        data = r.json()
        if data.get("error") == "ratelimited":
            retry_after = int(r.headers.get("Retry-After", "30"))
            raise RateLimitError(self.CONNECTOR_NAME, retry_after)
        return data

    async def _fetch_channels(self) -> list[dict]:
        all_channels = []
        cursor = None
        while True:
            params: dict[str, Any] = {"types": "public_channel,private_channel", "limit": 200}
            if cursor: params["cursor"] = cursor
            data = await self._api_get("/conversations.list", params)
            all_channels.extend(data.get("channels", []))
            cursor = data.get("response_metadata", {}).get("next_cursor")
            if not cursor: break
        return all_channels

    async def _fetch_channel_history(self, channel_id: str, oldest: str) -> list[dict]:
        all_messages = []
        cursor = None
        while True:
            params: dict[str, Any] = {"channel": channel_id, "oldest": oldest, "limit": 200}
            if cursor: params["cursor"] = cursor
            data = await self._api_get("/conversations.history", params)
            if not data.get("ok"): break
            all_messages.extend(data.get("messages", []))
            if not data.get("has_more"): break
            cursor = data.get("response_metadata", {}).get("next_cursor")
            if not cursor: break
        return all_messages
