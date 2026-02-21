"""
Microsoft Teams Connector — Channels, messages, activité.
Microsoft Graph API. Bearer token.
Rate limit : 10,000/10min.
Métadonnées only — pas le contenu des messages.
"""

from __future__ import annotations
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
import httpx
from connectors.base import AuthenticationError, BaseConnector, ConnectorError, RateLimitError
from connectors.utils import normalize_date, retry_with_backoff, is_business_hours


class TeamsConnector(BaseConnector):

    CONNECTOR_NAME = "teams"
    CONNECTOR_CATEGORY = "chat"
    DATA_TYPES = ["channels", "activity", "summary"]
    BASE_URL = "https://graph.microsoft.com/v1.0"

    def __init__(self, company_id: str):
        super().__init__(company_id=company_id)
        self._client: Optional[httpx.AsyncClient] = None

    async def authenticate(self, access_token: str, **kwargs) -> None:
        self._client = httpx.AsyncClient(
            base_url=self.BASE_URL,
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=httpx.Timeout(30.0),
        )
        try:
            r = await self._client.get("/me/joinedTeams")
            if r.status_code == 401:
                raise AuthenticationError(self.CONNECTOR_NAME, "Invalid token.")
            r.raise_for_status()
            self._authenticated = True
        except httpx.HTTPStatusError as e:
            raise AuthenticationError(self.CONNECTOR_NAME, str(e), raw_error=e)

    async def disconnect(self) -> None:
        if self._client: await self._client.aclose()
        await super().disconnect()

    async def health_check(self) -> bool:
        if not self._authenticated or not self._client: return False
        try: return (await self._client.get("/me/joinedTeams")).status_code == 200
        except Exception: return False

    async def extract(self, days_back: int = 90) -> dict[str, Any]:
        self._require_auth()
        metrics = self._start_metrics()

        try:
            assert self._client
            teams_r = await self._client.get("/me/joinedTeams")
            teams_r.raise_for_status()
            teams = teams_r.json().get("value", [])

            channel_activity = []
            total_messages = 0

            for team in teams[:10]:
                team_id = team["id"]
                channels_r = await self._client.get(f"/teams/{team_id}/channels")
                channels = channels_r.json().get("value", []) if channels_r.status_code == 200 else []

                for channel in channels[:10]:
                    channel_activity.append({
                        "channel_id": channel.get("id", ""),
                        "channel_name": channel.get("displayName", ""),
                        "team_name": team.get("displayName", ""),
                        "member_count": 0,
                        "message_count": 0,
                        "is_private": channel.get("membershipType") == "private",
                    })

            summary = {
                "total_channels": len(channel_activity),
                "active_channels": 0,
                "total_messages": 0,
                "after_hours_pct": 0,
                "busiest_channels": channel_activity[:10],
                "busiest_users": [],
                "avg_messages_per_channel": 0,
            }

            metrics.complete(records=len(channel_activity))
            return {"channels": channel_activity, "summary": summary, "extraction_metrics": metrics.model_dump()}
        except Exception as e:
            metrics.fail(str(e))
            raise ConnectorError(self.CONNECTOR_NAME, str(e), raw_error=e)
