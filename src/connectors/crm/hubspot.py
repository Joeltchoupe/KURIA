"""
HubSpot Connector — Connecteur CRM pour Kuria.

C'est le connecteur PRIORITAIRE de la V1.
HubSpot est le CRM le plus utilisé par les PME cibles (20-500 personnes).

Ce connecteur extrait :
- Deals (pipeline, stages, montants, activités)
- Contacts (infos, source, engagement)
- Owners (commerciaux)
- Pipeline stages (configuration)

Il NE lit PAS le contenu des emails/notes (privacy).
Il lit les MÉTADONNÉES : dates, types d'activité, fréquence.

Design decisions :
- Async httpx pour les appels HTTP (pas requests — on est en async)
- Pagination automatique (HubSpot limite à 100 résultats par page)
- Rate limit handling avec retry (HubSpot = 100 calls/10s sur API privée)
- Normalisation stricte vers les modèles /models
- Aucune donnée brute ne sort du connecteur — tout est normalisé
"""

from __future__ import annotations

import logging
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
    normalize_date,
    retry_with_backoff,
    safe_float,
    safe_int,
    truncate,
)


logger = logging.getLogger("kuria.connectors.hubspot")


# ──────────────────────────────────────────────
# NORMALIZED STRUCTURES
# ──────────────────────────────────────────────
# Ces TypedDicts définissent la structure EXACTE
# que le connecteur retourne. Les agents lisent ÇA,
# jamais du JSON HubSpot brut.


class NormalizedDeal:
    """Structure normalisée d'un deal."""

    __slots__ = (
        "deal_id",
        "name",
        "stage",
        "pipeline",
        "amount",
        "close_date",
        "create_date",
        "owner_id",
        "owner_name",
        "is_closed",
        "is_won",
        "days_in_current_stage",
        "days_since_last_activity",
        "last_activity_date",
        "last_activity_type",
        "activity_count_30d",
        "has_next_step",
        "contact_ids",
        "source",
        "raw_properties",
    )

    def __init__(self, **kwargs: Any):
        for slot in self.__slots__:
            setattr(self, slot, kwargs.get(slot))

    def to_dict(self) -> dict[str, Any]:
        return {slot: getattr(self, slot) for slot in self.__slots__}


class NormalizedContact:
    """Structure normalisée d'un contact."""

    __slots__ = (
        "contact_id",
        "email",
        "first_name",
        "last_name",
        "company",
        "job_title",
        "lifecycle_stage",
        "lead_status",
        "source",
        "source_detail",
        "create_date",
        "last_activity_date",
        "owner_id",
        "associated_deal_ids",
    )

    def __init__(self, **kwargs: Any):
        for slot in self.__slots__:
            setattr(self, slot, kwargs.get(slot))

    def to_dict(self) -> dict[str, Any]:
        return {slot: getattr(self, slot) for slot in self.__slots__}


# ──────────────────────────────────────────────
# HUBSPOT CONNECTOR
# ──────────────────────────────────────────────


class HubSpotConnector(BaseConnector):
    """Connecteur HubSpot CRM.

    Authentification : OAuth2 access token (Bearer).
    Le token est stocké chiffré en DB par services/auth.py.
    Ce connecteur ne stocke PAS le token — il le reçoit à l'init.
    """
# En tête de classe, ajouter :
    CONNECTOR_CATEGORY = "crm"
    DATA_TYPES = ["deals", "contacts", "owners", "stages"]
    CONNECTOR_NAME = "hubspot"
    BASE_URL = "https://api.hubapi.com"
    MAX_RECORDS_PER_REQUEST = 100  # Limite HubSpot

    # Rate limits HubSpot : 100 requests / 10 secondes (OAuth private apps)
    RATE_LIMIT_REQUESTS = 100
    RATE_LIMIT_WINDOW_SECONDS = 10

    # Propriétés à récupérer pour chaque objet
    DEAL_PROPERTIES = [
        "dealname",
        "dealstage",
        "pipeline",
        "amount",
        "closedate",
        "createdate",
        "hs_lastmodifieddate",
        "hubspot_owner_id",
        "hs_is_closed",
        "hs_is_closed_won",
        "hs_deal_stage_probability",
        "notes_last_updated",
        "num_notes",
        "hs_next_step",
        "hs_analytics_source",
        "hs_analytics_source_data_1",
    ]

    CONTACT_PROPERTIES = [
        "email",
        "firstname",
        "lastname",
        "company",
        "jobtitle",
        "lifecyclestage",
        "hs_lead_status",
        "hs_analytics_source",
        "hs_analytics_source_data_1",
        "createdate",
        "notes_last_updated",
        "hubspot_owner_id",
    ]

    def __init__(self, company_id: str):
        super().__init__(company_id=company_id)
        self._client: Optional[httpx.AsyncClient] = None
        self._access_token: Optional[str] = None
        self._owners_cache: dict[str, str] = {}
        self._stages_cache: dict[str, dict[str, Any]] = {}
        self._request_count = 0
        self._window_start = datetime.utcnow()

    # ── Authentication ──

    async def authenticate(self, access_token: str, **kwargs: Any) -> None:
        """Authentifie avec un OAuth access token.

        Args:
            access_token: Bearer token HubSpot.

        Raises:
            AuthenticationError: Si le token est invalide.
        """
        self._access_token = access_token
        self._client = httpx.AsyncClient(
            base_url=self.BASE_URL,
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(30.0, connect=10.0),
        )

        # Vérifier que le token marche
        try:
            response = await self._client.get("/crm/v3/objects/contacts?limit=1")
            if response.status_code == 401:
                raise AuthenticationError(
                    connector_name=self.CONNECTOR_NAME,
                    message="Invalid or expired access token.",
                )
            response.raise_for_status()
            self._authenticated = True
            self.logger.info("Authenticated successfully")
        except httpx.HTTPStatusError as e:
            raise AuthenticationError(
                connector_name=self.CONNECTOR_NAME,
                message=f"Authentication check failed: {e.response.status_code}",
                raw_error=e,
            )
        except httpx.RequestError as e:
            raise ConnectorError(
                connector_name=self.CONNECTOR_NAME,
                message=f"Network error during authentication: {e}",
                raw_error=e,
            )

    async def disconnect(self) -> None:
        """Ferme le client HTTP."""
        if self._client:
            await self._client.aclose()
            self._client = None
        await super().disconnect()

    async def health_check(self) -> bool:
        """Vérifie que la connexion HubSpot est active."""
        if not self._authenticated or not self._client:
            return False
        try:
            response = await self._client.get("/crm/v3/objects/contacts?limit=1")
            return response.status_code == 200
        except Exception:
            return False

    # ── Extraction principale ──

    async def extract(self, days_back: int = 90) -> dict[str, Any]:
        """Extraction complète : deals + contacts + owners + stages.

        Returns:
            {
                "deals": [NormalizedDeal.to_dict(), ...],
                "contacts": [NormalizedContact.to_dict(), ...],
                "owners": {"owner_id": "owner_name", ...},
                "stages": {"stage_id": {"label": ..., "probability": ...}, ...},
                "metrics": ExtractionMetrics
            }
        """
        self._require_auth()
        days_back = self._validate_days_back(days_back)
        metrics = self._start_metrics()

        since = datetime.now(tz=timezone.utc) - timedelta(days=days_back)
        total_records = 0
        total_errors = 0

        try:
            # 1. Owners (cache pour enrichir les deals/contacts)
            self._owners_cache = await self._fetch_owners()
            self.logger.info(f"Fetched {len(self._owners_cache)} owners")

            # 2. Pipeline stages (cache pour enrichir les deals)
            self._stages_cache = await self._fetch_pipeline_stages()
            self.logger.info(f"Fetched {len(self._stages_cache)} pipeline stages")

            # 3. Deals
            raw_deals = await self._fetch_all_deals(since=since)
            normalized_deals = []
            for raw in raw_deals:
                try:
                    deal = self._normalize_deal(raw)
                    normalized_deals.append(deal.to_dict())
                except Exception as e:
                    total_errors += 1
                    self.logger.warning(
                        f"Failed to normalize deal {raw.get('id', '?')}: {e}"
                    )
            total_records += len(normalized_deals)
            self.logger.info(f"Extracted {len(normalized_deals)} deals")

            # 4. Contacts
            raw_contacts = await self._fetch_all_contacts(since=since)
            normalized_contacts = []
            for raw in raw_contacts:
                try:
                    contact = self._normalize_contact(raw)
                    normalized_contacts.append(contact.to_dict())
                except Exception as e:
                    total_errors += 1
                    self.logger.warning(
                        f"Failed to normalize contact {raw.get('id', '?')}: {e}"
                    )
            total_records += len(normalized_contacts)
            self.logger.info(f"Extracted {len(normalized_contacts)} contacts")

            metrics.complete(records=total_records, errors=total_errors)

            return {
                "deals": normalized_deals,
                "contacts": normalized_contacts,
                "owners": self._owners_cache,
                "stages": self._stages_cache,
                "extraction_metrics": metrics.model_dump(),
            }

        except RateLimitError:
            raise
        except Exception as e:
            metrics.fail(str(e))
            raise ConnectorError(
                connector_name=self.CONNECTOR_NAME,
                message=f"Extraction failed: {e}",
                raw_error=e,
            )

    # ── Fetch methods (API calls) ──

    @retry_with_backoff(
        max_retries=3,
        retryable_exceptions=(RateLimitError, httpx.RequestError),
    )
    async def _api_get(
        self,
        endpoint: str,
        params: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """GET request avec rate limit handling.

        Toutes les requêtes passent par ici.
        C'est LE point de contrôle pour le rate limiting.
        """
        self._require_auth()
        assert self._client is not None

        await self._respect_rate_limit()

        response = await self._client.get(endpoint, params=params)

        if response.status_code == 429:
            retry_after = safe_int(
                response.headers.get("Retry-After", "10"), default=10
            )
            raise RateLimitError(
                connector_name=self.CONNECTOR_NAME,
                retry_after_seconds=retry_after,
            )

        if response.status_code == 401:
            raise AuthenticationError(
                connector_name=self.CONNECTOR_NAME,
                message="Token expired during extraction.",
            )

        response.raise_for_status()
        self._request_count += 1
        return response.json()

    async def _api_post(
        self,
        endpoint: str,
        json_body: dict[str, Any],
    ) -> dict[str, Any]:
        """POST request avec rate limit handling."""
        self._require_auth()
        assert self._client is not None

        await self._respect_rate_limit()

        response = await self._client.post(endpoint, json=json_body)

        if response.status_code == 429:
            retry_after = safe_int(
                response.headers.get("Retry-After", "10"), default=10
            )
            raise RateLimitError(
                connector_name=self.CONNECTOR_NAME,
                retry_after_seconds=retry_after,
            )

        response.raise_for_status()
        self._request_count += 1
        return response.json()

    async def _respect_rate_limit(self) -> None:
        """Throttle les requêtes pour respecter le rate limit HubSpot."""
        import asyncio

        now = datetime.utcnow()
        elapsed = (now - self._window_start).total_seconds()

        if elapsed >= self.RATE_LIMIT_WINDOW_SECONDS:
            self._request_count = 0
            self._window_start = now
            return

        if self._request_count >= self.RATE_LIMIT_REQUESTS - 5:  # marge de 5
            wait_time = self.RATE_LIMIT_WINDOW_SECONDS - elapsed + 0.5
            self.logger.debug(f"Rate limit approaching, waiting {wait_time:.1f}s")
            await asyncio.sleep(wait_time)
            self._request_count = 0
            self._window_start = datetime.utcnow()

    async def _fetch_all_deals(
        self, since: datetime
    ) -> list[dict[str, Any]]:
        """Récupère tous les deals avec pagination."""
        all_deals: list[dict[str, Any]] = []
        after: Optional[str] = None
        since_ms = int(since.timestamp() * 1000)

        while True:
            params: dict[str, Any] = {
                "limit": self.MAX_RECORDS_PER_REQUEST,
                "properties": ",".join(self.DEAL_PROPERTIES),
            }
            if after:
                params["after"] = after

            data = await self._api_get("/crm/v3/objects/deals", params=params)

            results = data.get("results", [])
            for deal in results:
                # Filtrer par date de création ou de modification
                created = deal.get("properties", {}).get("createdate", "")
                modified = deal.get("properties", {}).get("hs_lastmodifieddate", "")

                created_dt = normalize_date(created)
                modified_dt = normalize_date(modified)

                # Garder le deal si créé OU modifié dans la fenêtre
                if created_dt and created_dt.timestamp() * 1000 >= since_ms:
                    all_deals.append(deal)
                elif modified_dt and modified_dt.timestamp() * 1000 >= since_ms:
                    all_deals.append(deal)

            # Pagination
            paging = data.get("paging", {})
            next_link = paging.get("next", {})
            after = next_link.get("after")

            if not after or not results:
                break

        return all_deals

    async def _fetch_all_contacts(
        self, since: datetime
    ) -> list[dict[str, Any]]:
        """Récupère tous les contacts avec pagination."""
        all_contacts: list[dict[str, Any]] = []
        after: Optional[str] = None
        since_ms = int(since.timestamp() * 1000)

        while True:
            params: dict[str, Any] = {
                "limit": self.MAX_RECORDS_PER_REQUEST,
                "properties": ",".join(self.CONTACT_PROPERTIES),
            }
            if after:
                params["after"] = after

            data = await self._api_get("/crm/v3/objects/contacts", params=params)

            results = data.get("results", [])
            for contact in results:
                created = contact.get("properties", {}).get("createdate", "")
                created_dt = normalize_date(created)

                if created_dt and created_dt.timestamp() * 1000 >= since_ms:
                    all_contacts.append(contact)

            paging = data.get("paging", {})
            next_link = paging.get("next", {})
            after = next_link.get("after")

            if not after or not results:
                break

        return all_contacts

    async def _fetch_owners(self) -> dict[str, str]:
        """Récupère la liste des owners (commerciaux)."""
        owners: dict[str, str] = {}
        data = await self._api_get("/crm/v3/owners/")

        for owner in data.get("results", []):
            owner_id = str(owner.get("id", ""))
            first = owner.get("firstName", "")
            last = owner.get("lastName", "")
            name = f"{first} {last}".strip() or owner.get("email", "Unknown")
            owners[owner_id] = name

        return owners

    async def _fetch_pipeline_stages(self) -> dict[str, dict[str, Any]]:
        """Récupère la configuration des pipelines et stages."""
        stages: dict[str, dict[str, Any]] = {}
        data = await self._api_get("/crm/v3/pipelines/deals")

        for pipeline in data.get("results", []):
            pipeline_id = pipeline.get("id", "")
            pipeline_label = pipeline.get("label", "")

            for stage in pipeline.get("stages", []):
                stage_id = stage.get("id", "")
                stages[stage_id] = {
                    "label": stage.get("label", ""),
                    "display_order": stage.get("displayOrder", 0),
                    "probability": safe_float(
                        stage.get("metadata", {}).get("probability"), default=0.0
                    ),
                    "is_closed": stage.get("metadata", {}).get("isClosed", "false")
                    == "true",
                    "pipeline_id": pipeline_id,
                    "pipeline_label": pipeline_label,
                }

        return stages

    async def _fetch_deal_engagements(
        self, deal_id: str, days_back: int = 30
    ) -> list[dict[str, Any]]:
        """Récupère les engagements (activités) d'un deal."""
        try:
            data = await self._api_get(
                f"/crm/v3/objects/deals/{deal_id}/associations/engagements"
            )
            return data.get("results", [])
        except Exception as e:
            self.logger.debug(
                f"Failed to fetch engagements for deal {deal_id}: {e}"
            )
            return []

    # ── Normalization ──

    def _normalize_deal(self, raw: dict[str, Any]) -> NormalizedDeal:
        """Transforme un deal HubSpot brut en NormalizedDeal."""
        props = raw.get("properties", {})
        deal_id = str(raw.get("id", ""))

        # Dates
        create_date = normalize_date(props.get("createdate"))
        close_date = normalize_date(props.get("closedate"))
        last_modified = normalize_date(props.get("hs_lastmodifieddate"))
        last_note = normalize_date(props.get("notes_last_updated"))

        # Dernière activité = max(last_modified, last_note)
        last_activity = last_modified
        if last_note and (last_activity is None or last_note > last_activity):
            last_activity = last_note

        # Stage info
        stage_id = props.get("dealstage", "")
        stage_info = self._stages_cache.get(stage_id, {})

        # Owner
        owner_id = props.get("hubspot_owner_id", "")
        owner_name = self._owners_cache.get(str(owner_id), "Unassigned")

        # Calculs temporels
        now = datetime.now(tz=timezone.utc)

        days_in_stage = 0
        if last_modified:
            # Approximation : days_in_stage ≈ jours depuis dernière modification
            # (HubSpot ne donne pas directement quand le deal est entré dans le stage)
            days_in_stage = max(0, (now - last_modified).days)

        days_since_activity = 0
        if last_activity:
            days_since_activity = max(0, (now - last_activity).days)

        is_closed = stage_info.get("is_closed", False)
        is_won = props.get("hs_is_closed_won", "false") == "true"

        return NormalizedDeal(
            deal_id=deal_id,
            name=truncate(props.get("dealname", "Untitled"), max_length=256),
            stage=stage_info.get("label", stage_id),
            pipeline=stage_info.get("pipeline_label", ""),
            amount=safe_float(props.get("amount"), default=0.0),
            close_date=close_date,
            create_date=create_date,
            owner_id=str(owner_id),
            owner_name=owner_name,
            is_clo
