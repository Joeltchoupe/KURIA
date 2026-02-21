"""
QuickBooks Connector — Extraction de données financières.

Extrait :
- Position cash (comptes bancaires)
- Factures émises (Invoices) — montants, dates, statuts
- Factures reçues (Bills) — dépenses, échéances
- Profit & Loss (revenus et dépenses par catégorie)
- Données récurrentes (abonnements, salaires)

Design decisions :
- QuickBooks Online API (pas Desktop)
- OAuth2 avec refresh token (les tokens QuickBooks expirent vite : 1h)
- Normalisation stricte des montants (QuickBooks utilise des strings...)
- Calcul du burn rate et du runway côté connecteur
  (ce sont des calculs FACTUELS, pas de l'analyse — c'est la responsabilité du connecteur)
"""

from __future__ import annotations

import asyncio
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
    normalize_date,
    retry_with_backoff,
    safe_float,
    safe_int,
)

logger = logging.getLogger("kuria.connectors.quickbooks")


# ──────────────────────────────────────────────
# NORMALIZED STRUCTURES
# ──────────────────────────────────────────────


class NormalizedInvoice:
    """Facture client normalisée."""

    __slots__ = (
        "invoice_id",
        "customer_name",
        "amount",
        "amount_due",
        "issue_date",
        "due_date",
        "status",
        "is_overdue",
        "days_overdue",
        "days_to_payment",
        "currency",
    )

    def __init__(self, **kwargs: Any):
        for slot in self.__slots__:
            setattr(self, slot, kwargs.get(slot))

    def to_dict(self) -> dict[str, Any]:
        return {slot: getattr(self, slot) for slot in self.__slots__}


class NormalizedBill:
    """Facture fournisseur normalisée."""

    __slots__ = (
        "bill_id",
        "vendor_name",
        "amount",
        "amount_due",
        "issue_date",
        "due_date",
        "status",
        "is_overdue",
        "category",
        "currency",
    )

    def __init__(self, **kwargs: Any):
        for slot in self.__slots__:
            setattr(self, slot, kwargs.get(slot))

    def to_dict(self) -> dict[str, Any]:
        return {slot: getattr(self, slot) for slot in self.__slots__}


class NormalizedAccount:
    """Compte bancaire normalisé."""

    __slots__ = (
        "account_id",
        "name",
        "account_type",
        "balance",
        "currency",
    )

    def __init__(self, **kwargs: Any):
        for slot in self.__slots__:
            setattr(self, slot, kwargs.get(slot))

    def to_dict(self) -> dict[str, Any]:
        return {slot: getattr(self, slot) for slot in self.__slots__}


# ──────────────────────────────────────────────
# QUICKBOOKS CONNECTOR
# ──────────────────────────────────────────────


class QuickBooksConnector(BaseConnector):
    """Connecteur QuickBooks Online.

    Authentification : OAuth2 Bearer token.
    Base URL dépend de l'environnement (sandbox vs production).
    """

    CONNECTOR_NAME = "quickbooks"
    BASE_URL_PROD = "https://quickbooks.api.intuit.com"
    BASE_URL_SANDBOX = "https://sandbox-quickbooks.api.intuit.com"
    API_VERSION = "v3"

    def __init__(self, company_id: str, sandbox: bool = False):
        super().__init__(company_id=company_id)
        self._client: Optional[httpx.AsyncClient] = None
        self._realm_id: Optional[str] = None
        self._base_url = self.BASE_URL_SANDBOX if sandbox else self.BASE_URL_PROD

    async def authenticate(
        self,
        access_token: str,
        realm_id: str,
        **kwargs: Any,
    ) -> None:
        """Authentifie avec un OAuth access token QuickBooks.

        Args:
            access_token: Bearer token.
            realm_id: L'ID du "realm" (company) QuickBooks.
        """
        self._realm_id = realm_id
        self._client = httpx.AsyncClient(
            base_url=f"{self._base_url}/{self.API_VERSION}/company/{realm_id}",
            headers={
                "Authorization": f"Bearer {access_token}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(30.0, connect=10.0),
        )

        try:
            response = await self._client.get("/companyinfo/" + realm_id)
            if response.status_code == 401:
                raise AuthenticationError(
                    connector_name=self.CONNECTOR_NAME,
                    message="Invalid or expired QuickBooks token.",
                )
            response.raise_for_status()
            self._authenticated = True
            self.logger.info(f"Authenticated for realm {realm_id}")
        except httpx.HTTPStatusError as e:
            raise AuthenticationError(
                connector_name=self.CONNECTOR_NAME,
                message=f"QuickBooks auth failed: {e.response.status_code}",
                raw_error=e,
            )

    async def disconnect(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
        await super().disconnect()

    async def health_check(self) -> bool:
        if not self._authenticated or not self._client or not self._realm_id:
            return False
        try:
            response = await self._client.get(
                f"/companyinfo/{self._realm_id}"
            )
            return response.status_code == 200
        except Exception:
            return False

    async def extract(self, days_back: int = 90) -> dict[str, Any]:
        """Extraction complète des données financières.

        Returns:
            {
                "accounts": [NormalizedAccount.to_dict(), ...],
                "invoices": [NormalizedInvoice.to_dict(), ...],
                "bills": [NormalizedBill.to_dict(), ...],
                "summary": {
                    "cash_position": float,
                    "total_receivable": float,
                    "total_payable": float,
                    "invoices_overdue_count": int,
                    "invoices_overdue_value": float,
                    "avg_client_payment_days": float,
                    "avg_supplier_payment_days": float,
                    "monthly_burn_rate": float,
                    "monthly_revenue_avg": float,
                    "revenue_recurring_pct": float,
                    "runway_months": float,
                    "top_expenses": [...],
                },
                "extraction_metrics": {...}
            }
        """
        self._require_auth()
        days_back = self._validate_days_back(days_back)
        metrics = self._start_metrics()

        since = datetime.now(tz=timezone.utc) - timedelta(days=days_back)
        since_str = since.strftime("%Y-%m-%d")
        total_records = 0

        try:
            # 1. Comptes bancaires
            accounts = await self._fetch_accounts()
            total_records += len(accounts)

            # 2. Factures clients
            invoices = await self._fetch_invoices(since_str)
            total_records += len(invoices)

            # 3. Factures fournisseurs
            bills = await self._fetch_bills(since_str)
            total_records += len(bills)

            # 4. Calculer le résumé
            summary = self._calculate_summary(
                accounts, invoices, bills, days_back
            )

            metrics.complete(records=total_records)

            return {
                "accounts": [a.to_dict() for a in accounts],
                "invoices": [i.to_dict() for i in invoices],
                "bills": [b.to_dict() for b in bills],
                "summary": summary,
                "extraction_metrics": metrics.model_dump(),
            }

        except (RateLimitError, AuthenticationError):
            raise
        except Exception as e:
            metrics.fail(str(e))
            raise ConnectorError(
                connector_name=self.CONNECTOR_NAME,
                message=f"QuickBooks extraction failed: {e}",
                raw_error=e,
            )

    # ── Fetch methods ──

    @retry_with_backoff(
        max_retries=3,
        retryable_exceptions=(RateLimitError, httpx.RequestError),
    )
    async def _qb_query(self, query: str) -> list[dict[str, Any]]:
        """Exécute une requête QuickBooks Query Language."""
        self._require_auth()
        assert self._client is not None

        response = await self._client.get(
            "/query",
            params={"query": query},
        )

        if response.status_code == 429:
            raise RateLimitError(
                connector_name=self.CONNECTOR_NAME,
                retry_after_seconds=60,
            )
        if response.status_code == 401:
            raise AuthenticationError(
                connector_name=self.CONNECTOR_NAME,
                message="QuickBooks token expired during extraction.",
            )

        response.raise_for_status()
        data = response.json()

        # QuickBooks wraps results in QueryResponse
        query_response = data.get("QueryResponse", {})

        # Le nom de la clé des résultats dépend du type d'objet requêté
        for key in query_response:
            if key not in ("startPosition", "maxResults", "totalCount"):
                return query_response.get(key, [])

        return []

    async def _fetch_accounts(self) -> list[NormalizedAccount]:
        """Récupère les comptes bancaires."""
        raw = await self._qb_query(
            "SELECT * FROM Account WHERE AccountType IN ('Bank', 'Other Current Asset')"
        )

        accounts = []
        for item in raw:
            accounts.append(
                NormalizedAccount(
                    account_id=str(item.get("Id", "")),
                    name=item.get("Name", ""),
                    account_type=item.get("AccountType", ""),
                    balance=safe_float(
                        item.get("CurrentBalance"), default=0.0
                    ),
                    currency=item.get("CurrencyRef", {}).get("value", "EUR"),
                )
            )
        return accounts

    async def _fetch_invoices(self, since: str) -> list[NormalizedInvoice]:
        """Récupère les factures clients."""
        raw = await self._qb_query(
            f"SELECT * FROM Invoice WHERE TxnDate >= '{since}' ORDERBY TxnDate DESC"
        )

        now = datetime.now(tz=timezone.utc)
        invoices = []

        for item in raw:
            amount = safe_float(item.get("TotalAmt"), default=0.0)
            amount_due = safe_float(item.get("Balance"), default=0.0)
            due_date = normalize_date(item.get("DueDate"))
            issue_date = normalize_date(item.get("TxnDate"))

            is_overdue = False
            days_overdue = 0
            if due_date and amount_due > 0:
                is_overdue = now > due_date
                if is_overdue:
                    days_overdue = (now - due_date).days

            # Calculer le délai de paiement réel (si payée)
            days_to_payment: Optional[int] = None
            if amount_due == 0 and issue_date:
                # Payée : estimer via la date de dernière modification
                modified = normalize_date(item.get("MetaData", {}).get("LastUpdatedTime"))
                if modified and issue_date:
                    days_to_payment = max(0, (modified - issue_date).days)

            # Statut
            balance = safe_float(item.get("Balance"), default=0.0)
            if balance == 0:
                status = "paid"
            elif is_overdue:
                status = "overdue"
            else:
                status = "pending"

            customer_name = ""
            customer_ref = item.get("CustomerRef")
            if customer_ref:
                customer_name = customer_ref.get("name", "")

            invoices.append(
                NormalizedInvoice(
                    invoice_id=str(item.get("Id", "")),
                    customer_name=customer_name,
                    amount=amount,
                    amount_due=amount_due,
                    issue_date=issue_date,
                    due_date=due_date,
                    status=status,
                    is_overdue=is_overdue,
                    days_overdue=days_overdue,
                    days_to_payment=days_to_payment,
                    currency=item.get("CurrencyRef", {}).get("value", "EUR"),
                )
            )
        return invoices

    async def _fetch_bills(self, since: str) -> list[NormalizedBill]:
        """Récupère les factures fournisseurs."""
        raw = await self._qb_query(
            f"SELECT * FROM Bill WHERE TxnDate >= '{since}' ORDERBY TxnDate DESC"
        )

        now = datetime.now(tz=timezone.utc)
        bills = []

        for item in raw:
            amount = safe_float(item.get("TotalAmt"), default=0.0)
            amount_due = safe_float(item.get("Balance"), default=0.0)
            due_date = normalize_date(item.get("DueDate"))

            is_overdue = False
            if due_date and amount_due > 0:
                is_overdue = now > due_date

            status = "paid" if amount_due == 0 else ("overdue" if is_overdue else "pending")

            vendor_name = ""
            vendor_ref = item.get("VendorRef")
            if vendor_ref:
                vendor_name = vendor_ref.get("name", "")

            # Catégorie (première ligne de la facture)
            category = "uncategorized"
            lines = item.get("Line", [])
            for line in lines:
                detail = line.get("AccountBasedExpenseLineDetail", {})
                account_ref = detail.get("AccountRef", {})
                if account_ref.get("name"):
                    category = account_ref["name"]
                    break

            bills.append(
                NormalizedBill(
                    bill_id=str(item.get("Id", "")),
                    vendor_name=vendor_name,
                    amount=amount,
                    amount_due=amount_due,
                    issue_date=normalize_date(item.get("TxnDate")),
                    due_date=due_date,
                    status=status,
                    is_overdue=is_overdue,
                    category=category,
                    currency=item.get("CurrencyRef", {}).get("value", "EUR"),
                )
            )
        return bills

    # ── Summary calculation ──

    def _calculate_summary(
        self,
        accounts: list[NormalizedAccount],
        invoices: list[NormalizedInvoice],
        bills: list[NormalizedBill],
        days_back: int,
    ) -> dict[str, Any]:
        """Calcule le résumé financier.

        Ce sont des CALCULS FACTUELS, pas de l'analyse.
        L'analyse (scénarios, alertes) est le job de l'agent Cash Predictability.
        """
        # Cash position
        cash_position = sum(a.balance or 0 for a in accounts)

        # Receivable & Payable
        total_receivable = sum(
            i.amount_due or 0 for i in invoices if i.status != "paid"
        )
        total_payable = sum(
            b.amount_due or 0 for b in bills if b.status != "paid"
        )

        # Overdue invoices
        overdue_invoices = [i for i in invoices if i.is_overdue]
        invoices_overdue_count = len(overdue_invoices)
        invoices_overdue_value = sum(i.amount_due or 0 for i in overdue_invoices)

        # Average payment days (clients)
        payment_days_list = [
            i.days_to_payment
            for i in invoices
            if i.days_to_payment is not None and i.days_to_payment > 0
        ]
        avg_client_payment_days = 0.0
        if payment_days_list:
            avg_client_payment_days = sum(payment_days_list) / len(payment_days_list)

        # Average supplier payment days
        supplier_days: list[int] = []
        for b in bills:
            if b.status == "paid" and b.issue_date and b.due_date:
                delta = (b.due_date - b.issue_date).days
                if 0 < delta < 365:
                    supplier_days.append(delta)
        avg_supplier_payment_days = 0.0
        if supplier_days:
            avg_supplier_payment_days = sum(supplier_days) / len(supplier_days)

        # Monthly burn rate & revenue
        months = max(1, days_back / 30)
        total_expenses = sum(b.amount or 0 for b in bills)
        total_revenue = sum(i.amount or 0 for i in invoices)
        monthly_burn = total_expenses / months
        monthly_revenue = total_revenue / months

        # Runway
        runway_months = 0.0
        net_monthly = monthly_revenue - monthly_burn
        if monthly_burn > 0 and net_monthly < 0:
            runway_months = cash_position / abs(net_monthly)
        elif net_monthly >= 0:
            runway_months = 99.0  # Cash-positive = pas de runway concern

        # Recurring revenue estimation (heuristique : même client, même montant, > 2x)
        from collections import Counter

        invoice_patterns = Counter()
        for inv in invoices:
            key = (inv.customer_name, round(inv.amount or 0, -1))  # Arrondi à 10
            invoice_patterns[key] += 1

        recurring_revenue = sum(
            (inv.amount or 0)
            for inv in invoices
            if invoice_patterns[(inv.customer_name, round(inv.amount or 0, -1))] >= 2
        )
        revenue_recurring_pct = 0.0
        if total_revenue > 0:
            revenue_recurring_pct = recurring_revenue / total_revenue

        # Top expenses par catégorie
        expense_by_category: dict[str, float] = {}
        for b in bills:
            cat = b.category or "uncategorized"
            expense_by_category[cat] = expense_by_category.get(cat, 0) + (b.amount or 0)

        top_expenses = sorted(
            expense_by_category.items(), key=lambda x: x[1], reverse=True
        )[:10]

        return {
            "cash_position": round(cash_position, 2),
            "total_receivable": round(total_receivable, 2),
            "total_payable": round(total_payable, 2),
            "invoices_overdue_count": invoices_overdue_count,
            "invoices_overdue_value": round(invoices_overdue_value, 2),
            "avg_client_payment_days": round(avg_client_payment_days, 1),
            "avg_supplier_payment_days": round(avg_supplier_payment_days, 1),
            "monthly_burn_rate": round(monthly_burn, 2),
            "monthly_revenue_avg": round(monthly_revenue, 2),
            "revenue_recurring_pct": round(revenue_recurring_pct, 3),
            "runway_months": round(min(runway_months, 99.0), 1),
            "top_expenses": [
                {
                    "category": cat,
                    "total": round(amount, 2),
                    "monthly_avg": round(amount / months, 2),
                    "pct_of_total": round(amount / total_expenses, 3)
                    if total_expenses > 0
                    else 0,
                }
                for cat, amount in top_expenses
            ],
  }
