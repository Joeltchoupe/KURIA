"""
Xero Connector — Comptabilité cloud populaire en Europe et Australie.

API REST 2.0. OAuth 2.0.
Rate limit : 60 calls/minute (par tenant).

Xero utilise un modèle similaire à QuickBooks :
- Invoices (factures clients)
- Bills (factures fournisseurs)
- BankTransactions (mouvements bancaires)
- Accounts (plan comptable)

Output normalisé IDENTIQUE à QuickBooks :
même structure accounts/invoices/bills/summary.
"""

from __future__ import annotations

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


logger = __import__("logging").getLogger("kuria.connectors.finance.xero")


class NormalizedInvoice:
    __slots__ = (
        "invoice_id", "customer_name", "amount", "amount_due",
        "issue_date", "due_date", "status", "is_overdue",
        "days_overdue", "days_to_payment", "currency",
    )

    def __init__(self, **kwargs):
        for slot in self.__slots__:
            setattr(self, slot, kwargs.get(slot))

    def to_dict(self) -> dict[str, Any]:
        return {slot: getattr(self, slot) for slot in self.__slots__}


class NormalizedBill:
    __slots__ = (
        "bill_id", "vendor_name", "amount", "amount_due",
        "issue_date", "due_date", "status", "is_overdue",
        "category", "currency",
    )

    def __init__(self, **kwargs):
        for slot in self.__slots__:
            setattr(self, slot, kwargs.get(slot))

    def to_dict(self) -> dict[str, Any]:
        return {slot: getattr(self, slot) for slot in self.__slots__}


class NormalizedAccount:
    __slots__ = ("account_id", "name", "account_type", "balance", "currency")

    def __init__(self, **kwargs):
        for slot in self.__slots__:
            setattr(self, slot, kwargs.get(slot))

    def to_dict(self) -> dict[str, Any]:
        return {slot: getattr(self, slot) for slot in self.__slots__}


class XeroConnector(BaseConnector):

    CONNECTOR_NAME = "xero"
    CONNECTOR_CATEGORY = "finance"
    DATA_TYPES = ["accounts", "invoices", "bills", "summary"]

    BASE_URL = "https://api.xero.com/api.xro/2.0"

    def __init__(self, company_id: str):
        super().__init__(company_id=company_id)
        self._client: Optional[httpx.AsyncClient] = None
        self._tenant_id: Optional[str] = None

    async def authenticate(
        self,
        access_token: str,
        tenant_id: str,
        **kwargs: Any,
    ) -> None:
        """Xero requiert un tenant_id (organisation) en plus du token."""
        self._tenant_id = tenant_id
        self._client = httpx.AsyncClient(
            base_url=self.BASE_URL,
            headers={
                "Authorization": f"Bearer {access_token}",
                "Xero-Tenant-Id": tenant_id,
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(30.0, connect=10.0),
        )

        try:
            r = await self._client.get("/Organisation")
            if r.status_code == 401:
                raise AuthenticationError(self.CONNECTOR_NAME, "Invalid Xero token.")
            r.raise_for_status()
            self._authenticated = True
            self.logger.info(f"Authenticated for tenant {tenant_id}")
        except httpx.HTTPStatusError as e:
            raise AuthenticationError(
                self.CONNECTOR_NAME, f"Xero auth failed: {e.response.status_code}", raw_error=e,
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
            return (await self._client.get("/Organisation")).status_code == 200
        except Exception:
            return False

    async def extract(self, days_back: int = 90) -> dict[str, Any]:
        self._require_auth()
        days_back = self._validate_days_back(days_back)
        metrics = self._start_metrics()
        since = datetime.now(tz=timezone.utc) - timedelta(days=days_back)
        since_str = since.strftime("%Y-%m-%d")
        total_records = 0

        try:
            accounts = await self._fetch_accounts()
            total_records += len(accounts)

            invoices = await self._fetch_invoices(since_str)
            total_records += len(invoices)

            bills = await self._fetch_bills(since_str)
            total_records += len(bills)

            summary = self._calculate_summary(accounts, invoices, bills, days_back)
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
            raise ConnectorError(self.CONNECTOR_NAME, f"Xero extraction failed: {e}", raw_error=e)

    @retry_with_backoff(max_retries=3, retryable_exceptions=(RateLimitError, httpx.RequestError))
    async def _api_get(self, endpoint: str, params: Optional[dict] = None) -> dict:
        assert self._client
        r = await self._client.get(endpoint, params=params)
        if r.status_code == 429:
            retry_after = int(r.headers.get("Retry-After", "60"))
            raise RateLimitError(self.CONNECTOR_NAME, retry_after)
        if r.status_code == 401:
            raise AuthenticationError(self.CONNECTOR_NAME, "Token expired.")
        r.raise_for_status()
        return r.json()

    async def _fetch_accounts(self) -> list[NormalizedAccount]:
        data = await self._api_get("/Accounts")
        accounts = []
        for a in data.get("Accounts", []):
            acc_type = a.get("Type", "")
            if acc_type in ("BANK", "CURRENT"):
                accounts.append(NormalizedAccount(
                    account_id=a.get("AccountID", ""),
                    name=a.get("Name", ""),
                    account_type=acc_type,
                    balance=safe_float(a.get("BankAccountBalance"), 0),
                    currency=a.get("CurrencyCode", "EUR"),
                ))
        return accounts

    async def _fetch_invoices(self, since: str) -> list[NormalizedInvoice]:
        """Xero Invoices = ACCREC (accounts receivable)."""
        data = await self._api_get(
            "/Invoices",
            params={
                "where": f'Type=="ACCREC" AND Date>=DateTime({since.replace("-", ",")})',
                "order": "Date DESC",
            },
        )
        now = datetime.now(tz=timezone.utc)
        invoices = []

        for inv in data.get("Invoices", []):
            amount = safe_float(inv.get("Total"), 0)
            amount_due = safe_float(inv.get("AmountDue"), 0)
            due_date = self._parse_xero_date(inv.get("DueDate"))
            issue_date = self._parse_xero_date(inv.get("Date"))

            is_overdue = False
            days_overdue = 0
            if due_date and amount_due > 0:
                is_overdue = now > due_date
                if is_overdue:
                    days_overdue = (now - due_date).days

            # Payment days
            days_to_payment = None
            status_raw = inv.get("Status", "")
            if status_raw == "PAID" and issue_date:
                paid_date = self._parse_xero_date(inv.get("FullyPaidOnDate"))
                if paid_date:
                    days_to_payment = max(0, (paid_date - issue_date).days)

            status = "paid" if status_raw == "PAID" else ("overdue" if is_overdue else "pending")

            contact = inv.get("Contact", {})
            invoices.append(NormalizedInvoice(
                invoice_id=inv.get("InvoiceID", ""),
                customer_name=contact.get("Name", ""),
                amount=amount,
                amount_due=amount_due,
                issue_date=issue_date,
                due_date=due_date,
                status=status,
                is_overdue=is_overdue,
                days_overdue=days_overdue,
                days_to_payment=days_to_payment,
                currency=inv.get("CurrencyCode", "EUR"),
            ))
        return invoices

    async def _fetch_bills(self, since: str) -> list[NormalizedBill]:
        """Xero Bills = ACCPAY (accounts payable)."""
        data = await self._api_get(
            "/Invoices",
            params={
                "where": f'Type=="ACCPAY" AND Date>=DateTime({since.replace("-", ",")})',
                "order": "Date DESC",
            },
        )
        now = datetime.now(tz=timezone.utc)
        bills = []

        for bill in data.get("Invoices", []):
            amount = safe_float(bill.get("Total"), 0)
            amount_due = safe_float(bill.get("AmountDue"), 0)
            due_date = self._parse_xero_date(bill.get("DueDate"))
            is_overdue = False
            if due_date and amount_due > 0:
                is_overdue = now > due_date

            status_raw = bill.get("Status", "")
            status = "paid" if status_raw == "PAID" else ("overdue" if is_overdue else "pending")

            contact = bill.get("Contact", {})

            # Catégorie (premier line item)
            category = "uncategorized"
            for line in bill.get("LineItems", []):
                if line.get("AccountCode"):
                    category = line.get("AccountCode", "uncategorized")
                    break

            bills.append(NormalizedBill(
                bill_id=bill.get("InvoiceID", ""),
                vendor_name=contact.get("Name", ""),
                amount=amount,
                amount_due=amount_due,
                issue_date=self._parse_xero_date(bill.get("Date")),
                due_date=due_date,
                status=status,
                is_overdue=is_overdue,
                category=category,
                currency=bill.get("CurrencyCode", "EUR"),
            ))
        return bills
