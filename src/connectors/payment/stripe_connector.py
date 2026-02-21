"""
Stripe Connector — Paiements, abonnements, factures.

API REST v2023-10-16. Bearer token (Secret Key).
Rate limit : 100 req/s en mode live (très généreux).

Stripe est une mine d'or pour les agents :
- Revenus réels (pas déclarés, RÉELS)
- MRR / ARR calculable directement
- Churn détectable via les subscriptions
- Payment failures = signal d'alerte cash

Output normalisé vers le format finance standard :
invoices, bills (refunds), summary.
+ données spécifiques : subscriptions, MRR.
"""

from __future__ import annotations

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
from connectors.utils import normalize_date, retry_with_backoff, safe_float, safe_int


class StripeConnector(BaseConnector):

    CONNECTOR_NAME = "stripe"
    CONNECTOR_CATEGORY = "payment"
    DATA_TYPES = ["invoices", "subscriptions", "charges", "summary"]

    BASE_URL = "https://api.stripe.com/v1"

    def __init__(self, company_id: str):
        super().__init__(company_id=company_id)
        self._client: Optional[httpx.AsyncClient] = None

    async def authenticate(self, secret_key: str, **kwargs: Any) -> None:
        self._client = httpx.AsyncClient(
            base_url=self.BASE_URL,
            auth=(secret_key, ""),
            timeout=httpx.Timeout(30.0, connect=10.0),
        )
        try:
            r = await self._client.get("/balance")
            if r.status_code == 401:
                raise AuthenticationError(self.CONNECTOR_NAME, "Invalid Stripe key.")
            r.raise_for_status()
            self._authenticated = True
            self.logger.info("Stripe authenticated")
        except httpx.HTTPStatusError as e:
            raise AuthenticationError(self.CONNECTOR_NAME, str(e), raw_error=e)

    async def disconnect(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
        await super().disconnect()

    async def health_check(self) -> bool:
        if not self._authenticated or not self._client:
            return False
        try:
            return (await self._client.get("/balance")).status_code == 200
        except Exception:
            return False

    async def extract(self, days_back: int = 90) -> dict[str, Any]:
        self._require_auth()
        days_back = self._validate_days_back(days_back)
        metrics = self._start_metrics()
        since_ts = int((datetime.now(tz=timezone.utc) - timedelta(days=days_back)).timestamp())

        try:
            # Balance
            balance_data = await self._api_get("/balance")
            available = balance_data.get("available", [])
            cash_balance = sum(b.get("amount", 0) for b in available) / 100

            # Charges (paiements reçus)
            raw_charges = await self._paginate("/charges", {"created[gte]": since_ts})
            charges = [self._normalize_charge(c) for c in raw_charges]

            # Invoices
            raw_invoices = await self._paginate("/invoices", {"created[gte]": since_ts})
            invoices = [self._normalize_invoice(i) for i in raw_invoices]

            # Subscriptions actives
            raw_subs = await self._paginate("/subscriptions", {"status": "active"})
            subscriptions = [self._normalize_subscription(s) for s in raw_subs]

            # Refunds
            raw_refunds = await self._paginate("/refunds", {"created[gte]": since_ts})
            refunds = [self._normalize_refund(r) for r in raw_refunds]

            # MRR
            mrr = sum(s.get("mrr", 0) for s in subscriptions)

            # Summary (compatible format finance)
            summary = self._calculate_summary(
                cash_balance, charges, invoices, subscriptions, refunds, days_back
            )

            metrics.complete(
                records=len(charges) + len(invoices) + len(subscriptions)
            )

            return {
                "accounts": [{"account_id": "stripe_balance", "name": "Stripe Balance", "account_type": "PAYMENT", "balance": cash_balance, "currency": "EUR"}],
                "invoices": invoices,
                "bills": refunds,  # Refunds comme "sorties"
                "charges": charges,
                "subscriptions": subscriptions,
                "mrr": round(mrr, 2),
                "summary": summary,
                "extraction_metrics": metrics.model_dump(),
            }
        except (RateLimitError, AuthenticationError):
            raise
        except Exception as e:
            metrics.fail(str(e))
            raise ConnectorError(self.CONNECTOR_NAME, str(e), raw_error=e)

    @retry_with_backoff(max_retries=3, retryable_exceptions=(RateLimitError, httpx.RequestError))
    async def _api_get(self, endpoint: str, params: Optional[dict] = None) -> dict:
        assert self._client
        r = await self._client.get(endpoint, params=params)
        if r.status_code == 429:
            raise RateLimitError(self.CONNECTOR_NAME, 2)
        if r.status_code == 401:
            raise AuthenticationError(self.CONNECTOR_NAME, "Key expired.")
        r.raise_for_status()
        return r.json()

    async def _paginate(self, endpoint: str, params: Optional[dict] = None) -> list[dict]:
        all_items = []
        p = dict(params or {})
        p["limit"] = 100

        while True:
            data = await self._api_get(endpoint, p)
            items = data.get("data", [])
            all_items.extend(items)

            if not data.get("has_more", False) or not items:
                break
            p["starting_after"] = items[-1]["id"]

        return all_items

    def _normalize_charge(self, raw: dict) -> dict[str, Any]:
        return {
            "charge_id": raw.get("id", ""),
            "amount": raw.get("amount", 0) / 100,
            "currency": raw.get("currency", "eur").upper(),
            "status": raw.get("status", ""),
            "paid": raw.get("paid", False),
            "customer_id": raw.get("customer", ""),
            "customer_email": raw.get("billing_details", {}).get("email"),
            "date": normalize_date(raw.get("created")),
            "description": raw.get("description"),
            "failure_code": raw.get("failure_code"),
            "refunded": raw.get("refunded", False),
        }

    def _normalize_invoice(self, raw: dict) -> dict[str, Any]:
        amount = raw.get("amount_due", 0) / 100
        amount_paid = raw.get("amount_paid", 0) / 100
        amount_remaining = raw.get("amount_remaining", 0) / 100
        due_date = normalize_date(raw.get("due_date"))
        now = datetime.now(tz=timezone.utc)

        is_overdue = False
        days_overdue = 0
        if due_date and amount_remaining > 0:
            is_overdue = now > due_date
            if is_overdue:
                days_overdue = (now - due_date).days

        status_raw = raw.get("status", "")
        status = "paid" if status_raw == "paid" else ("overdue" if is_overdue else "pending")

        return {
            "invoice_id": raw.get("id", ""),
            "customer_name": raw.get("customer_name", raw.get("customer_email", "")),
            "amount": amount,
            "amount_due": amount_remaining,
            "issue_date": normalize_date(raw.get("created")),
            "due_date": due_date,
            "status": status,
            "is_overdue": is_overdue,
            "days_overdue": days_overdue,
            "days_to_payment": None,
            "currency": raw.get("currency", "eur").upper(),
        }

    def _normalize_subscription(self, raw: dict) -> dict[str, Any]:
        items = raw.get("items", {}).get("data", [])
        mrr = 0
        for item in items:
            price = item.get("price", {})
            unit_amount = (price.get("unit_amount", 0) or 0) / 100
            quantity = item.get("quantity", 1) or 1
            interval = price.get("recurring", {}).get("interval", "month")

            if interval == "year":
                mrr += (unit_amount * quantity) / 12
            elif interval == "month":
                mrr += unit_amount * quantity
            elif interval == "week":
                mrr += unit_amount * quantity * 4.33

        return {
            "subscription_id": raw.get("id", ""),
            "customer_id": raw.get("customer", ""),
            "status": raw.get("status", ""),
            "mrr": round(mrr, 2),
            "current_period_start": normalize_date(raw.get("current_period_start")),
            "current_period_end": normalize_date(raw.get("current_period_end")),
            "cancel_at_period_end": raw.get("cancel_at_period_end", False),
            "created": normalize_date(raw.get("created")),
        }

    def _normalize_refund(self, raw: dict) -> dict[str, Any]:
        amount = raw.get("amount", 0) / 100
        return {
            "bill_id": raw.get("id", ""),
            "vendor_name": "Stripe Refund",
            "amount": amount,
            "amount_due": 0,
            "issue_date": normalize_date(raw.get("created")),
            "due_date": None,
            "status": "paid",
            "is_overdue": False,
            "category": "refund",
            "currency": raw.get("currency", "eur").upper(),
        }

    def _calculate_summary(self, cash, charges, invoices, subs, refunds, days_back):
        months = max(1, days_back / 30)
        total_revenue = sum(c.get("amount", 0) for c in charges if c.get("paid"))
        total_refunds = sum(r.get("amount", 0) for r in refunds)
        net_revenue = total_revenue - total_refunds
        overdue = [i for i in invoices if i.get("is_overdue")]
        mrr = sum(s.get("mrr", 0) for s in subs)
        churn_risk = sum(1 for s in subs if s.get("cancel_at_period_end"))

        return {
            "cash_position": round(cash, 2),
            "total_receivable": round(sum(i.get("amount_due", 0) for i in invoices if i["status"] != "paid"), 2),
            "total_payable": round(total_refunds, 2),
            "invoices_overdue_count": len(overdue),
            "invoices_overdue_value": round(sum(i.get("amount_due", 0) for i in overdue), 2),
            "avg_client_payment_days": 0,
            "avg_supplier_payment_days": 0,
            "monthly_burn_rate": round(total_refunds / months, 2),
            "monthly_revenue_avg": round(net_revenue / months, 2),
            "revenue_recurring_pct": round(mrr * 12 / max(net_revenue, 1), 3) if net_revenue > 0 else 0,
            "runway_months": 99.0,
            "top_expenses": [],
            # Stripe-specific
            "mrr": round(mrr, 2),
            "arr": round(mrr * 12, 2),
            "active_subscriptions": len(subs),
            "churn_risk_count": churn_risk,
            "failed_charges": sum(1 for c in charges if not c.get("paid")),
            "total_charges": len(charges),
}
