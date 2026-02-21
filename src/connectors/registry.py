"""
ConnectorRegistry — Registre dynamique de tous les connecteurs.

Permet à l'orchestrateur de :
1. Lister les connecteurs disponibles par catégorie
2. Instancier le bon connecteur à partir du provider name
3. Vérifier si un outil est supporté

Design decisions :
- Auto-registration via decorator @register_connector
- Lazy import pour ne pas charger toutes les dépendances au boot
- Metadata riche pour le dashboard (nom, catégorie, types de données)
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Optional, Type

from connectors.base import BaseConnector

logger = logging.getLogger("kuria.connectors.registry")


# ──────────────────────────────────────────────
# REGISTRY
# ──────────────────────────────────────────────

# Mapping complet : provider_name → (module_path, class_name)
# C'est le CATALOGUE de tout ce que Kuria peut connecter.

_CONNECTOR_MAP: dict[str, tuple[str, str]] = {
    # CRM
    "hubspot": ("connectors.crm.hubspot", "HubSpotConnector"),
    "pipedrive": ("connectors.crm.pipedrive", "PipedriveConnector"),
    "salesforce": ("connectors.crm.salesforce", "SalesforceConnector"),
    "zoho": ("connectors.crm.zoho", "ZohoConnector"),

    # Email
    "gmail": ("connectors.email.gmail", "GmailConnector"),
    "outlook": ("connectors.email.outlook", "OutlookConnector"),

    # Finance
    "quickbooks": ("connectors.finance.quickbooks", "QuickBooksConnector"),
    "xero": ("connectors.finance.xero", "XeroConnector"),
    "pennylane": ("connectors.finance.pennylane", "PennylaneConnector"),
    "freshbooks": ("connectors.finance.freshbooks", "FreshBooksConnector"),
    "sage": ("connectors.finance.sage", "SageConnector"),

    # Payment
    "stripe": ("connectors.payment.stripe_connector", "StripeConnector"),
    "gocardless": ("connectors.payment.gocardless", "GoCardlessConnector"),

    # Project
    "asana": ("connectors.project.asana", "AsanaConnector"),
    "monday": ("connectors.project.monday", "MondayConnector"),
    "notion": ("connectors.project.notion", "NotionConnector"),
    "trello": ("connectors.project.trello", "TrelloConnector"),
    "clickup": ("connectors.project.clickup", "ClickUpConnector"),
    "linear": ("connectors.project.linear", "LinearConnector"),

    # Chat
    "slack": ("connectors.chat.slack", "SlackConnector"),
    "teams": ("connectors.chat.teams", "TeamsConnector"),

    # Marketing
    "google_ads": ("connectors.marketing.google_ads", "GoogleAdsConnector"),
    "meta_ads": ("connectors.marketing.meta_ads", "MetaAdsConnector"),
    "linkedin_ads": ("connectors.marketing.linkedin_ads", "LinkedInAdsConnector"),
    "mailchimp": ("connectors.marketing.mailchimp", "MailchimpConnector"),
    "brevo": ("connectors.marketing.brevo", "BrevoConnector"),

    # Calendar
    "google_calendar": ("connectors.calendar.google_calendar", "GoogleCalendarConnector"),
    "outlook_calendar": ("connectors.calendar.outlook_calendar", "OutlookCalendarConnector"),

    # Support
    "zendesk": ("connectors.support.zendesk", "ZendeskConnector"),
    "intercom": ("connectors.support.intercom", "IntercomConnector"),
    "crisp": ("connectors.support.crisp", "CrispConnector"),
    "freshdesk": ("connectors.support.freshdesk", "FreshdeskConnector"),

    # HR
    "personio": ("connectors.hr.personio", "PersonioConnector"),
    "lucca": ("connectors.hr.lucca", "LuccaConnector"),
    "payfit": ("connectors.hr.payfit", "PayFitConnector"),

    # Analytics
    "google_analytics": ("connectors.analytics.google_analytics", "GoogleAnalyticsConnector"),
    "mixpanel": ("connectors.analytics.mixpanel", "MixpanelConnector"),

    # Signature
    "docusign": ("connectors.signature.docusign", "DocuSignConnector"),
    "yousign": ("connectors.signature.yousign", "YousignConnector"),
    "pandadoc": ("connectors.signature.pandadoc", "PandaDocConnector"),

    # E-commerce
    "shopify": ("connectors.ecommerce.shopify", "ShopifyConnector"),
    "woocommerce": ("connectors.ecommerce.woocommerce", "WooCommerceConnector"),

    # Fallback
    "csv": ("connectors.csv_fallback", "CSVConnector"),
}

# Catégories
_CATEGORY_MAP: dict[str, list[str]] = {
    "crm": ["hubspot", "pipedrive", "salesforce", "zoho"],
    "email": ["gmail", "outlook"],
    "finance": ["quickbooks", "xero", "pennylane", "freshbooks", "sage"],
    "payment": ["stripe", "gocardless"],
    "project": ["asana", "monday", "notion", "trello", "clickup", "linear"],
    "chat": ["slack", "teams"],
    "marketing": ["google_ads", "meta_ads", "linkedin_ads", "mailchimp", "brevo"],
    "calendar": ["google_calendar", "outlook_calendar"],
    "support": ["zendesk", "intercom", "crisp", "freshdesk"],
    "hr": ["personio", "lucca", "payfit"],
    "analytics": ["google_analytics", "mixpanel"],
    "signature": ["docusign", "yousign", "pandadoc"],
    "ecommerce": ["shopify", "woocommerce"],
}


class ConnectorRegistry:
    """Registre central de tous les connecteurs Kuria.

    Usage :
        # Lister tout
        ConnectorRegistry.list_all()

        # Lister par catégorie
        ConnectorRegistry.list_by_category("crm")

        # Instancier
        connector = ConnectorRegistry.create("hubspot", company_id="acme")

        # Vérifier
        ConnectorRegistry.is_supported("hubspot")  # True
        ConnectorRegistry.is_supported("random_crm")  # False
    """

    @staticmethod
    def create(
        provider: str,
        company_id: str,
        **kwargs: Any,
    ) -> BaseConnector:
        """Instancie un connecteur par son nom de provider.

        Args:
            provider: Nom du provider (ex: "hubspot", "stripe")
            company_id: ID de l'entreprise cliente
            **kwargs: Arguments additionnels passés au constructeur

        Returns:
            Instance du connecteur.

        Raises:
            ConnectorError si le provider n'est pas supporté.
        """
        provider = provider.lower().strip()

        if provider not in _CONNECTOR_MAP:
            from connectors.base import ConnectorError
            raise ConnectorError(
                connector_name=provider,
                message=(
                    f"Provider '{provider}' not supported. "
                    f"Supported: {sorted(_CONNECTOR_MAP.keys())}"
                ),
                recoverable=False,
            )

        module_path, class_name = _CONNECTOR_MAP[provider]

        try:
            module = importlib.import_module(module_path)
            connector_class = getattr(module, class_name)
            return connector_class(company_id=company_id, **kwargs)
        except ImportError as e:
            logger.error(f"Failed to import connector {provider}: {e}")
            from connectors.base import ConnectorError
            raise ConnectorError(
                connector_name=provider,
                message=(
                    f"Connector '{provider}' is registered but its module "
                    f"could not be imported. Missing dependency? {e}"
                ),
                recoverable=False,
                raw_error=e,
            )
        except Exception as e:
            logger.error(f"Failed to create connector {provider}: {e}")
            from connectors.base import ConnectorError
            raise ConnectorError(
                connector_name=provider,
                message=f"Failed to create connector: {e}",
                recoverable=False,
                raw_error=e,
            )

    @staticmethod
    def is_supported(provider: str) -> bool:
        return provider.lower().strip() in _CONNECTOR_MAP

    @staticmethod
    def list_all() -> list[str]:
        return sorted(_CONNECTOR_MAP.keys())

    @staticmethod
    def list_by_category(category: str) -> list[str]:
        return _CATEGORY_MAP.get(category.lower(), [])

    @staticmethod
    def list_categories() -> list[str]:
        return sorted(_CATEGORY_MAP.keys())

    @staticmethod
    def get_category(provider: str) -> Optional[str]:
        provider = provider.lower()
        for category, providers in _CATEGORY_MAP.items():
            if provider in providers:
                return category
        return None

    @staticmethod
    def count() -> int:
        return len(_CONNECTOR_MAP)

    @staticmethod
    def summary() -> dict[str, int]:
        """Résumé par catégorie pour le dashboard."""
        return {
            category: len(providers)
            for category, providers in sorted(_CATEGORY_MAP.items())
}
