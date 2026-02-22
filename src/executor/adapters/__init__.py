"""
KURIA Executor Adapters — Pont vers les APIs externes.

Chaque adapter encapsule UNE catégorie d'APIs :
  - crm.py       → HubSpot, Salesforce, Pipedrive
  - messaging.py → Slack, Email (Resend)
  - docs.py      → Notion, Google Drive
  - ads.py       → Google Ads, Meta Ads
"""

from executor.adapters.crm import CRMAdapter
from executor.adapters.messaging import MessagingAdapter
from executor.adapters.docs import DocsAdapter
from executor.adapters.ads import AdsAdapter

__all__ = [
    "CRMAdapter",
    "MessagingAdapter",
    "DocsAdapter",
    "AdsAdapter",
]
