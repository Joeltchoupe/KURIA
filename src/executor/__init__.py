"""
KURIA Executor — Le pont entre les décisions IA et le monde réel.

Le LLM DÉCIDE. L'Executor FAIT.

Flux :
  Decision.actions[] → ActionExecutor.execute() → Adapter → API externe

Safety :
  A → exécution directe
  B → queue pending_actions (72h)
  C → briefing only, jamais exécuté

Tout est loggé. Rien n'échappe à l'audit trail.
"""

from executor.engine import ActionExecutor

__all__ = [
    "ActionExecutor",
]
