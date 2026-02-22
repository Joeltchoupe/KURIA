"""
PromptManager — Gestion des templates de prompts.

Les prompts sont stockés dans des fichiers texte sous prompts/.
Chaque agent a son dossier. Chaque prompt a des variables {{var}}.

Structure :
  prompts/
  ├── scanner/
  │   ├── initial_scan.txt
  │   └── data_quality.txt
  ├── revenue_velocity/
  │   ├── pipeline_analysis.txt
  │   ├── lead_scoring.txt
  │   └── forecast.txt
  ├── process_clarity/
  │   ├── process_detection.txt
  │   ├── bottleneck_analysis.txt
  │   └── ops_report.txt
  ├── cash_predictability/
  │   ├── cash_position.txt
  │   └── scenario_analysis.txt
  ├── acquisition_efficiency/
  │   ├── channel_analysis.txt
  │   └── cac_calculation.txt
  └── orchestrator/
      ├── profile_generation.txt
      └── adaptation.txt

Usage :
    from services.llm import PromptManager

    pm = PromptManager()
    prompt = pm.render(
        "revenue_velocity/pipeline_analysis",
        deals=deals_json,
        company_name="Acme Corp",
        cycle_moyen=35,
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from services.config import get_settings


class PromptTemplate(BaseModel):
    """Template de prompt chargé depuis un fichier."""
    name: str
    path: str
    content: str
    variables: list[str] = Field(default_factory=list)
    agent: str = ""
    version: str = "1.0"

    @property
    def variable_count(self) -> int:
        return len(self.variables)


class PromptManager:
    """
    Charge et rend les templates de prompts.

    Les templates utilisent {{variable}} pour les placeholders.
    """

    def __init__(
        self,
        prompts_dir: str | None = None,
    ) -> None:
        settings = get_settings()
        self._prompts_dir = Path(prompts_dir or settings.prompts_dir)
        self._cache: dict[str, PromptTemplate] = {}

    def render(
        self,
        template_name: str,
        **variables: Any,
    ) -> str:
        """
        Charge et rend un template de prompt.

        Args:
            template_name: Chemin relatif sans extension.
                Ex: "revenue_velocity/pipeline_analysis"
            **variables: Variables à injecter dans le template.

        Returns:
            Le prompt rendu avec les variables.

        Raises:
            FileNotFoundError: Si le template n'existe pas.
            ValueError: Si une variable requise manque.
        """
        template = self._load(template_name)
        return self._render(template, variables)

    def render_with_system(
        self,
        template_name: str,
        system_template: str | None = None,
        **variables: Any,
    ) -> tuple[str, str | None]:
        """
        Charge un prompt + un system prompt optionnel.

        Args:
            template_name: Template principal (user prompt).
            system_template: Template du system prompt (optionnel).
            **variables: Variables partagées entre les deux.

        Returns:
            Tuple (user_prompt, system_prompt).
        """
        user_prompt = self.render(template_name, **variables)

        system_prompt = None
        if system_template:
            system_prompt = self.render(system_template, **variables)

        return user_prompt, system_prompt

    def list_templates(self, agent: str | None = None) -> list[str]:
        """
        Liste les templates disponibles.

        Args:
            agent: Filtrer par agent (optionnel).
        """
        if not self._prompts_dir.exists():
            return []

        if agent:
            agent_dir = self._prompts_dir / agent
            if not agent_dir.exists():
                return []
            return [
                f"{agent}/{f.stem}"
                for f in agent_dir.glob("*.txt")
            ]

        templates = []
        for agent_dir in self._prompts_dir.iterdir():
            if agent_dir.is_dir():
                for f in agent_dir.glob("*.txt"):
                    templates.append(f"{agent_dir.name}/{f.stem}")

        return sorted(templates)

    def exists(self, template_name: str) -> bool:
        """Vérifie si un template existe."""
        path = self._resolve_path(template_name)
        return path.exists()

    # ──────────────────────────────────────────────────────
    # INLINE TEMPLATES (pas de fichier)
    # ──────────────────────────────────────────────────────

    def render_inline(
        self,
        template: str,
        **variables: Any,
    ) -> str:
        """
        Rend un template inline (string directe, pas un fichier).

        Utile pour les tests ou les prompts dynamiques.
        """
        tpl = PromptTemplate(
            name="inline",
            path="",
            content=template,
            variables=self._extract_variables(template),
        )
        return self._render(tpl, variables)

    # ──────────────────────────────────────────────────────
    # INTERNAL
    # ──────────────────────────────────────────────────────

    def _load(self, template_name: str) -> PromptTemplate:
        """Charge un template depuis le cache ou le disque."""
        if template_name in self._cache:
            return self._cache[template_name]

        path = self._resolve_path(template_name)
        if not path.exists():
            raise FileNotFoundError(
                f"Template non trouvé : {template_name} "
                f"(cherché dans {path})"
            )

        content = path.read_text(encoding="utf-8")
        variables = self._extract_variables(content)

        parts = template_name.split("/")
        agent = parts[0] if len(parts) > 1 else ""

        template = PromptTemplate(
            name=template_name,
            path=str(path),
            content=content,
            variables=variables,
            agent=agent,
        )

        self._cache[template_name] = template
        return template

    def _resolve_path(self, template_name: str) -> Path:
        """Résout le chemin d'un template."""
        # Essayer .txt d'abord, puis .md, puis .prompt
        for ext in (".txt", ".md", ".prompt"):
            path = self._prompts_dir / f"{template_name}{ext}"
            if path.exists():
                return path
        return self._prompts_dir / f"{template_name}.txt"

    @staticmethod
    def _extract_variables(content: str) -> list[str]:
        """Extrait les noms de variables {{var}} d'un template."""
        import re
        pattern = r"\{\{(\w+)\}\}"
        return list(set(re.findall(pattern, content)))

    @staticmethod
    def _render(template: PromptTemplate, variables: dict[str, Any]) -> str:
        """Rend un template avec les variables."""
        content = template.content

        for var_name in template.variables:
            placeholder = "{{" + var_name + "}}"
            if var_name in variables:
                value = variables[var_name]
                if isinstance(value, (dict, list)):
                    import json
                    value = json.dumps(value, ensure_ascii=False, indent=2)
                content = content.replace(placeholder, str(value))
            else:
                # Variable non fournie — laisser le placeholder
                # (le LLM comprendra qu'il manque de l'info)
                pass

        return content
