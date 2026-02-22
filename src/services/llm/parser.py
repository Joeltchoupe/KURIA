"""
ResponseParser — Extraction structurée des réponses LLM.

Les LLMs retournent du texte. On veut des données structurées.
Ce module extrait le JSON, valide contre un schéma Pydantic,
et gère les cas d'erreur.

Usage :
    from services.llm import ResponseParser, LLMResponse

    parser = ResponseParser()

    # Extraire du JSON brut
    data = parser.extract_json(response.content)

    # Valider contre un modèle Pydantic
    from models.friction import Friction
    friction = parser.parse_as(response.content, Friction)

    # Extraire une liste
    frictions = parser.parse_list(response.content, Friction)
"""

from __future__ import annotations

import json
import re
from typing import Any, TypeVar, Type

from pydantic import BaseModel


T = TypeVar("T", bound=BaseModel)


class ParseError(Exception):
    """Erreur de parsing d'une réponse LLM."""

    def __init__(self, message: str, raw_content: str = "") -> None:
        self.raw_content = raw_content
        super().__init__(message)


class ResponseParser:
    """
    Parse les réponses LLM en données structurées.

    Gère :
      - Extraction JSON depuis du texte mixte
      - Nettoyage des markdown code blocks
      - Validation Pydantic
      - Extraction de listes
      - Fallback gracieux
    """

    def extract_json(self, content: str) -> dict[str, Any] | list[Any]:
        """
        Extrait le premier objet JSON valide depuis le contenu.

        Gère les cas :
          - JSON pur
          - JSON dans un code block ```json ... ```
          - JSON précédé/suivi de texte
          - JSON avec trailing commas (nettoyage)

        Returns:
            dict ou list parsé.

        Raises:
            ParseError si aucun JSON valide trouvé.
        """
        content = content.strip()

        # Tentative 1 : JSON pur
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Tentative 2 : Extraire depuis code block
        cleaned = self._strip_code_blocks(content)
        if cleaned != content:
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                pass

        # Tentative 3 : Trouver le premier { ou [
        json_str = self._find_json_substring(content)
        if json_str:
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass

        # Tentative 4 : Nettoyer et réessayer
        cleaned = self._clean_json(content)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        raise ParseError(
            f"Impossible d'extraire du JSON valide depuis la réponse "
            f"({len(content)} caractères)",
            raw_content=content,
        )

    def parse_as(self, content: str, model: Type[T]) -> T:
        """
        Parse le contenu comme une instance d'un modèle Pydantic.

        Args:
            content: Réponse LLM brute.
            model: Classe Pydantic cible.

        Returns:
            Instance du modèle.

        Raises:
            ParseError si le parsing ou la validation échoue.
        """
        try:
            data = self.extract_json(content)
        except ParseError:
            raise

        if not isinstance(data, dict):
            raise ParseError(
                f"Attendu un objet JSON, reçu {type(data).__name__}",
                raw_content=content,
            )

        try:
            return model.model_validate(data)
        except Exception as e:
            raise ParseError(
                f"Validation Pydantic échouée pour {model.__name__}: {e}",
                raw_content=content,
            )

    def parse_list(self, content: str, model: Type[T]) -> list[T]:
        """
        Parse le contenu comme une liste d'instances Pydantic.

        Args:
            content: Réponse LLM brute.
            model: Classe Pydantic cible pour chaque élément.

        Returns:
            Liste d'instances du modèle.
        """
        try:
            data = self.extract_json(content)
        except ParseError:
            raise

        if isinstance(data, dict):
            # Peut-être un wrapper {"items": [...]} ou {"results": [...]}
            for key in ("items", "results", "data", "list", "elements"):
                if key in data and isinstance(data[key], list):
                    data = data[key]
                    break
            else:
                # Un seul élément → wrap dans une liste
                data = [data]

        if not isinstance(data, list):
            raise ParseError(
                f"Attendu une liste JSON, reçu {type(data).__name__}",
                raw_content=content,
            )

        results: list[T] = []
        errors: list[str] = []

        for i, item in enumerate(data):
            if not isinstance(item, dict):
                errors.append(f"Élément {i} n'est pas un objet JSON")
                continue
            try:
                results.append(model.model_validate(item))
            except Exception as e:
                errors.append(f"Élément {i} invalide : {e}")

        if not results and errors:
            raise ParseError(
                f"Aucun élément valide parsé. Erreurs : {'; '.join(errors)}",
                raw_content=content,
            )

        return results

    def extract_field(
        self,
        content: str,
        field: str,
        default: Any = None,
    ) -> Any:
        """
        Extrait un champ spécifique depuis le JSON de la réponse.

        Args:
            content: Réponse LLM brute.
            field: Nom du champ à extraire.
            default: Valeur par défaut si le champ n'existe pas.
        """
        try:
            data = self.extract_json(content)
        except ParseError:
            return default

        if isinstance(data, dict):
            return data.get(field, default)

        return default

    def extract_number(
        self,
        content: str,
        default: float = 0.0,
    ) -> float:
        """
        Extrait le premier nombre depuis le contenu.

        Utile pour les réponses simples type "Le score est 73.5".
        """
        numbers = re.findall(r"-?\d+\.?\d*", content)
        if numbers:
            try:
                return float(numbers[0])
            except ValueError:
                pass
        return default

    # ──────────────────────────────────────────────────────
    # CLEANING
    # ──────────────────────────────────────────────────────

    @staticmethod
    def _strip_code_blocks(content: str) -> str:
        """Retire les code blocks markdown."""
        # ```json ... ``` ou ``` ... ```
        pattern = r"```(?:json)?\s*\n?(.*?)\n?\s*```"
        match = re.search(pattern, content, re.DOTALL)
        if match:
            return match.group(1).strip()
        return content

    @staticmethod
    def _find_json_substring(content: str) -> str | None:
        """Trouve le premier objet/array JSON dans le texte."""
        # Chercher le premier { ou [
        for start_char, end_char in [("{", "}"), ("[", "]")]:
            start = content.find(start_char)
            if start == -1:
                continue

            # Trouver la fin correspondante (gestion des imbrications)
            depth = 0
            for i in range(start, len(content)):
                if content[i] == start_char:
                    depth += 1
                elif content[i] == end_char:
                    depth -= 1
                    if depth == 0:
                        return content[start : i + 1]

        return None

    @staticmethod
    def _clean_json(content: str) -> str:
        """Nettoie le JSON (trailing commas, commentaires)."""
        # Retirer les commentaires // ...
        content = re.sub(r"//.*?\n", "\n", content)

        # Retirer les trailing commas
        content = re.sub(r",\s*([}\]])", r"\1", content)

        # Retirer les caractères non-JSON au début et à la fin
        content = content.strip()
        if content and content[0] not in "{[":
            idx = min(
                (content.find(c) for c in "{[" if content.find(c) != -1),
                default=-1,
            )
            if idx > 0:
                content = content[idx:]

        return content
