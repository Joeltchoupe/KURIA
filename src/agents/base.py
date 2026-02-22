    @property
    def system_prompt(self) -> str:
        """
        Charge le system prompt depuis prompts/system/{agent_type}.txt
        et injecte les variables de config.

        Fallback : retourne un prompt générique si le fichier n'existe pas.
        """
        template_name = f"system/{self.agent_type.value}"

        if self._prompts.exists(template_name):
            return self._prompts.render(
                template_name,
                **self._prompt_variables(),
            )

        # Fallback générique
        return (
            f"You are Kuria {self.agent_type.value} AI agent. "
            f"Respond with valid JSON only."
        )

    def _prompt_variables(self) -> dict[str, Any]:
        """
        Variables injectées dans les prompts.

        Combine les paramètres de config + les variables custom de l'agent.
        """
        base_vars = {
            "company_id": self.company_id,
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Injecter tous les paramètres de config
        if hasattr(self.config, "parameters"):
            base_vars.update(self.config.parameters)

        # Variables custom de l'agent
        base_vars.update(self._custom_prompt_variables())

        return base_vars

    def _custom_prompt_variables(self) -> dict[str, Any]:
        """
        Override par chaque agent pour ajouter ses variables spécifiques.

        Ex: RevenueVelocity ajoute score_fit, score_activity, etc.
        """
        return {}
