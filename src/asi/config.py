"""
ASI v8 Configuration
All configuration loaded from environment variables.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    # API Keys
    groq_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None

    # Provider preferences
    preferred_provider: str = "groq"

    # Paths
    db_path: str = "./data/asi.db"
    audit_dir: str = "./data/audit/"
    workspace_dir: str = "/tmp/asi_workspace"

    # Limits
    max_query_len: int = 8000
    model_timeout: int = 30

    # Model names
    groq_model: str = "mixtral-8x7b-32768"
    openai_model: str = "gpt-4o-mini"
    anthropic_model: str = "claude-3-haiku-20240307"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "mistral"

    @classmethod
    def from_env(cls) -> "Config":
        """Load config from environment. Raises ValueError listing ALL missing required vars."""
        groq_key = os.getenv("GROQ_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")

        # At least one LLM provider must be configured
        errors = []
        if not any([groq_key, openai_key, anthropic_key, ollama_url]):
            errors.append(
                "At least one of GROQ_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, or OLLAMA_URL must be set"
            )

        if errors:
            raise ValueError(
                "ASI v8 startup failed — missing required configuration:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

        return cls(
            groq_api_key=groq_key,
            openai_api_key=openai_key,
            anthropic_api_key=anthropic_key,
            preferred_provider=os.getenv("PREFERRED_PROVIDER", "groq"),
            db_path=os.getenv("ASI_DB_PATH", "./data/asi.db"),
            audit_dir=os.getenv("ASI_AUDIT_DIR", "./data/audit/"),
            workspace_dir=os.getenv("ASI_WORKSPACE_DIR", "/tmp/asi_workspace"),
            max_query_len=int(os.getenv("ASI_MAX_QUERY_LEN", "8000")),
            model_timeout=int(os.getenv("ASI_MODEL_TIMEOUT", "30")),
            groq_model=os.getenv("GROQ_MODEL", "mixtral-8x7b-32768"),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307"),
            ollama_url=ollama_url,
            ollama_model=os.getenv("OLLAMA_MODEL", "mistral"),
        )

    def available_providers(self) -> list:
        """Return list of providers that have credentials configured."""
        providers = []
        if self.groq_api_key:
            providers.append("groq")
        if self.openai_api_key:
            providers.append("openai")
        if self.anthropic_api_key:
            providers.append("anthropic")
        # Ollama is always potentially available (local)
        providers.append("ollama")
        return providers
