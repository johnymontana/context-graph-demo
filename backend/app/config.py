"""
Configuration management for the Context Graph application.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from pydantic import SecretStr

load_dotenv()


@dataclass
class Neo4jConfig:
    """Neo4j connection configuration."""

    uri: str
    username: str
    password: str
    database: str = "neo4j"

    @classmethod
    def from_env(cls) -> "Neo4jConfig":
        return cls(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            username=os.getenv("NEO4J_USERNAME", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "password"),
            database=os.getenv("NEO4J_DATABASE", "neo4j"),
        )


@dataclass
class OpenAIConfig:
    """OpenAI configuration for text embeddings."""

    api_key: str
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    @classmethod
    def from_env(cls) -> "OpenAIConfig":
        return cls(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            embedding_dimensions=int(os.getenv("OPENAI_EMBEDDING_DIMENSIONS", "1536")),
        )


@dataclass
class AnthropicConfig:
    """Anthropic configuration for Claude Agent SDK."""

    api_key: str

    @classmethod
    def from_env(cls) -> "AnthropicConfig":
        return cls(
            api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        )


@dataclass
class AppConfig:
    """Main application configuration."""

    neo4j: Neo4jConfig
    openai: OpenAIConfig
    anthropic: AnthropicConfig

    # FastRP embedding dimensions (structural)
    fastrp_dimensions: int = 128

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    @classmethod
    def from_env(cls) -> "AppConfig":
        return cls(
            neo4j=Neo4jConfig.from_env(),
            openai=OpenAIConfig.from_env(),
            anthropic=AnthropicConfig.from_env(),
            fastrp_dimensions=int(os.getenv("FASTRP_DIMENSIONS", "128")),
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", "8000")),
            debug=os.getenv("DEBUG", "false").lower() == "true",
        )


# Global config instance
config = AppConfig.from_env()


def get_memory_settings():
    """
    Create MemorySettings for neo4j-agent-memory package.
    Lazy import to avoid circular dependencies.
    """
    from neo4j_agent_memory import (
        EmbeddingConfig,
        EmbeddingProvider,
        MemorySettings,
    )
    from neo4j_agent_memory import (
        Neo4jConfig as NAMNeo4jConfig,
    )

    return MemorySettings(
        neo4j=NAMNeo4jConfig(
            uri=config.neo4j.uri,
            username=config.neo4j.username,
            password=SecretStr(config.neo4j.password),
            database=config.neo4j.database,
        ),
        embedding=EmbeddingConfig(
            provider=EmbeddingProvider.OPENAI,
            model=config.openai.embedding_model,
            api_key=SecretStr(config.openai.api_key) if config.openai.api_key else None,
        ),
    )
