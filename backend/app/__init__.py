"""
Context Graph Demo - Decision traces for AI agents using Neo4j

This application now uses neo4j-agent-memory for:
- Short-term memory (conversation persistence)
- Reasoning memory (decision traces)
- Long-term memory (entity storage via POLE+O model)

The legacy clients (context_graph_client, vector_client, gds_client) are
retained for backward compatibility and domain-specific operations.
"""

__version__ = "0.1.0"
