"""
Neo4j Agent Memory client wrapper.

Provides a unified interface to the neo4j-agent-memory package's
short-term, long-term, and reasoning memory capabilities.
"""

from contextlib import asynccontextmanager
from typing import Any, Optional

from neo4j_agent_memory import MemoryClient

from .config import get_memory_settings

# Global memory client instance (initialized in app lifespan)
_memory_client: Optional[MemoryClient] = None


async def init_memory_client() -> MemoryClient:
    """Initialize the global memory client."""
    global _memory_client
    settings = get_memory_settings()
    _memory_client = MemoryClient(settings)
    await _memory_client.__aenter__()
    return _memory_client


async def close_memory_client() -> None:
    """Close the global memory client."""
    global _memory_client
    if _memory_client:
        await _memory_client.__aexit__(None, None, None)
        _memory_client = None


def get_memory_client() -> MemoryClient:
    """Get the global memory client instance."""
    if _memory_client is None:
        raise RuntimeError("Memory client not initialized. Call init_memory_client() first.")
    return _memory_client


@asynccontextmanager
async def memory_context():
    """Context manager for memory client access."""
    client = get_memory_client()
    yield client


# ============================================
# SHORT-TERM MEMORY (Conversations)
# ============================================


async def add_message(
    session_id: str,
    role: str,
    content: str,
    metadata: Optional[dict[str, Any]] = None,
) -> str:
    """Add a message to the conversation history."""
    client = get_memory_client()
    return await client.short_term.add_message(
        session_id=session_id,
        role=role,
        content=content,
        metadata=metadata,
    )


async def get_conversation(
    session_id: str,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Get conversation history for a session."""
    client = get_memory_client()
    messages = await client.short_term.get_conversation(
        session_id=session_id,
        limit=limit,
    )
    return [
        {
            "role": msg.role,
            "content": msg.content,
            "timestamp": msg.timestamp.isoformat() if msg.timestamp else None,
            "metadata": msg.metadata,
        }
        for msg in messages
    ]


async def list_sessions(
    limit: int = 20,
) -> list[dict[str, Any]]:
    """List recent conversation sessions."""
    client = get_memory_client()
    sessions = await client.short_term.list_sessions(limit=limit)
    return [
        {
            "session_id": s.session_id,
            "message_count": s.message_count,
            "first_message": s.first_message.isoformat() if s.first_message else None,
            "last_message": s.last_message.isoformat() if s.last_message else None,
        }
        for s in sessions
    ]


async def get_conversation_summary(
    session_id: str,
) -> Optional[str]:
    """Get AI-generated summary of a conversation."""
    client = get_memory_client()
    return await client.short_term.get_conversation_summary(session_id=session_id)


# ============================================
# LONG-TERM MEMORY (Entities)
# ============================================


async def add_entity(
    name: str,
    entity_type: str,
    subtype: Optional[str] = None,
    description: Optional[str] = None,
    properties: Optional[dict[str, Any]] = None,
) -> str:
    """Add an entity to long-term memory."""
    client = get_memory_client()
    entity = await client.long_term.add_entity(
        name=name,
        entity_type=entity_type,
        subtype=subtype,
        description=description,
        properties=properties,
    )
    return entity.id


async def search_entities(
    query: str,
    entity_type: Optional[str] = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Search entities by text query."""
    client = get_memory_client()
    entities = await client.long_term.search_entities(
        query=query,
        entity_type=entity_type,
        limit=limit,
    )
    return [
        {
            "id": e.id,
            "name": e.name,
            "entity_type": e.entity_type,
            "subtype": e.subtype,
            "description": e.description,
            "properties": e.properties,
        }
        for e in entities
    ]


async def add_preference(
    session_id: str,
    category: str,
    value: str,
    confidence: float = 1.0,
) -> str:
    """Record a user preference."""
    client = get_memory_client()
    pref = await client.long_term.add_preference(
        session_id=session_id,
        category=category,
        value=value,
        confidence=confidence,
    )
    return pref.id


async def add_fact(
    subject: str,
    predicate: str,
    object_value: str,
    confidence: float = 1.0,
    source: Optional[str] = None,
) -> str:
    """Record a fact in long-term memory."""
    client = get_memory_client()
    fact = await client.long_term.add_fact(
        subject=subject,
        predicate=predicate,
        object=object_value,
        confidence=confidence,
        source=source,
    )
    return fact.id


# ============================================
# REASONING MEMORY (Decision Traces)
# ============================================


async def start_trace(
    session_id: str,
    task: str,
    metadata: Optional[dict[str, Any]] = None,
) -> str:
    """Start a new reasoning trace."""
    client = get_memory_client()
    trace = await client.reasoning.start_trace(
        session_id=session_id,
        task=task,
        metadata=metadata,
    )
    return trace.id


async def add_trace_step(
    trace_id: str,
    step_type: str,
    content: str,
    metadata: Optional[dict[str, Any]] = None,
) -> str:
    """Add a step to a reasoning trace."""
    client = get_memory_client()
    step = await client.reasoning.add_step(
        trace_id=trace_id,
        step_type=step_type,
        content=content,
        metadata=metadata,
    )
    return step.id


async def record_tool_call(
    trace_id: str,
    tool_name: str,
    tool_input: dict[str, Any],
    tool_output: Any,
    duration_ms: Optional[int] = None,
    success: bool = True,
) -> str:
    """Record a tool call within a reasoning trace."""
    client = get_memory_client()
    tool_call = await client.reasoning.record_tool_call(
        trace_id=trace_id,
        tool_name=tool_name,
        tool_input=tool_input,
        tool_output=tool_output,
        duration_ms=duration_ms,
        success=success,
    )
    return tool_call.id


async def complete_trace(
    trace_id: str,
    outcome: str,
    success: bool = True,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    """Complete a reasoning trace with an outcome."""
    client = get_memory_client()
    await client.reasoning.complete_trace(
        trace_id=trace_id,
        outcome=outcome,
        success=success,
        metadata=metadata,
    )


async def get_similar_traces(
    query: str,
    limit: int = 5,
    session_id: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Find similar reasoning traces by semantic search."""
    client = get_memory_client()
    traces = await client.reasoning.get_similar_traces(
        query=query,
        limit=limit,
        session_id=session_id,
    )
    return [
        {
            "id": t.id,
            "session_id": t.session_id,
            "task": t.task,
            "outcome": t.outcome,
            "success": t.success,
            "started_at": t.started_at.isoformat() if t.started_at else None,
            "completed_at": t.completed_at.isoformat() if t.completed_at else None,
            "similarity": getattr(t, "similarity", None),
        }
        for t in traces
    ]


async def get_trace(
    trace_id: str,
) -> Optional[dict[str, Any]]:
    """Get a specific reasoning trace with its steps."""
    client = get_memory_client()
    trace = await client.reasoning.get_trace(trace_id=trace_id)
    if not trace:
        return None
    return {
        "id": trace.id,
        "session_id": trace.session_id,
        "task": trace.task,
        "outcome": trace.outcome,
        "success": trace.success,
        "started_at": trace.started_at.isoformat() if trace.started_at else None,
        "completed_at": trace.completed_at.isoformat() if trace.completed_at else None,
        "steps": [
            {
                "id": s.id,
                "step_type": s.step_type,
                "content": s.content,
                "metadata": s.metadata,
            }
            for s in (trace.steps or [])
        ],
        "tool_calls": [
            {
                "id": tc.id,
                "tool_name": tc.tool_name,
                "tool_input": tc.tool_input,
                "tool_output": tc.tool_output,
                "success": tc.success,
            }
            for tc in (trace.tool_calls or [])
        ],
    }


# ============================================
# COMBINED CONTEXT
# ============================================


async def get_context(
    query: str,
    session_id: Optional[str] = None,
    include_conversation: bool = True,
    include_entities: bool = True,
    include_traces: bool = True,
    conversation_limit: int = 10,
    entity_limit: int = 5,
    trace_limit: int = 3,
) -> dict[str, Any]:
    """
    Get combined context from all memory types.

    This is useful for building agent context before responding.
    """
    context = {}

    if include_conversation and session_id:
        context["conversation"] = await get_conversation(
            session_id=session_id,
            limit=conversation_limit,
        )

    if include_entities:
        context["related_entities"] = await search_entities(
            query=query,
            limit=entity_limit,
        )

    if include_traces:
        context["similar_traces"] = await get_similar_traces(
            query=query,
            limit=trace_limit,
            session_id=session_id,
        )

    return context
