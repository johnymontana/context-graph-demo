"""
FastAPI application for the Context Graph demo.
Provides REST API endpoints for the frontend and agent interactions.
"""

import json
import logging
import traceback
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse

from .agent import ContextGraphAgent
from .config import config
from .context_graph_client import context_graph_client
from .gds_client import gds_client
from .models import (
    ChatRequest,
    ChatResponse,
    DecisionRequest,
    GraphData
)
from .vector_client import vector_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info("Starting Context Graph API...")
    if context_graph_client.verify_connectivity():
        logger.info("Connected to Neo4j successfully!")

        # Ensure all required indexes exist
        logger.info("Checking database indexes...")
        index_results = context_graph_client.ensure_indexes()
        if index_results["created"]:
            logger.info(f"Created indexes: {index_results['created']}")
        if index_results["existing"]:
            logger.info(f"Existing indexes: {len(index_results['existing'])} already present")
        if index_results["errors"]:
            logger.warning(f"Index errors: {index_results['errors']}")

        gds_client.refresh_gds_analyses()

    else:
        logger.warning("Could not connect to Neo4j")
    yield
    # Shutdown
    logger.info("Shutting down Context Graph API...")
    context_graph_client.close()
    gds_client.close()
    vector_client.close()


app = FastAPI(
    title="Context Graph API",
    description="Decision traces for AI agents using Neo4j",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
        "https://context-graph-demo.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# HEALTH CHECK
# ============================================


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    neo4j_connected = context_graph_client.verify_connectivity()
    return {
        "status": "healthy" if neo4j_connected else "degraded",
        "neo4j_connected": neo4j_connected,
    }


# ============================================
# CHAT / AGENT ENDPOINTS
# ============================================


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Send a message to the Claude agent.
    Agent has access to context graph tools.
    """
    session_id = request.session_id or str(uuid.uuid4())
    logger.info(f"Chat request received: {request.message[:100]}...")

    try:
        # Convert conversation history to list of dicts
        history = [
            {"role": msg.role, "content": msg.content} for msg in request.conversation_history
        ]

        logger.info("Creating ContextGraphAgent...")
        async with ContextGraphAgent() as agent:
            logger.info("Agent connected, sending query...")
            result = await agent.query(request.message, conversation_history=history)
            logger.info("Query completed successfully")

            return ChatResponse(
                response=result["response"],
                session_id=session_id,
                tool_calls=result.get("tool_calls", []),
                decisions_made=result.get("decisions_made", []),
            )
    except Exception as e:
        logger.error(f"Chat error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Send a message to the Claude agent with streaming response.
    Returns Server-Sent Events (SSE) for real-time streaming.
    """
    import asyncio

    session_id = request.session_id or str(uuid.uuid4())
    logger.info(f"Stream chat request received: {request.message[:100]}...")

    async def event_generator():
        try:
            # Convert conversation history to list of dicts
            history = [
                {"role": msg.role, "content": msg.content} for msg in request.conversation_history
            ]

            logger.info("Creating ContextGraphAgent for streaming...")
            async with ContextGraphAgent() as agent:
                logger.info("Agent connected, starting stream...")

                # Use an async queue to enable keep-alive pings during long operations
                event_queue: asyncio.Queue = asyncio.Queue()
                stream_done = asyncio.Event()

                async def process_agent_stream():
                    """Process agent events and put them in the queue."""
                    try:
                        async for event in agent.query_stream(
                            request.message, conversation_history=history
                        ):
                            await event_queue.put(event)
                        await event_queue.put(None)  # Signal completion
                    except Exception as e:
                        await event_queue.put({"type": "error", "error": str(e)})
                        await event_queue.put(None)
                    finally:
                        stream_done.set()

                # Start processing in background
                agent_task = asyncio.create_task(process_agent_stream())

                try:
                    while True:
                        try:
                            # Wait for event with timeout for keep-alive
                            event = await asyncio.wait_for(event_queue.get(), timeout=15.0)

                            if event is None:
                                # Stream completed
                                break

                            # Send different event types
                            if event["type"] == "agent_context":
                                yield {
                                    "event": "agent_context",
                                    "data": json.dumps(event["context"]),
                                }
                            elif event["type"] == "text":
                                yield {
                                    "event": "text",
                                    "data": json.dumps({"content": event["content"]}),
                                }
                            elif event["type"] == "tool_use":
                                logger.info(f"Tool use: {event['name']}")
                                yield {
                                    "event": "tool_use",
                                    "data": json.dumps(
                                        {
                                            "name": event["name"],
                                            "input": event.get("input", {}),
                                        }
                                    ),
                                }
                            elif event["type"] == "tool_result":
                                logger.info(f"Tool result: {event['name']}")
                                yield {
                                    "event": "tool_result",
                                    "data": json.dumps(
                                        {
                                            "name": event["name"],
                                            "output": event.get("output"),
                                        }
                                    ),
                                }
                            elif event["type"] == "done":
                                logger.info("Stream completed successfully")
                                yield {
                                    "event": "done",
                                    "data": json.dumps(
                                        {
                                            "session_id": session_id,
                                            "tool_calls": event.get("tool_calls", []),
                                            "decisions_made": event.get("decisions_made", []),
                                        }
                                    ),
                                }
                            elif event["type"] == "error":
                                logger.error(f"Agent error: {event.get('error')}")
                                yield {
                                    "event": "error",
                                    "data": json.dumps({"error": event.get("error")}),
                                }

                        except asyncio.TimeoutError:
                            # Send keep-alive ping to prevent connection timeout
                            yield {
                                "event": "ping",
                                "data": json.dumps({"keepalive": True}),
                            }
                finally:
                    # Ensure the agent task is cleaned up
                    if not agent_task.done():
                        agent_task.cancel()
                        try:
                            await agent_task
                        except asyncio.CancelledError:
                            pass

        except Exception as e:
            logger.error(f"Stream error: {traceback.format_exc()}")
            yield {
                "event": "error",
                "data": json.dumps({"error": str(e)}),
            }

    return EventSourceResponse(event_generator(), ping=20)


# ============================================
# CUSTOMER ENDPOINTS
# ============================================


@app.get("/api/customers/search")
async def search_customers(query: str, limit: int = 10):
    """Search for customers by name, email, or account number."""
    try:
        results = context_graph_client.search_customers(query, limit)
        return {"customers": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/customers/{customer_id}")
async def get_customer(customer_id: str):
    """Get a customer by ID with related entities."""
    customer = context_graph_client.get_customer(customer_id)
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")
    return customer


@app.get("/api/customers/{customer_id}/decisions")
async def get_customer_decisions(
    customer_id: str,
    decision_type: Optional[str] = None,
    limit: int = 20,
):
    """Get all decisions about a customer."""
    try:
        decisions = context_graph_client.get_customer_decisions(customer_id, decision_type, limit)
        return {"decisions": decisions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# DECISION ENDPOINTS
# ============================================


@app.get("/api/decisions")
async def list_decisions(
    category: Optional[str] = None,
    decision_type: Optional[str] = None,
    limit: int = 20,
):
    """List recent decisions with optional filters."""
    try:
        decisions = context_graph_client.list_decisions(
            category=category,
            decision_type=decision_type,
            limit=limit,
        )
        return {"decisions": decisions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/decisions/{decision_id}")
async def get_decision(decision_id: str):
    """Get a decision by ID with full context."""
    decision = context_graph_client.get_decision(decision_id)
    if not decision:
        raise HTTPException(status_code=404, detail="Decision not found")
    return decision


@app.post("/api/decisions")
async def create_decision(request: DecisionRequest):
    """Record a new decision."""
    try:
        # Generate reasoning embedding
        reasoning_embedding = None
        try:
            reasoning_embedding = vector_client.generate_embedding(request.reasoning)
        except Exception:
            pass

        decision_id = context_graph_client.record_decision(
            decision_type=request.decision_type,
            category=request.category,
            reasoning=request.reasoning,
            customer_id=request.customer_id,
            account_id=request.account_id,
            transaction_id=request.transaction_id,
            risk_factors=request.risk_factors,
            precedent_ids=request.precedent_ids,
            confidence_score=request.confidence_score,
            reasoning_embedding=reasoning_embedding,
        )
        return {"decision_id": decision_id, "success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/decisions/{decision_id}/similar")
async def find_similar_decisions(decision_id: str, limit: int = 5):
    """Find structurally similar decisions using FastRP embeddings."""
    try:
        similar = gds_client.find_similar_decisions_knn(decision_id, limit)
        return {"similar_decisions": similar}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/decisions/{decision_id}/causal-chain")
async def get_causal_chain(decision_id: str, depth: int = 3):
    """Get the causal chain for a decision."""
    try:
        chain = context_graph_client.get_causal_chain(decision_id, "both", depth)
        return chain
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/decisions/search/precedents")
async def find_precedents(scenario: str, category: Optional[str] = None, limit: int = 5):
    """Find precedent decisions using hybrid search."""
    try:
        precedents = vector_client.find_precedents_hybrid(scenario, category, limit=limit)
        return {"precedents": precedents}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# POLICY ENDPOINTS
# ============================================


@app.get("/api/policies")
async def list_policies(category: Optional[str] = None):
    """List all policies, optionally filtered by category."""
    try:
        policies = context_graph_client.get_policies(category)
        return {"policies": policies}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/policies/{policy_id}")
async def get_policy(policy_id: str):
    """Get a policy by ID."""
    policy = context_graph_client.get_policy(policy_id)
    if not policy:
        raise HTTPException(status_code=404, detail="Policy not found")
    return policy


# ============================================
# GRAPH VISUALIZATION ENDPOINTS
# ============================================


@app.get("/api/graph", response_model=GraphData)
async def get_graph(
    center_node_id: Optional[str] = None,
    center_node_type: Optional[str] = None,
    depth: int = 2,
    include_decisions: bool = True,
    limit: int = 100,
):
    """Get graph data for NVL visualization."""
    try:
        graph = context_graph_client.get_graph_data(
            center_node_id=center_node_id,
            center_node_type=center_node_type,
            depth=depth,
            include_decisions=include_decisions,
            limit=limit,
        )
        return graph
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/graph/statistics")
async def get_statistics():
    """Get graph statistics."""
    try:
        stats = context_graph_client.get_statistics()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/graph/expand/{node_id}", response_model=GraphData)
async def expand_node(node_id: str, limit: int = 50):
    """Get all nodes connected to a given node (for graph expansion on double-click)."""
    try:
        graph = context_graph_client.get_connected_nodes(node_id=node_id, limit=limit)
        return graph
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/graph/relationships")
async def get_relationships_between(node_ids: list[str]):
    """Get all relationships between a set of nodes."""
    try:
        relationships = context_graph_client.get_relationships_between_nodes(node_ids)
        return {
            "relationships": [
                {
                    "id": rel.id,
                    "type": rel.type,
                    "startNodeId": rel.start_node_id,
                    "endNodeId": rel.end_node_id,
                    "properties": rel.properties,
                }
                for rel in relationships
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/graph/schema")
async def get_graph_schema():
    """Get the graph schema for visualization."""
    try:
        schema = context_graph_client.get_schema()
        return schema
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# GDS / ANALYTICS ENDPOINTS
# ============================================


@app.post("/api/analytics/fastrp")
async def run_fastrp_embeddings():
    """Generate FastRP embeddings for all nodes."""
    try:
        # Create projection
        projection = gds_client.create_decision_graph_projection()

        # Generate embeddings
        result = gds_client.generate_fastrp_embeddings()

        # Write back to database
        write_result = gds_client.write_fastrp_embeddings()

        return {
            "projection": projection,
            "embeddings": result,
            "written": write_result,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/analytics/communities")
async def get_decision_communities():
    """Get detected decision communities."""
    try:
        communities = gds_client.detect_decision_communities()
        return {"communities": communities}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/analytics/influence")
async def get_influence_scores():
    """Get influence scores for decisions using PageRank."""
    try:
        scores = gds_client.calculate_influence_scores()
        return {"influence_scores": scores}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/analytics/fraud-patterns")
async def detect_fraud_patterns(
    account_id: Optional[str] = None,
    similarity_threshold: float = 0.7,
):
    """Detect potential fraud patterns."""
    try:
        patterns = gds_client.detect_fraud_patterns(account_id, similarity_threshold)
        return {"fraud_patterns": patterns}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/analytics/entity-resolution")
async def find_entity_matches(similarity_threshold: float = 0.7):
    """Find potential duplicate entities."""
    try:
        matches = gds_client.find_potential_duplicates(similarity_threshold)
        return {"entity_matches": matches}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/analytics/projections")
async def list_graph_projections():
    """List all GDS graph projections."""
    try:
        projections = gds_client.list_graph_projections()
        return {"projections": projections}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# VECTOR SEARCH ENDPOINTS
# ============================================


@app.get("/api/search/decisions")
async def search_decisions_semantic(
    query: str,
    category: Optional[str] = None,
    limit: int = 10,
):
    """Search decisions by semantic similarity."""
    try:
        results = vector_client.search_decisions_semantic(query, limit, category)
        return {"decisions": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/search/policies")
async def search_policies_semantic(query: str, limit: int = 5):
    """Search policies by semantic similarity."""
    try:
        results = vector_client.search_policies_semantic(query, limit)
        return {"policies": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/embeddings/batch-update")
async def batch_update_embeddings(limit: int = 100):
    """Generate embeddings for decisions that don't have them."""
    try:
        count = vector_client.batch_update_decision_embeddings(limit)
        return {"updated_count": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config.host, port=config.port)
