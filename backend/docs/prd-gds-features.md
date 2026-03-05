# PRD: Graph Data Science Features for neo4j-agent-memory

**Author:** Context Graph Demo Team  
**Date:** 2026-01-29  
**Status:** Proposed  
**Version:** 1.0

---

## Executive Summary

Add Graph Data Science (GDS) algorithm support to the neo4j-agent-memory package, enabling structural similarity search, community detection, and influence analysis on agent memory graphs.

This PRD documents the GDS features currently used in the Context Graph application that would benefit from native integration into the neo4j-agent-memory package.

---

## Problem Statement

The neo4j-agent-memory package provides semantic similarity via embeddings, but lacks structural similarity capabilities. AI agent memory benefits from understanding not just *what* was said (semantic), but *how* memories relate to each other structurally (graph patterns).

### Current Gap

When searching for similar reasoning traces, we can only match by text embedding similarity. We cannot find traces that have similar *patterns* of entity relationships, tool usage, or causal chains.

### Impact

Without structural analysis:
- Precedent search misses decisions with similar entity patterns but different wording
- Cannot identify communities of related decisions
- Cannot measure influence/impact of past decisions
- Fraud/anomaly detection limited to text-based patterns

---

## Use Cases

### 1. Hybrid Precedent Search

**Description:** Find past decisions that are both semantically similar AND structurally similar (same entities involved, same tool sequence used).

**Current Implementation:**
```python
# We currently combine two separate searches
semantic_results = vector_client.search_decisions_semantic(query)
structural_results = gds_client.find_similar_decisions_knn(decision_id)
# Manual combination with weighted scoring
```

**Desired API:**
```python
results = await memory.analytics.find_similar_hybrid(
    query="customer requesting credit limit increase",
    semantic_weight=0.6,
    structural_weight=0.4,
    limit=10,
)
```

### 2. Community Detection

**Description:** Group related reasoning traces into clusters based on their graph connectivity, not just topic similarity.

**Current Implementation:**
```python
# Run Louvain on Decision nodes
gds_client.write_community_ids()
# Query decisions by community_id
```

**Desired API:**
```python
communities = await memory.analytics.detect_communities(
    algorithm="louvain",
    relationship_types=["CAUSED", "INFLUENCED", "PRECEDENT_FOR"],
)

related_traces = await memory.analytics.get_trace_community(
    trace_id="abc123",
    limit=10,
)
```

### 3. Influence Analysis

**Description:** Identify which reasoning traces or decisions had the most downstream impact via PageRank on causal chains.

**Current Implementation:**
```python
scores = gds_client.calculate_influence_scores()
# Returns decisions with high PageRank on CAUSED/INFLUENCED relationships
```

**Desired API:**
```python
influential_traces = await memory.analytics.calculate_influence_scores(
    algorithm="pagerank",
    relationship_types=["CAUSED", "INFLUENCED"],
    damping_factor=0.85,
)
```

### 4. Structural Similarity for Anomaly Detection

**Description:** Find traces with unusual structural patterns (different from typical tool usage sequences).

**Current Implementation:**
```python
# Generate FastRP embeddings
gds_client.generate_fastrp_embeddings()
# Use KNN to find similar/dissimilar patterns
similar = gds_client.find_similar_decisions_knn(decision_id)
```

**Desired API:**
```python
# Generate structural embeddings on demand or automatically
await memory.analytics.generate_structural_embeddings(
    algorithm="fastrp",
    dimensions=128,
)

anomalies = await memory.analytics.find_anomalies(
    threshold=0.3,  # Similarity below this is anomalous
    limit=20,
)
```

---

## Proposed API

### New `analytics` Property on MemoryClient

```python
class MemoryClient:
    # Existing properties...
    
    @property
    def analytics(self) -> AnalyticsMemory:
        """Access to GDS-powered analytics."""
        ...
```

### AnalyticsMemory Class

```python
class AnalyticsMemory:
    """Graph Data Science analytics for agent memory."""
    
    # ============================================
    # STRUCTURAL EMBEDDINGS
    # ============================================
    
    async def generate_structural_embeddings(
        self,
        algorithm: Literal["fastrp", "node2vec"] = "fastrp",
        dimensions: int = 128,
        node_labels: list[str] | None = None,
        relationship_types: list[str] | None = None,
        iteration_weights: list[float] | None = None,
    ) -> EmbeddingResult:
        """
        Generate graph-structure embeddings for nodes.
        
        These embeddings capture the structural position of nodes in the graph,
        enabling similarity search based on connection patterns rather than content.
        
        Args:
            algorithm: Embedding algorithm to use
            dimensions: Dimensionality of output embeddings
            node_labels: Which node types to generate embeddings for
            relationship_types: Which relationships to traverse
            iteration_weights: FastRP iteration weights
            
        Returns:
            EmbeddingResult with counts and timing
        """
        ...
    
    # ============================================
    # SIMILARITY SEARCH
    # ============================================
    
    async def find_similar_by_structure(
        self,
        trace_id: str,
        limit: int = 10,
        min_similarity: float = 0.5,
    ) -> list[SimilarTrace]:
        """
        Find traces with similar graph structure using KNN on structural embeddings.
        
        Args:
            trace_id: The trace to find similar matches for
            limit: Maximum number of results
            min_similarity: Minimum similarity score (0-1)
            
        Returns:
            List of similar traces with similarity scores
        """
        ...
    
    async def find_similar_hybrid(
        self,
        query: str,
        trace_id: str | None = None,
        semantic_weight: float = 0.5,
        structural_weight: float = 0.5,
        limit: int = 10,
    ) -> list[SimilarTrace]:
        """
        Combine semantic and structural similarity for hybrid search.
        
        Args:
            query: Text query for semantic similarity
            trace_id: Optional trace for structural similarity
            semantic_weight: Weight for semantic similarity (0-1)
            structural_weight: Weight for structural similarity (0-1)
            limit: Maximum number of results
            
        Returns:
            List of traces with combined similarity scores
        """
        ...
    
    # ============================================
    # COMMUNITY DETECTION
    # ============================================
    
    async def detect_communities(
        self,
        algorithm: Literal["louvain", "leiden"] = "louvain",
        relationship_types: list[str] | None = None,
        resolution: float = 1.0,
    ) -> list[Community]:
        """
        Detect communities of related traces using graph clustering.
        
        Args:
            algorithm: Clustering algorithm
            relationship_types: Relationships to consider for community detection
            resolution: Resolution parameter (higher = more communities)
            
        Returns:
            List of detected communities with metadata
        """
        ...
    
    async def get_trace_community(
        self,
        trace_id: str,
        limit: int = 10,
    ) -> list[ReasoningTrace]:
        """
        Get other traces in the same community as the given trace.
        
        Args:
            trace_id: The trace to find community members for
            limit: Maximum number of results
            
        Returns:
            List of traces in the same community
        """
        ...
    
    # ============================================
    # INFLUENCE ANALYSIS
    # ============================================
    
    async def calculate_influence_scores(
        self,
        algorithm: Literal["pagerank", "betweenness"] = "pagerank",
        relationship_types: list[str] = ["CAUSED", "INFLUENCED"],
        damping_factor: float = 0.85,
        max_iterations: int = 20,
    ) -> list[InfluenceScore]:
        """
        Calculate influence/centrality scores for traces.
        
        Args:
            algorithm: Centrality algorithm to use
            relationship_types: Relationships to traverse
            damping_factor: PageRank damping factor
            max_iterations: Maximum algorithm iterations
            
        Returns:
            List of traces with influence scores
        """
        ...
    
    async def get_most_influential(
        self,
        limit: int = 10,
        trace_types: list[str] | None = None,
    ) -> list[InfluenceScore]:
        """
        Get the most influential traces.
        
        Args:
            limit: Maximum number of results
            trace_types: Filter by trace types
            
        Returns:
            Top influential traces with scores
        """
        ...
    
    # ============================================
    # ANOMALY DETECTION
    # ============================================
    
    async def find_anomalies(
        self,
        threshold: float = 0.3,
        limit: int = 20,
    ) -> list[AnomalyResult]:
        """
        Find traces with unusual structural patterns.
        
        Args:
            threshold: Similarity threshold below which traces are anomalous
            limit: Maximum number of results
            
        Returns:
            Anomalous traces with deviation scores
        """
        ...
```

### Data Models

```python
from dataclasses import dataclass
from datetime import datetime


@dataclass
class SimilarTrace:
    """A trace with similarity score."""
    trace: ReasoningTrace
    semantic_similarity: float | None
    structural_similarity: float | None
    combined_score: float


@dataclass
class Community:
    """A detected community of traces."""
    id: int
    trace_count: int
    sample_traces: list[str]
    common_tools: list[str]
    common_entities: list[str]
    modularity_contribution: float | None


@dataclass
class InfluenceScore:
    """Influence score for a trace."""
    trace_id: str
    score: float
    algorithm: str
    rank: int


@dataclass
class AnomalyResult:
    """An anomalous trace."""
    trace: ReasoningTrace
    deviation_score: float
    nearest_normal: str | None
    anomaly_reason: str


@dataclass
class EmbeddingResult:
    """Result of embedding generation."""
    nodes_processed: int
    embeddings_generated: int
    compute_time_ms: int
    algorithm: str
    dimensions: int
```

---

## Configuration

### GDSConfig

```python
from pydantic import BaseModel


class GDSConfig(BaseModel):
    """Configuration for GDS features."""
    
    # Feature toggle
    enabled: bool = True
    
    # Embedding settings
    fastrp_dimensions: int = 128
    fastrp_iteration_weights: list[float] = [0.0, 1.0, 1.0, 0.8, 0.6]
    auto_generate_embeddings: bool = True  # Generate on trace completion
    
    # Community detection
    community_algorithm: str = "louvain"
    community_detection_schedule: str | None = None  # Cron expression for batch updates
    
    # Graph projection
    projection_name: str = "agent-memory-graph"
    include_node_labels: list[str] = ["ReasoningTrace", "Message", "Entity"]
    include_relationship_types: list[str] = ["NEXT", "CAUSED", "INFLUENCED", "ABOUT"]


class MemorySettings(BaseModel):
    """Extended settings with GDS configuration."""
    neo4j: Neo4jConfig
    embedding: EmbeddingConfig
    gds: GDSConfig = GDSConfig()  # Optional, defaults to enabled
```

---

## Implementation Considerations

### 1. GDS Plugin Requirement

The Neo4j Graph Data Science plugin must be installed. The package should:

- Check for GDS availability at startup
- Provide clear error message if GDS is missing
- Gracefully degrade (disable analytics features) if GDS unavailable

```python
async def check_gds_available(self) -> bool:
    """Check if GDS plugin is installed."""
    try:
        result = await self._run_query("RETURN gds.version() AS version")
        return True
    except Exception:
        return False
```

### 2. Graph Projections

GDS algorithms operate on in-memory graph projections. The package should:

- Create projections on demand
- Cache projections for reuse
- Provide refresh mechanism when graph changes
- Clean up projections on shutdown

```python
async def ensure_projection_exists(self, refresh: bool = False) -> str:
    """Ensure graph projection exists, creating if needed."""
    if refresh:
        await self.drop_projection()
    
    exists = await self.projection_exists()
    if not exists:
        await self.create_projection()
    
    return self.settings.gds.projection_name
```

### 3. Async Support

All GDS operations should be async-compatible. This may require:

- Using async Neo4j driver for GDS calls
- Background task for long-running algorithms
- Progress callbacks for large graphs

### 4. Automatic Embedding Updates

When `auto_generate_embeddings` is enabled:

```python
async def complete_trace(self, trace_id: str, ...) -> None:
    """Complete trace and optionally update embeddings."""
    await self._complete_trace_internal(trace_id, ...)
    
    if self.settings.gds.auto_generate_embeddings:
        # Queue embedding update for this trace
        await self._queue_embedding_update(trace_id)
```

---

## Migration Path

### For Existing Users

1. **No breaking changes** - GDS features are additive
2. **Opt-in activation** - `gds.enabled = True` (default)
3. **Graceful degradation** - Works without GDS plugin installed

### For This Application

1. Keep existing `gds_client.py` until package supports all features
2. Gradually migrate to `memory.analytics` as features become available
3. Remove `gds_client.py` once feature parity achieved

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Precedent search relevance | +20% improvement | A/B test hybrid vs semantic-only |
| Similar trace discovery | <100ms p99 | Latency monitoring |
| Community detection utility | >80% user satisfaction | Survey |
| API adoption | >50% of memory users | Usage analytics |

---

## Alternatives Considered

### 1. External GDS Service

Run GDS as a separate microservice.

**Rejected:** Adds deployment complexity, network latency, and operational overhead.

### 2. Pure Python Graph Algorithms

Implement algorithms in Python using NetworkX or similar.

**Rejected:** Performance issues at scale. GDS is optimized for large graphs.

### 3. Separate Analytics Package

Create `neo4j-agent-memory-analytics` as a separate package.

**Possible alternative:** Could work, but fragments the API and complicates dependency management. Prefer integrated solution.

---

## Open Questions

1. **Embedding storage:** Store structural embeddings on nodes or in separate property?
2. **Projection lifecycle:** How long to keep projections in memory?
3. **Algorithm parameters:** Expose full algorithm configuration or simplified presets?
4. **Batch vs streaming:** Support for incremental embedding updates?

---

## Appendix: Current GDS Usage in Context Graph

### Algorithms Used

| Algorithm | Purpose | Node Labels | Relationship Types |
|-----------|---------|-------------|-------------------|
| FastRP | Structural embeddings | Decision, Person, Account | ABOUT, CAUSED, INFLUENCED, OWNS |
| KNN | Similar decision search | Decision | (uses embeddings) |
| Louvain | Community detection | Decision | CAUSED, INFLUENCED, PRECEDENT_FOR |
| PageRank | Influence scoring | Decision | CAUSED, INFLUENCED |
| Node Similarity | Duplicate detection | Person, Account | OWNS, FROM_ACCOUNT, TO_ACCOUNT |

### Graph Projections

```python
# Decision graph projection
gds.graph.project(
    'decision-graph',
    {
        Decision: {properties: ['fastrp_embedding']},
        Person: {properties: ['fastrp_embedding']},
        Account: {properties: ['fastrp_embedding']},
    },
    {
        ABOUT: {orientation: 'UNDIRECTED'},
        CAUSED: {orientation: 'NATURAL'},
        INFLUENCED: {orientation: 'NATURAL'},
        OWNS: {orientation: 'UNDIRECTED'},
    }
)
```

### Performance Characteristics

| Operation | Typical Time | Graph Size |
|-----------|--------------|------------|
| FastRP generation | 2-5 seconds | 10K nodes |
| KNN search | <100ms | 10K nodes |
| Louvain detection | 1-3 seconds | 10K nodes |
| PageRank | <500ms | 10K nodes |
