# Lenny's Memory - Neo4j Agent Memory Demo

**Lenny's Memory** is the official demo application for [Neo4j Agent Memory](https://github.com/neo4j-labs/agent-memory), showcasing how to build **Context Graphs** with Neo4j for AI-powered decision tracing in financial institutions.

![Architecture Diagram](img/arch_diagram.png)

## What is a Context Graph?

Context Graphs are the infrastructure layer that enables AI agents to make enterprise-grade decisions. They capture **decision traces** - the tribal knowledge, exception logic, and reasoning that traditionally lives only in human experts' heads.

Key concepts:
- **State Clock**: What is true now (traditional databases)
- **Event Clock**: What happened, when, and why (context graphs)
- **Decision Traces**: Full reasoning, risk factors, confidence scores, and causal relationships


## AI-powered Decision Tracing For Financial Institutions

![Graph Data Model](img/graph_data_model.png)

### Mobile-First Agent Chat Interface

Modern chat interface with embedded graph visualizations. Each tool result displays an interactive subgraph showing exactly which entities and relationships were used to answer your question.

**Key Features:**
- **Tool Result Cards**: Every agent tool call renders as a card with embedded graph visualization
- **Responsive Design**: Works seamlessly on phones, tablets, and desktop (375px to 1920px+)
- **Always-Visible Tools**: Sidebar drawer with all 12 available tools
- **Data Model Context**: See which node types and relationships were traversed for each query


## Demo Scenarios

![Trading Limit Override Toolcall](img/trading_limit_override_toolcall.png)

### Scenario 1: Credit Decision with Precedent Lookup

```
User: "Should we approve a $50K credit line increase for John Smith?"

Agent:
1. Searches customer profile
2. Finds similar past decisions via FastRP embeddings
3. Cites relevant precedents
4. Makes recommendation with confidence score
5. Records decision trace with full reasoning
```

### Scenario 2: Fraud Pattern Detection

```
User: "Analyze account #12345 for fraud patterns"

Agent:
1. Uses FastRP embeddings to compare with known fraud typologies
2. Applies Node Similarity to find structural matches
3. Visualizes suspicious patterns in graph
4. Returns risk score with explanation
```

### Scenario 3: Exception Request with Audit Trail

```
User: "We need to override the trading limit for XYZ Corp"

Agent:
1. Finds past exceptions and their outcomes
2. Traces causal chain of similar exceptions
3. Records exception with justification and precedent links
4. Links to applicable policies
```

## Key Features

![Trading Limit Override Toolcard](img/trading_limit_override_toolcard.png)

This demo showcases three key differentiators of Neo4j for context graphs:

### 1. Data Model Fit
Neo4j is the natural substrate for context graphs - entities, decisions, and causal relationships map directly to nodes and relationships.

### 2. Graph Data Science Algorithms
Graph Data Science algorithms provide **structural/topological similarity** that's impossible on Postgres without a PhD in graph theory:
- **FastRP**: 75,000x faster than node2vec, captures structural embeddings
- **KNN**: Find k-nearest neighbors based on graph structure
- **Node Similarity**: Compare neighborhood patterns (fraud detection)
- **Louvain**: Community detection for decision clusters
- **PageRank**: Influence scoring for decision impact

### 3. Vector Embeddings + Graph
Combine semantic similarity (text embeddings) with structural similarity (FastRP) for hybrid search that understands both meaning and context.

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Next.js UI    │────▶│  FastAPI + SDK  │────▶│  Neo4j + GDS    │
│  Chakra UI v3   │     │  Claude Agent   │     │  Vector Search  │
│  Mobile-First   │     │  12 MCP Tools   │     │  FastRP/KNN     │
│  Tool Cards     │     │  Graph Data     │     │  Node Similarity│
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

**New in v2.0:**
- **Mobile-first single-column layout** - removed 3-column grid for better mobile UX
- **Tool result cards** - embedded NVL graph visualizations for every tool call
- **Neo4j Needle branding** - Baltic Sea Blue (#018BFF), Public Sans typography
- **Multi-hop traversal** - new `find_related_context` tool with 2-3 hop queries
- **SIMILAR_TO relationships** - GDS Node Similarity for richer context discovery

## Prerequisites

- Python 3.11+ with [uv](https://docs.astral.sh/uv/) package manager
- Node.js 18+
- Neo4j AuraDS instance (or local Neo4j Enterprise with GDS plugin)
- Anthropic API Key
- OpenAI API Key (for embeddings)

## Quick Start

### 1. Clone and Setup Environment

```bash
cd context-graph

# Create environment file
cat > .env << 'EOF'
# Neo4j Connection
NEO4J_URI=neo4j+s://xxxx.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password

# Anthropic API Key (for Claude Agent SDK)
ANTHROPIC_API_KEY=your_anthropic_key

# OpenAI API Key (for text embeddings)
OPENAI_API_KEY=your_openai_key
EOF
```

### 2. Install Backend Dependencies

```bash
cd backend

# Create virtual environment and install dependencies with uv
uv venv
uv pip install -e .
```

### 3. Generate Sample Data

```bash
cd backend
source .venv/bin/activate
export $(grep -v '^#' ../.env | xargs)
python scripts/generate_sample_data.py
```

This creates:
- 200 persons
- 350 accounts  
- 2000 transactions
- 600 decisions with causal chains
- 50 organizations
- 30 employees
- 15 policies

### 4. Start Backend

```bash
cd backend
source .venv/bin/activate
export $(grep -v '^#' ../.env | xargs)
uvicorn app.main:app --port 8000
```

Backend runs at http://localhost:8000

### 5. Start Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at http://localhost:3000

## Using Neo4j AuraDS (Recommended)

For the best experience with GDS algorithms:

1. Create an AuraDS instance at https://console.neo4j.io
2. Note the connection URI (format: `neo4j+s://xxxx.databases.neo4j.io`)
3. Update your `.env` with the connection details

AuraDS includes all GDS algorithms (FastRP, KNN, Node Similarity, Louvain, PageRank) without additional configuration.

## Using Local Neo4j with Docker

If you have a Neo4j Enterprise license:

```bash
docker-compose up -d
```

Wait for Neo4j to be ready at http://localhost:7474, then update `.env`:

```bash
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password
```


## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health check |
| `POST /api/chat` | Chat with AI agent |
| `POST /api/chat/stream` | Streaming chat with SSE |
| `GET /api/customers/search?query=` | Search customers |
| `GET /api/customers/{id}` | Get customer details |
| `GET /api/customers/{id}/decisions` | Get customer decisions |
| `GET /api/decisions` | List decisions |
| `GET /api/decisions/{id}` | Get decision details |
| `POST /api/decisions` | Create new decision |
| `GET /api/decisions/{id}/similar` | Find similar decisions |
| `GET /api/decisions/{id}/causal-chain` | Get causal chain |
| `GET /api/graph` | Get graph visualization data |
| `GET /api/graph/expand/{node_id}` | Expand node connections |
| `POST /api/graph/relationships` | Get relationships between nodes |
| `GET /api/graph/schema` | Get graph schema |
| `GET /api/graph/statistics` | Get graph statistics |
| `GET /api/policies` | List policies |
| `GET /api/policies/{id}` | Get policy details |

## Project Structure

```
context-graph/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI application
│   │   ├── agent.py             # Claude Agent SDK with MCP tools
│   │   ├── context_graph_client.py  # Neo4j operations
│   │   ├── gds_client.py        # GDS algorithms
│   │   ├── vector_client.py     # Vector search
│   │   └── models/              # Pydantic models
│   └── scripts/
│       └── generate_sample_data.py
├── frontend/
│   ├── app/                     # Next.js pages (mobile-first layout)
│   ├── components/
│   │   ├── ChatInterface.tsx    # 🆕 Redesigned chat with avatars
│   │   ├── ToolResultCard.tsx   # 🆕 Tool results with embedded graphs
│   │   ├── ToolsSidebar.tsx     # 🆕 Always-visible tools drawer
│   │   ├── DataModelBadge.tsx   # 🆕 Schema context for queries
│   │   ├── ContextGraphView.tsx # NVL visualization
│   │   └── DecisionTracePanel.tsx
│   └── lib/
│       ├── api.ts               # API client
│       └── system.ts            # 🆕 Neo4j Needle theme
├── cypher/
│   ├── schema.cypher            # Neo4j schema
│   └── gds_projections.cypher   # GDS algorithms
├── docker-compose.yml           # Local Neo4j setup
└── .env                         # Environment variables
```

## Agent Tools (MCP)

The Claude Agent has access to 12 custom tools (via MCP):

| Tool | Description | Graph Output |
|------|-------------|--------------|
| `search_customer` | Search customers by name, email, account number | ✅ Customer + 1-hop relationships |
| `get_customer_decisions` | Get all decisions about a customer | ✅ Customer + decision network |
| `find_similar_decisions` | FastRP-based structural similarity search | ✅ Decision + similar decisions |
| `find_precedents` | Semantic + structural precedent search | ✅ Precedent chain subgraph |
| `get_causal_chain` | Trace causes and effects of a decision | ✅ Causal flow visualization |
| `record_decision` | Create new decision trace with reasoning | ✅ New decision node created |
| `detect_fraud_patterns` | Graph-based fraud analysis | ✅ Fraud pattern network |
| `find_decision_community` | Louvain community detection | ✅ Community cluster graph |
| `get_policy` | Get current policy rules | - |
| `find_related_context` | 🆕 Multi-hop (2-3 hop) context discovery | ✅ Extended context subgraph |
| `execute_cypher` | Read-only Cypher for custom analysis | - |
| `get_schema` | Retrieve the current Neo4j schema | - |

**New Tool:** `find_related_context` enables richer context discovery through multi-hop graph traversal, optionally including SIMILAR_TO relationships generated via GDS Node Similarity.

## Neo4j Data Model

### Core Entities
- `Person` - Customer/employee with FastRP embedding
- `Account` - Bank/trading accounts with risk tiers
- `Transaction` - Financial transactions with fraud scores
- `Organization` - Companies with sanctions status

### Decision Trace Nodes
- `Decision` - Core decision with reasoning, confidence
- `DecisionContext` - State snapshot at decision time
- `Precedent` - Links to similar past decisions
- `Policy` - Rules governing decisions
- `Exception` - Documented exceptions

### Key Relationships
```
(:Decision)-[:CAUSED]->(:Decision)
(:Decision)-[:INFLUENCED]->(:Decision)
(:Decision)-[:PRECEDENT_FOR]->(:Decision)
(:Decision)-[:ABOUT]->(:Person|:Account|:Transaction)
(:Decision)-[:APPLIED_POLICY]->(:Policy)
(:Decision)-[:GRANTED_EXCEPTION]->(:Exception)
```

## GDS Algorithms

### FastRP Embeddings
```cypher
CALL gds.fastRP.mutate('decision-graph', {
  embeddingDimension: 128,
  iterationWeights: [0.0, 1.0, 1.0, 0.8, 0.6],
  mutateProperty: 'fastrp_embedding'
})
```

### Hybrid Search (Semantic + Structural)
```cypher
// Find decisions similar in both meaning and graph structure
CALL db.index.vector.queryNodes('decision_semantic_idx', 10, $query_embedding)
YIELD node, score AS semantic_score
WITH node, semantic_score
CALL db.index.vector.queryNodes('decision_fastrp_idx', 10, node.fastrp_embedding)
YIELD node AS similar, score AS structural_score
RETURN similar, (semantic_score + structural_score) / 2 AS combined_score
ORDER BY combined_score DESC
```

## What's New in v2.0

### User Experience
- ✅ **Mobile-first responsive design** - works flawlessly on phones (375px+), tablets, and desktop
- ✅ **Tool result cards** - every tool call shows an embedded NVL graph with the subgraph used
- ✅ **Modern chat UI** - avatars, improved message bubbles, collapsible tool results
- ✅ **Always-visible tools** - sidebar drawer with all 12 tools and their parameters
- ✅ **Data model transparency** - see which node types and relationships were traversed

### Backend Enhancements
- ✅ **Multi-hop traversal** - new `find_related_context` tool with 2-3 hop queries
- ✅ **SIMILAR_TO relationships** - GDS Node Similarity creates co-occurrence relationships
- ✅ **Richer context** - all tools return graph_data for visualization

### Design System
- ✅ **Neo4j Needle branding** - Baltic Sea Blue (#018BFF), nature-inspired green (#10B860)
- ✅ **Public Sans typography** - Neo4j's official typeface
- ✅ **Semantic tokens** - consistent light/dark mode support

## References

- [Neo4j Agent Memory Package](https://github.com/neo4j-labs/agent-memory) - Official Python package
- [Lenny's Memory Blog Post](https://medium.com/neo4j/meet-lennys-memory-building-context-graphs-for-ai-agents-24cb102fb91a) - Announcement
- [AI's Trillion-Dollar Opportunity: Context Graphs](https://foundationcapital.com/context-graphs-ais-trillion-dollar-opportunity/) - Foundation Capital
- [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk-python) - Anthropic
- [NVL Visualization Library](https://neo4j.com/docs/nvl/) - Neo4j

## License

MIT
