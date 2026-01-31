# Lenny's Memory - Demo Examples

These examples showcase the key features of **Lenny's Memory**, the official demo application for Neo4j Agent Memory. Each example demonstrates how AI agents can leverage graph-based decision tracing for financial institutions.

## Using the Mobile-First Interface

**Lenny's Memory** features a modern, mobile-first chat interface where:
- **Tool results appear as cards** below the assistant's message
- **Each card contains an embedded graph** showing the exact subgraph used to answer your question
- **Click the graph tab** to see the full interactive visualization
- **Expand/collapse cards** to focus on what matters
- **Tap the Tools button** in the header to see all 12 available tools

---

## 1. Fraud Detection & Pattern Analysis

**Best for:** Showing fraud detection with graph visualization and flagged transactions

```
Valerie Howard has multiple flagged transactions on her checking account. Analyze her account for fraud patterns and find similar cases.
```

**What this demonstrates:**
- Customer search populates a tool result card with account relationships graph
- Fraud detection tool card shows suspicious network patterns via embedded NVL visualization
- Each tool result (search, fraud detection) appears as a separate card with its own subgraph
- You can see exactly which nodes and relationships the agent used to make its determination
- Flagged transactions are highlighted in the graph visualization

**Alternative:**
```
Check Alan Kramer's margin account for suspicious activity. He has several flagged transactions.
```

---

## 2. Credit Decision with Precedent Search

**Best for:** Semantic search using text embeddings to find similar past decisions

```
Should we approve a $25,000 credit limit increase? The customer has a margin account and moderate income. Find similar past credit decisions to guide this recommendation.
```

**What this demonstrates:**
- `find_precedents` tool uses OpenAI embeddings for semantic similarity
- Returns past decisions with similar reasoning text
- Shows confidence scores and decision outcomes
- Policy lookup for Credit Limit Policy

---

## 3. Customer Account Overview

**Best for:** Graph visualization showing entity relationships

```
Show me everything about Jacob Fitzpatrick - all his accounts, transactions, and any decisions made about him.
```

**What this demonstrates:**
- Customer has 4 accounts (savings, trading, checking)
- Graph expands to show account relationships
- Double-click nodes to explore further connections
- Decision Trace panel shows decisions from graph

---

## 4. Policy Compliance Check

**Best for:** Policy lookup and compliance verification

```
A customer wants to make a $15,000 wire transfer to an international account. What policies apply and what verification is needed?
```

**What this demonstrates:**
- `get_policy` tool finds relevant policies (High-Value Transaction Review, Wire Transfer Verification)
- Shows policy thresholds and requirements
- Finds precedent decisions for similar transactions

---

## 5. Decision Community Analysis

**Best for:** Louvain community detection showing related decisions

```
Find decisions related to the recent trading exception we approved. What other decisions are in the same cluster?
```

**What this demonstrates:**
- `find_decision_community` uses Louvain algorithm
- Community nodes connect related decisions via BELONGS_TO relationships
- Shows how decisions are structurally grouped by causal chains

---

## 6. Trading Limit Exception

**Best for:** Exception handling with precedent lookup

```
Samuel Jones wants to exceed his trading limit on his margin account. Find precedents for similar trading exceptions and what the outcomes were.
```

**What this demonstrates:**
- Customer has margin and trading accounts
- Searches for exception decisions in trading category
- Shows precedent outcomes (approved vs rejected)
- Trading Limit Policy and Margin Call Requirements

---

## 7. Multi-Step Investigation

**Best for:** Demonstrating causal chain tracing

```
We had a fraud rejection last week that led to an account freeze. Trace the causal chain - what triggered it and what decisions followed?
```

**What this demonstrates:**
- `get_causal_chain` tool traces CAUSED and INFLUENCED relationships
- Shows upstream causes and downstream effects
- Decision Trace panel displays the full chain
- Graph visualization shows decision flow

---

## Recommended Demo Flow

### Opening Demo (2 minutes)
Start with this comprehensive example:

```
Valerie Howard has 6 flagged transactions on her account. Check for fraud patterns, find similar cases, and recommend next steps based on our policies and past decisions.
```

This single message will:
1. **Search customer** - populates graph with Valerie Howard's account network
2. **Detect fraud patterns** - uses GDS Node Similarity against known fraud cases
3. **Find precedents** - semantic search for similar fraud scenarios
4. **Get policies** - retrieves Account Freeze Protocol and fraud policies
5. **Show graph** - visualizes customer, accounts, transactions, and related decisions

### Key Points to Highlight

1. **Tool Result Cards**: Each agent tool call appears as a card with tabs (Graph | Summary | Input | Output)
2. **Embedded Graphs**: Every card shows the exact subgraph used - no more disconnected nodes!
3. **Mobile-Optimized**: The interface works perfectly on phones, tablets, and desktop
4. **Data Model Transparency**: Hover over the info badge to see which node types and relationships were traversed
5. **Multiple Similarity Methods**:
   - Semantic (text embeddings) - for finding decisions with similar reasoning
   - Structural (FastRP + KNN) - for finding decisions with similar graph patterns
   - Community (Louvain) - for finding decisions in the same cluster
   - Multi-hop (new!) - traverse 2-3 hops including SIMILAR_TO relationships

### Data Highlights

| Customer | Notable Feature |
|----------|-----------------|
| Valerie Howard | 6 flagged transactions (most in system) |
| Alan Kramer | 5 flagged transactions on margin account |
| Jacob Fitzpatrick | 4 accounts across 3 types |
| Christian Martinez | Margin, savings, and trading accounts |
| Laurie Li | 4 flagged transactions on trading account |

| Policy | Category | Use Case |
|--------|----------|----------|
| Credit Limit Policy | credit | Credit increase requests |
| High-Value Transaction Review | fraud | Transactions over $10,000 |
| Account Freeze Protocol | fraud | Immediate freeze conditions |
| Trading Limit Policy | trading | Position size limits |
| Wire Transfer Verification | fraud | Outgoing wire verification |
