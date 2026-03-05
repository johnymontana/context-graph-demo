"""
Neo4j client for context graph operations.
Handles entities, decisions, and causal relationships.

Note: This client is being migrated to use neo4j-agent-memory for memory operations.
Some operations now delegate to memory_client, while graph visualization and
domain-specific queries remain here.
"""

import uuid
from datetime import date, datetime
from typing import Any, Optional

from neo4j import AsyncGraphDatabase, GraphDatabase
from neo4j.exceptions import ServiceUnavailable
from neo4j.time import Date as Neo4jDate
from neo4j.time import DateTime as Neo4jDateTime

from .config import config
from .models import (
    Account,
    CausalChain,
    Decision,
    GraphData,
    GraphNode,
    GraphRelationship,
    Person,
    Transaction,
)


def convert_neo4j_value(value: Any) -> Any:
    """Convert Neo4j types to JSON-serializable Python types."""
    if isinstance(value, Neo4jDateTime):
        return value.isoformat()
    elif isinstance(value, Neo4jDate):
        return value.isoformat()
    elif isinstance(value, (datetime, date)):
        return value.isoformat()
    elif isinstance(value, list):
        return [convert_neo4j_value(v) for v in value]
    elif isinstance(value, dict):
        return {k: convert_neo4j_value(v) for k, v in value.items()}
    return value


def convert_node_properties(props: dict) -> dict:
    """Convert all properties in a node to JSON-serializable types."""
    return {k: convert_neo4j_value(v) for k, v in props.items()}


class ContextGraphClient:
    """Neo4j client for context graph operations."""

    def __init__(self):
        self.driver = GraphDatabase.driver(
            config.neo4j.uri,
            auth=(config.neo4j.username, config.neo4j.password),
        )
        self.database = config.neo4j.database

    def close(self):
        self.driver.close()

    def ensure_indexes(self) -> dict:
        """Ensure all required indexes exist, creating them if necessary."""
        results = {"created": [], "existing": [], "errors": []}

        # Define required indexes
        indexes = [
            # Constraints (unique IDs)
            (
                "constraint",
                "person_id_unique",
                "CREATE CONSTRAINT person_id_unique IF NOT EXISTS FOR (p:Person) REQUIRE p.id IS UNIQUE",
            ),
            (
                "constraint",
                "account_id_unique",
                "CREATE CONSTRAINT account_id_unique IF NOT EXISTS FOR (a:Account) REQUIRE a.id IS UNIQUE",
            ),
            (
                "constraint",
                "transaction_id_unique",
                "CREATE CONSTRAINT transaction_id_unique IF NOT EXISTS FOR (t:Transaction) REQUIRE t.id IS UNIQUE",
            ),
            (
                "constraint",
                "decision_id_unique",
                "CREATE CONSTRAINT decision_id_unique IF NOT EXISTS FOR (d:Decision) REQUIRE d.id IS UNIQUE",
            ),
            (
                "constraint",
                "policy_id_unique",
                "CREATE CONSTRAINT policy_id_unique IF NOT EXISTS FOR (p:Policy) REQUIRE p.id IS UNIQUE",
            ),
            (
                "constraint",
                "employee_id_unique",
                "CREATE CONSTRAINT employee_id_unique IF NOT EXISTS FOR (e:Employee) REQUIRE e.id IS UNIQUE",
            ),
            (
                "constraint",
                "organization_id_unique",
                "CREATE CONSTRAINT organization_id_unique IF NOT EXISTS FOR (o:Organization) REQUIRE o.id IS UNIQUE",
            ),
            # Text indexes for search
            (
                "index",
                "person_name_idx",
                "CREATE INDEX person_name_idx IF NOT EXISTS FOR (p:Person) ON (p.normalized_name)",
            ),
            (
                "index",
                "account_number_idx",
                "CREATE INDEX account_number_idx IF NOT EXISTS FOR (a:Account) ON (a.account_number)",
            ),
            (
                "index",
                "decision_type_category_idx",
                "CREATE INDEX decision_type_category_idx IF NOT EXISTS FOR (d:Decision) ON (d.decision_type, d.category)",
            ),
            (
                "index",
                "decision_timestamp_idx",
                "CREATE INDEX decision_timestamp_idx IF NOT EXISTS FOR (d:Decision) ON (d.decision_timestamp)",
            ),
            (
                "index",
                "policy_category_idx",
                "CREATE INDEX policy_category_idx IF NOT EXISTS FOR (p:Policy) ON (p.category)",
            ),
            # Vector indexes for semantic search (1536 dims for OpenAI embeddings)
            (
                "vector",
                "decision_reasoning_idx",
                """
                CREATE VECTOR INDEX decision_reasoning_idx IF NOT EXISTS
                FOR (d:Decision) ON (d.reasoning_embedding)
                OPTIONS {indexConfig: {`vector.dimensions`: 1536, `vector.similarity_function`: 'cosine'}}
            """,
            ),
            (
                "vector",
                "policy_description_idx",
                """
                CREATE VECTOR INDEX policy_description_idx IF NOT EXISTS
                FOR (p:Policy) ON (p.description_embedding)
                OPTIONS {indexConfig: {`vector.dimensions`: 1536, `vector.similarity_function`: 'cosine'}}
            """,
            ),
            # Vector indexes for FastRP structural embeddings (128 dims)
            (
                "vector",
                "decision_fastrp_idx",
                """
                CREATE VECTOR INDEX decision_fastrp_idx IF NOT EXISTS
                FOR (d:Decision) ON (d.fastrp_embedding)
                OPTIONS {indexConfig: {`vector.dimensions`: 128, `vector.similarity_function`: 'cosine'}}
            """,
            ),
            (
                "vector",
                "person_fastrp_idx",
                """
                CREATE VECTOR INDEX person_fastrp_idx IF NOT EXISTS
                FOR (p:Person) ON (p.fastrp_embedding)
                OPTIONS {indexConfig: {`vector.dimensions`: 128, `vector.similarity_function`: 'cosine'}}
            """,
            ),
            (
                "vector",
                "account_fastrp_idx",
                """
                CREATE VECTOR INDEX account_fastrp_idx IF NOT EXISTS
                FOR (a:Account) ON (a.fastrp_embedding)
                OPTIONS {indexConfig: {`vector.dimensions`: 128, `vector.similarity_function`: 'cosine'}}
            """,
            ),
        ]

        with self.driver.session(database=self.database) as session:
            for index_type, name, cypher in indexes:
                try:
                    session.run(cypher.strip())
                    results["created"].append(f"{index_type}:{name}")
                except Exception as e:
                    error_msg = str(e)
                    if "already exists" in error_msg.lower() or "equivalent" in error_msg.lower():
                        results["existing"].append(f"{index_type}:{name}")
                    else:
                        results["errors"].append(f"{index_type}:{name} - {error_msg}")

        return results

    def verify_connectivity(self) -> bool:
        """Verify connection to Neo4j."""
        try:
            self.driver.verify_connectivity()
            return True
        except ServiceUnavailable:
            return False

    # ============================================
    # CUSTOMER/PERSON OPERATIONS
    # ============================================

    def search_customers(self, query: str, limit: int = 10) -> list[dict]:
        """Search for customers by name, email, or account number."""
        with self.driver.session(database=self.database) as session:
            result = session.run(
                """
                MATCH (p:Person)
                WHERE toLower(p.name) CONTAINS toLower($query)
                   OR toLower(p.email) CONTAINS toLower($query)
                   OR EXISTS {
                       MATCH (p)-[:OWNS]->(a:Account)
                       WHERE a.account_number CONTAINS $query
                   }
                OPTIONAL MATCH (p)-[:OWNS]->(a:Account)
                OPTIONAL MATCH (d:Decision)-[:ABOUT]->(p)
                RETURN p.id AS id,
                       p.name AS name,
                       p.email AS email,
                       p.risk_score AS risk_score,
                       count(DISTINCT a) AS account_count,
                       count(DISTINCT d) AS decision_count
                ORDER BY p.risk_score DESC
                LIMIT $limit
                """,
                {"query": query, "limit": limit},
            )
            return [dict(record) for record in result]

    def get_customer(self, customer_id: str) -> Optional[dict]:
        """Get a customer by ID with related entities."""
        with self.driver.session(database=self.database) as session:
            result = session.run(
                """
                MATCH (p:Person {id: $customer_id})
                OPTIONAL MATCH (p)-[:OWNS]->(a:Account)
                OPTIONAL MATCH (p)-[:WORKS_FOR]->(o:Organization)
                RETURN p {
                    .*,
                    accounts: collect(DISTINCT a {.*}),
                    organizations: collect(DISTINCT o {.*})
                } AS customer
                """,
                {"customer_id": customer_id},
            )
            record = result.single()
            return record["customer"] if record else None

    def get_customer_decisions(
        self,
        customer_id: str,
        decision_type: Optional[str] = None,
        limit: int = 20,
    ) -> list[dict]:
        """Get all decisions made about a customer."""
        type_filter = "AND d.decision_type = $decision_type" if decision_type else ""

        with self.driver.session(database=self.database) as session:
            result = session.run(
                f"""
                MATCH (d:Decision)-[:ABOUT]->(p:Person {{id: $customer_id}})
                WHERE true {type_filter}
                OPTIONAL MATCH (d)-[:MADE_BY]->(maker)
                OPTIONAL MATCH (d)-[:APPLIED_POLICY]->(policy:Policy)
                WITH d, maker, collect(DISTINCT policy.name) AS policies_applied
                RETURN d {{
                    .*,
                    made_by: maker.name,
                    policies_applied: policies_applied
                }} AS decision
                ORDER BY decision.decision_timestamp DESC
                LIMIT $limit
                """,
                {
                    "customer_id": customer_id,
                    "decision_type": decision_type,
                    "limit": limit,
                },
            )
            return [record["decision"] for record in result]

    # ============================================
    # DECISION OPERATIONS
    # ============================================

    def get_decision(self, decision_id: str) -> Optional[dict]:
        """Get a decision by ID with full context."""
        with self.driver.session(database=self.database) as session:
            result = session.run(
                """
                MATCH (d:Decision {id: $decision_id})
                OPTIONAL MATCH (d)-[:ABOUT]->(entity)
                OPTIONAL MATCH (d)-[:MADE_BY]->(maker)
                OPTIONAL MATCH (d)-[:APPLIED_POLICY]->(policy:Policy)
                OPTIONAL MATCH (d)-[:GRANTED_EXCEPTION]->(exception:Exception)
                OPTIONAL MATCH (d)-[:TRIGGERED]->(escalation:Escalation)
                OPTIONAL MATCH (d)-[:HAD_CONTEXT]->(context:DecisionContext)
                RETURN d {
                    .*,
                    about_entities: collect(DISTINCT {id: entity.id, labels: labels(entity), name: entity.name}),
                    made_by: maker {.*},
                    policies: collect(DISTINCT policy {.*}),
                    exceptions: collect(DISTINCT exception {.*}),
                    escalations: collect(DISTINCT escalation {.*}),
                    contexts: collect(DISTINCT context {.*})
                } AS decision
                """,
                {"decision_id": decision_id},
            )
            record = result.single()
            return record["decision"] if record else None

    def record_decision(
        self,
        decision_type: str,
        category: str,
        reasoning: str,
        customer_id: Optional[str] = None,
        account_id: Optional[str] = None,
        transaction_id: Optional[str] = None,
        risk_factors: list[str] = None,
        precedent_ids: list[str] = None,
        confidence_score: float = 0.8,
        session_id: Optional[str] = None,
        reasoning_embedding: Optional[list[float]] = None,
    ) -> str:
        """Record a new decision with full context."""
        decision_id = str(uuid.uuid4())
        risk_factors = risk_factors or []
        precedent_ids = precedent_ids or []

        with self.driver.session(database=self.database) as session:
            # Create the decision
            session.run(
                """
                CREATE (d:Decision {
                    id: $decision_id,
                    decision_type: $decision_type,
                    category: $category,
                    status: 'completed',
                    decision_timestamp: datetime(),
                    reasoning: $reasoning,
                    reasoning_summary: $reasoning_summary,
                    confidence_score: $confidence_score,
                    risk_factors: $risk_factors,
                    session_id: $session_id,
                    reasoning_embedding: $reasoning_embedding,
                    created_at: datetime()
                })
                """,
                {
                    "decision_id": decision_id,
                    "decision_type": decision_type,
                    "category": category,
                    "reasoning": reasoning,
                    "reasoning_summary": reasoning[:100] + "..."
                    if len(reasoning) > 100
                    else reasoning,
                    "confidence_score": confidence_score,
                    "risk_factors": risk_factors,
                    "session_id": session_id,
                    "reasoning_embedding": reasoning_embedding,
                },
            )

            # Link to customer
            if customer_id:
                session.run(
                    """
                    MATCH (d:Decision {id: $decision_id})
                    MATCH (p:Person {id: $customer_id})
                    MERGE (d)-[:ABOUT]->(p)
                    """,
                    {"decision_id": decision_id, "customer_id": customer_id},
                )

            # Link to account
            if account_id:
                session.run(
                    """
                    MATCH (d:Decision {id: $decision_id})
                    MATCH (a:Account {id: $account_id})
                    MERGE (d)-[:ABOUT]->(a)
                    """,
                    {"decision_id": decision_id, "account_id": account_id},
                )

            # Link to transaction
            if transaction_id:
                session.run(
                    """
                    MATCH (d:Decision {id: $decision_id})
                    MATCH (t:Transaction {id: $transaction_id})
                    MERGE (d)-[:ABOUT]->(t)
                    """,
                    {"decision_id": decision_id, "transaction_id": transaction_id},
                )

            # Link to precedents
            for precedent_id in precedent_ids:
                session.run(
                    """
                    MATCH (d:Decision {id: $decision_id})
                    MATCH (p:Decision {id: $precedent_id})
                    MERGE (d)-[:FOLLOWED_PRECEDENT]->(p)
                    """,
                    {"decision_id": decision_id, "precedent_id": precedent_id},
                )

        return decision_id

    def list_decisions(
        self,
        category: Optional[str] = None,
        decision_type: Optional[str] = None,
        limit: int = 20,
    ) -> list[dict]:
        """List recent decisions with optional filters."""
        filters = []
        params = {"limit": limit}

        if category:
            filters.append("d.category = $category")
            params["category"] = category
        if decision_type:
            filters.append("d.decision_type = $decision_type")
            params["decision_type"] = decision_type

        where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""

        with self.driver.session(database=self.database) as session:
            result = session.run(
                f"""
                MATCH (d:Decision)
                {where_clause}
                OPTIONAL MATCH (d)-[:ABOUT]->(target)
                WITH d, collect(DISTINCT labels(target)[0]) AS target_types
                RETURN d {{
                    .*,
                    target_types: target_types
                }} AS decision
                ORDER BY decision.decision_timestamp DESC
                LIMIT $limit
                """,
                params,
            )
            decisions = []
            for record in result:
                decision = dict(record["decision"])
                # Convert Neo4j types
                decision = convert_node_properties(decision)
                decisions.append(decision)
            return decisions

    def get_causal_chain(
        self,
        decision_id: str,
        direction: str = "both",
        depth: int = 3,
    ) -> dict:
        """Trace the causal chain of a decision."""
        with self.driver.session(database=self.database) as session:
            causes = []
            effects = []

            if direction in ("both", "causes"):
                result = session.run(
                    f"""
                    MATCH (d:Decision {{id: $decision_id}})
                    MATCH path = (cause:Decision)-[:CAUSED|INFLUENCED*1..{depth}]->(d)
                    WITH cause, length(path) AS distance
                    RETURN cause {{.*, distance: distance}} AS decision
                    ORDER BY distance
                    """,
                    {"decision_id": decision_id},
                )
                causes = [record["decision"] for record in result]

            if direction in ("both", "effects"):
                result = session.run(
                    f"""
                    MATCH (d:Decision {{id: $decision_id}})
                    MATCH path = (d)-[:CAUSED|INFLUENCED*1..{depth}]->(effect:Decision)
                    WITH effect, length(path) AS distance
                    RETURN effect {{.*, distance: distance}} AS decision
                    ORDER BY distance
                    """,
                    {"decision_id": decision_id},
                )
                effects = [record["decision"] for record in result]

            return {
                "decision_id": decision_id,
                "causes": causes,
                "effects": effects,
                "depth": depth,
            }

    # ============================================
    # POLICY OPERATIONS
    # ============================================

    def get_policies(self, category: Optional[str] = None) -> list[dict]:
        """Get policies, optionally filtered by category."""
        category_filter = "WHERE p.category = $category" if category else ""

        with self.driver.session(database=self.database) as session:
            result = session.run(
                f"""
                MATCH (p:Policy)
                {category_filter}
                RETURN p {{.*}} AS policy
                ORDER BY p.name
                """,
                {"category": category},
            )
            return [record["policy"] for record in result]

    def get_policy(self, policy_id: str) -> Optional[dict]:
        """Get a policy by ID."""
        with self.driver.session(database=self.database) as session:
            result = session.run(
                """
                MATCH (p:Policy {id: $policy_id})
                OPTIONAL MATCH (d:Decision)-[:APPLIED_POLICY]->(p)
                RETURN p {
                    .*,
                    usage_count: count(d)
                } AS policy
                """,
                {"policy_id": policy_id},
            )
            record = result.single()
            return record["policy"] if record else None

    # ============================================
    # GRAPH VISUALIZATION
    # ============================================

    def get_graph_data(
        self,
        center_node_id: Optional[str] = None,
        center_node_type: Optional[str] = None,
        depth: int = 2,
        include_decisions: bool = True,
        limit: int = 100,
    ) -> GraphData:
        """Get graph data for NVL visualization."""
        with self.driver.session(database=self.database) as session:
            if center_node_id:
                # Get subgraph centered on a specific node using variable-length paths
                # Support both UUID property and element ID
                result = session.run(
                    """
                    MATCH (center)
                    WHERE center.id = $center_id OR elementId(center) = $center_id
                    OPTIONAL MATCH (center)-[r1:!HAS_SIMILAR_FACTORS&!BELONGS_TO_DECISION_COMMUNITY&!BELONGS_TO_ACCOUNT_COMMUNITY]-(n1)
                    OPTIONAL MATCH (n1)-[r2:!HAS_SIMILAR_FACTORS&!BELONGS_TO_DECISION_COMMUNITY&!BELONGS_TO_ACCOUNT_COMMUNITY]-(n2) WHERE n2 <> center
                    WITH center,
                         collect(DISTINCT n1) + collect(DISTINCT n2) AS connectedNodes,
                         collect(DISTINCT r1) + collect(DISTINCT r2) AS allRels
                    WITH [center] + connectedNodes[0..$limit] AS nodes, allRels AS relationships
                    RETURN nodes, relationships
                    """,
                    {"center_id": center_node_id, "limit": limit},
                )
            else:
                # Get a sample of the graph - mix of different node types
                decision_filter = "" if include_decisions else "WHERE NOT 'Decision' IN labels(n)"
                result = session.run(
                    f"""
                    MATCH (n)
                    {decision_filter}
                    WITH n LIMIT $limit
                    OPTIONAL MATCH (n)-[r]-(m)
                    WITH collect(DISTINCT n) + collect(DISTINCT m) AS nodes,
                         collect(DISTINCT r) AS relationships
                    RETURN nodes[0..$limit] AS nodes, relationships
                    """,
                    {"limit": limit},
                )

            record = result.single()
            if not record:
                return GraphData(nodes=[], relationships=[])

            nodes = []
            seen_node_ids = set()
            for node in record["nodes"] or []:
                if node and node.element_id not in seen_node_ids:
                    seen_node_ids.add(node.element_id)
                    nodes.append(
                        GraphNode(
                            id=str(node.element_id),
                            labels=list(node.labels),
                            properties=convert_node_properties(dict(node)),
                        )
                    )

            relationships = []
            seen_rel_ids = set()
            for rel in record["relationships"] or []:
                if rel is not None and rel.element_id not in seen_rel_ids:
                    seen_rel_ids.add(rel.element_id)
                    relationships.append(
                        GraphRelationship(
                            id=str(rel.element_id),
                            type=rel.type,
                            start_node_id=str(rel.start_node.element_id),
                            end_node_id=str(rel.end_node.element_id),
                            properties=convert_node_properties(dict(rel)),
                        )
                    )

            return GraphData(nodes=nodes, relationships=relationships)

    def get_connected_nodes(
        self,
        node_id: str,
        limit: int = 50,
    ) -> GraphData:
        """Get all nodes directly connected to a given node."""
        with self.driver.session(database=self.database) as session:
            result = session.run(
                """
                MATCH (center)
                WHERE center.id = $node_id OR elementId(center) = $node_id
                OPTIONAL MATCH (center)-[r]-(connected)
                WITH center, collect(DISTINCT connected)[0..$limit] AS connectedNodes,
                     collect(DISTINCT r) AS rels
                RETURN [center] + connectedNodes AS nodes, rels AS relationships
                """,
                {"node_id": node_id, "limit": limit},
            )

            record = result.single()
            if not record:
                return GraphData(nodes=[], relationships=[])

            nodes = []
            seen_node_ids = set()
            for node in record["nodes"] or []:
                if node and node.element_id not in seen_node_ids:
                    seen_node_ids.add(node.element_id)
                    nodes.append(
                        GraphNode(
                            id=str(node.element_id),
                            labels=list(node.labels),
                            properties=convert_node_properties(dict(node)),
                        )
                    )

            relationships = []
            seen_rel_ids = set()
            for rel in record["relationships"] or []:
                if rel is not None and rel.element_id not in seen_rel_ids:
                    seen_rel_ids.add(rel.element_id)
                    relationships.append(
                        GraphRelationship(
                            id=str(rel.element_id),
                            type=rel.type,
                            start_node_id=str(rel.start_node.element_id),
                            end_node_id=str(rel.end_node.element_id),
                            properties=convert_node_properties(dict(rel)),
                        )
                    )

            return GraphData(nodes=nodes, relationships=relationships)

    def get_relationships_between_nodes(
        self,
        node_ids: list[str],
    ) -> list[GraphRelationship]:
        """Get all relationships between a set of nodes."""
        if len(node_ids) < 2:
            return []

        with self.driver.session(database=self.database) as session:
            # Query for relationships where both endpoints are in our node list
            result = session.run(
                """
                MATCH (a)-[r]->(b)
                WHERE (a.id IN $node_ids OR elementId(a) IN $node_ids)
                  AND (b.id IN $node_ids OR elementId(b) IN $node_ids)
                RETURN DISTINCT r
                """,
                {"node_ids": node_ids},
            )

            relationships = []
            seen_rel_ids = set()
            for record in result:
                rel = record["r"]
                if rel is not None and rel.element_id not in seen_rel_ids:
                    seen_rel_ids.add(rel.element_id)
                    relationships.append(
                        GraphRelationship(
                            id=str(rel.element_id),
                            type=rel.type,
                            start_node_id=str(rel.start_node.element_id),
                            end_node_id=str(rel.end_node.element_id),
                            properties=convert_node_properties(dict(rel)),
                        )
                    )

            return relationships

    # ============================================
    # STATISTICS
    # ============================================

    def get_statistics(self) -> dict:
        """Get graph statistics."""
        with self.driver.session(database=self.database) as session:
            result = session.run(
                """
                MATCH (n)
                WITH labels(n) AS nodeLabels
                UNWIND nodeLabels AS label
                WITH label, count(*) AS count
                RETURN collect({label: label, count: count}) AS node_counts
                """
            )
            record = result.single()
            node_counts = {item["label"]: item["count"] for item in record["node_counts"]}

            result = session.run(
                """
                MATCH ()-[r]->()
                WITH type(r) AS relType
                WITH relType, count(*) AS count
                RETURN collect({type: relType, count: count}) AS rel_counts
                """
            )
            record = result.single()
            rel_counts = {item["type"]: item["count"] for item in record["rel_counts"]}

            return {
                "node_counts": node_counts,
                "relationship_counts": rel_counts,
                "total_nodes": sum(node_counts.values()),
                "total_relationships": sum(rel_counts.values()),
            }

    # ============================================
    # CYPHER EXECUTION (Read-only)
    # ============================================

    def execute_cypher(self, cypher: str, parameters: dict = None) -> list[dict]:
        """Execute a read-only Cypher query."""
        # Basic safety check - only allow read operations
        cypher_upper = cypher.upper().strip()
        if any(
            keyword in cypher_upper
            for keyword in ["CREATE", "MERGE", "DELETE", "SET", "REMOVE", "DROP"]
        ):
            raise ValueError("Only read operations are allowed")

        with self.driver.session(database=self.database) as session:
            result = session.run(cypher, parameters or {})
            return [dict(record) for record in result]

    def get_schema(self) -> dict[str, Any]:
        """Get the graph database schema including node labels, relationship types, and properties."""
        with self.driver.session(database=self.database) as session:
            # Get node labels and their properties
            node_labels_result = session.run("""
                CALL db.labels() YIELD label
                RETURN label ORDER BY label
            """)
            node_labels = [record["label"] for record in node_labels_result]

            # Get relationship types
            rel_types_result = session.run("""
                CALL db.relationshipTypes() YIELD relationshipType
                RETURN relationshipType ORDER BY relationshipType
            """)
            relationship_types = [record["relationshipType"] for record in rel_types_result]

            # Get property keys
            prop_keys_result = session.run("""
                CALL db.propertyKeys() YIELD propertyKey
                RETURN propertyKey ORDER BY propertyKey
            """)
            property_keys = [record["propertyKey"] for record in prop_keys_result]

            # Get node counts by label
            node_counts = {}
            for label in node_labels:
                count_result = session.run(f"MATCH (n:`{label}`) RETURN count(n) as count")
                node_counts[label] = count_result.single()["count"]

            # Get relationship counts by type
            rel_counts = {}
            for rel_type in relationship_types:
                count_result = session.run(
                    f"MATCH ()-[r:`{rel_type}`]->() RETURN count(r) as count"
                )
                rel_counts[rel_type] = count_result.single()["count"]

            # Get relationship patterns (which labels connect via which relationship types)
            patterns_result = session.run("""
                MATCH (a)-[r]->(b)
                WITH labels(a) AS from_labels, type(r) AS rel_type, labels(b) AS to_labels, count(*) AS count
                UNWIND from_labels AS from_label
                UNWIND to_labels AS to_label
                RETURN DISTINCT from_label, rel_type, to_label, sum(count) AS count
                ORDER BY from_label, rel_type, to_label
            """)
            relationship_patterns = [
                {
                    "from_label": record["from_label"],
                    "rel_type": record["rel_type"],
                    "to_label": record["to_label"],
                    "count": record["count"],
                }
                for record in patterns_result
            ]

            # Get indexes
            indexes_result = session.run("""
                SHOW INDEXES YIELD name, type, labelsOrTypes, properties, state
                RETURN name, type, labelsOrTypes, properties, state
            """)
            indexes = [
                {
                    "name": record["name"],
                    "type": record["type"],
                    "labels_or_types": record["labelsOrTypes"],
                    "properties": record["properties"],
                    "state": record["state"],
                }
                for record in indexes_result
            ]

            # Get constraints
            constraints_result = session.run("""
                SHOW CONSTRAINTS YIELD name, type, labelsOrTypes, properties
                RETURN name, type, labelsOrTypes, properties
            """)
            constraints = [
                {
                    "name": record["name"],
                    "type": record["type"],
                    "labels_or_types": record["labelsOrTypes"],
                    "properties": record["properties"],
                }
                for record in constraints_result
            ]

            return {
                "node_labels": node_labels,
                "node_counts": node_counts,
                "relationship_types": relationship_types,
                "relationship_counts": rel_counts,
                "relationship_patterns": relationship_patterns,
                "property_keys": property_keys,
                "indexes": indexes,
                "constraints": constraints,
            }


# Singleton instance
context_graph_client = ContextGraphClient()
