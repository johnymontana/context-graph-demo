"""
Neo4j Graph Data Science (GDS) client.
Implements FastRP, KNN, Node Similarity, Louvain, and PageRank.
Uses the graphdatascience Python package for AuraDB compatibility.
"""

from typing import Optional

from graphdatascience import GraphDataScience

from .config import config


class GDSClient:
    """Neo4j GDS client for graph algorithms using the graphdatascience package."""

    def __init__(self):
        # Connect using graphdatascience with AuraDS support
        self.gds = GraphDataScience(
            config.neo4j.uri,
            auth=(config.neo4j.username, config.neo4j.password),
            database=config.neo4j.database,
            aura_ds=False,  # Enable AuraDS-compatible settings
        )
        self.fastrp_dimensions = config.fastrp_dimensions
        # Cache for graph projections
        self._decision_graph = None
        self._entity_graph = None

    def close(self):
        """Close the GDS connection."""
        # Drop any cached graph projections
        if self._decision_graph is not None:
            try:
                self._decision_graph.drop()
            except Exception:
                pass
        if self._entity_graph is not None:
            try:
                self._entity_graph.drop()
            except Exception:
                pass

    # ============================================
    # GRAPH PROJECTION MANAGEMENT
    # ============================================

    def create_decision_graph_projection(self) -> dict:
        """Create the decision graph projection for GDS algorithms."""
        # Drop if exists
        if self.gds.graph.exists("decision-graph").iloc[0]:
            self.gds.graph.drop("decision-graph")

        G, result = self.gds.graph.project(
            "decision-graph",
            ["Decision", "Person", "Account", "Transaction", "Organization", "Policy", "Employee"],
            {
                "ABOUT": {"orientation": "UNDIRECTED"},
                "CAUSED": {"orientation": "NATURAL", "properties": ["confidence"]},
                "INFLUENCED": {"orientation": "NATURAL", "properties": ["weight"]},
                "PRECEDENT_FOR": {"orientation": "NATURAL", "properties": ["similarity_score"]},
                "OWNS": {"orientation": "UNDIRECTED"},
                "MADE_BY": {"orientation": "NATURAL"},
                "APPLIED_POLICY": {"orientation": "NATURAL"},
                "FROM_ACCOUNT": {"orientation": "NATURAL"},
                "TO_ACCOUNT": {"orientation": "NATURAL"},
            },
        )
        self._decision_graph = G
        return {
            "graphName": result["graphName"],
            "nodeCount": result["nodeCount"],
            "relationshipCount": result["relationshipCount"],
        }

    def create_entity_graph_projection(self) -> dict:
        """Create the entity graph projection for fraud detection."""
        # Drop if exists
        if self.gds.graph.exists("entity-graph").iloc[0]:
            self.gds.graph.drop("entity-graph")

        G, result = self.gds.graph.project(
            "entity-graph",
            ["Person", "Account", "Transaction", "Organization"],
            {
                "OWNS": {"orientation": "UNDIRECTED"},
                "FROM_ACCOUNT": {"orientation": "NATURAL"},
                "TO_ACCOUNT": {"orientation": "NATURAL"},
                "INVOLVING": {"orientation": "UNDIRECTED"},
                "RELATED_TO": {"orientation": "UNDIRECTED"},
            },
        )
        self._entity_graph = G
        return {
            "graphName": result["graphName"],
            "nodeCount": result["nodeCount"],
            "relationshipCount": result["relationshipCount"],
        }

    def _ensure_decision_graph_exists(self) -> None:
        """Ensure the decision-graph projection exists, creating it if necessary."""
        if self._decision_graph is None:
            if not self.gds.graph.exists("decision-graph").iloc[0]:
                self.create_decision_graph_projection()
            else:
                self._decision_graph = self.gds.graph.get("decision-graph")

    def _ensure_entity_graph_exists(self) -> None:
        """Ensure the entity-graph projection exists, creating it if necessary."""
        if self._entity_graph is None:
            if not self.gds.graph.exists("entity-graph").iloc[0]:
                self.create_entity_graph_projection()
            else:
                self._entity_graph = self.gds.graph.get("entity-graph")

    def list_graph_projections(self) -> list[dict]:
        """List all graph projections."""
        result = self.gds.graph.list()
        return result.to_dict("records")

    # ============================================
    # FASTRP EMBEDDINGS
    # ============================================

    def generate_fastrp_embeddings(
        self,
        graph_name: str = "decision-graph",
        node_labels: Optional[list[str]] = None,
    ) -> dict:
        """Generate FastRP embeddings for nodes."""
        self._ensure_decision_graph_exists()

        result = self.gds.fastRP.mutate(
            self._decision_graph,
            embeddingDimension=self.fastrp_dimensions,
            iterationWeights=[0.0, 1.0, 1.0, 0.8, 0.6],
            normalizationStrength=0.5,
            randomSeed=42,
            mutateProperty="fastrp_embedding",
        )
        return {
            "nodePropertiesWritten": result["nodePropertiesWritten"],
            "computeMillis": result["computeMillis"],
        }

    def write_fastrp_embeddings(
        self,
        graph_name: str = "decision-graph",
        node_labels: Optional[list[str]] = None,
    ) -> dict:
        """Write FastRP embeddings back to the database."""
        node_labels = node_labels or ["Decision", "Person", "Account", "Transaction"]
        self._ensure_decision_graph_exists()

        result = self.gds.graph.nodeProperties.write(
            self._decision_graph,
            ["fastrp_embedding"],
            node_labels,
        )
        return {"propertiesWritten": result["propertiesWritten"]}

    # ============================================
    # K-NEAREST NEIGHBORS (KNN)
    # ============================================

    def find_similar_decisions_knn(
        self,
        decision_id: str,
        limit: int = 10,
        graph_name: str = "decision-graph",
    ) -> list[dict]:
        """Find similar decisions using KNN on FastRP embeddings."""
        self._ensure_decision_graph_exists()

        result_df = self.gds.knn.stream(
            self._decision_graph,
            nodeLabels=["Decision"],
            nodeProperties=["fastrp_embedding"],
            topK=limit,
            sampleRate=1.0,
            randomSeed=42,
        )

        # Filter results for the specific decision and get node details
        similar_decisions = []
        for _, row in result_df.iterrows():
            node1 = self.gds.util.asNode(row["node1"])
            node2 = self.gds.util.asNode(row["node2"])
            if node1.get("id") == decision_id:
                similar_decisions.append(
                    {
                        "id": node2.get("id"),
                        "decision_type": node2.get("decision_type"),
                        "category": node2.get("category"),
                        "reasoning_summary": node2.get("reasoning_summary"),
                        "decision_timestamp": node2.get("decision_timestamp"),
                        "similarity": row["similarity"],
                    }
                )

        return sorted(similar_decisions, key=lambda x: x["similarity"], reverse=True)

    def run_knn_all(
        self,
        graph_name: str = "decision-graph",
        node_label: str = "Decision",
        top_k: int = 5,
    ) -> dict:
        """Run KNN on all nodes and create SIMILAR_TO relationships."""
        self._ensure_decision_graph_exists()

        result = self.gds.knn.mutate(
            self._decision_graph,
            nodeLabels=[node_label],
            nodeProperties=["fastrp_embedding"],
            topK=top_k,
            mutateRelationshipType="SIMILAR_TO",
            mutateProperty="score",
        )
        return {
            "relationshipsWritten": result["relationshipsWritten"],
            "computeMillis": result["computeMillis"],
        }

    # ============================================
    # NODE SIMILARITY
    # ============================================

    def find_similar_accounts(
        self,
        account_id: str,
        limit: int = 10,
        similarity_cutoff: float = 0.5,
        graph_name: str = "entity-graph",
    ) -> list[dict]:
        """Find accounts with similar neighborhood structures."""
        self._ensure_entity_graph_exists()

        result_df = self.gds.nodeSimilarity.stream(
            self._entity_graph,
            nodeLabels=["Account"],
            topK=limit,
            similarityCutoff=similarity_cutoff,
        )

        # Filter results for the specific account and get node details
        similar_accounts = []
        for _, row in result_df.iterrows():
            node1 = self.gds.util.asNode(row["node1"])
            node2 = self.gds.util.asNode(row["node2"])
            if node1.get("id") == account_id:
                similar_accounts.append(
                    {
                        "id": node2.get("id"),
                        "account_number": node2.get("account_number"),
                        "account_type": node2.get("account_type"),
                        "risk_tier": node2.get("risk_tier"),
                        "similarity": row["similarity"],
                    }
                )

        return sorted(similar_accounts, key=lambda x: x["similarity"], reverse=True)

    def find_potential_duplicates(
        self,
        similarity_cutoff: float = 0.7,
        graph_name: str = "entity-graph",
    ) -> list[dict]:
        """Find potential duplicate persons using Node Similarity."""
        self._ensure_entity_graph_exists()

        result_df = self.gds.nodeSimilarity.stream(
            self._entity_graph,
            nodeLabels=["Person"],
            topK=10,
            similarityCutoff=similarity_cutoff,
        )

        # Process results
        duplicates = []
        seen_pairs = set()
        for _, row in result_df.iterrows():
            node1 = self.gds.util.asNode(row["node1"])
            node2 = self.gds.util.asNode(row["node2"])
            id1, id2 = node1.get("id"), node2.get("id")

            # Ensure we don't add duplicate pairs
            pair_key = tuple(sorted([id1, id2]))
            if pair_key not in seen_pairs:
                seen_pairs.add(pair_key)
                duplicates.append(
                    {
                        "person1_id": id1,
                        "person1_name": node1.get("name"),
                        "person1_sources": node1.get("source_systems"),
                        "person2_id": id2,
                        "person2_name": node2.get("name"),
                        "person2_sources": node2.get("source_systems"),
                        "similarity": row["similarity"],
                    }
                )

        return sorted(duplicates, key=lambda x: x["similarity"], reverse=True)[:20]

    # ============================================
    # FRAUD PATTERN DETECTION
    # ============================================

    def detect_fraud_patterns(
        self,
        account_id: Optional[str] = None,
        similarity_threshold: float = 0.7,
        graph_name: str = "entity-graph",
    ) -> list[dict]:
        """Detect accounts with similar structures to known fraud cases."""
        self._ensure_entity_graph_exists()

        # Run node similarity on accounts
        result_df = self.gds.nodeSimilarity.stream(
            self._entity_graph,
            nodeLabels=["Account"],
            topK=10,
            similarityCutoff=similarity_threshold,
        )

        # If account_id specified, filter for that account
        # Otherwise, we need to identify fraud accounts via a separate query
        patterns = []
        for _, row in result_df.iterrows():
            node1 = self.gds.util.asNode(row["node1"])
            node2 = self.gds.util.asNode(row["node2"])

            if account_id:
                # Check if this row involves the target account
                if node1.get("id") == account_id or node2.get("id") == account_id:
                    target = node1 if node1.get("id") == account_id else node2
                    similar = node2 if node1.get("id") == account_id else node1
                    patterns.append(
                        {
                            "target_id": target.get("id"),
                            "target_account": target.get("account_number"),
                            "similar_account_id": similar.get("id"),
                            "similar_account": similar.get("account_number"),
                            "structural_similarity": row["similarity"],
                        }
                    )
            else:
                # Return all similar account pairs
                patterns.append(
                    {
                        "account1_id": node1.get("id"),
                        "account1_number": node1.get("account_number"),
                        "account1_risk_tier": node1.get("risk_tier"),
                        "account2_id": node2.get("id"),
                        "account2_number": node2.get("account_number"),
                        "account2_risk_tier": node2.get("risk_tier"),
                        "structural_similarity": row["similarity"],
                    }
                )

        return sorted(patterns, key=lambda x: x.get("structural_similarity", 0), reverse=True)[:20]

    # ============================================
    # LOUVAIN COMMUNITY DETECTION
    # ============================================

    def detect_decision_communities(
        self,
        graph_name: str = "decision-graph",
    ) -> list[dict]:
        """Detect communities of related decisions using Louvain."""
        self._ensure_decision_graph_exists()

        result_df = self.gds.louvain.stream(
            self._decision_graph,
            nodeLabels=["Decision"],
            relationshipTypes=["CAUSED", "INFLUENCED", "PRECEDENT_FOR"],
        )

        # Aggregate results by community
        communities = {}
        for _, row in result_df.iterrows():
            node = self.gds.util.asNode(row["nodeId"])
            community_id = row["communityId"]

            if community_id not in communities:
                communities[community_id] = {
                    "communityId": community_id,
                    "decision_count": 0,
                    "decision_types": set(),
                    "categories": set(),
                    "sample_decision_ids": [],
                }

            communities[community_id]["decision_count"] += 1
            if node.get("decision_type"):
                communities[community_id]["decision_types"].add(node.get("decision_type"))
            if node.get("category"):
                communities[community_id]["categories"].add(node.get("category"))
            if len(communities[community_id]["sample_decision_ids"]) < 5:
                communities[community_id]["sample_decision_ids"].append(node.get("id"))

        # Convert sets to lists for JSON serialization
        result = []
        for comm in communities.values():
            comm["decision_types"] = list(comm["decision_types"])
            comm["categories"] = list(comm["categories"])
            result.append(comm)

        return sorted(result, key=lambda x: x["decision_count"], reverse=True)[:20]

    def write_community_ids(
        self,
        graph_name: str = "decision-graph",
    ) -> dict:
        """Write community IDs to decision nodes."""
        self._ensure_decision_graph_exists()

        result = self.gds.louvain.mutate(
            self._decision_graph,
            nodeLabels=["Decision"],
            relationshipTypes=["CAUSED", "INFLUENCED", "PRECEDENT_FOR"],
            mutateProperty="community_id",
        )
        return {
            "communityCount": result["communityCount"],
            "modularity": result["modularity"],
            "computeMillis": result["computeMillis"],
        }

    # ============================================
    # PAGERANK - INFLUENCE SCORING
    # ============================================

    def calculate_influence_scores(
        self,
        graph_name: str = "decision-graph",
    ) -> list[dict]:
        """Calculate PageRank influence scores for decisions."""
        self._ensure_decision_graph_exists()

        result_df = self.gds.pageRank.stream(
            self._decision_graph,
            nodeLabels=["Decision"],
            relationshipTypes=["CAUSED", "INFLUENCED"],
            maxIterations=20,
            dampingFactor=0.85,
        )

        # Filter for exception/override/escalation decisions and get node details
        influence_scores = []
        for _, row in result_df.iterrows():
            node = self.gds.util.asNode(row["nodeId"])
            decision_type = node.get("decision_type")
            if decision_type in ["exception", "override", "escalation"]:
                influence_scores.append(
                    {
                        "id": node.get("id"),
                        "decision_type": decision_type,
                        "category": node.get("category"),
                        "reasoning_summary": node.get("reasoning_summary"),
                        "influence_score": row["score"],
                    }
                )

        return sorted(influence_scores, key=lambda x: x["influence_score"], reverse=True)[:20]

    def write_influence_scores(
        self,
        graph_name: str = "decision-graph",
    ) -> dict:
        """Write PageRank scores to decision nodes."""
        self._ensure_decision_graph_exists()

        result = self.gds.pageRank.mutate(
            self._decision_graph,
            nodeLabels=["Decision"],
            relationshipTypes=["CAUSED", "INFLUENCED"],
            mutateProperty="influence_score",
        )
        return {
            "nodePropertiesWritten": result["nodePropertiesWritten"],
            "computeMillis": result["computeMillis"],
        }


# Singleton instance
gds_client = GDSClient()
