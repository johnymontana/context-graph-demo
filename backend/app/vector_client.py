"""
Neo4j vector search client.
Handles semantic similarity using text embeddings and hybrid search.
"""

from typing import Optional

from neo4j import GraphDatabase
from openai import OpenAI

from .config import config


class VectorClient:
    """Neo4j vector search client for semantic similarity."""

    def __init__(self):
        self.driver = GraphDatabase.driver(
            config.neo4j.uri,
            auth=(config.neo4j.username, config.neo4j.password),
        )
        self.database = config.neo4j.database
        self.openai_client = (
            OpenAI(api_key=config.openai.api_key) if config.openai.api_key else None
        )
        self.embedding_model = config.openai.embedding_model
        self.embedding_dimensions = config.openai.embedding_dimensions

    def close(self):
        self.driver.close()

    # ============================================
    # EMBEDDING GENERATION
    # ============================================

    def generate_embedding(self, text: str) -> list[float]:
        """Generate an embedding for the given text using OpenAI."""
        if not self.openai_client:
            raise ValueError("OpenAI API key not configured")

        response = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=text,
        )
        return response.data[0].embedding

    def generate_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        if not self.openai_client:
            raise ValueError("OpenAI API key not configured")

        response = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=texts,
        )
        return [item.embedding for item in response.data]

    # ============================================
    # SEMANTIC SEARCH
    # ============================================

    def search_decisions_semantic(
        self,
        query: str,
        limit: int = 10,
        category: Optional[str] = None,
    ) -> list[dict]:
        """Search decisions by semantic similarity to query."""
        query_embedding = self.generate_embedding(query)

        category_filter = "WHERE d.category = $category" if category else ""

        with self.driver.session(database=self.database) as session:
            result = session.run(
                f"""
                MATCH (d:Decision)
                {category_filter}
                CALL db.index.vector.queryNodes(
                    'decision_reasoning_idx',
                    $limit,
                    $query_embedding
                ) YIELD node, score
                WHERE node = d
                RETURN d.id AS id,
                       d.decision_type AS decision_type,
                       d.category AS category,
                       d.reasoning_summary AS reasoning_summary,
                       d.decision_timestamp AS decision_timestamp,
                       d.confidence_score AS confidence_score,
                       score AS semantic_similarity
                ORDER BY score DESC
                """,
                {
                    "query_embedding": query_embedding,
                    "limit": limit,
                    "category": category,
                },
            )
            return [dict(record) for record in result]

    def search_policies_semantic(
        self,
        query: str,
        limit: int = 5,
    ) -> list[dict]:
        """Search policies by semantic similarity."""
        query_embedding = self.generate_embedding(query)

        with self.driver.session(database=self.database) as session:
            result = session.run(
                """
                CALL db.index.vector.queryNodes(
                    'policy_description_idx',
                    $limit,
                    $query_embedding
                ) YIELD node, score
                RETURN node.id AS id,
                       node.name AS name,
                       node.description AS description,
                       node.category AS category,
                       score AS semantic_similarity
                ORDER BY score DESC
                """,
                {"query_embedding": query_embedding, "limit": limit},
            )
            return [dict(record) for record in result]

    # ============================================
    # HYBRID SEARCH (Semantic + Structural)
    # ============================================

    def find_precedents_hybrid(
        self,
        scenario: str,
        category: Optional[str] = None,
        semantic_weight: float = 0.6,
        structural_weight: float = 0.4,
        limit: int = 5,
    ) -> list[dict]:
        """
        Find precedent decisions using semantic similarity.

        Uses text embeddings (reasoning_embedding) to find decisions with
        similar reasoning to the given scenario.
        """
        query_embedding = self.generate_embedding(scenario)

        category_filter = "AND d.category = $category" if category else ""

        with self.driver.session(database=self.database) as session:
            result = session.run(
                f"""
                CALL db.index.vector.queryNodes(
                    'decision_reasoning_idx',
                    $limit,
                    $query_embedding
                ) YIELD node AS d, score AS semantic_score
                WHERE d:Decision {category_filter}
                RETURN d.id AS id,
                       d.decision_type AS decision_type,
                       d.category AS category,
                       d.reasoning_summary AS reasoning_summary,
                       d.decision_timestamp AS decision_timestamp,
                       semantic_score AS combined_score,
                       semantic_score AS semantic_similarity,
                       null AS structural_similarity
                ORDER BY semantic_score DESC
                LIMIT $limit
                """,
                {
                    "query_embedding": query_embedding,
                    "category": category,
                    "limit": limit,
                },
            )
            return [dict(record) for record in result]

    def find_similar_decisions_hybrid(
        self,
        decision_id: str,
        semantic_weight: float = 0.5,
        structural_weight: float = 0.5,
        limit: int = 5,
    ) -> list[dict]:
        """
        Find decisions similar to a given decision using hybrid similarity.
        """
        with self.driver.session(database=self.database) as session:
            result = session.run(
                """
                // Get the source decision
                MATCH (source:Decision {id: $decision_id})
                // Ensure valid embeddings exist
                WHERE any(val in source.fastrp_embedding WHERE val <> 0.0)
                AND any(val in source.reasoning_embedding WHERE val <> 0.0)
                // Use union to find both most semantically and structurally similar
                CALL (source) {

                    // Find semantically similar
                    CALL db.index.vector.queryNodes(
                        'decision_reasoning_idx',
                        $limit * 2 + 1,
                        source.reasoning_embedding
                    ) YIELD node, score AS semantic_score
                    WHERE node <> source
                    AND any(val in node.fastrp_embedding WHERE val <> 0.0)
                    RETURN node, semantic_score, vector.similarity.cosine(source.fastrp_embedding, node.fastrp_embedding) AS structural_score

                    UNION

                    // Find structurally similar
                    CALL db.index.vector.queryNodes(
                        'decision_fastrp_idx',
                        $limit * 2 + 1,
                        source.fastrp_embedding
                    ) YIELD node, score AS structural_score
                    WHERE node <> source
                    AND any(val in node.reasoning_embedding WHERE val <> 0.0)
                    RETURN node, vector.similarity.cosine(source.reasoning_embedding, node.reasoning_embedding) AS semantic_score, structural_score
                }

                WITH node AS decision, semantic_score, structural_score,
                     (semantic_score * $semantic_weight + structural_score * $structural_weight) AS combined_score
                ORDER BY combined_score DESC
                LIMIT $limit
                
                RETURN decision.id AS id,
                       decision.decision_type AS decision_type,
                       decision.category AS category,
                       decision.reasoning_summary AS reasoning_summary,
                       decision.decision_timestamp AS decision_timestamp,
                       combined_score,
                       semantic_score AS semantic_similarity,
                       structural_score AS structural_similarity
                """,
                {
                    "decision_id": decision_id,
                    "semantic_weight": semantic_weight,
                    "structural_weight": structural_weight,
                    "limit": limit,
                },
            )
            return [dict(record) for record in result]

    # ============================================
    # EMBEDDING STORAGE
    # ============================================

    def update_decision_reasoning_embedding(
        self,
        decision_id: str,
        reasoning: str,
    ) -> bool:
        """Generate and store reasoning embedding for a decision."""
        embedding = self.generate_embedding(reasoning)

        with self.driver.session(database=self.database) as session:
            result = session.run(
                """
                MATCH (d:Decision {id: $decision_id})
                SET d.reasoning_embedding = $embedding
                RETURN d.id AS id
                """,
                {"decision_id": decision_id, "embedding": embedding},
            )
            return result.single() is not None

    def update_policy_description_embedding(
        self,
        policy_id: str,
        description: str,
    ) -> bool:
        """Generate and store description embedding for a policy."""
        embedding = self.generate_embedding(description)

        with self.driver.session(database=self.database) as session:
            result = session.run(
                """
                MATCH (p:Policy {id: $policy_id})
                SET p.description_embedding = $embedding
                RETURN p.id AS id
                """,
                {"policy_id": policy_id, "embedding": embedding},
            )
            return result.single() is not None


# Singleton instance
vector_client = VectorClient()
