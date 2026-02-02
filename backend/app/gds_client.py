"""
Neo4j Graph Data Science (GDS) client.
Implements FastRP, KNN, Node Similarity, Louvain, and PageRank.
"""

from typing import Any
from venv import logger

from neo4j import GraphDatabase
from graphdatascience import GraphDataScience
from graphdatascience.graph.graph_object import Graph
import logging

from .config import config


class GDSClient:
    """Neo4j GDS client for graph algorithms."""

    def __init__(self):
        self.driver = GraphDatabase.driver(
            config.neo4j.uri,
            auth=(config.neo4j.username, config.neo4j.password),
        )
        self.database = config.neo4j.database
        self.fastrp_dimensions = config.fastrp_dimensions
        self.gds = GraphDataScience(
            config.neo4j.uri,
            auth=(config.neo4j.username, config.neo4j.password),
            database=config.neo4j.database,
        )
        self.logger = logging.getLogger(__name__)

    def close(self):
        self.driver.close()
        self.gds.close()

    # ============================================
    # GRAPH PROJECTION MANAGEMENT
    # ============================================

    def create_transaction_graph_projection(self) -> Graph:
        """Create the transaction graph projection for influence scoring."""

        self.gds.graph.drop('transaction-graph', False)
        g_transactions, _ = self.gds.graph.cypher.project(
            """//cypher
            cypher runtime = parallel
            MATCH (a:Account)
            OPTIONAL MATCH (a)<-[t:FROM_ACCOUNT|TO_ACCOUNT]-(tr:Transaction)
            RETURN gds.graph.project('transaction-graph',
                a, tr,
                {
                    sourceNodeLabels: ['Transaction'],
                    targetNodeLabels: ['Account'],
                    relationshipType: 'HAS_TRANSACTION',
                    relationshipProperties: {amount: tr.amount}
                }
                )
            """)

        return g_transactions
    
    def create_account_graph_projection(self) -> Graph:
        """Create the account graph projection showing accounts that share transactions."""

        self.gds.graph.drop('account-graph', False)
        g_account, _ = self.gds.graph.cypher.project(
            """//cypher
            cypher runtime = parallel
            MATCH (a1:Account)
            OPTIONAL MATCH (a1:Account)<-[:FROM_ACCOUNT]-(t)-[:TO_ACCOUNT]->(a2:Account)
            WITH CASE WHEN a1.id < a2.id OR a2.id IS NULL THEN [a1, a2] ELSE [a2, a1] END AS pair, sum(t.amount) AS amount
            RETURN gds.graph.project('account-graph',
                pair[0], pair[1],
                {
                    sourceNodeLabels: ['Account'],
                    targetNodeLabels: ['Account'],
                    relationshipType: 'SHARE_TRANSACTIONS',
                    relationshipProperties: {amount: amount}
                },
                {
                    undirectedRelationshipTypes: ['SHARE_TRANSACTIONS']
                }
                )
            """)
        return g_account
    

    def create_decision_graph_projection(self) -> Graph:
        """Create the decision graph projection for finding similarity between decisions algorithms.
        """

        self.gds.graph.drop('decision-graph', False)
        g_decisions, _ = self.gds.graph.cypher.project(
            """//cypher
            cypher runtime = parallel
            CALL () {
            // Decisions that are direct neighbors
            MATCH (d1)
            OPTIONAL MATCH (d1)-[:INFLUENCED|CAUSED|PRECEDENT_FOR]-(d2)
            WHERE d1 < d2
            RETURN d1, d2

            UNION 
            
            // Decisions about accounts owned by the same entity
            MATCH (d1:Decision)-[:ABOUT]->()(()-[:OWNS|OWNS_ACCOUNT]->()<-[:OWNS|OWNS_ACCOUNT]-()){0,1}()<-[:ABOUT]-(d2)
            WHERE d1 < d2
            RETURN d1, d2
            
            UNION 
            
            // Decisions about accounts sharing high transaction amounts
            MATCH (d1:Decision)-[:ABOUT]->()-[:SHARES_HIGH_PERCENTAGE_OF_TRANSACTIONS_WITH]-()<-[:ABOUT]-(d2)
            WHERE d1 < d2
            RETURN d1, d2
            }
            RETURN gds.graph.project(
            'decision-graph',
            d1, d2,
            {
                sourceNodeLabels: ['Decision'],
                targetNodeLabels: ['Decision'],
                relationshipType: "SHARES_FACTORS"
            },
            {
                undirectedRelationshipTypes: ["SHARES_FACTORS"]
            }
            )
            """)
        return g_decisions

        
    # ============================================
    # RELATED ACCOUNTS VIA NODE SIMILARITY
    # ============================================

    def find_related_accounts(
        self,
    ) ->  Any:
        """Find accounts that share common transactions."""
        
        with self.driver.session(database=self.database) as session:
            session.run("""//cypher
                        MATCH ()-[r:SHARES_HIGH_PERCENTAGE_OF_TRANSACTIONS_WITH]->()
                        CALL (r)
                        {
                            DELETE r
                        } IN TRANSACTIONS
                        """)

        g_transactions = self.create_transaction_graph_projection()

        node_similarity_result = self.gds.v2.node_similarity.write(
            g_transactions,
            relationship_types = ['HAS_TRANSACTION'],
            relationship_weight_property = 'amount',
            top_k = 5,
            similarity_cutoff = 0.2,
            write_relationship_type = 'SHARES_HIGH_PERCENTAGE_OF_TRANSACTIONS_WITH',
            write_property = 'weighted_jaccard_similarity',
            use_components = True
        )

        g_transactions.drop()
        return node_similarity_result

    # ============================================
    # ACCOUNT COMMUNITIES
    # ============================================
    def find_account_communities(
        self,
    ) -> Any:
        """Detect communities of related accounts using Leiden."""
        #Clean up previous community assignments
        with self.driver.session(database=self.database) as session:
            session.run("""//cypher
                        MATCH (d:Account)
                        CALL (d)
                        {
                        REMOVE d.community_id
                        } IN CONCURRENT TRANSACTIONS
                        """)
            
            session.run("""//cypher
                        MATCH ()-[b:BELONGS_TO_ACCOUNT_COMMUNITY]->()
                        CALL (b)
                        {
                            DELETE b
                        } IN TRANSACTIONS
                        """)
            
            session.run("""//cypher
                        MATCH (c:AccountCommunity)
                        CALL (c)
                        {
                            DELETE c
                        } IN CONCURRENT TRANSACTIONS
                        """)
        
        g_accounts = self.create_account_graph_projection()

        leiden_result = self.gds.v2.leiden.write(
            g_accounts,
            random_seed = 42,
            write_property = 'community_id'
        )

        with self.driver.session(database=self.database) as session:
            # Create Community nodes and connect them to Account nodes
            session.run(
                """//cypher
                CYPHER 25
                MATCH (a:Account)
                WHERE a.community_id IS NOT NULL
                WITH a.community_id AS communityId, collect(a) AS accounts
                CALL (communityId, accounts)
                {
                    MERGE (c:AccountCommunity {id: communityId})
                    SET c.account_count = size(accounts),
                    c.account_types = coll.distinct([d IN accounts | d.account_type])
                    WITH c, accounts
                    UNWIND accounts AS a
                    MERGE (a)-[:BELONGS_TO_ACCOUNT_COMMUNITY]->(c)
                    WITH c, accounts, avg(a.balance) AS avg_balance,
                    percentileCont(a.balance, 0.5) AS median_balance
                    SET c.avg_account_balance = avg_balance,
                    c.median_account_balance = median_balance
                    UNWIND accounts AS a
                    OPTIONAL MATCH (a)<-[:FROM_ACCOUNT|TO_ACCOUNT]-(t:Transaction)
                    WITH c, count(*) AS transaction_count, SUM(CASE WHEN t.status = 'flagged' THEN 1 ELSE 0 END) AS flagged_count
                    SET c.total_transactions = transaction_count,
                    c.flagged_transactions = flagged_count,
                    c.percent_flagged_transactions = CASE WHEN transaction_count > 0 THEN 1.0 * flagged_count / transaction_count ELSE 0 END
                } IN CONCURRENT TRANSACTIONS
                """
            )
        
        g_accounts.drop()

        return leiden_result

    # ============================================
    # FASTRP EMBEDDINGS
    # ============================================

    def generate_fastrp_embeddings(
        self,
    ) -> Any:
        """Generate FastRP embeddings for nodes and use them to create similarity relationships.
        """

        # Drop previously created FastRP embeddings and KNN relationships if they exist
        with self.driver.session(database=self.database) as session:
            session.run("""//cypher
                        MATCH (d:Decision)
                        CALL (d)
                        {
                        REMOVE d.fast_rp_embedding
                        } IN CONCURRENT TRANSACTIONS
                        """)
            session.run("""//cypher
                        MATCH ()-[r:HAS_SIMILAR_FACTORS]->()
                        CALL (r)
                        {
                            DELETE r
                        } IN TRANSACTIONS
                        """)

        g_decisions = self.create_decision_graph_projection()


        # Calculate FastRP embeddings
        self.gds.v2.fast_rp.mutate(
            g_decisions,
            embedding_dimension = self.fastrp_dimensions,
            iteration_weights = [0.0, 0.0, 1.0, 1.0],
            mutate_property = 'fast_rp_embedding'
        )

        fast_rp_write_result = self.gds.v2.graph.node_properties.write(
            g_decisions,
            ['fast_rp_embedding']
        )

        # Create KNN relationships based on FastRP embeddings
        knn_write_result = self.gds.v2.knn.write(
            g_decisions,
            node_properties = ['fast_rp_embedding'],
            top_k = 10,
            similarity_cutoff = 0.6,
            initial_sampler="randomWalk",
            write_relationship_type = 'HAS_SIMILAR_FACTORS',
            write_property = 'fast_rp_cosine_similarity'
            )
        
        # Delete KNN relationships not in same WCC component
        self.gds.v2.wcc.write(
            g_decisions,
            write_property = 'decision_wcc_id')
        
        with self.driver.session(database=self.database) as session:
            delete_result = session.run(
                """//cypher
                MATCH (d1:Decision)-[r:HAS_SIMILAR_FACTORS]-(d2:Decision)
                WHERE d1.decision_wcc_id <> d2.decision_wcc_id
                CALL (r) {
                    DELETE r
                } IN CONCURRENT TRANSACTIONS
                """)
            delete_summary = delete_result.consume().counters
        
            session.run(
                """//cypher
                MATCH (d:Decision)
                CALL (d) {
                    REMOVE d.decision_wcc_id
                } IN CONCURRENT TRANSACTIONS
                """)

        g_decisions.drop()

        return fast_rp_write_result, knn_write_result, delete_summary

    # ============================================
    # DECISION COMMUNITY DETECTION
    # ============================================

    def find_decision_communities(
        self,
    ) -> Any:
        """Detect communities of related decisions using Louvain."""
        
        #Clean up previous community assignments
        with self.driver.session(database=self.database) as session:
            session.run("""//cypher
                        MATCH (d:Decision)
                        CALL (d)
                        {
                        REMOVE d.community_id
                        } IN CONCURRENT TRANSACTIONS
                        """)
            
            session.run("""//cypher
                        MATCH ()-[b:BELONGS_TO_DECISION_COMMUNITY]->()
                        CALL (b)
                        {
                            DELETE b
                        } IN TRANSACTIONS
                        """)
            
            session.run("""//cypher
                        MATCH (c:DecisionCommunity)
                        CALL (c)
                        {
                            DELETE c
                        } IN CONCURRENT TRANSACTIONS
                        """)
        
        g_decisions = self.create_decision_graph_projection()

        leiden_result = self.gds.v2.leiden.write(
            g_decisions,
            random_seed = 42,
            write_property = 'community_id'
        )

        with self.driver.session(database=self.database) as session:
            # Create Community nodes and connect them to Decision nodes
            session.run(
                """//cypher
                CYPHER 25
                MATCH (d:Decision)
                WHERE d.community_id IS NOT NULL
                WITH d.community_id AS communityId, collect(d) AS decisions
                CALL (communityId, decisions)
                {
                    MERGE (c:DecisionCommunity {id: communityId})
                    SET c.decision_count = size(decisions),
                    c.categories = coll.distinct([d IN decisions | d.category]),
                    c.decision_types = coll.distinct([d IN decisions | d.decision_type]),
                    c.rejection_rate = reduce(rejectCount = 0.0, d in decisions | rejectCount + CASE WHEN d.decision_type = 'rejection' THEN 1.0 ELSE 0.0 END)/size(decisions)
                    WITH c, decisions
                    UNWIND decisions AS d
                    MERGE (d)-[:BELONGS_TO_DECISION_COMMUNITY]->(c)
                } IN CONCURRENT TRANSACTIONS
                """
            )
        g_decisions.drop()

        return leiden_result
    
    # ============================================
    # PAGE RANK NODE INFLUENCE SCORING
    # ============================================
    def calculate_flagged_transaction_influence(
        self
    ) -> Any:
        """Calculate influence scores for accounts showing how much they are influenced by flagged transactions."""
        # Create graph projection
        g_transactions = self.create_transaction_graph_projection()

        self.gds.v2.graph.relationships.to_undirected(
            g_transactions,
            relationship_type = 'HAS_TRANSACTION',
            mutate_relationship_type = 'UNDIRECTED_HAS_TRANSACTION'
        )

        flagged_node_records, _, _ = self.driver.execute_query(
            """//cypher
            MATCH (tr:Transaction)
            WHERE tr.status = 'flagged'
            RETURN collect(id(tr)) AS flagged_node_ids
            """)
        
        flagged_node_ids = flagged_node_records[0]['flagged_node_ids']

        self.gds.v2.page_rank.mutate(
            g_transactions,
            relationship_types = ['UNDIRECTED_HAS_TRANSACTION'],
            source_nodes = flagged_node_ids,
            mutate_property = 'flagged_transaction_influence',
            scaler = 'MinMax'
        )

        write_result = self.gds.v2.graph.node_properties.write(
            g_transactions,
            ['flagged_transaction_influence'],
            node_labels = ['Account']
        )

        g_transactions.drop()

        return write_result
        
    
    # ============================================
    # FULL GDS WORKFLOW
    # ============================================      

    def refresh_gds_analyses(
        self,
    ) -> None:
        """Run full GDS workflow: 
            1. Find related accounts via Node Similarity
            2. Create Leiden communities for accounts
            3. Generate FastRP embeddings for decisions and create KNN relationships
            4. Run Leiden community detection to compute community IDs for decisions
            5. Run PageRank influence scoring for flagged transactions
        """
        self.logger.info("Starting GDS analyses refresh...")

        # 1. Find related accounts via Node Similarity
        self.logger.info("Finding accounts that share transactions...")
        try:
            node_similarity_result = self.find_related_accounts()
            self.logger.info(
                f"Account similarity computation complete: {node_similarity_result.relationships_written} relationships created"
            )
        except Exception as e:
            self.logger.warning(f"Could not compute account similarities: {e}")

        # 2. Create Leiden communities for accounts
        self.logger.info("Finding account communities...")
        try:
            account_communities_result = self.find_account_communities()
            self.logger.info(
                f"Account community detection complete: {account_communities_result.community_count} communities found"
            )
        except Exception as e:
            self.logger.warning(f"Could not compute account communities: {e}")

        # 3. Generate FastRP embeddings for decisions and create KNN relationships
        self.logger.info("Running fastRP embedding generation...")
        try:
            fast_rp_result, knn_result, delete_summary = self.generate_fastrp_embeddings()
            self.logger.info(
                f"FastRP embedding generation complete: {fast_rp_result.properties_written} nodes embedded"
                f"; {knn_result.relationships_written} similarity relationships created; {delete_summary.relationships_deleted} relationships deleted."
            )
        except Exception as e:
            self.logger.warning(f"Could not run FastRP: {e}")

        # 4. Run Leiden community detection to compute community IDs for decisions
        self.logger.info("Running decision community detection...")
        try:
            decision_communities_result = self.find_decision_communities()
            self.logger.info(
                f"Decision community detection complete: {decision_communities_result.community_count} communities found"
            )
        except Exception as e:
            self.logger.warning(f"Could not compute decision communities: {e}")

        # 5. Run PageRank influence scoring for flagged transactions
        self.logger.info("Running flagged transaction influence scoring...")
        try:
            pagerank_result = self.calculate_flagged_transaction_influence()
            self.logger.info(
                f"Flagged transaction influence scoring complete: {pagerank_result.properties_written} nodes scored"
            )
        except Exception as e:
            self.logger.warning(f"Could not run PageRank: {e}")

        self.logger.info("GDS analyses refresh complete!")

    # ============================================
    # RETURN GDS-BASED RESULTS
    # ============================================  

    def find_similar_decisions(self, decision_id: str, limit: int =10) -> list[dict[str, Any]]:
        """Get decisions similar to the given decision based on FastRP embeddings."""
        records, _, _ = self.driver.execute_query(
            """//cypher
            MATCH (d:Decision {id: $decision_id})-[s:HAS_SIMILAR_FACTORS]->(decision2)
            RETURN decision2.id AS id,
            decision2.decision_type AS decision_type,
            decision2.category AS category,
            decision2.reasoning_summary AS reasoning_summary,
            decision2.decision_timestamp AS decision_timestamp,
            s.fast_rp_cosine_similarity AS fast_rp_cosine_similarity
            ORDER BY fast_rp_cosine_similarity DESC
            LIMIT $limit
            """,
            {"decision_id": decision_id, "limit": limit},
        )
    
        similar_decisions = [dict(record) for record in records]

        return similar_decisions
    
    def get_decision_community(self, decision_id: str, example_count: int) -> dict[str, Any]:
        """Get information about the decision community for a given decision."""
        records, _, _ = self.driver.execute_query(
            """//cypher
            MATCH (d:Decision {id:$decision_id})
            OPTIONAL MATCH (d)-[:BELONGS_TO_DECISION_COMMUNITY]->(c)
            RETURN 
            c.decision_types AS community_decision_types,
            c.categories AS community_categories,
            c.decision_count AS community_decision_count,
            c.rejection_rate AS community_rejection_rate,
            COLLECT {
                MATCH (c)<-[:BELONGS_TO_DECISION_COMMUNITY]-(n)
                WHERE n <> d
                RETURN
                n{.decision_type, .category, .status, .reasoning_summary, .decision_timestamp}
                ORDER BY vector.similarity.cosine(n.fast_rp_embedding, d.fast_rp_embedding) DESC
                LIMIT $example_count
            } AS sample_community_decisions
            """,
            {"decision_id": decision_id, "example_count": example_count},
        )
    
        community_info = dict(records[0]) if records else {}

        return community_info
    

    def detect_fraud_patterns(self, account_id: str, neighbor_count: int =5) -> dict[str, Any]:
        """Analyze accounts or transactions for potential fraud patterns using graph structure analysis.
        Checks an account's proximity to flagged transactions as well as the prevalance of flagged transactions in the community of related accounts.
        """
        records, _, _ = self.driver.execute_query(
            """//cypher
            MATCH (a:Account {id:$account_id})
            OPTIONAL MATCH (a)-[:BELONGS_TO_ACCOUNT_COMMUNITY]->(c)
            CALL (a) {
            MATCH (a)<-[:FROM_ACCOUNT|TO_ACCOUNT]-(t)
            RETURN count(*) AS account_related_transaction_count,
            sum(CASE WHEN t.status='flagged' THEN 1 ELSE 0 END) AS account_flagged_transaction_count
            }
            RETURN 
            a.flagged_transaction_influence AS account_flagged_transaction_influence_score,
            account_related_transaction_count,
            account_flagged_transaction_count,
            c.percent_flagged_transactions AS community_percent_flagged_transactions,
            COLLECT {
            MATCH (c)<-[:BELONGS_TO_ACCOUNT_COMMUNITY]-(n)
            WHERE n <> a
            WITH n, COUNT {(n)<-[:FROM_ACCOUNT|TO_ACCOUNT]-({status:"flagged"})} AS flagged_transaction_count
            WHERE flagged_transaction_count > 0
            RETURN {account_id: n.id, flagged_transaction_count: flagged_transaction_count}
            ORDER BY flagged_transaction_count DESC 
            LIMIT $neighbor_count
            } AS community_accounts_with_most_flagged_transactions
            """,
            {"account_id": account_id, "neighbor_count": neighbor_count},
        )
    
        fraud_analysis = dict(records[0]) if records else {}

        return fraud_analysis
    
    def find_accounts_with_high_shared_transaction_volume(self, account_id: str) -> list[dict[str, Any]]:
        """Get accounts with high shared transaction volume."""
        records, _, _ = self.driver.execute_query(
            """//cypher
            MATCH (a:Account {id: $account_id})-[s:SHARES_HIGH_PERCENTAGE_OF_TRANSACTIONS_WITH]->(other_account)
            RETURN other_account.id AS id,
            other_account.account_number AS account_number,
            other_account.account_type AS account_type,
            other_account.status AS status,
            COLLECT { MATCH (other_account)<-[:OWNS|OWNS_ACCOUNT]-(owner) RETURN owner.name } AS owners,
            s.weighted_jaccard_similarity AS percentage_of_shared_transactions
            ORDER BY percentage_of_shared_transactions DESC
            """,
            {"account_id": account_id},
        )
    
        related_accounts = [dict(record) for record in records]

        return related_accounts

# Singleton instance
gds_client = GDSClient()
