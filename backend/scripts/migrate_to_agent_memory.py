#!/usr/bin/env python3
"""
Migration script to convert existing Decision nodes to neo4j-agent-memory reasoning traces.

This script:
1. Reads existing Decision nodes from the graph
2. Creates equivalent reasoning traces using neo4j-agent-memory
3. Preserves relationships and metadata
4. Optionally creates a backup before migration

Usage:
    python migrate_to_agent_memory.py --dry-run     # Preview changes
    python migrate_to_agent_memory.py --execute     # Run migration
    python migrate_to_agent_memory.py --backup      # Backup first, then migrate
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from neo4j import GraphDatabase
from neo4j_agent_memory import MemoryClient

from app.config import config, get_memory_settings


class MigrationStats:
    """Track migration statistics."""

    def __init__(self):
        self.decisions_found = 0
        self.traces_created = 0
        self.steps_created = 0
        self.errors = []
        self.skipped = 0

    def summary(self) -> dict:
        return {
            "decisions_found": self.decisions_found,
            "traces_created": self.traces_created,
            "steps_created": self.steps_created,
            "errors": len(self.errors),
            "skipped": self.skipped,
        }


async def get_existing_decisions(driver, database: str, batch_size: int = 100) -> list[dict]:
    """Fetch all existing Decision nodes."""
    decisions = []

    with driver.session(database=database) as session:
        # Get total count
        result = session.run("MATCH (d:Decision) RETURN count(d) AS count")
        total = result.single()["count"]
        print(f"Found {total} Decision nodes to migrate")

        # Fetch in batches
        offset = 0
        while offset < total:
            result = session.run(
                """
                MATCH (d:Decision)
                OPTIONAL MATCH (d)-[:ABOUT]->(entity)
                OPTIONAL MATCH (d)-[:FOLLOWED_PRECEDENT]->(precedent:Decision)
                OPTIONAL MATCH (d)-[:CAUSED]->(effect:Decision)
                OPTIONAL MATCH (d)-[:INFLUENCED]->(influenced:Decision)
                WITH d,
                     collect(DISTINCT {id: entity.id, labels: labels(entity)}) AS entities,
                     collect(DISTINCT precedent.id) AS precedent_ids,
                     collect(DISTINCT effect.id) AS effect_ids,
                     collect(DISTINCT influenced.id) AS influenced_ids
                RETURN d {
                    .*,
                    about_entities: entities,
                    precedent_ids: [p IN precedent_ids WHERE p IS NOT NULL],
                    effect_ids: [e IN effect_ids WHERE e IS NOT NULL],
                    influenced_ids: [i IN influenced_ids WHERE i IS NOT NULL]
                } AS decision
                ORDER BY d.decision_timestamp
                SKIP $offset
                LIMIT $batch_size
                """,
                {"offset": offset, "batch_size": batch_size},
            )

            batch = [record["decision"] for record in result]
            decisions.extend(batch)
            offset += batch_size
            print(f"  Fetched {len(decisions)}/{total} decisions...")

    return decisions


async def migrate_decision_to_trace(
    memory: MemoryClient,
    decision: dict,
    stats: MigrationStats,
    dry_run: bool = True,
) -> str | None:
    """Convert a single Decision node to a reasoning trace."""

    decision_id = decision.get("id")
    decision_type = decision.get("decision_type", "unknown")
    category = decision.get("category", "unknown")
    reasoning = decision.get("reasoning", "")
    reasoning_summary = decision.get("reasoning_summary", "")
    confidence_score = decision.get("confidence_score", 0.8)
    risk_factors = decision.get("risk_factors", [])
    session_id = decision.get("session_id") or f"migration-{decision_id}"

    # Build task description
    task = f"{decision_type} decision for {category}"
    if reasoning_summary:
        task = f"{task}: {reasoning_summary[:100]}"

    # Build metadata
    metadata = {
        "legacy_decision_id": decision_id,
        "decision_type": decision_type,
        "category": category,
        "migrated_at": datetime.utcnow().isoformat(),
        "about_entities": decision.get("about_entities", []),
        "precedent_ids": decision.get("precedent_ids", []),
        "effect_ids": decision.get("effect_ids", []),
        "influenced_ids": decision.get("influenced_ids", []),
    }

    if dry_run:
        print(f"  [DRY RUN] Would create trace for decision {decision_id}")
        print(f"    Task: {task[:80]}...")
        print(f"    Session: {session_id}")
        return None

    try:
        # Create the trace
        trace = await memory.reasoning.start_trace(
            session_id=session_id,
            task=task,
            metadata=metadata,
        )
        trace_id = trace.id
        stats.traces_created += 1

        # Add reasoning as a step
        if reasoning:
            await memory.reasoning.add_step(
                trace_id=trace_id,
                step_type="analysis",
                content=reasoning,
                metadata={
                    "confidence_score": confidence_score,
                    "risk_factors": risk_factors,
                },
            )
            stats.steps_created += 1

        # Complete the trace
        outcome = f"Decision recorded: {decision_type}"
        await memory.reasoning.complete_trace(
            trace_id=trace_id,
            outcome=outcome,
            success=True,
        )

        return trace_id

    except Exception as e:
        stats.errors.append(
            {
                "decision_id": decision_id,
                "error": str(e),
            }
        )
        print(f"  [ERROR] Failed to migrate decision {decision_id}: {e}")
        return None


async def create_backup(driver, database: str, backup_path: Path) -> None:
    """Create a JSON backup of all Decision nodes."""
    print(f"Creating backup at {backup_path}...")

    decisions = []
    with driver.session(database=database) as session:
        result = session.run(
            """
            MATCH (d:Decision)
            OPTIONAL MATCH (d)-[r]->(target)
            WITH d, collect({type: type(r), target_id: target.id, target_labels: labels(target)}) AS relationships
            RETURN d {.*, outgoing_relationships: relationships} AS decision
            """
        )
        decisions = [record["decision"] for record in result]

    # Convert Neo4j types to JSON-serializable
    def serialize(obj):
        if hasattr(obj, "isoformat"):
            return obj.isoformat()
        if isinstance(obj, (list, tuple)):
            return [serialize(item) for item in obj]
        if isinstance(obj, dict):
            return {k: serialize(v) for k, v in obj.items()}
        return obj

    backup_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "database": database,
        "decision_count": len(decisions),
        "decisions": [serialize(d) for d in decisions],
    }

    backup_path.parent.mkdir(parents=True, exist_ok=True)
    with open(backup_path, "w") as f:
        json.dump(backup_data, f, indent=2, default=str)

    print(f"  Backup complete: {len(decisions)} decisions saved")


async def run_migration(
    dry_run: bool = True,
    backup: bool = False,
    batch_size: int = 100,
) -> MigrationStats:
    """Run the migration process."""

    stats = MigrationStats()

    # Connect to Neo4j
    driver = GraphDatabase.driver(
        config.neo4j.uri,
        auth=(config.neo4j.username, config.neo4j.password),
    )

    try:
        # Create backup if requested
        if backup and not dry_run:
            backup_path = (
                Path(__file__).parent
                / "backups"
                / f"decisions_backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
            )
            await create_backup(driver, config.neo4j.database, backup_path)

        # Fetch existing decisions
        decisions = await get_existing_decisions(driver, config.neo4j.database, batch_size)
        stats.decisions_found = len(decisions)

        if not decisions:
            print("No decisions found to migrate")
            return stats

        # Initialize memory client
        settings = get_memory_settings()

        async with MemoryClient(settings) as memory:
            print(f"\nMigrating {len(decisions)} decisions to reasoning traces...")

            for i, decision in enumerate(decisions):
                if (i + 1) % 10 == 0:
                    print(f"  Progress: {i + 1}/{len(decisions)}")

                trace_id = await migrate_decision_to_trace(
                    memory=memory,
                    decision=decision,
                    stats=stats,
                    dry_run=dry_run,
                )

                if trace_id:
                    print(f"  Migrated decision {decision.get('id')} -> trace {trace_id}")

        return stats

    finally:
        driver.close()


def main():
    parser = argparse.ArgumentParser(
        description="Migrate Decision nodes to neo4j-agent-memory reasoning traces"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without executing migration",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the migration",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create backup before migration (only with --execute)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of decisions to process per batch",
    )

    args = parser.parse_args()

    if not args.dry_run and not args.execute:
        print("Error: Must specify either --dry-run or --execute")
        parser.print_help()
        sys.exit(1)

    if args.dry_run and args.execute:
        print("Error: Cannot specify both --dry-run and --execute")
        sys.exit(1)

    dry_run = args.dry_run

    print("=" * 60)
    print("Decision to Reasoning Trace Migration")
    print("=" * 60)
    print(f"Mode: {'DRY RUN (no changes)' if dry_run else 'EXECUTE'}")
    print(f"Backup: {'Yes' if args.backup else 'No'}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 60)
    print()

    # Run migration
    stats = asyncio.run(
        run_migration(
            dry_run=dry_run,
            backup=args.backup,
            batch_size=args.batch_size,
        )
    )

    # Print summary
    print()
    print("=" * 60)
    print("Migration Summary")
    print("=" * 60)
    summary = stats.summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")

    if stats.errors:
        print()
        print("Errors:")
        for error in stats.errors[:10]:  # Show first 10 errors
            print(f"  - {error['decision_id']}: {error['error']}")
        if len(stats.errors) > 10:
            print(f"  ... and {len(stats.errors) - 10} more errors")

    print("=" * 60)

    if dry_run:
        print("\nThis was a dry run. Use --execute to perform the actual migration.")
    else:
        print("\nMigration complete!")


if __name__ == "__main__":
    main()
