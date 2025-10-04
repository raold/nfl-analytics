#!/usr/bin/env python3
"""
Google Drive sync conflict detection and resolution for distributed compute.

Handles sync conflicts that can occur when multiple machines (MacBook M4 and Windows 4090)
simultaneously access SQLite databases through Google Drive synchronization.
"""

import logging
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from .file_locks import database_lock, sync_manager
from .machine_manager import get_machine_id

logger = logging.getLogger(__name__)


class ConflictResolutionError(Exception):
    """Raised when conflict resolution fails."""

    pass


class DatabaseConflictResolver:
    """Resolves SQLite database conflicts from Google Drive sync."""

    def __init__(self, auto_resolve: bool = True):
        """
        Initialize conflict resolver.

        Args:
            auto_resolve: If True, automatically resolve conflicts when detected
        """
        self.auto_resolve = auto_resolve
        self.machine_id = get_machine_id()

    def detect_database_conflicts(self, db_path: str) -> list[str]:
        """
        Detect Google Drive sync conflicts for a database.

        Args:
            db_path: Path to the main database file

        Returns:
            List of conflict file paths
        """
        conflicts = sync_manager.detect_sync_conflicts(db_path)

        # Also check for WAL and SHM conflicts
        db_file = Path(db_path)
        wal_file = db_file.with_suffix(db_file.suffix + "-wal")
        shm_file = db_file.with_suffix(db_file.suffix + "-shm")

        if wal_file.exists():
            conflicts.extend(sync_manager.detect_sync_conflicts(str(wal_file)))
        if shm_file.exists():
            conflicts.extend(sync_manager.detect_sync_conflicts(str(shm_file)))

        return conflicts

    def analyze_database_conflicts(self, db_path: str, conflicts: list[str]) -> dict[str, Any]:
        """
        Analyze database conflicts to determine resolution strategy.

        Args:
            db_path: Path to main database
            conflicts: List of conflict file paths

        Returns:
            Analysis results with recommended resolution
        """
        analysis = {
            "main_db": db_path,
            "conflicts": conflicts,
            "analysis_time": datetime.now().isoformat(),
            "machine_id": self.machine_id,
            "resolution_strategy": None,
            "conflict_details": [],
        }

        try:
            # Analyze each conflict file
            main_stats = self._get_database_stats(db_path)
            analysis["main_stats"] = main_stats

            for conflict_path in conflicts:
                conflict_stats = self._get_database_stats(conflict_path)
                conflict_details = {
                    "path": conflict_path,
                    "stats": conflict_stats,
                    "newer_than_main": False,
                    "more_data_than_main": False,
                }

                # Compare modification times
                if conflict_stats and main_stats:
                    conflict_details["newer_than_main"] = (
                        conflict_stats["modified_time"] > main_stats["modified_time"]
                    )
                    conflict_details["more_data_than_main"] = (
                        conflict_stats["total_records"] > main_stats["total_records"]
                    )

                analysis["conflict_details"].append(conflict_details)

            # Determine resolution strategy
            analysis["resolution_strategy"] = self._determine_resolution_strategy(analysis)

        except Exception as e:
            analysis["error"] = str(e)
            logger.error(f"Error analyzing conflicts for {db_path}: {e}")

        return analysis

    def _get_database_stats(self, db_path: str) -> dict[str, Any] | None:
        """Get statistics about a database file."""
        try:
            db_file = Path(db_path)
            if not db_file.exists():
                return None

            stats = {
                "file_size": db_file.stat().st_size,
                "modified_time": db_file.stat().st_mtime,
                "total_records": 0,
                "table_counts": {},
                "last_task_id": None,
                "machine_records": {},
            }

            # Connect and analyze database content
            with database_lock(db_path, timeout=10):
                conn = sqlite3.connect(db_path)
                conn.row_factory = sqlite3.Row

                try:
                    # Get table list
                    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
                    tables = [row["name"] for row in cursor]

                    # Count records in each table
                    total_records = 0
                    for table in tables:
                        try:
                            cursor = conn.execute(f"SELECT COUNT(*) as count FROM {table}")
                            count = cursor.fetchone()["count"]
                            stats["table_counts"][table] = count
                            total_records += count
                        except Exception:
                            pass

                    stats["total_records"] = total_records

                    # Get last task ID if tasks table exists
                    if "tasks" in tables:
                        try:
                            cursor = conn.execute(
                                "SELECT id FROM tasks ORDER BY created_at DESC LIMIT 1"
                            )
                            row = cursor.fetchone()
                            if row:
                                stats["last_task_id"] = row["id"]
                        except Exception:
                            pass

                    # Count records by machine if applicable
                    if "tasks" in tables:
                        try:
                            # Check if machine_id column exists
                            cursor = conn.execute("PRAGMA table_info(tasks)")
                            columns = [row[1] for row in cursor]
                            if "machine_id" in columns:
                                cursor = conn.execute(
                                    "SELECT machine_id, COUNT(*) as count FROM tasks GROUP BY machine_id"
                                )
                                for row in cursor:
                                    stats["machine_records"][row["machine_id"]] = row["count"]
                        except Exception:
                            pass

                finally:
                    conn.close()

            return stats

        except Exception as e:
            logger.warning(f"Could not get stats for {db_path}: {e}")
            return None

    def _determine_resolution_strategy(self, analysis: dict[str, Any]) -> str:
        """
        Determine the best resolution strategy based on conflict analysis.

        Args:
            analysis: Conflict analysis results

        Returns:
            Resolution strategy name
        """
        if not analysis["conflict_details"]:
            return "no_conflicts"

        main_stats = analysis.get("main_stats")
        if not main_stats:
            return "use_newest_conflict"

        # Strategy 1: If any conflict has significantly more data, use it
        for conflict in analysis["conflict_details"]:
            if conflict.get("more_data_than_main", False):
                conflict_records = conflict["stats"]["total_records"]
                main_records = main_stats["total_records"]
                if conflict_records > main_records * 1.1:  # 10% more data
                    return "use_largest_database"

        # Strategy 2: If any conflict is significantly newer, use it
        for conflict in analysis["conflict_details"]:
            if conflict.get("newer_than_main", False):
                conflict_time = conflict["stats"]["modified_time"]
                main_time = main_stats["modified_time"]
                if conflict_time > main_time + 300:  # 5 minutes newer
                    return "use_newest_database"

        # Strategy 3: Try to merge databases if possible
        if len(analysis["conflict_details"]) == 1:
            return "attempt_merge"

        # Strategy 4: Default to keeping main database
        return "keep_main_database"

    def resolve_conflicts(self, db_path: str, strategy: str | None = None) -> dict[str, Any]:
        """
        Resolve database conflicts.

        Args:
            db_path: Path to main database
            strategy: Override resolution strategy

        Returns:
            Resolution results
        """
        conflicts = self.detect_database_conflicts(db_path)
        if not conflicts:
            return {"status": "no_conflicts", "db_path": db_path}

        logger.info(f"Resolving {len(conflicts)} conflicts for {db_path}")

        # Analyze conflicts
        analysis = self.analyze_database_conflicts(db_path, conflicts)
        resolution_strategy = strategy or analysis["resolution_strategy"]

        try:
            if resolution_strategy == "use_largest_database":
                return self._resolve_use_largest(db_path, conflicts, analysis)
            elif resolution_strategy == "use_newest_database":
                return self._resolve_use_newest(db_path, conflicts, analysis)
            elif resolution_strategy == "attempt_merge":
                return self._resolve_attempt_merge(db_path, conflicts, analysis)
            elif resolution_strategy == "keep_main_database":
                return self._resolve_keep_main(db_path, conflicts, analysis)
            else:
                return self._resolve_use_newest(db_path, conflicts, analysis)

        except Exception as e:
            logger.error(f"Error resolving conflicts for {db_path}: {e}")
            # Fallback: backup conflicts and keep main
            return self._resolve_backup_conflicts(db_path, conflicts)

    def _resolve_use_largest(
        self, db_path: str, conflicts: list[str], analysis: dict[str, Any]
    ) -> dict[str, Any]:
        """Resolve by using the database with the most data."""
        # Find the database with the most records
        largest_db = db_path
        largest_records = analysis["main_stats"]["total_records"]

        for conflict in analysis["conflict_details"]:
            if conflict["stats"]["total_records"] > largest_records:
                largest_db = conflict["path"]
                largest_records = conflict["stats"]["total_records"]

        if largest_db != db_path:
            # Replace main with largest
            self._backup_database(db_path)
            shutil.move(largest_db, db_path)
            logger.info(f"Replaced {db_path} with larger database {largest_db}")

        # Clean up remaining conflicts
        self._cleanup_conflicts(conflicts, exclude=largest_db)

        return {
            "status": "resolved_use_largest",
            "chosen_database": largest_db,
            "records": largest_records,
        }

    def _resolve_use_newest(
        self, db_path: str, conflicts: list[str], analysis: dict[str, Any]
    ) -> dict[str, Any]:
        """Resolve by using the newest database."""
        # Find the newest database
        newest_db = db_path
        newest_time = analysis["main_stats"]["modified_time"]

        for conflict in analysis["conflict_details"]:
            if conflict["stats"]["modified_time"] > newest_time:
                newest_db = conflict["path"]
                newest_time = conflict["stats"]["modified_time"]

        if newest_db != db_path:
            # Replace main with newest
            self._backup_database(db_path)
            shutil.move(newest_db, db_path)
            logger.info(f"Replaced {db_path} with newer database {newest_db}")

        # Clean up remaining conflicts
        self._cleanup_conflicts(conflicts, exclude=newest_db)

        return {
            "status": "resolved_use_newest",
            "chosen_database": newest_db,
            "modified_time": newest_time,
        }

    def _resolve_attempt_merge(
        self, db_path: str, conflicts: list[str], analysis: dict[str, Any]
    ) -> dict[str, Any]:
        """Attempt to merge database conflicts."""
        if len(conflicts) != 1:
            # Fallback to use largest for multiple conflicts
            return self._resolve_use_largest(db_path, conflicts, analysis)

        conflict_path = conflicts[0]

        try:
            # Create backup
            backup_path = self._backup_database(db_path)

            # Attempt merge
            merged_records = self._merge_databases(db_path, conflict_path)

            # Clean up conflict
            Path(conflict_path).unlink()

            return {
                "status": "resolved_merge",
                "merged_records": merged_records,
                "backup_path": backup_path,
            }

        except Exception as e:
            logger.error(f"Merge failed for {db_path}: {e}")
            # Restore backup and fallback
            if "backup_path" in locals():
                shutil.move(backup_path, db_path)
            return self._resolve_use_largest(db_path, conflicts, analysis)

    def _resolve_keep_main(
        self, db_path: str, conflicts: list[str], analysis: dict[str, Any]
    ) -> dict[str, Any]:
        """Resolve by keeping the main database and backing up conflicts."""
        backup_paths = []
        for conflict_path in conflicts:
            backup_path = self._backup_conflict(conflict_path)
            backup_paths.append(backup_path)

        return {
            "status": "resolved_keep_main",
            "backup_paths": backup_paths,
        }

    def _resolve_backup_conflicts(self, db_path: str, conflicts: list[str]) -> dict[str, Any]:
        """Fallback resolution: backup all conflicts."""
        backup_paths = []
        for conflict_path in conflicts:
            try:
                backup_path = self._backup_conflict(conflict_path)
                backup_paths.append(backup_path)
            except Exception as e:
                logger.error(f"Failed to backup conflict {conflict_path}: {e}")

        return {
            "status": "resolved_backup_only",
            "backup_paths": backup_paths,
        }

    def _backup_database(self, db_path: str) -> str:
        """Create a timestamped backup of a database."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        db_file = Path(db_path)
        backup_path = db_file.with_name(
            f"{db_file.stem}_backup_{timestamp}_{self.machine_id}{db_file.suffix}"
        )

        shutil.copy2(db_path, backup_path)
        logger.info(f"Created backup: {backup_path}")
        return str(backup_path)

    def _backup_conflict(self, conflict_path: str) -> str:
        """Backup a conflict file and remove it."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        conflict_file = Path(conflict_path)
        backup_path = conflict_file.with_name(
            f"{conflict_file.stem}_conflict_{timestamp}_{self.machine_id}{conflict_file.suffix}"
        )

        shutil.move(conflict_path, backup_path)
        logger.info(f"Backed up conflict: {backup_path}")
        return str(backup_path)

    def _cleanup_conflicts(self, conflicts: list[str], exclude: str | None = None):
        """Clean up conflict files."""
        for conflict_path in conflicts:
            if exclude and conflict_path == exclude:
                continue
            try:
                Path(conflict_path).unlink()
                logger.info(f"Cleaned up conflict: {conflict_path}")
            except Exception as e:
                logger.warning(f"Could not clean up {conflict_path}: {e}")

    def _merge_databases(self, main_db: str, conflict_db: str) -> int:
        """
        Merge conflict database into main database.

        Args:
            main_db: Path to main database
            conflict_db: Path to conflict database

        Returns:
            Number of records merged
        """
        merged_records = 0

        with database_lock(main_db):
            main_conn = sqlite3.connect(main_db)
            main_conn.row_factory = sqlite3.Row

            conflict_conn = sqlite3.connect(conflict_db)
            conflict_conn.row_factory = sqlite3.Row

            try:
                # Get table schemas
                cursor = conflict_conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row["name"] for row in cursor]

                for table in tables:
                    try:
                        # Get table structure
                        cursor = main_conn.execute(f"PRAGMA table_info({table})")
                        columns = [row[1] for row in cursor]

                        if not columns:
                            continue

                        # Check for unique constraints (simple approach)
                        has_id = "id" in columns
                        has_created_at = "created_at" in columns

                        # Read conflict records
                        cursor = conflict_conn.execute(f"SELECT * FROM {table}")
                        conflict_records = cursor.fetchall()

                        for record in conflict_records:
                            try:
                                if has_id:
                                    # Check if record already exists
                                    existing = main_conn.execute(
                                        f"SELECT id FROM {table} WHERE id = ?",
                                        (record["id"],),
                                    ).fetchone()
                                    if existing:
                                        continue  # Skip duplicate

                                # Insert record
                                placeholders = ", ".join(["?" for _ in columns])
                                values = [record[col] for col in columns]

                                main_conn.execute(
                                    f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})",
                                    values,
                                )
                                merged_records += 1

                            except sqlite3.IntegrityError:
                                # Skip duplicates or constraint violations
                                continue

                        main_conn.commit()

                    except Exception as e:
                        logger.warning(f"Could not merge table {table}: {e}")
                        continue

            finally:
                conflict_conn.close()
                main_conn.close()

        logger.info(f"Merged {merged_records} records from {conflict_db}")
        return merged_records


class AutoConflictMonitor:
    """Automatically monitors and resolves database conflicts."""

    def __init__(self, db_paths: list[str], check_interval: int = 300):
        """
        Initialize conflict monitor.

        Args:
            db_paths: List of database paths to monitor
            check_interval: Check interval in seconds
        """
        self.db_paths = db_paths
        self.check_interval = check_interval
        self.resolver = DatabaseConflictResolver(auto_resolve=True)
        self.last_check = {}

    def check_all_databases(self) -> dict[str, Any]:
        """Check all monitored databases for conflicts."""
        results = {}

        for db_path in self.db_paths:
            try:
                conflicts = self.resolver.detect_database_conflicts(db_path)
                if conflicts:
                    logger.warning(f"Found {len(conflicts)} conflicts for {db_path}, resolving...")
                    resolution = self.resolver.resolve_conflicts(db_path)
                    results[db_path] = resolution
                else:
                    results[db_path] = {"status": "no_conflicts"}

            except Exception as e:
                logger.error(f"Error checking {db_path}: {e}")
                results[db_path] = {"status": "error", "error": str(e)}

        return results


# Global resolver instance
conflict_resolver = DatabaseConflictResolver()


def check_and_resolve_conflicts(db_path: str) -> dict[str, Any]:
    """Convenience function to check and resolve conflicts."""
    return conflict_resolver.resolve_conflicts(db_path)


if __name__ == "__main__":
    # Test conflict detection and resolution
    print("=== Testing Conflict Resolution ===")

    # This would normally be run when conflicts are detected
    test_db = "/tmp/test_conflicts.db"

    # Create test database
    import sqlite3

    conn = sqlite3.connect(test_db)
    conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, data TEXT)")
    conn.execute("INSERT INTO test (data) VALUES ('test1')")
    conn.commit()
    conn.close()

    # Test conflict detection
    conflicts = conflict_resolver.detect_database_conflicts(test_db)
    print(f"Detected conflicts: {conflicts}")

    # Test conflict analysis
    if conflicts:
        analysis = conflict_resolver.analyze_database_conflicts(test_db, conflicts)
        print(f"Analysis: {analysis}")

    print("Conflict resolution system ready!")
