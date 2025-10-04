"""
Google Drive sync infrastructure for distributed compute.

Provides machine identification, file locking, and conflict resolution
for seamless distributed computing across multiple machines.
"""

from .machine_manager import MachineManager, get_machine_id, get_machine_summary
from .file_locks import FileLock, DatabaseLock, file_lock, database_lock, sync_manager
from .conflict_resolver import DatabaseConflictResolver, check_and_resolve_conflicts

__all__ = [
    "MachineManager",
    "get_machine_id",
    "get_machine_summary",
    "FileLock",
    "DatabaseLock",
    "file_lock",
    "database_lock",
    "sync_manager",
    "DatabaseConflictResolver",
    "check_and_resolve_conflicts",
]