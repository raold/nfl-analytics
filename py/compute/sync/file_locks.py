#!/usr/bin/env python3
"""
Cross-platform file locking for Google Drive synced compute.

Provides robust file locking mechanisms to prevent database corruption
during concurrent access across MacBook M4 and Windows 4090 systems.
"""

import fcntl
import logging
import os
import platform
import tempfile
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

if platform.system() == "Windows":
    import msvcrt

logger = logging.getLogger(__name__)


class FileLockError(Exception):
    """Raised when file locking operations fail."""

    pass


class FileLock:
    """Cross-platform file locking implementation."""

    def __init__(self, file_path: str, timeout: float = 30.0):
        """
        Initialize file lock.

        Args:
            file_path: Path to file or database to lock
            timeout: Maximum time to wait for lock acquisition
        """
        self.file_path = Path(file_path)
        self.lock_file_path = self.file_path.with_suffix(self.file_path.suffix + ".lock")
        self.timeout = timeout
        self.lock_file: object | None = None
        self._acquired = False

    def acquire(self, blocking: bool = True) -> bool:
        """
        Acquire the file lock.

        Args:
            blocking: If True, wait for lock. If False, return immediately.

        Returns:
            True if lock acquired, False otherwise

        Raises:
            FileLockError: If lock cannot be acquired within timeout
        """
        if self._acquired:
            return True

        start_time = time.time()
        while True:
            try:
                # Create lock file with exclusive access
                self.lock_file = open(self.lock_file_path, "w")

                if platform.system() == "Windows":
                    # Windows file locking
                    try:
                        msvcrt.locking(self.lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                        self._acquired = True
                        break
                    except OSError:
                        if not blocking:
                            self.lock_file.close()
                            return False
                else:
                    # Unix-like systems (macOS, Linux)
                    try:
                        if blocking:
                            fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_EX)
                        else:
                            fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        self._acquired = True
                        break
                    except OSError:
                        if not blocking:
                            self.lock_file.close()
                            return False

            except OSError:
                if not blocking:
                    return False

            # Check timeout
            if blocking and time.time() - start_time > self.timeout:
                if self.lock_file:
                    self.lock_file.close()
                raise FileLockError(
                    f"Could not acquire lock for {self.file_path} within {self.timeout} seconds"
                )

            # Wait briefly before retrying
            time.sleep(0.1)

        # Write lock metadata
        if self._acquired and self.lock_file:
            try:
                self.lock_file.write(f"locked_by_pid:{os.getpid()}\n")
                self.lock_file.write(f"locked_at:{time.time()}\n")
                self.lock_file.write(f"hostname:{platform.node()}\n")
                self.lock_file.flush()
            except Exception:
                pass  # Lock metadata is optional

        return self._acquired

    def release(self):
        """Release the file lock."""
        if not self._acquired or not self.lock_file:
            return

        try:
            if platform.system() == "Windows":
                msvcrt.locking(self.lock_file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_UN)

            self.lock_file.close()
            self.lock_file = None

            # Clean up lock file
            try:
                self.lock_file_path.unlink(missing_ok=True)
            except Exception:
                pass  # Lock file cleanup is best-effort

        except Exception as e:
            logger.warning(f"Error releasing lock for {self.file_path}: {e}")
        finally:
            self._acquired = False

    def is_locked(self) -> bool:
        """Check if lock is currently held."""
        return self._acquired

    def __enter__(self):
        """Context manager entry."""
        if not self.acquire():
            raise FileLockError(f"Could not acquire lock for {self.file_path}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()

    def __del__(self):
        """Cleanup on destruction."""
        if self._acquired:
            self.release()


class DatabaseLock(FileLock):
    """Specialized lock for SQLite database files."""

    def __init__(self, db_path: str, timeout: float = 30.0):
        """
        Initialize database lock.

        Args:
            db_path: Path to SQLite database file
            timeout: Maximum time to wait for lock acquisition
        """
        super().__init__(db_path, timeout)

        # Also lock WAL and SHM files if they exist
        self.wal_lock = None
        self.shm_lock = None

        db_file = Path(db_path)
        wal_file = db_file.with_suffix(db_file.suffix + "-wal")
        shm_file = db_file.with_suffix(db_file.suffix + "-shm")

        if wal_file.exists():
            self.wal_lock = FileLock(str(wal_file), timeout)
        if shm_file.exists():
            self.shm_lock = FileLock(str(shm_file), timeout)

    def acquire(self, blocking: bool = True) -> bool:
        """Acquire locks on database and related files."""
        # First acquire main database lock
        if not super().acquire(blocking):
            return False

        # Then acquire WAL and SHM locks if needed
        try:
            if self.wal_lock and not self.wal_lock.acquire(blocking):
                self.release()
                return False

            if self.shm_lock and not self.shm_lock.acquire(blocking):
                if self.wal_lock:
                    self.wal_lock.release()
                self.release()
                return False

            return True

        except Exception:
            # Cleanup on failure
            self.release()
            return False

    def release(self):
        """Release all database-related locks."""
        if self.shm_lock:
            self.shm_lock.release()
        if self.wal_lock:
            self.wal_lock.release()
        super().release()


@contextmanager
def file_lock(file_path: str, timeout: float = 30.0) -> Generator[FileLock, None, None]:
    """
    Context manager for file locking.

    Args:
        file_path: Path to file to lock
        timeout: Maximum time to wait for lock

    Yields:
        FileLock instance

    Example:
        with file_lock("important_file.txt") as lock:
            # File is exclusively locked here
            with open("important_file.txt", "w") as f:
                f.write("Safe to write")
    """
    lock = FileLock(file_path, timeout)
    try:
        if not lock.acquire():
            raise FileLockError(f"Could not acquire lock for {file_path}")
        yield lock
    finally:
        lock.release()


@contextmanager
def database_lock(db_path: str, timeout: float = 30.0) -> Generator[DatabaseLock, None, None]:
    """
    Context manager for database locking.

    Args:
        db_path: Path to SQLite database
        timeout: Maximum time to wait for lock

    Yields:
        DatabaseLock instance

    Example:
        with database_lock("compute_queue.db") as lock:
            # Database is exclusively locked here
            conn = sqlite3.connect("compute_queue.db")
            # Safe to perform operations
            conn.close()
    """
    lock = DatabaseLock(db_path, timeout)
    try:
        if not lock.acquire():
            raise FileLockError(f"Could not acquire database lock for {db_path}")
        yield lock
    finally:
        lock.release()


class SyncSafetyManager:
    """Manages file operations with Google Drive sync safety."""

    def __init__(self, sync_delay: float = 2.0):
        """
        Initialize sync safety manager.

        Args:
            sync_delay: Time to wait after operations for sync to settle
        """
        self.sync_delay = sync_delay

    @contextmanager
    def safe_database_operation(self, db_path: str, timeout: float = 30.0):
        """
        Perform database operation safely with Google Drive sync.

        Args:
            db_path: Path to database file
            timeout: Lock timeout
        """
        with database_lock(db_path, timeout) as lock:
            try:
                # Perform the operation
                yield lock

                # Wait for potential sync conflicts to resolve
                time.sleep(self.sync_delay)

            except Exception:
                # On error, wait for sync and re-raise
                time.sleep(self.sync_delay)
                raise

    def checkpoint_database(self, db_path: str):
        """
        Safely checkpoint a SQLite database for sync.

        Args:
            db_path: Path to SQLite database
        """
        import sqlite3

        with self.safe_database_operation(db_path):
            conn = sqlite3.connect(db_path)
            try:
                # Force WAL checkpoint
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                conn.commit()
                logger.info(f"Checkpointed database: {db_path}")
            finally:
                conn.close()

    def detect_sync_conflicts(self, file_path: str) -> list[str]:
        """
        Detect Google Drive sync conflict files.

        Args:
            file_path: Path to check for conflicts

        Returns:
            List of conflict file paths
        """
        base_path = Path(file_path)
        parent = base_path.parent

        # Google Drive creates conflict files with pattern:
        # filename (Conflicted copy 2024-01-01 123456).ext
        conflict_pattern = f"{base_path.stem} (Conflicted copy*){base_path.suffix}"

        conflicts = []
        try:
            for conflict_file in parent.glob(conflict_pattern):
                conflicts.append(str(conflict_file))
        except Exception:
            pass

        return conflicts

    def resolve_sync_conflicts(self, file_path: str, strategy: str = "newest") -> bool:
        """
        Resolve Google Drive sync conflicts.

        Args:
            file_path: Original file path
            strategy: Resolution strategy ("newest", "largest", "manual")

        Returns:
            True if conflicts were resolved
        """
        conflicts = self.detect_sync_conflicts(file_path)
        if not conflicts:
            return True

        logger.warning(f"Found {len(conflicts)} sync conflicts for {file_path}")

        if strategy == "newest":
            # Keep the newest file
            all_files = [file_path] + conflicts
            newest_file = max(all_files, key=lambda f: Path(f).stat().st_mtime)

            # Replace original with newest
            if newest_file != file_path:
                import shutil

                shutil.move(newest_file, file_path)
                logger.info(f"Resolved conflict: kept newest file {newest_file}")

            # Remove other conflict files
            for conflict in conflicts:
                if conflict != newest_file:
                    try:
                        Path(conflict).unlink()
                    except Exception:
                        pass

            return True

        elif strategy == "largest":
            # Keep the largest file
            all_files = [file_path] + conflicts
            largest_file = max(all_files, key=lambda f: Path(f).stat().st_size)

            if largest_file != file_path:
                import shutil

                shutil.move(largest_file, file_path)
                logger.info(f"Resolved conflict: kept largest file {largest_file}")

            # Remove other conflict files
            for conflict in conflicts:
                if conflict != largest_file:
                    try:
                        Path(conflict).unlink()
                    except Exception:
                        pass

            return True

        else:
            # Manual resolution required
            logger.error(f"Manual conflict resolution required for {file_path}")
            return False


# Global sync safety manager
sync_manager = SyncSafetyManager()


if __name__ == "__main__":
    # Test file locking
    print("=== Testing File Locking ===")

    test_file = Path(tempfile.gettempdir()) / "test_lock.txt"

    # Test basic file lock
    with file_lock(str(test_file)) as lock:
        print(f"Acquired lock for {test_file}")
        with open(test_file, "w") as f:
            f.write("test content")
        print("File operation completed safely")

    # Test database lock
    test_db = Path(tempfile.gettempdir()) / "test.db"
    import sqlite3

    # Create test database
    conn = sqlite3.connect(str(test_db))
    conn.execute("CREATE TABLE IF NOT EXISTS test (id INTEGER PRIMARY KEY)")
    conn.close()

    with database_lock(str(test_db)) as db_lock:
        print(f"Acquired database lock for {test_db}")
        conn = sqlite3.connect(str(test_db))
        conn.execute("INSERT INTO test DEFAULT VALUES")
        conn.commit()
        conn.close()
        print("Database operation completed safely")

    # Cleanup
    test_file.unlink(missing_ok=True)
    test_db.unlink(missing_ok=True)
    Path(str(test_db) + "-wal").unlink(missing_ok=True)
    Path(str(test_db) + "-shm").unlink(missing_ok=True)

    print("File locking tests completed successfully!")
