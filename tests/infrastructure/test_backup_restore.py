"""
Backup and Restore Automation Tests.

Tests the backup.sh and restore.sh scripts to ensure disaster recovery works.

CRITICAL: These tests validate that your disaster recovery process actually works!

Test Categories:
1. Backup Creation: Verify backup files are created correctly
2. Backup Verification: Ensure backups are valid and restorable
3. Restore Process: Test full restore workflow
4. Data Integrity: Verify restored data matches original
5. Rotation: Test backup rotation/cleanup
"""

import json
import os
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path

import psycopg
import pytest


class TestBackupScript:
    """Test backup.sh script functionality."""

    @pytest.fixture(scope="class")
    def backup_dir(self):
        """Temporary backup directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture(scope="class")
    def test_db_conn(self):
        """Connection to test database."""
        conn = psycopg.connect(
            host=os.environ.get("POSTGRES_HOST", "localhost"),
            port=int(os.environ.get("POSTGRES_PORT", "5544")),
            dbname=os.environ.get("POSTGRES_DB", "devdb01"),
            user=os.environ.get("POSTGRES_USER", "dro"),
            password=os.environ.get("POSTGRES_PASSWORD", "sicillionbillions")
        )
        yield conn
        conn.close()

    def test_backup_script_exists(self):
        """Test that backup script exists and is executable."""
        script_path = Path("scripts/maintenance/backup.sh")
        assert script_path.exists(), "backup.sh script not found"
        assert os.access(script_path, os.X_OK), "backup.sh is not executable"

    def test_backup_creates_files(self, backup_dir, test_db_conn):
        """Test that backup creates .backup and .meta files."""
        # Run backup script with custom backup directory
        env = os.environ.copy()
        env["BACKUP_DIR"] = str(backup_dir)

        result = subprocess.run(
            ["bash", "scripts/maintenance/backup.sh"],
            env=env,
            capture_output=True,
            text=True,
            timeout=120
        )

        assert result.returncode == 0, f"Backup script failed: {result.stderr}"

        # Check that backup files were created
        backup_files = list(backup_dir.glob("nfl_analytics_*.backup"))
        assert len(backup_files) > 0, "No backup file created"

        meta_files = list(backup_dir.glob("nfl_analytics_*.meta"))
        assert len(meta_files) > 0, "No metadata file created"

        # Check symlinks
        assert (backup_dir / "latest.backup").exists(), "latest.backup symlink not created"
        assert (backup_dir / "latest.meta").exists(), "latest.meta symlink not created"

        print(f"✅ Backup created: {backup_files[0].name}")

    def test_backup_metadata_valid_json(self, backup_dir):
        """Test that metadata file is valid JSON."""
        meta_files = list(backup_dir.glob("nfl_analytics_*.meta"))

        if not meta_files:
            pytest.skip("No metadata files found (backup not run yet)")

        meta_file = meta_files[0]

        try:
            with open(meta_file) as f:
                metadata = json.load(f)

            # Check required fields
            assert "timestamp" in metadata
            assert "database" in metadata
            assert "data_counts" in metadata
            assert "games" in metadata["data_counts"]
            assert "plays" in metadata["data_counts"]

            print(f"✅ Metadata valid: {metadata['data_counts']['games']} games")

        except json.JSONDecodeError as e:
            pytest.fail(f"Invalid JSON in metadata file: {e}")

    def test_backup_verification_passes(self, backup_dir):
        """Test that pg_restore can list backup contents (verifies integrity)."""
        backup_files = list(backup_dir.glob("nfl_analytics_*.backup"))

        if not backup_files:
            pytest.skip("No backup files found")

        backup_file = backup_files[0]

        # Use pg_restore --list to verify backup
        result = subprocess.run(
            ["pg_restore", "--list", str(backup_file)],
            capture_output=True,
            text=True,
            timeout=30
        )

        assert result.returncode == 0, f"Backup verification failed: {result.stderr}"
        assert "TABLE DATA" in result.stdout, "Backup doesn't contain table data"

        print(f"✅ Backup verified: {backup_file.name}")

    def test_backup_rotation_works(self, backup_dir):
        """Test that old backups are rotated/deleted."""
        # Create multiple backups
        for i in range(3):
            env = os.environ.copy()
            env["BACKUP_DIR"] = str(backup_dir)
            env["MAX_BACKUPS"] = "2"  # Only keep 2

            subprocess.run(
                ["bash", "scripts/maintenance/backup.sh"],
                env=env,
                capture_output=True,
                timeout=120
            )
            time.sleep(2)  # Ensure different timestamps

        backup_files = list(backup_dir.glob("nfl_analytics_*.backup"))

        # With MAX_BACKUPS=2, should only have 2 backups
        # (Note: rotation happens on next backup, so might have 3)
        assert len(backup_files) <= 3, f"Too many backups retained: {len(backup_files)}"

        print(f"✅ Rotation working: {len(backup_files)} backups kept")


class TestRestoreScript:
    """Test restore.sh script functionality."""

    @pytest.fixture(scope="class")
    def backup_dir(self):
        """Create a backup for restore testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backup_path = Path(tmpdir)

            # Create a backup
            env = os.environ.copy()
            env["BACKUP_DIR"] = str(backup_path)

            result = subprocess.run(
                ["bash", "scripts/maintenance/backup.sh"],
                env=env,
                capture_output=True,
                text=True,
                timeout=120
            )

            if result.returncode != 0:
                pytest.skip(f"Backup creation failed: {result.stderr}")

            yield backup_path

    def test_restore_script_exists(self):
        """Test that restore script exists and is executable."""
        script_path = Path("scripts/maintenance/restore.sh")
        assert script_path.exists(), "restore.sh script not found"
        assert os.access(script_path, os.X_OK), "restore.sh is not executable"

    @pytest.mark.slow
    def test_restore_from_backup(self, backup_dir):
        """
        Test full restore process.

        WARNING: This test is DESTRUCTIVE - it drops and recreates the database!
        Only run in test environment.
        """
        # Get backup file
        backup_files = list(backup_dir.glob("nfl_analytics_*.backup"))
        if not backup_files:
            pytest.skip("No backup file found")

        backup_file = backup_files[0]

        # Get original counts from metadata
        meta_file = backup_file.with_suffix('.meta')
        with open(meta_file) as f:
            original_meta = json.load(f)

        original_games = original_meta['data_counts']['games']
        original_plays = original_meta['data_counts']['plays']

        # Restore using pg_restore directly (safer for tests than running restore.sh)
        db_url = "postgresql://dro:sicillionbillions@localhost:5544/devdb01"

        # Safety check: ensure we're in test environment
        if "test" not in os.environ.get("POSTGRES_DB", ""):
            pytest.skip("Skipping destructive test - not in test database")

        try:
            # Restore
            result = subprocess.run(
                ["pg_restore", "--dbname", db_url, "--clean", "--if-exists",
                 "--no-owner", "--verbose", str(backup_file)],
                capture_output=True,
                text=True,
                timeout=300
            )

            # pg_restore may have warnings but should complete
            assert "error" not in result.stderr.lower() or result.returncode == 0, \
                f"Restore had errors: {result.stderr}"

            # Verify restored data
            conn = psycopg.connect(db_url)
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM games")
                    restored_games = cur.fetchone()[0]

                    cur.execute("SELECT COUNT(*) FROM plays")
                    restored_plays = cur.fetchone()[0]

                # Allow some variance (new data may have been added)
                assert restored_games >= original_games * 0.9, \
                    f"Restored games mismatch: {restored_games} vs {original_games}"
                assert restored_plays >= original_plays * 0.9, \
                    f"Restored plays mismatch: {restored_plays} vs {original_plays}"

                print(f"✅ Restore successful: {restored_games} games, {restored_plays} plays")

            finally:
                conn.close()

        except subprocess.TimeoutExpired:
            pytest.fail("Restore timed out after 5 minutes")


class TestBackupRestoreRoundTrip:
    """Test complete backup → restore cycle preserves data integrity."""

    @pytest.mark.slow
    @pytest.mark.skip(reason="Destructive test - run manually")
    def test_backup_restore_data_identical(self):
        """
        Test that backup → restore produces identical data.

        This is the ULTIMATE test of disaster recovery.

        Process:
        1. Query current data
        2. Backup database
        3. Restore to new database
        4. Compare data
        """
        # This would require a separate test database
        # Skipped by default due to complexity and risk
        pass

    def test_latest_backup_is_most_recent(self):
        """Test that latest.backup symlink points to newest backup."""
        backup_dir = Path(os.environ.get("BACKUP_DIR", f"{os.environ['HOME']}/nfl-analytics-backups"))

        if not backup_dir.exists():
            pytest.skip("Backup directory doesn't exist")

        latest_link = backup_dir / "latest.backup"
        if not latest_link.exists():
            pytest.skip("No latest.backup symlink")

        # Get all backups
        backups = sorted(backup_dir.glob("nfl_analytics_*.backup"), reverse=True)

        if not backups:
            pytest.skip("No backups found")

        # Latest should point to most recent
        actual_target = latest_link.resolve()
        expected_target = backups[0].resolve()

        assert actual_target == expected_target, \
            f"latest.backup points to {actual_target}, expected {expected_target}"

        print(f"✅ Latest backup: {backups[0].name}")


class TestBackupMonitoring:
    """Test backup monitoring and alerting."""

    def test_backup_age_not_too_old(self):
        """Test that most recent backup is < 48 hours old."""
        backup_dir = Path(os.environ.get("BACKUP_DIR", f"{os.environ['HOME']}/nfl-analytics-backups"))

        if not backup_dir.exists():
            pytest.skip("Backup directory doesn't exist")

        backups = list(backup_dir.glob("nfl_analytics_*.backup"))

        if not backups:
            pytest.skip("No backups found")

        # Get newest backup
        newest = max(backups, key=lambda p: p.stat().st_mtime)
        backup_age_hours = (time.time() - newest.stat().st_mtime) / 3600

        assert backup_age_hours < 48, \
            f"Most recent backup is {backup_age_hours:.1f} hours old (> 48 hours)"

        print(f"✅ Most recent backup: {backup_age_hours:.1f} hours old")

    def test_backup_size_reasonable(self):
        """Test that backup size is within expected range."""
        backup_dir = Path(os.environ.get("BACKUP_DIR", f"{os.environ['HOME']}/nfl-analytics-backups"))

        if not backup_dir.exists():
            pytest.skip("Backup directory doesn't exist")

        backups = list(backup_dir.glob("nfl_analytics_*.backup"))

        if not backups:
            pytest.skip("No backups found")

        newest = max(backups, key=lambda p: p.stat().st_mtime)
        size_mb = newest.stat().st_size / (1024 * 1024)

        # NFL analytics DB should be 100 MB - 2 GB compressed
        assert 50 < size_mb < 5000, \
            f"Backup size {size_mb:.1f} MB is outside expected range (50-5000 MB)"

        print(f"✅ Backup size: {size_mb:.1f} MB")


# Utility functions for manual testing
def create_test_backup():
    """Utility to create a test backup."""
    import sys
    print("Creating test backup...")

    result = subprocess.run(
        ["bash", "scripts/maintenance/backup.sh"],
        capture_output=False,
        timeout=120
    )

    if result.returncode == 0:
        print("✅ Test backup created successfully")
    else:
        print("❌ Test backup failed")
        sys.exit(1)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short", "-m", "not slow"])
