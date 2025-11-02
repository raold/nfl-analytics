#!/usr/bin/env python3
"""
Export PostgreSQL database to SQL file for GitHub storage.
Usage: python scripts/export_db_snapshot.py
"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Configuration
DB_HOST = "localhost"
DB_PORT = "5544"
DB_USER = "dro"
DB_NAME = "devdb01"
DB_PASSWORD = "sicillionbillions"

# Output file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = Path("db/backups")
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / f"devdb01_{timestamp}_windows.sql"

print(f"Exporting database {DB_NAME} to {output_file}...")

# Use pg_dump via subprocess (works on Windows if PostgreSQL is installed)
try:
    # Try pg_dump in PATH first
    result = subprocess.run(
        [
            "pg_dump",
            f"--host={DB_HOST}",
            f"--port={DB_PORT}",
            f"--username={DB_USER}",
            f"--dbname={DB_NAME}",
            "--clean",
            "--if-exists",
            "--no-owner",
            "--no-acl",
            f"--file={output_file}",
        ],
        env={"PGPASSWORD": DB_PASSWORD},
        capture_output=True,
        text=True,
        check=True,
    )

    # Get file size
    size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"✓ Export complete: {output_file}")
    print(f"✓ File size: {size_mb:.2f} MB")
    print(f"\nNext steps:")
    print(f"  git add {output_file}")
    print(f"  git commit -m \"chore: database snapshot for Mac transition ({timestamp})\"")
    print(f"  git push origin main")

except FileNotFoundError:
    print("ERROR: pg_dump not found in PATH")
    print("\nOptions:")
    print("1. Install PostgreSQL client tools for Windows")
    print("2. Use Docker: docker compose exec -T pg pg_dump ...")
    print("3. Manually export from pgAdmin or DBeaver")
    sys.exit(1)

except subprocess.CalledProcessError as e:
    print(f"ERROR: pg_dump failed")
    print(f"stdout: {e.stdout}")
    print(f"stderr: {e.stderr}")
    sys.exit(1)
