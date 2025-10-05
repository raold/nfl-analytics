"""
NFLverse data extractor - wraps R ingestion scripts.

This extractor coordinates R-based data ingestion from nflverse packages
(nflfastR, nflreadr) with Python ETL infrastructure for:
- Schedules & betting lines
- Play-by-play data
- Rosters & player information
- Game metadata

Author: NFL Analytics Team
Date: October 2025
"""

import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd

from etl.extract.base import BaseExtractor, ExtractionResult, SourceType

logger = logging.getLogger(__name__)


class NFLVerseExtractor(BaseExtractor):
    """
    Extractor for nflverse data sources via R scripts.

    Wraps existing R ingestion scripts (ingest_schedules.R, ingest_pbp.R, etc.)
    with retry logic, error handling, and monitoring.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize NFLverse extractor.

        Args:
            config: Configuration dict with:
                - r_scripts_path: Path to R ingestion scripts
                - rscript_bin: Path to Rscript binary (default: 'Rscript')
                - timeout: Script timeout in seconds (default: 600)
        """
        super().__init__(source_type=SourceType.R_PACKAGE, config=config)

        self.r_scripts_path = Path(config.get('r_scripts_path', 'R/ingestion'))
        self.rscript_bin = config.get('rscript_bin', 'Rscript')
        self.timeout = config.get('timeout', 600)  # 10 minutes default

        logger.info(f"NFLVerseExtractor initialized with scripts at {self.r_scripts_path}")

    def extract_schedules(
        self,
        seasons: Optional[List[int]] = None,
        **kwargs
    ) -> ExtractionResult:
        """
        Extract NFL schedules and betting lines.

        Args:
            seasons: List of seasons to fetch (None = all available)
            **kwargs: Additional arguments

        Returns:
            ExtractionResult with schedules data
        """
        script_path = self.r_scripts_path / "ingest_schedules.R"

        if not script_path.exists():
            return ExtractionResult(
                success=False,
                data=None,
                row_count=0,
                extraction_time=datetime.now(),
                duration_seconds=0.0,
                source="nflverse_schedules",
                endpoint=str(script_path),
                error=f"R script not found: {script_path}"
            )

        logger.info(f"Extracting schedules via {script_path}")
        start_time = datetime.now()

        try:
            # Run R script
            result = subprocess.run(
                [self.rscript_bin, '--vanilla', str(script_path)],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=True
            )

            duration = (datetime.now() - start_time).total_seconds()

            # Log R output
            if result.stdout:
                logger.info(f"R stdout: {result.stdout}")
            if result.stderr:
                logger.warning(f"R stderr: {result.stderr}")

            # Parse output for row count (if R script prints it)
            row_count = self._parse_row_count(result.stdout)

            return ExtractionResult(
                success=True,
                data=None,  # Data loaded directly to DB by R script
                row_count=row_count,
                extraction_time=start_time,
                duration_seconds=duration,
                source="nflverse_schedules",
                endpoint=str(script_path),
                metadata={"seasons": seasons, "stdout": result.stdout}
            )

        except subprocess.TimeoutExpired as e:
            duration = (datetime.now() - start_time).total_seconds()
            logger.error(f"R script timeout after {duration}s: {e}")
            return ExtractionResult(
                success=False,
                data=None,
                row_count=0,
                extraction_time=start_time,
                duration_seconds=duration,
                source="nflverse_schedules",
                endpoint=str(script_path),
                error=f"Timeout after {self.timeout}s"
            )

        except subprocess.CalledProcessError as e:
            duration = (datetime.now() - start_time).total_seconds()
            logger.error(f"R script failed with code {e.returncode}: {e.stderr}")
            return ExtractionResult(
                success=False,
                data=None,
                row_count=0,
                extraction_time=start_time,
                duration_seconds=duration,
                source="nflverse_schedules",
                endpoint=str(script_path),
                error=f"R script error: {e.stderr}"
            )

    def extract_pbp(
        self,
        seasons: Optional[List[int]] = None,
        **kwargs
    ) -> ExtractionResult:
        """
        Extract play-by-play data with EPA.

        Args:
            seasons: List of seasons (None = all 1999-present)
            **kwargs: Additional arguments

        Returns:
            ExtractionResult with PBP data
        """
        script_path = self.r_scripts_path / "ingest_pbp.R"

        if not script_path.exists():
            return ExtractionResult(
                success=False,
                data=None,
                row_count=0,
                extraction_time=datetime.now(),
                duration_seconds=0.0,
                source="nflverse_pbp",
                endpoint=str(script_path),
                error=f"R script not found: {script_path}"
            )

        logger.info(f"Extracting PBP data via {script_path} (may take 5+ minutes)")
        start_time = datetime.now()

        try:
            result = subprocess.run(
                [self.rscript_bin, '--vanilla', str(script_path)],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=True
            )

            duration = (datetime.now() - start_time).total_seconds()

            if result.stdout:
                logger.info(f"R stdout: {result.stdout}")
            if result.stderr:
                logger.warning(f"R stderr: {result.stderr}")

            row_count = self._parse_row_count(result.stdout)

            return ExtractionResult(
                success=True,
                data=None,
                row_count=row_count,
                extraction_time=start_time,
                duration_seconds=duration,
                source="nflverse_pbp",
                endpoint=str(script_path),
                metadata={"seasons": seasons, "stdout": result.stdout}
            )

        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            logger.error(f"PBP extraction failed: {e}")
            return ExtractionResult(
                success=False,
                data=None,
                row_count=0,
                extraction_time=start_time,
                duration_seconds=duration,
                source="nflverse_pbp",
                endpoint=str(script_path),
                error=str(e)
            )

    def extract_rosters(
        self,
        seasons: Optional[List[int]] = None,
        **kwargs
    ) -> ExtractionResult:
        """
        Extract player rosters.

        Args:
            seasons: List of seasons
            **kwargs: Additional arguments

        Returns:
            ExtractionResult with roster data
        """
        script_path = self.r_scripts_path / "ingest_2025_season.R"

        if not script_path.exists():
            # Fallback to backfill script
            script_path = Path("R/backfill_rosters.R")

        if not script_path.exists():
            return ExtractionResult(
                success=False,
                data=None,
                row_count=0,
                extraction_time=datetime.now(),
                duration_seconds=0.0,
                source="nflverse_rosters",
                endpoint=str(script_path),
                error=f"R script not found: {script_path}"
            )

        logger.info(f"Extracting rosters via {script_path}")
        start_time = datetime.now()

        try:
            result = subprocess.run(
                [self.rscript_bin, '--vanilla', str(script_path)],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=True
            )

            duration = (datetime.now() - start_time).total_seconds()
            row_count = self._parse_row_count(result.stdout)

            return ExtractionResult(
                success=True,
                data=None,
                row_count=row_count,
                extraction_time=start_time,
                duration_seconds=duration,
                source="nflverse_rosters",
                endpoint=str(script_path),
                metadata={"seasons": seasons, "stdout": result.stdout}
            )

        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            logger.error(f"Roster extraction failed: {e}")
            return ExtractionResult(
                success=False,
                data=None,
                row_count=0,
                extraction_time=start_time,
                duration_seconds=duration,
                source="nflverse_rosters",
                endpoint=str(script_path),
                error=str(e)
            )

    def _parse_row_count(self, stdout: str) -> int:
        """
        Parse row count from R script output.

        Looks for patterns like:
        - "Loaded 1234 rows"
        - "Inserted 5678 games"
        - "[1] 9012 rows"

        Args:
            stdout: R script stdout

        Returns:
            Parsed row count or 0 if not found
        """
        if not stdout:
            return 0

        import re

        # Try different patterns
        patterns = [
            r'(\d+)\s+rows?',
            r'(\d+)\s+games?',
            r'Loaded\s+(\d+)',
            r'Inserted\s+(\d+)',
            r'\[1\]\s+(\d+)'
        ]

        for pattern in patterns:
            match = re.search(pattern, stdout, re.IGNORECASE)
            if match:
                try:
                    return int(match.group(1))
                except (ValueError, IndexError):
                    continue

        return 0

    def health_check(self) -> bool:
        """
        Verify R environment is available.

        Returns:
            True if Rscript is available and scripts exist
        """
        try:
            # Check Rscript binary
            result = subprocess.run(
                [self.rscript_bin, '--version'],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode != 0:
                logger.error(f"Rscript not available: {result.stderr}")
                return False

            # Check key scripts exist
            schedules_script = self.r_scripts_path / "ingest_schedules.R"
            if not schedules_script.exists():
                logger.error(f"Schedule ingestion script not found: {schedules_script}")
                return False

            logger.info("NFLverse extractor health check passed")
            return True

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    config = {
        'r_scripts_path': 'R/ingestion',
        'rscript_bin': 'Rscript',
        'timeout': 600
    }

    extractor = NFLVerseExtractor(config)

    # Health check
    if extractor.health_check():
        print("✅ NFLverse extractor ready")

        # Extract schedules
        result = extractor.extract_schedules()
        print(f"Schedules: {result.success}, {result.row_count} rows in {result.duration_seconds:.1f}s")
    else:
        print("❌ NFLverse extractor not available")
