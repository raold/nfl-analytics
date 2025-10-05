"""
Alert management for ETL pipelines.

Sends alerts on pipeline failures, data quality issues, and anomalies.
"""

import logging
import smtplib
from dataclasses import dataclass
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Alert:
    """Single alert."""

    level: AlertLevel
    title: str
    message: str
    timestamp: datetime
    source: str = "ETL Pipeline"
    context: Optional[Dict] = None

    def __str__(self) -> str:
        level_emoji = {
            AlertLevel.INFO: "ℹ️",
            AlertLevel.WARNING: "⚠️",
            AlertLevel.ERROR: "❌",
            AlertLevel.CRITICAL: "🚨"
        }
        emoji = level_emoji.get(self.level, "")
        return f"{emoji} [{self.level.value.upper()}] {self.title}: {self.message}"


class AlertManager:
    """
    Manages alerts for ETL pipelines.

    Supports multiple alert channels:
    - Logging (always enabled)
    - Email (if configured)
    - File (for record keeping)
    - Slack (future)
    - PagerDuty (future)
    """

    def __init__(
        self,
        alert_log_path: Optional[Path] = None,
        email_config: Optional[Dict] = None,
        min_level: AlertLevel = AlertLevel.WARNING
    ):
        """
        Initialize alert manager.

        Args:
            alert_log_path: Path to write alert log file
            email_config: Email configuration (smtp_host, smtp_port, from_addr, to_addrs)
            min_level: Minimum alert level to process
        """
        self.min_level = min_level
        self.email_config = email_config

        if alert_log_path is None:
            alert_log_path = Path("logs/etl/alerts.log")
        self.alert_log_path = alert_log_path
        self.alert_log_path.parent.mkdir(parents=True, exist_ok=True)

        # Alert history (in-memory)
        self.alert_history: List[Alert] = []
        self.max_history = 1000

        logger.info(f"AlertManager initialized (min_level={min_level.value})")

    def send(
        self,
        level: AlertLevel,
        title: str,
        message: str,
        context: Optional[Dict] = None
    ):
        """
        Send an alert.

        Args:
            level: Alert severity
            title: Short alert title
            message: Detailed alert message
            context: Additional context (dict)
        """
        if level.value not in [l.value for l in AlertLevel if l.value >= self.min_level.value]:
            return

        alert = Alert(
            level=level,
            title=title,
            message=message,
            timestamp=datetime.now(),
            context=context
        )

        # Add to history
        self.alert_history.append(alert)
        if len(self.alert_history) > self.max_history:
            self.alert_history.pop(0)

        # Send to channels
        self._log_alert(alert)
        self._write_to_file(alert)

        if level in [AlertLevel.ERROR, AlertLevel.CRITICAL] and self.email_config:
            self._send_email(alert)

    def _log_alert(self, alert: Alert):
        """Log alert to application logger."""
        level_map = {
            AlertLevel.INFO: logging.INFO,
            AlertLevel.WARNING: logging.WARNING,
            AlertLevel.ERROR: logging.ERROR,
            AlertLevel.CRITICAL: logging.CRITICAL
        }
        log_level = level_map.get(alert.level, logging.INFO)
        logger.log(log_level, str(alert))

    def _write_to_file(self, alert: Alert):
        """Write alert to file."""
        try:
            with open(self.alert_log_path, 'a') as f:
                timestamp_str = alert.timestamp.isoformat()
                f.write(f"{timestamp_str} | {alert.level.value.upper()} | {alert.title} | {alert.message}\n")
        except Exception as e:
            logger.error(f"Failed to write alert to file: {e}")

    def _send_email(self, alert: Alert):
        """Send email alert."""
        if not self.email_config:
            return

        try:
            smtp_host = self.email_config.get('smtp_host')
            smtp_port = self.email_config.get('smtp_port', 587)
            from_addr = self.email_config.get('from_addr')
            to_addrs = self.email_config.get('to_addrs', [])
            username = self.email_config.get('username')
            password = self.email_config.get('password')

            if not all([smtp_host, from_addr, to_addrs]):
                logger.warning("Email config incomplete, skipping email alert")
                return

            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"[{alert.level.value.upper()}] ETL Alert: {alert.title}"
            msg['From'] = from_addr
            msg['To'] = ', '.join(to_addrs)

            # Create email body
            text_body = f"""
ETL Pipeline Alert

Level: {alert.level.value.upper()}
Time: {alert.timestamp.isoformat()}
Title: {alert.title}

Message:
{alert.message}

Context:
{alert.context if alert.context else 'None'}

---
This is an automated alert from the NFL Analytics ETL system.
            """

            html_body = f"""
<html>
<body>
<h2>ETL Pipeline Alert</h2>
<p><strong>Level:</strong> {alert.level.value.upper()}</p>
<p><strong>Time:</strong> {alert.timestamp.isoformat()}</p>
<p><strong>Title:</strong> {alert.title}</p>
<h3>Message:</h3>
<p>{alert.message}</p>
<h3>Context:</h3>
<pre>{alert.context if alert.context else 'None'}</pre>
<hr>
<p><em>This is an automated alert from the NFL Analytics ETL system.</em></p>
</body>
</html>
            """

            part1 = MIMEText(text_body, 'plain')
            part2 = MIMEText(html_body, 'html')
            msg.attach(part1)
            msg.attach(part2)

            # Send email
            with smtplib.SMTP(smtp_host, smtp_port) as server:
                if username and password:
                    server.starttls()
                    server.login(username, password)
                server.sendmail(from_addr, to_addrs, msg.as_string())

            logger.info(f"Email alert sent to {to_addrs}")

        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")

    def get_recent_alerts(self, limit: int = 10) -> List[Alert]:
        """Get most recent alerts."""
        return self.alert_history[-limit:]

    def get_alert_summary(self) -> Dict[str, int]:
        """Get summary of alerts by level."""
        summary = {level.value: 0 for level in AlertLevel}
        for alert in self.alert_history:
            summary[alert.level.value] += 1
        return summary

    # Convenience methods for common alerts
    def pipeline_failed(self, pipeline_name: str, error_message: str, context: Optional[Dict] = None):
        """Send pipeline failure alert."""
        self.send(
            level=AlertLevel.ERROR,
            title=f"Pipeline Failed: {pipeline_name}",
            message=f"Pipeline '{pipeline_name}' failed with error: {error_message}",
            context=context
        )

    def data_quality_issue(self, table: str, issue: str, severity: str = "warning", context: Optional[Dict] = None):
        """Send data quality issue alert."""
        level = AlertLevel.WARNING if severity == "warning" else AlertLevel.ERROR
        self.send(
            level=level,
            title=f"Data Quality Issue: {table}",
            message=issue,
            context=context
        )

    def api_rate_limit_warning(self, api_name: str, remaining: int, limit: int):
        """Send API rate limit warning."""
        pct_used = (1 - remaining / limit) * 100
        self.send(
            level=AlertLevel.WARNING,
            title=f"API Rate Limit Warning: {api_name}",
            message=f"API usage at {pct_used:.1f}% ({remaining}/{limit} remaining)",
            context={"api": api_name, "remaining": remaining, "limit": limit}
        )

    def pipeline_slow(self, pipeline_name: str, duration: float, expected: float):
        """Send slow pipeline alert."""
        slowdown_pct = (duration / expected - 1) * 100
        self.send(
            level=AlertLevel.WARNING,
            title=f"Slow Pipeline: {pipeline_name}",
            message=f"Pipeline took {duration:.1f}s (expected {expected:.1f}s, +{slowdown_pct:.0f}%)",
            context={"pipeline": pipeline_name, "duration": duration, "expected": expected}
        )


# Example usage
if __name__ == "__main__":
    # Initialize alert manager
    alert_manager = AlertManager(min_level=AlertLevel.INFO)

    # Send various alerts
    alert_manager.send(
        AlertLevel.INFO,
        "Pipeline Started",
        "Daily schedules pipeline started"
    )

    alert_manager.pipeline_failed(
        "odds_ingestion",
        "API key invalid",
        context={"api": "the-odds-api", "status_code": 401}
    )

    alert_manager.data_quality_issue(
        "games",
        "10 games missing scores",
        severity="warning",
        context={"missing_count": 10, "total_games": 272}
    )

    alert_manager.api_rate_limit_warning(
        "the-odds-api",
        remaining=50,
        limit=500
    )

    # Get summary
    print("\nAlert Summary:")
    summary = alert_manager.get_alert_summary()
    for level, count in summary.items():
        print(f"  {level}: {count}")

    print("\nRecent Alerts:")
    for alert in alert_manager.get_recent_alerts():
        print(f"  {alert}")
