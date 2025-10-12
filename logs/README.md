# Logs Directory

This directory contains log files organized by category:

## Subdirectories

- **backfill/**: Logs from data backfill operations (nflverse data ingestion)
- **backtest/**: Logs from model backtest runs and results
- **training/**: Logs from R and Python model training sessions
- **workers/**: Logs from distributed training worker processes

## Log Retention

Logs are kept for debugging and audit purposes. Old logs (>90 days) can be archived or removed periodically.
