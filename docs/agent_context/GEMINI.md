# NFL Analytics Project

## Project Overview

This project is an NFL analytics platform that uses a combination of R and Python to ingest, process, and analyze NFL data. The goal is to build predictive models for game outcomes and spreads, and to provide insights into team performance.

The project uses a PostgreSQL database with TimescaleDB for storing time-series data like odds and play-by-play data. Data is ingested from various sources, including `nflverse` for schedules and play-by-play data, and "The Odds API" for historical odds.

The analysis is performed using a series of Quarto notebooks, which cover data ingestion, feature engineering, model training, and simulation. The project also includes R and Python scripts for specific tasks like data ingestion and reporting.

## Building and Running

### Prerequisites

*   Docker and Docker Compose
*   R and the `renv` package
*   Python 3 and the dependencies listed in `requirements.txt`
*   An API key for "The Odds API" stored in a `.env` file as `ODDS_API_KEY`.

### Running the Project

1.  **Start the database:**
    ```bash
    docker compose up -d pg
    ```

2.  **Install R dependencies:**
    ```R
    renv::restore()
    ```

3.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the data ingestion and analysis notebooks:**
    The notebooks in the `notebooks/` directory are designed to be run in order. You can use the Quarto CLI to render and execute them:
    ```bash
    quarto render notebooks/
    ```

## Development Conventions

*   **Database:** The project uses a PostgreSQL database with TimescaleDB. The schema is defined in `db/001_init.sql`.
*   **Data Ingestion:** Data is ingested using R and Python scripts. The `nflverse` R package is used for schedules and play-by-play data, and a Python script is used to fetch historical odds from "The Odds API".
*   **Analysis:** The analysis is performed using Quarto notebooks. These notebooks are parameterized to allow for easy configuration and reuse.
*   **Modeling:** The project uses XGBoost for predictive modeling and a Skellam distribution for Monte Carlo simulations.
*   **Code Style:** The project uses a mix of R and Python. The code should be well-documented and follow the conventions of each language.
