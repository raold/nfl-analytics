# Contributing

Thank you for contributing! This project includes data pipelines, models, and analysis that can generate large artifacts. To keep the repository performant, please follow these guidelines.

Large files and generated artifacts
- Do not commit large generated artifacts (results, predictions, processed data). These paths are ignored in `.gitignore`:
  - results/  predictions/  data/processed/  logs/r_etl/
- Prefer to publish curated artifacts via project releases or external storage, and link to them from `docs/`.

Git LFS for model binaries
- We use Git LFS for model binaries and UBJSON artifacts:
  - Patterns: `*.pth`, `*.ubj`
- Local setup:
  - brew install git-lfs
  - git lfs install
- Adding new binary patterns (if truly needed):
  - git lfs track "*.newbinary"
  - git add .gitattributes
  - git commit -m "chore(lfs): track *.newbinary"

Adding small readmes in ignored folders
- We allow README.md inside ignored directories so users know how to reproduce contents:
  - results/README.md, predictions/README.md, data/processed/README.md
- If you add more ignored directories, consider adding a small README and a corresponding `!dir/README.md` exception in `.gitignore`.

Secrets and configuration
- Never commit secrets. Use environment variables or a secret manager.
- If a script needs credentials, document the variable names (e.g., `ODDS_API_KEY`) but do not print their values.

Pull requests
- Include tests for new modules where feasible (unit or smoke tests).
- For heavy training/experiments, provide a small, reproducible subset or a dry-run mode.
- For database changes, include SQL migration files and note any required environment changes.

Style and docs
- Keep docs under `docs/` up to date and add links to any external artifacts.
- Prefer adding summary markdown tables/figures (small text assets) over committing large binary artifacts.
