# Contributing

Thank you for contributing! This project includes data pipelines, models, and analysis that can generate large artifacts. To keep the repository performant, please follow these guidelines.

Large files and generated artifacts
- Do not commit large generated artifacts (results, predictions, processed data). These paths are ignored in `.gitignore`:
  - results/  predictions/  data/processed/  logs/r_etl/
- Prefer to publish curated artifacts via project releases or external storage, and link to them from `docs/`.

Git LFS for model binaries
- **IMPORTANT**: We use Git LFS for ALL model binaries to avoid GitHub's 100MB limit
- Currently tracked patterns (see `.gitattributes`):
  - Python models: `*.pkl` (343MB+ via LFS)
  - R models: `*.rds`, `*.RDS` (28MB+ via LFS)
  - PyTorch: `*.pth`
  - UBJSON: `*.ubj`

- **Initial setup** (required once per machine):
  ```bash
  brew install git-lfs  # or: curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
  git lfs install
  ```

- **Before committing new model files**:
  1. Check if file type is already tracked: `cat .gitattributes`
  2. If not tracked, add pattern FIRST:
     ```bash
     git lfs track "*.newbinary"
     git add .gitattributes
     git commit -m "chore(lfs): track *.newbinary"
     ```
  3. Then add your model file:
     ```bash
     git add models/your_new_model.newbinary
     git commit -m "feat: add new model"
     ```

- **Verify LFS is working**:
  ```bash
  git lfs ls-files  # Should show your model files
  ```

- **Common mistakes to avoid**:
  - ❌ Committing large files without LFS tracking → GitHub will reject push
  - ❌ Adding to `.gitignore` when you want version control → Use LFS instead
  - ❌ Tracking then untracking without cleaning history → Requires BFG Repo-Cleaner

- **If you accidentally commit a large file without LFS**:
  1. Stop - don't push yet!
  2. Add LFS tracking: `git lfs track "*.extension"`
  3. Redo commit: `git reset --soft HEAD~1 && git add . && git commit`

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
