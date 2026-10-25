"""Offline smoke tests for the frozen Game Outcome v2 code path.

The model module, the prediction service and scripts/generate_2025_game_predictions_fixed.py
must import without the optional 'causal' extra (shap) and without database credentials.
The frozen v2 boosters must load, and predict() must zero-fill and report missing inputs.
"""

import importlib
import importlib.util
import logging
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
V2_DIR = REPO / "models" / "production" / "game_outcome_v2"
MODULES = [
    "nfl_analytics.models.game_outcome_model_fixed",
    "nfl_analytics.services.game_outcome_runs",
]


@pytest.fixture
def no_shap_no_db(monkeypatch):
    """Make `import shap` fail and drop DB credentials, then re-import the modules fresh."""
    monkeypatch.setitem(sys.modules, "shap", None)  # None in sys.modules -> ImportError
    monkeypatch.delenv("DB_PASSWORD", raising=False)
    for name in MODULES:
        monkeypatch.delitem(sys.modules, name, raising=False)
    yield


def test_model_and_service_import_without_shap(no_shap_no_db):
    for name in MODULES:
        module = importlib.import_module(name)
        assert module is not None
    assert sys.modules["shap"] is None  # nothing replaced the blocked import


def test_prediction_script_imports_without_shap_or_credentials(no_shap_no_db):
    path = REPO / "scripts" / "generate_2025_game_predictions_fixed.py"
    spec = importlib.util.spec_from_file_location("generate_2025_game_predictions_fixed", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # top level only; main() is behind __main__
    assert callable(module.main)


def test_missing_feature_summary_groups_by_prefix():
    from nfl_analytics.models.game_outcome_model_fixed import summarize_missing_features

    missing = ["sem_home_sentiment_24h", "sem_injury_advantage", "syn_syn_qb_ol_synergy_diff"]
    missing += ["home_ngs_cpoe", "away_ngs_cpoe"]
    assert summarize_missing_features(missing) == "sem_* x2, ngs_* x2, syn_* x1"


class _NoSemanticExtractor:
    """Returns every v2 input except the removed sem_* and failing syn_* groups (all 0.0)."""

    def __init__(self, names):
        self.names = [n for n in names if not n.startswith(("sem_", "syn_"))]

    def extract_game_features(self, home_team, away_team, season, week):
        return {name: 0.0 for name in self.names}


def test_frozen_v2_refuses_to_load_by_default(no_shap_no_db):
    from nfl_analytics.models.game_outcome_model_fixed import GameOutcomeModel, is_frozen_reference

    assert is_frozen_reference(V2_DIR)
    model = GameOutcomeModel(model_dir=str(V2_DIR))
    with pytest.raises(RuntimeError, match="frozen reference"):
        model.load_models()
    assert not model.models_trained


def test_frozen_v2_predict_zero_fills_and_warns_once(no_shap_no_db, caplog):
    from nfl_analytics.models.game_outcome_model_fixed import GameOutcomeModel

    model = GameOutcomeModel(model_dir=str(V2_DIR))
    model.load_models(allow_frozen_reference=True)
    assert len(model.feature_names) == 382
    model.feature_extractor = _NoSemanticExtractor(model.feature_names)

    with caplog.at_level(logging.WARNING, logger="nfl_analytics.models.game_outcome_model_fixed"):
        pred = model.predict("LAC", "PIT", 2025, 10)

    warnings = [r for r in caplog.records if "model features missing" in r.getMessage()]
    assert len(warnings) == 1
    assert "50 of 382" in warnings[0].getMessage()
    assert "sem_* x38" in warnings[0].getMessage() and "syn_* x12" in warnings[0].getMessage()
    assert 0.0 <= pred["home_win_probability"] <= 1.0
    assert pred["total_points"] > 0
