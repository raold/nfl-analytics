"""Unit tests for state-space ratings model."""
import numpy as np
import pandas as pd
import pytest

# Need to add py/ to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "py"))

from models.state_space import StateSpaceRatings


class TestStateSpaceRatings:
    """Test state-space ratings model."""
    
    @pytest.fixture
    def model(self):
        """Create test model with default params."""
        return StateSpaceRatings(
            q=3.0,
            r=13.5,
            init_rating=0.0,
            init_variance=100.0,
            hfa_init=2.5
        )
    
    def test_initialization(self, model):
        """Test model initializes with correct parameters."""
        assert model.q == 3.0
        assert model.r == 13.5
        assert model.init_rating == 0.0
        assert model.init_variance == 100.0
        assert model.hfa == 2.5
        assert len(model.ratings) == 0
        assert len(model.history) == 0
    
    def test_initialize_team(self, model):
        """Test team initialization."""
        model.initialize_team("KC")
        
        assert "KC" in model.ratings
        mean, var = model.ratings["KC"]
        assert mean == 0.0
        assert var == 100.0
    
    def test_predict_step(self, model):
        """Test predict step adds process noise."""
        model.initialize_team("KC")
        initial_var = model.ratings["KC"][1]
        
        model.predict_step()
        
        new_var = model.ratings["KC"][1]
        assert new_var == initial_var + model.q**2
        # Mean should stay same
        assert model.ratings["KC"][0] == 0.0
    
    def test_update_game_basic(self, model):
        """Test game update modifies ratings."""
        home_team = "KC"
        away_team = "LV"
        margin = 14.0  # KC wins by 14
        
        model.update_game(home_team, away_team, margin)
        
        # Both teams should be in ratings now
        assert home_team in model.ratings
        assert away_team in model.ratings
        
        # Home team (winner) should have positive rating, away negative
        kc_rating, _ = model.ratings[home_team]
        lv_rating, _ = model.ratings[away_team]
        
        # KC should be stronger than LV after this game
        assert kc_rating > lv_rating
    
    def test_update_game_reduces_variance(self, model):
        """Test that observing games reduces uncertainty."""
        home_team = "KC"
        away_team = "LV"
        
        model.initialize_team(home_team)
        model.initialize_team(away_team)
        
        initial_var_home = model.ratings[home_team][1]
        initial_var_away = model.ratings[away_team][1]
        
        model.update_game(home_team, away_team, 7.0)
        
        new_var_home = model.ratings[home_team][1]
        new_var_away = model.ratings[away_team][1]
        
        # Variance should decrease after observation
        assert new_var_home < initial_var_home
        assert new_var_away < initial_var_away
    
    def test_fit_on_dataframe(self, model):
        """Test fitting on a small DataFrame."""
        # Create mock game data
        games = pd.DataFrame({
            "season": [2023, 2023, 2023, 2023],
            "week": [1, 1, 2, 2],
            "home_team": ["KC", "BUF", "KC", "BUF"],
            "away_team": ["DET", "NYJ", "LV", "MIA"],
            "margin": [21, 14, 7, -3]  # KC wins, BUF wins, KC wins, BUF loses
        })
        
        model.fit(games)
        
        # Should have 6 teams (4 home + 4 away = 6 unique)
        assert len(model.ratings) == 6
        
        # Should have history entries (6 teams * 2 weeks = 12, but some teams appear in both weeks)
        assert len(model.history) > 0
        
        # KC won both games, should have positive rating
        kc_rating, _ = model.ratings["KC"]
        assert kc_rating > 0
    
    def test_predict_margin(self, model):
        """Test margin prediction."""
        # Set up some known ratings
        model.ratings["KC"] = (5.0, 25.0)  # Strong team
        model.ratings["LV"] = (-3.0, 25.0)  # Weak team
        
        mean_margin, std_margin = model.predict_margin("KC", "LV")
        
        # KC at home should be favored by 5 - (-3) + 2.5 = 10.5
        assert mean_margin == pytest.approx(10.5, abs=0.1)
        
        # Std should incorporate uncertainties
        assert std_margin > 0
    
    def test_predict_prob_home_win(self, model):
        """Test win probability prediction."""
        # Strong home team vs weak away team
        model.ratings["KC"] = (10.0, 25.0)
        model.ratings["LV"] = (-5.0, 25.0)
        
        prob = model.predict_prob_home_win("KC", "LV")
        
        # KC heavily favored
        assert prob > 0.8
        
        # Test close matchup
        model.ratings["BUF"] = (2.0, 25.0)
        model.ratings["MIA"] = (1.0, 25.0)
        
        prob_close = model.predict_prob_home_win("BUF", "MIA")
        
        # Should be close to 50-50 with HFA giving slight edge
        assert 0.5 < prob_close < 0.7
    
    def test_predict_prob_ats(self, model):
        """Test ATS probability prediction."""
        model.ratings["KC"] = (5.0, 25.0)
        model.ratings["LV"] = (-3.0, 25.0)
        
        # KC favored by ~10.5 at home
        # With spread of -7, KC needs to win by >7
        prob_cover = model.predict_prob_ats("KC", "LV", spread=-7.0)
        
        # KC expected to cover 7-point spread
        assert prob_cover > 0.5
        
        # With spread of -14, KC unlikely to cover
        prob_no_cover = model.predict_prob_ats("KC", "LV", spread=-14.0)
        assert prob_no_cover < 0.5
    
    def test_get_ratings_df(self, model):
        """Test converting history to DataFrame."""
        games = pd.DataFrame({
            "season": [2023, 2023],
            "week": [1, 1],
            "home_team": ["KC", "BUF"],
            "away_team": ["LV", "MIA"],
            "margin": [14, 7]
        })
        
        model.fit(games)
        ratings_df = model.get_ratings_df()
        
        assert isinstance(ratings_df, pd.DataFrame)
        assert "season" in ratings_df.columns
        assert "week" in ratings_df.columns
        assert "team" in ratings_df.columns
        assert "rating" in ratings_df.columns
        assert "variance" in ratings_df.columns
        assert len(ratings_df) == 4  # 4 teams * 1 week
    
    def test_multiple_weeks_increases_certainty(self, model):
        """Test that ratings become more certain over multiple weeks."""
        games = []
        for week in range(1, 11):  # 10 weeks
            games.append({
                "season": 2023,
                "week": week,
                "home_team": "KC",
                "away_team": f"TEAM_{week}",
                "margin": 10 + np.random.randn() * 3
            })
        
        games_df = pd.DataFrame(games)
        model.fit(games_df)
        
        ratings_df = model.get_ratings_df()
        kc_ratings = ratings_df[ratings_df["team"] == "KC"]
        
        # Variance should generally decrease over time
        variances = kc_ratings["variance"].values
        # First week variance should be higher than last week
        assert variances[0] > variances[-1]


class TestKalmanMechanics:
    """Test specific Kalman filter mechanics."""
    
    def test_information_gain(self):
        """Test that surprising results lead to bigger rating changes."""
        model = StateSpaceRatings(q=3.0, r=13.5)
        
        # Start with equal teams
        model.ratings["A"] = (0.0, 50.0)
        model.ratings["B"] = (0.0, 50.0)
        
        # Team A blows out Team B at home
        model.update_game("A", "B", margin=30.0)
        
        rating_a_after_blowout, _ = model.ratings["A"]
        
        # Reset and test close game
        model.ratings["A"] = (0.0, 50.0)
        model.ratings["B"] = (0.0, 50.0)
        
        model.update_game("A", "B", margin=5.0)
        
        rating_a_after_close, _ = model.ratings["A"]
        
        # Blowout should lead to bigger rating change
        assert abs(rating_a_after_blowout) > abs(rating_a_after_close)
    
    def test_home_field_advantage_effect(self):
        """Test HFA properly affects predictions."""
        model = StateSpaceRatings(hfa_init=3.0)
        
        # Equal teams
        model.ratings["A"] = (5.0, 25.0)
        model.ratings["B"] = (5.0, 25.0)
        
        # A at home should be favored by HFA
        margin_a_home, _ = model.predict_margin("A", "B")
        assert margin_a_home == pytest.approx(3.0, abs=0.1)
        
        # B at home should be favored by HFA
        margin_b_home, _ = model.predict_margin("B", "A")
        assert margin_b_home == pytest.approx(3.0, abs=0.1)
    
    def test_process_noise_accumulation(self):
        """Test that process noise accumulates over multiple weeks."""
        model = StateSpaceRatings(q=3.0)
        model.ratings["KC"] = (10.0, 25.0)
        
        initial_var = model.ratings["KC"][1]
        
        # Simulate 5 weeks with no games (just predict steps)
        for _ in range(5):
            model.predict_step()
        
        final_var = model.ratings["KC"][1]
        
        # Variance should have increased by 5 * q^2
        expected_var = initial_var + 5 * (model.q ** 2)
        assert final_var == pytest.approx(expected_var, abs=0.1)


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_dataframe(self):
        """Test fitting on empty DataFrame."""
        model = StateSpaceRatings()
        empty_df = pd.DataFrame(columns=["season", "week", "home_team", "away_team", "margin"])
        
        model.fit(empty_df)
        
        assert len(model.ratings) == 0
        assert len(model.history) == 0
    
    def test_single_game(self):
        """Test fitting on single game."""
        model = StateSpaceRatings()
        game = pd.DataFrame({
            "season": [2023],
            "week": [1],
            "home_team": ["KC"],
            "away_team": ["LV"],
            "margin": [14]
        })
        
        model.fit(game)
        
        assert len(model.ratings) == 2
        assert "KC" in model.ratings
        assert "LV" in model.ratings
    
    def test_predict_unseen_team(self):
        """Test prediction for team not in training data."""
        model = StateSpaceRatings()
        model.ratings["KC"] = (5.0, 25.0)
        
        # Predict against unseen team (should initialize with defaults)
        mean, std = model.predict_margin("KC", "UNSEEN")
        
        assert mean == pytest.approx(5.0 + 2.5, abs=0.1)  # KC rating + HFA
        assert std > 0
    
    def test_extreme_margin(self):
        """Test model handles extreme margins."""
        model = StateSpaceRatings()
        
        # Very large margin (e.g., forfeit or running up score)
        model.update_game("A", "B", margin=70.0)
        
        # Should still produce finite ratings
        rating_a, var_a = model.ratings["A"]
        assert np.isfinite(rating_a)
        assert np.isfinite(var_a)
        assert var_a > 0


class TestIntegration:
    """Integration tests with realistic scenarios."""
    
    def test_season_simulation(self):
        """Test fitting on simulated season data."""
        np.random.seed(42)
        
        teams = ["KC", "BUF", "MIA", "LV", "LAC", "DEN"]
        true_strengths = {
            "KC": 8, "BUF": 6, "MIA": 3, "LV": -2, "LAC": 0, "DEN": -4
        }
        
        # Simulate 10 weeks of games
        games = []
        for week in range(1, 11):
            # Each team plays once per week (3 games)
            matchups = [("KC", "LV"), ("BUF", "DEN"), ("MIA", "LAC")]
            for home, away in matchups:
                # Simulate margin based on true strengths + HFA + noise
                expected = true_strengths[home] - true_strengths[away] + 2.5
                margin = expected + np.random.randn() * 13.5
                games.append({
                    "season": 2023,
                    "week": week,
                    "home_team": home,
                    "away_team": away,
                    "margin": margin
                })
        
        games_df = pd.DataFrame(games)
        
        # Fit model
        model = StateSpaceRatings(q=2.0, r=13.5, hfa_init=2.5)
        model.fit(games_df)
        
        # Check that estimated ratings correlate with true strengths
        estimated = {team: model.ratings[team][0] for team in teams}
        
        # KC should be rated highest
        assert estimated["KC"] == max(estimated.values())
        
        # LV should be rated lowest (lost all games in our simulation)
        assert estimated["LV"] == min(estimated.values())
        
        # Relative ordering should roughly match for clear cases
        assert estimated["KC"] > estimated["BUF"]  # 8 vs 6
        assert estimated["BUF"] > estimated["LV"]  # 6 vs -2
        # MIA vs LAC can vary due to noise, so skip that specific comparison
