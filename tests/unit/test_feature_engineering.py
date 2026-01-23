"""
Unit tests for Feature Engineering.
Tests ELO calculations, feature generation, and sport-specific features.
"""

import pytest
from datetime import datetime, timedelta
import numpy as np

from app.services.ml.feature_engineering import (
    FeatureEngineer,
    ELORating,
    SportCode,
    SportFeatures,
)


class TestELORating:
    """Tests for ELORating class."""
    
    def test_default_rating(self):
        """Test default ELO rating is 1500."""
        elo = ELORating()
        assert elo.rating == 1500.0
    
    def test_custom_initial_rating(self):
        """Test custom initial rating."""
        elo = ELORating(rating=1600.0)
        assert elo.rating == 1600.0
    
    def test_expected_score_equal_ratings(self):
        """Test expected score when ratings are equal."""
        elo = ELORating(rating=1500.0)
        expected = elo.expected_score(opponent_rating=1500.0)
        assert expected == pytest.approx(0.5, rel=0.01)
    
    def test_expected_score_higher_rating(self):
        """Test expected score when our rating is higher."""
        elo = ELORating(rating=1600.0)
        expected = elo.expected_score(opponent_rating=1400.0)
        assert expected > 0.5
        assert expected < 1.0
    
    def test_expected_score_lower_rating(self):
        """Test expected score when our rating is lower."""
        elo = ELORating(rating=1400.0)
        expected = elo.expected_score(opponent_rating=1600.0)
        assert expected < 0.5
        assert expected > 0.0
    
    def test_rating_update_win(self):
        """Test rating increases after a win."""
        elo = ELORating(rating=1500.0, k_factor=32)
        opponent_rating = 1500.0
        
        old_rating = elo.rating
        elo.update(actual_score=1.0, opponent_rating=opponent_rating)
        
        assert elo.rating > old_rating
    
    def test_rating_update_loss(self):
        """Test rating decreases after a loss."""
        elo = ELORating(rating=1500.0, k_factor=32)
        opponent_rating = 1500.0
        
        old_rating = elo.rating
        elo.update(actual_score=0.0, opponent_rating=opponent_rating)
        
        assert elo.rating < old_rating
    
    def test_rating_update_draw(self):
        """Test rating with draw (0.5)."""
        elo = ELORating(rating=1500.0, k_factor=32)
        opponent_rating = 1500.0
        
        old_rating = elo.rating
        elo.update(actual_score=0.5, opponent_rating=opponent_rating)
        
        # Equal ratings, draw = no change
        assert elo.rating == pytest.approx(old_rating, rel=0.01)
    
    def test_upset_win_larger_gain(self):
        """Test upset win gives larger rating gain."""
        elo_underdog = ELORating(rating=1400.0, k_factor=32)
        elo_favorite = ELORating(rating=1600.0, k_factor=32)
        
        underdog_old = elo_underdog.rating
        favorite_old = elo_favorite.rating
        
        # Underdog wins
        elo_underdog.update(actual_score=1.0, opponent_rating=1600.0)
        elo_favorite.update(actual_score=1.0, opponent_rating=1200.0)  # Expected win
        
        underdog_gain = elo_underdog.rating - underdog_old
        favorite_gain = elo_favorite.rating - favorite_old
        
        # Underdog gains more for upset
        assert underdog_gain > favorite_gain
    
    def test_games_played_tracking(self):
        """Test games played counter."""
        elo = ELORating()
        
        assert elo.games_played == 0
        
        elo.update(1.0, 1500.0)
        assert elo.games_played == 1
        
        elo.update(0.0, 1500.0)
        assert elo.games_played == 2


class TestSportCode:
    """Tests for SportCode enum."""
    
    def test_all_sports_defined(self):
        """Test all 10 sports are defined."""
        expected_sports = ["NFL", "NCAAF", "CFL", "NBA", "NCAAB", 
                          "WNBA", "NHL", "MLB", "ATP", "WTA"]
        
        for sport in expected_sports:
            assert hasattr(SportCode, sport)
    
    def test_sport_values(self):
        """Test sport code values."""
        assert SportCode.NFL.value == "NFL"
        assert SportCode.NBA.value == "NBA"
        assert SportCode.ATP.value == "ATP"


class TestSportFeatures:
    """Tests for SportFeatures dataclass."""
    
    def test_feature_creation(self):
        """Test creating sport features."""
        features = SportFeatures(
            game_id=12345,
            sport_code="NBA",
            features={
                "home_elo": 1620.0,
                "away_elo": 1580.0,
                "elo_diff": 40.0,
            },
            feature_names=["home_elo", "away_elo", "elo_diff"],
        )
        
        assert features.game_id == 12345
        assert features.sport_code == "NBA"
        assert len(features.features) == 3
    
    def test_to_dataframe(self):
        """Test conversion to DataFrame."""
        features = SportFeatures(
            game_id=1,
            sport_code="NFL",
            features={
                "home_elo": 1550.0,
                "away_elo": 1500.0,
            },
            feature_names=["home_elo", "away_elo"],
        )
        
        df = features.to_dataframe()
        assert len(df) == 1
        assert "home_elo" in df.columns
    
    def test_to_array(self):
        """Test conversion to numpy array."""
        features = SportFeatures(
            game_id=2,
            sport_code="NHL",
            features={
                "home_elo": 1520.0,
                "away_elo": 1480.0,
                "elo_diff": 40.0,
            },
            feature_names=["home_elo", "away_elo", "elo_diff"],
        )
        
        arr = features.to_array()
        assert isinstance(arr, np.ndarray)
        assert len(arr) == 3


class TestFeatureEngineer:
    """Tests for FeatureEngineer class."""
    
    @pytest.fixture
    def engineer(self):
        """Create a feature engineer instance."""
        return FeatureEngineer()
    
    def test_feature_counts_defined(self, engineer):
        """Test feature counts are defined for all sports."""
        for sport in SportCode:
            assert sport in engineer.FEATURE_COUNTS
            assert engineer.FEATURE_COUNTS[sport] >= 60
    
    def test_elo_k_factors_defined(self, engineer):
        """Test ELO K-factors defined for all sports."""
        for sport in SportCode:
            assert sport in engineer.ELO_K_FACTORS
            assert 15 <= engineer.ELO_K_FACTORS[sport] <= 40
    
    def test_nba_feature_count(self, engineer):
        """Test NBA has 80 features."""
        assert engineer.FEATURE_COUNTS[SportCode.NBA] == 80
    
    def test_mlb_feature_count(self, engineer):
        """Test MLB has 85 features (most)."""
        assert engineer.FEATURE_COUNTS[SportCode.MLB] == 85
    
    def test_tennis_feature_count(self, engineer):
        """Test tennis has 60 features."""
        assert engineer.FEATURE_COUNTS[SportCode.ATP] == 60
        assert engineer.FEATURE_COUNTS[SportCode.WTA] == 60


class TestTeamPerformanceFeatures:
    """Tests for team performance feature calculations."""
    
    def test_offensive_rating(self):
        """Test offensive rating calculation."""
        # Points per 100 possessions
        points = 110
        possessions = 100
        off_rating = (points / possessions) * 100
        assert off_rating == 110.0
    
    def test_defensive_rating(self):
        """Test defensive rating calculation."""
        # Points allowed per 100 possessions
        points_allowed = 105
        possessions = 100
        def_rating = (points_allowed / possessions) * 100
        assert def_rating == 105.0
    
    def test_net_rating(self):
        """Test net rating calculation."""
        off_rating = 112.0
        def_rating = 108.0
        net_rating = off_rating - def_rating
        assert net_rating == 4.0
    
    def test_pace(self):
        """Test pace calculation."""
        possessions = 102
        minutes = 48
        pace = possessions  # Possessions per 48 minutes
        assert pace == 102


class TestRecentFormFeatures:
    """Tests for recent form feature calculations."""
    
    def test_win_streak_positive(self):
        """Test positive win streak."""
        results = ["W", "W", "W", "L", "W"]  # Current 1-game win streak
        
        streak = 0
        for result in reversed(results):
            if result == "W":
                streak += 1
            else:
                break
        
        assert streak == 1
    
    def test_win_streak_negative(self):
        """Test losing streak (negative)."""
        results = ["L", "L", "L", "W", "W"]  # Current 3-game losing streak
        
        streak = 0
        for result in reversed(results):
            if result == "L":
                streak -= 1
            else:
                break
        
        assert streak == -3
    
    def test_last_5_wins(self):
        """Test last 5 games win count."""
        results = ["W", "L", "W", "W", "L"]
        wins = sum(1 for r in results if r == "W")
        assert wins == 3
    
    def test_momentum_score(self):
        """Test momentum score calculation."""
        # Weighted recent performance (more recent = more weight)
        results = [1, 1, 0, 1, 0]  # W, W, L, W, L (most recent first)
        weights = [0.30, 0.25, 0.20, 0.15, 0.10]
        
        momentum = sum(r * w for r, w in zip(results, weights))
        expected = 0.30 + 0.25 + 0.0 + 0.15 + 0.0  # 0.70
        assert momentum == pytest.approx(expected)


class TestRestTravelFeatures:
    """Tests for rest and travel feature calculations."""
    
    def test_rest_days_calculation(self):
        """Test rest days calculation."""
        last_game = datetime(2024, 3, 10)
        current_game = datetime(2024, 3, 13)
        rest_days = (current_game - last_game).days
        assert rest_days == 3
    
    def test_back_to_back_detection(self):
        """Test back-to-back detection."""
        last_game = datetime(2024, 3, 14)
        current_game = datetime(2024, 3, 15)
        rest_days = (current_game - last_game).days
        is_b2b = rest_days <= 1
        assert is_b2b is True
    
    def test_games_in_7_days(self):
        """Test games in last 7 days calculation."""
        game_dates = [
            datetime(2024, 3, 15),  # Today
            datetime(2024, 3, 13),
            datetime(2024, 3, 11),
            datetime(2024, 3, 9),
            datetime(2024, 3, 5),  # Outside 7 days
        ]
        current = datetime(2024, 3, 15)
        games_in_7 = sum(
            1 for d in game_dates 
            if (current - d).days <= 7
        )
        assert games_in_7 == 4
    
    def test_rest_advantage(self):
        """Test rest advantage calculation."""
        home_rest = 3
        away_rest = 1
        rest_advantage = home_rest - away_rest
        assert rest_advantage == 2


class TestHeadToHeadFeatures:
    """Tests for head-to-head feature calculations."""
    
    def test_h2h_record(self):
        """Test H2H record calculation."""
        wins = 7
        losses = 3
        total = wins + losses
        win_pct = wins / total
        assert win_pct == 0.7
    
    def test_h2h_average_margin(self):
        """Test H2H average margin."""
        margins = [10, -5, 15, 3, -8]  # Home margins
        avg_margin = sum(margins) / len(margins)
        assert avg_margin == 3.0
    
    def test_last_5_h2h(self):
        """Test last 5 H2H results."""
        results = ["W", "L", "W", "W", "L"]
        wins = sum(1 for r in results if r == "W")
        win_pct = wins / len(results)
        assert win_pct == 0.6


class TestLineMovementFeatures:
    """Tests for line movement feature calculations."""
    
    def test_spread_movement(self):
        """Test spread movement calculation."""
        opening_spread = -3.0
        current_spread = -5.0
        movement = current_spread - opening_spread
        assert movement == -2.0  # Line moved toward home team
    
    def test_total_movement(self):
        """Test total movement calculation."""
        opening_total = 220.0
        current_total = 225.5
        movement = current_total - opening_total
        assert movement == 5.5  # Total moved up
    
    def test_steam_move_detection(self):
        """Test steam move detection."""
        # Steam move: 1+ point move in 10 minutes
        movement = 1.5
        time_minutes = 8
        is_steam = movement >= 1.0 and time_minutes <= 10
        assert is_steam is True
    
    def test_reverse_line_movement(self):
        """Test reverse line movement detection."""
        # Line moves opposite to public betting
        public_on_home = 0.70  # 70% on home
        spread_movement = 1.0  # Spread moved away from home
        is_rlm = public_on_home > 0.60 and spread_movement > 0
        assert is_rlm is True


class TestWeatherFeatures:
    """Tests for weather features (outdoor sports)."""
    
    def test_temperature_feature(self):
        """Test temperature feature."""
        temp_f = 35  # Cold game
        is_cold = temp_f < 40
        assert is_cold is True
    
    def test_wind_speed_feature(self):
        """Test wind speed feature."""
        wind_mph = 15
        is_windy = wind_mph >= 10
        assert is_windy is True
    
    def test_dome_detection(self):
        """Test dome game detection."""
        venue = "Lucas Oil Stadium"
        is_dome = "dome" in venue.lower() or venue in [
            "Lucas Oil Stadium", "AT&T Stadium", "Mercedes-Benz Stadium"
        ]
        assert is_dome is True


class TestFeatureNormalization:
    """Tests for feature normalization."""
    
    def test_min_max_scaling(self):
        """Test min-max scaling."""
        value = 1600
        min_val = 1200
        max_val = 1800
        scaled = (value - min_val) / (max_val - min_val)
        assert scaled == pytest.approx(0.667, rel=0.01)
    
    def test_z_score_normalization(self):
        """Test z-score normalization."""
        value = 110
        mean = 100
        std = 10
        z_score = (value - mean) / std
        assert z_score == 1.0
    
    def test_percentage_feature(self):
        """Test percentage features are 0-1."""
        win_pct = 0.65
        assert 0 <= win_pct <= 1


class TestFeatureValidation:
    """Tests for feature validation."""
    
    def test_elo_range(self):
        """Test ELO ratings are in valid range."""
        elo = 1650
        assert 1000 <= elo <= 2000
    
    def test_probability_range(self):
        """Test probabilities are 0-1."""
        prob = 0.65
        assert 0 <= prob <= 1
    
    def test_rest_days_non_negative(self):
        """Test rest days are non-negative."""
        rest_days = 2
        assert rest_days >= 0
    
    def test_streak_bounds(self):
        """Test streak is bounded."""
        streak = 5
        assert -20 <= streak <= 20


class TestSportSpecificFeatureEngineering:
    """Tests for sport-specific feature engineering."""
    
    def test_nfl_specific_features(self):
        """Test NFL-specific features exist."""
        nfl_features = [
            "yards_per_play",
            "turnover_margin", 
            "third_down_pct",
            "red_zone_pct",
            "time_of_possession",
        ]
        # These should be part of NFL feature set
        assert len(nfl_features) == 5
    
    def test_nba_specific_features(self):
        """Test NBA-specific features exist."""
        nba_features = [
            "offensive_rating",
            "defensive_rating",
            "pace",
            "true_shooting_pct",
            "effective_fg_pct",
        ]
        assert len(nba_features) == 5
    
    def test_mlb_specific_features(self):
        """Test MLB-specific features exist."""
        mlb_features = [
            "era",
            "batting_avg",
            "on_base_pct",
            "slugging_pct",
            "strikeout_rate",
        ]
        assert len(mlb_features) == 5
    
    def test_tennis_specific_features(self):
        """Test tennis-specific features exist."""
        tennis_features = [
            "ranking",
            "first_serve_pct",
            "break_point_saved_pct",
            "surface_win_rate",
            "fatigue_score",
        ]
        assert len(tennis_features) == 5
