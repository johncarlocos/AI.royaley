"""
ROYALEY - ELO Rating System
Phase 2: Custom ELO implementation with sport-specific parameters
"""

import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging

from .config import SPORT_CONFIGS, default_ml_config

logger = logging.getLogger(__name__)


@dataclass
class ELOResult:
    """Result of an ELO calculation"""
    team_id: str
    old_rating: float
    new_rating: float
    rating_change: float
    expected_score: float
    actual_score: float
    game_id: str
    calculated_at: datetime


@dataclass
class TeamELO:
    """Team ELO tracking"""
    team_id: str
    sport_code: str
    current_rating: float
    peak_rating: float
    lowest_rating: float
    games_played: int
    last_updated: datetime
    rating_history: List[Tuple[datetime, float]] = field(default_factory=list)


class ELOSystem:
    """
    ELO Rating System for sports predictions.
    
    Implements the standard ELO formula with sport-specific K-factors
    and home advantage adjustments as specified in the documentation.
    
    Formula:
        New Rating = Old Rating + K × (Actual - Expected)
        Expected = 1 / (1 + 10^((Opponent Rating - Team Rating) / 400))
        Actual = 1 for win, 0.5 for tie, 0 for loss
    """
    
    def __init__(
        self,
        sport_code: str,
        base_rating: float = None,
        scale: float = None,
    ):
        """
        Initialize ELO system for a specific sport.
        
        Args:
            sport_code: Sport code (NFL, NBA, etc.)
            base_rating: Starting ELO rating (default from config)
            scale: ELO scale factor (default 400)
        """
        self.sport_code = sport_code
        self.config = SPORT_CONFIGS.get(sport_code)
        
        if self.config is None:
            raise ValueError(f"Unknown sport code: {sport_code}")
        
        self.k_factor = self.config.elo_k_factor
        self.home_advantage = self.config.elo_home_advantage
        self.base_rating = base_rating or default_ml_config.elo_base_rating
        self.scale = scale or default_ml_config.elo_scale
        
        # Team ratings storage
        self._ratings: Dict[str, TeamELO] = {}
        
        logger.info(
            f"Initialized ELO system for {sport_code}: "
            f"K={self.k_factor}, Home={self.home_advantage}"
        )
    
    def get_rating(self, team_id: str) -> float:
        """Get current ELO rating for a team"""
        if team_id not in self._ratings:
            return self.base_rating
        return self._ratings[team_id].current_rating
    
    def get_team_elo(self, team_id: str) -> TeamELO:
        """Get full ELO tracking for a team"""
        if team_id not in self._ratings:
            return TeamELO(
                team_id=team_id,
                sport_code=self.sport_code,
                current_rating=self.base_rating,
                peak_rating=self.base_rating,
                lowest_rating=self.base_rating,
                games_played=0,
                last_updated=datetime.utcnow(),
            )
        return self._ratings[team_id]
    
    def calculate_expected_score(
        self,
        team_rating: float,
        opponent_rating: float,
        is_home: bool = True,
    ) -> float:
        """
        Calculate expected score (win probability) using ELO formula.
        
        Args:
            team_rating: Team's current ELO rating
            opponent_rating: Opponent's current ELO rating
            is_home: Whether team is playing at home
            
        Returns:
            Expected score (probability of winning)
        """
        # Apply home advantage
        if is_home:
            adjusted_rating = team_rating + self.home_advantage
        else:
            adjusted_rating = team_rating
        
        # Standard ELO expected score formula
        rating_diff = opponent_rating - adjusted_rating
        expected = 1.0 / (1.0 + math.pow(10, rating_diff / self.scale))
        
        return expected
    
    def calculate_win_probability(
        self,
        home_team_id: str,
        away_team_id: str,
    ) -> Tuple[float, float]:
        """
        Calculate win probabilities for both teams.
        
        Args:
            home_team_id: Home team identifier
            away_team_id: Away team identifier
            
        Returns:
            Tuple of (home_win_prob, away_win_prob)
        """
        home_rating = self.get_rating(home_team_id)
        away_rating = self.get_rating(away_team_id)
        
        home_expected = self.calculate_expected_score(
            home_rating, away_rating, is_home=True
        )
        away_expected = self.calculate_expected_score(
            away_rating, home_rating, is_home=False
        )
        
        # Normalize to ensure they sum to 1
        total = home_expected + away_expected
        home_prob = home_expected / total
        away_prob = away_expected / total
        
        return home_prob, away_prob
    
    def get_rating_advantage(
        self,
        home_team_id: str,
        away_team_id: str,
    ) -> float:
        """
        Calculate rating advantage (home - away with home boost).
        
        Args:
            home_team_id: Home team identifier
            away_team_id: Away team identifier
            
        Returns:
            Rating advantage value
        """
        home_rating = self.get_rating(home_team_id)
        away_rating = self.get_rating(away_team_id)
        
        return (home_rating + self.home_advantage) - away_rating
    
    def update_ratings(
        self,
        home_team_id: str,
        away_team_id: str,
        home_score: int,
        away_score: int,
        game_id: str,
        game_date: datetime = None,
        margin_of_victory_adjustment: bool = True,
    ) -> Tuple[ELOResult, ELOResult]:
        """
        Update ELO ratings after a game.
        
        Args:
            home_team_id: Home team identifier
            away_team_id: Away team identifier
            home_score: Home team final score
            away_score: Away team final score
            game_id: Game identifier for tracking
            game_date: Date of the game
            margin_of_victory_adjustment: Whether to adjust K based on MOV
            
        Returns:
            Tuple of (home_elo_result, away_elo_result)
        """
        game_date = game_date or datetime.utcnow()
        
        # Get current ratings
        home_rating = self.get_rating(home_team_id)
        away_rating = self.get_rating(away_team_id)
        
        # Calculate expected scores
        home_expected = self.calculate_expected_score(
            home_rating, away_rating, is_home=True
        )
        away_expected = self.calculate_expected_score(
            away_rating, home_rating, is_home=False
        )
        
        # Determine actual scores
        if home_score > away_score:
            home_actual = 1.0
            away_actual = 0.0
        elif home_score < away_score:
            home_actual = 0.0
            away_actual = 1.0
        else:
            home_actual = 0.5
            away_actual = 0.5
        
        # Calculate K-factor (optionally adjusted for margin of victory)
        k = self.k_factor
        if margin_of_victory_adjustment:
            margin = abs(home_score - away_score)
            k = self._adjust_k_for_margin(k, margin, home_rating, away_rating)
        
        # Calculate new ratings
        home_new = home_rating + k * (home_actual - home_expected)
        away_new = away_rating + k * (away_actual - away_expected)
        
        # Update team records
        self._update_team_record(home_team_id, home_new, game_date)
        self._update_team_record(away_team_id, away_new, game_date)
        
        # Create result objects
        home_result = ELOResult(
            team_id=home_team_id,
            old_rating=home_rating,
            new_rating=home_new,
            rating_change=home_new - home_rating,
            expected_score=home_expected,
            actual_score=home_actual,
            game_id=game_id,
            calculated_at=game_date,
        )
        
        away_result = ELOResult(
            team_id=away_team_id,
            old_rating=away_rating,
            new_rating=away_new,
            rating_change=away_new - away_rating,
            expected_score=away_expected,
            actual_score=away_actual,
            game_id=game_id,
            calculated_at=game_date,
        )
        
        logger.debug(
            f"ELO update: {home_team_id} {home_rating:.1f}->{home_new:.1f}, "
            f"{away_team_id} {away_rating:.1f}->{away_new:.1f}"
        )
        
        return home_result, away_result
    
    def _adjust_k_for_margin(
        self,
        base_k: float,
        margin: int,
        winner_rating: float,
        loser_rating: float,
    ) -> float:
        """
        Adjust K-factor based on margin of victory.
        
        This prevents excessive rating changes in blowout games
        and gives more credit to underdogs winning by large margins.
        """
        # Log transformation for diminishing returns on margin
        margin_multiplier = math.log(max(margin, 1) + 1)
        
        # Rating difference adjustment (larger changes for upsets)
        rating_diff = winner_rating - loser_rating
        upset_multiplier = 1.0
        if rating_diff < 0:  # Underdog won
            upset_multiplier = 1.0 + abs(rating_diff) / 400.0
        
        # Cap the adjustment
        adjusted_k = base_k * margin_multiplier * upset_multiplier
        return min(adjusted_k, base_k * 3)  # Cap at 3x base K
    
    def _update_team_record(
        self,
        team_id: str,
        new_rating: float,
        game_date: datetime,
    ) -> None:
        """Update team's ELO tracking record"""
        if team_id not in self._ratings:
            self._ratings[team_id] = TeamELO(
                team_id=team_id,
                sport_code=self.sport_code,
                current_rating=new_rating,
                peak_rating=new_rating,
                lowest_rating=new_rating,
                games_played=1,
                last_updated=game_date,
                rating_history=[(game_date, new_rating)],
            )
        else:
            team = self._ratings[team_id]
            team.current_rating = new_rating
            team.peak_rating = max(team.peak_rating, new_rating)
            team.lowest_rating = min(team.lowest_rating, new_rating)
            team.games_played += 1
            team.last_updated = game_date
            team.rating_history.append((game_date, new_rating))
            
            # Keep only last 365 days of history
            cutoff = game_date - timedelta(days=365)
            team.rating_history = [
                (dt, r) for dt, r in team.rating_history
                if dt >= cutoff
            ]
    
    def apply_season_regression(
        self,
        regression_factor: float = 0.25,
    ) -> None:
        """
        Apply regression to the mean at season start.
        
        This prevents ratings from becoming too extreme and
        accounts for roster changes between seasons.
        
        Args:
            regression_factor: Amount to regress (0.25 = 25% toward mean)
        """
        for team_id in self._ratings:
            current = self._ratings[team_id].current_rating
            regressed = current + regression_factor * (self.base_rating - current)
            self._ratings[team_id].current_rating = regressed
            
        logger.info(
            f"Applied {regression_factor:.0%} season regression for {self.sport_code}"
        )
    
    def get_power_rankings(
        self,
        top_n: int = None,
    ) -> List[Tuple[str, float]]:
        """
        Get power rankings based on ELO ratings.
        
        Args:
            top_n: Number of teams to return (all if None)
            
        Returns:
            List of (team_id, rating) tuples sorted by rating
        """
        rankings = [
            (team_id, team.current_rating)
            for team_id, team in self._ratings.items()
        ]
        rankings.sort(key=lambda x: x[1], reverse=True)
        
        if top_n:
            return rankings[:top_n]
        return rankings
    
    def initialize_from_history(
        self,
        games: List[Dict],
        reset_ratings: bool = True,
    ) -> int:
        """
        Initialize ELO ratings from historical game data.
        
        Args:
            games: List of game dictionaries with required fields:
                   - home_team_id, away_team_id
                   - home_score, away_score
                   - game_id, game_date
            reset_ratings: Whether to reset all ratings before processing
            
        Returns:
            Number of games processed
        """
        if reset_ratings:
            self._ratings = {}
        
        # Sort games by date
        sorted_games = sorted(games, key=lambda x: x['game_date'])
        
        processed = 0
        for game in sorted_games:
            try:
                self.update_ratings(
                    home_team_id=game['home_team_id'],
                    away_team_id=game['away_team_id'],
                    home_score=game['home_score'],
                    away_score=game['away_score'],
                    game_id=game['game_id'],
                    game_date=game['game_date'],
                )
                processed += 1
            except Exception as e:
                logger.warning(f"Failed to process game {game.get('game_id')}: {e}")
        
        logger.info(f"Initialized ELO from {processed} historical games")
        return processed
    
    def export_ratings(self) -> Dict[str, Dict]:
        """Export all ratings for persistence"""
        return {
            team_id: {
                'team_id': team.team_id,
                'sport_code': team.sport_code,
                'current_rating': team.current_rating,
                'peak_rating': team.peak_rating,
                'lowest_rating': team.lowest_rating,
                'games_played': team.games_played,
                'last_updated': team.last_updated.isoformat(),
            }
            for team_id, team in self._ratings.items()
        }
    
    def import_ratings(self, ratings_data: Dict[str, Dict]) -> None:
        """Import ratings from persisted data"""
        for team_id, data in ratings_data.items():
            self._ratings[team_id] = TeamELO(
                team_id=data['team_id'],
                sport_code=data['sport_code'],
                current_rating=data['current_rating'],
                peak_rating=data['peak_rating'],
                lowest_rating=data['lowest_rating'],
                games_played=data['games_played'],
                last_updated=datetime.fromisoformat(data['last_updated']),
            )


class MultiSportELOManager:
    """
    Manager for ELO systems across all sports.
    
    Provides a unified interface for managing ELO ratings
    across all 10 supported sports leagues.
    """
    
    def __init__(self):
        """Initialize ELO systems for all sports"""
        self._systems: Dict[str, ELOSystem] = {}
        
        for sport_code in SPORT_CONFIGS:
            self._systems[sport_code] = ELOSystem(sport_code)
        
        logger.info(f"Initialized ELO manager for {len(self._systems)} sports")
    
    def get_system(self, sport_code: str) -> ELOSystem:
        """Get ELO system for a specific sport"""
        if sport_code not in self._systems:
            raise ValueError(f"Unknown sport code: {sport_code}")
        return self._systems[sport_code]
    
    def get_rating(self, sport_code: str, team_id: str) -> float:
        """Get team rating for a specific sport"""
        return self.get_system(sport_code).get_rating(team_id)
    
    def calculate_win_probability(
        self,
        sport_code: str,
        home_team_id: str,
        away_team_id: str,
    ) -> Tuple[float, float]:
        """Calculate win probabilities for a matchup"""
        return self.get_system(sport_code).calculate_win_probability(
            home_team_id, away_team_id
        )
    
    def update_ratings(
        self,
        sport_code: str,
        home_team_id: str,
        away_team_id: str,
        home_score: int,
        away_score: int,
        game_id: str,
        game_date: datetime = None,
    ) -> Tuple[ELOResult, ELOResult]:
        """Update ratings after a game"""
        return self.get_system(sport_code).update_ratings(
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            home_score=home_score,
            away_score=away_score,
            game_id=game_id,
            game_date=game_date,
        )
    
    def export_all_ratings(self) -> Dict[str, Dict]:
        """Export ratings for all sports"""
        return {
            sport_code: system.export_ratings()
            for sport_code, system in self._systems.items()
        }
    
    def import_all_ratings(self, all_ratings: Dict[str, Dict]) -> None:
        """Import ratings for all sports"""
        for sport_code, ratings in all_ratings.items():
            if sport_code in self._systems:
                self._systems[sport_code].import_ratings(ratings)


# =============================================================================
# ELORATING CLASS FOR INDIVIDUAL RATING OPERATIONS
# =============================================================================

@dataclass
class ELORating:
    """
    Individual ELO rating handler for team/player ratings.
    Used for feature engineering and prediction calculations.
    """
    rating: float = 1500.0
    k_factor: float = 20.0
    games_played: int = 0
    wins: int = 0
    losses: int = 0
    
    def expected_score(self, opponent_rating: float) -> float:
        """Calculate expected score against opponent."""
        return 1.0 / (1.0 + 10 ** ((opponent_rating - self.rating) / 400))
    
    def update(self, opponent_rating: float, actual_score: float) -> float:
        """
        Update rating based on game result.
        
        Args:
            opponent_rating: Opponent's ELO rating
            actual_score: 1.0 for win, 0.5 for draw, 0.0 for loss
            
        Returns:
            New rating after update
        """
        expected = self.expected_score(opponent_rating)
        self.rating += self.k_factor * (actual_score - expected)
        self.games_played += 1
        if actual_score > 0.5:
            self.wins += 1
        elif actual_score < 0.5:
            self.losses += 1
        return self.rating
    
    def win_probability(self, opponent_rating: float) -> float:
        """Calculate probability of winning against opponent."""
        return self.expected_score(opponent_rating)
    
    def rating_change(self, opponent_rating: float, won: bool) -> float:
        """Calculate rating change for a result without applying it."""
        actual_score = 1.0 if won else 0.0
        expected = self.expected_score(opponent_rating)
        return self.k_factor * (actual_score - expected)
    
    def reset(self) -> None:
        """Reset to default rating."""
        self.rating = 1500.0
        self.games_played = 0
        self.wins = 0
        self.losses = 0


# =============================================================================
# EXPORT ALIASES FOR BACKWARD COMPATIBILITY
# =============================================================================

ELORatingSystem = ELOSystem
"""Alias for ELOSystem for backward compatibility."""

__all__ = [
    # Main Classes
    "ELOSystem",
    "ELORatingSystem",  # Alias
    "ELORating",  # Individual rating class
    "MultiSportELOManager",
    # Data Classes
    "ELOResult",
    "TeamELO",
]
