"""
LOYALEY - Feature Engineering Pipeline
Phase 2: Comprehensive feature engineering for all 10 sports

This module generates 60-85 features per sport across categories:
- Team Performance (ELO, ratings, efficiency)
- Recent Form (streaks, momentum, rolling stats)
- Rest & Travel (fatigue, back-to-back, travel distance)
- Head-to-Head (historical matchup data)
- Line Movement (spread changes, steam moves)
- Weather (outdoor sports)
- Injuries (impact scores)
- Situational (home/away, rivalry, prime time)
- Advanced Metrics (sport-specific advanced stats)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import logging
import math

from .config import SPORT_CONFIGS, SportConfig, default_ml_config, BetType
from .elo_rating import ELOSystem, MultiSportELOManager, ELORating, ELORatingSystem

logger = logging.getLogger(__name__)


@dataclass
class FeatureSet:
    """Container for computed features"""
    game_id: str
    sport_code: str
    computed_at: datetime
    features: Dict[str, float]
    feature_names: List[str]
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame row"""
        return pd.DataFrame([self.features])
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array in consistent order"""
        return np.array([self.features[name] for name in self.feature_names])


@dataclass
class GameContext:
    """Context information for feature computation"""
    game_id: str
    sport_code: str
    home_team_id: str
    away_team_id: str
    game_date: datetime
    venue_id: Optional[str] = None
    is_playoff: bool = False
    week: Optional[int] = None
    season: Optional[int] = None
    
    # Pre-fetched data for efficiency
    home_team_stats: Dict = field(default_factory=dict)
    away_team_stats: Dict = field(default_factory=dict)
    home_recent_games: List[Dict] = field(default_factory=list)
    away_recent_games: List[Dict] = field(default_factory=list)
    h2h_games: List[Dict] = field(default_factory=list)
    odds_data: Dict = field(default_factory=dict)
    weather_data: Dict = field(default_factory=dict)
    injury_data: Dict = field(default_factory=dict)


class BaseFeatureGenerator(ABC):
    """Base class for sport-specific feature generators"""
    
    def __init__(self, sport_code: str, elo_system: ELOSystem = None):
        self.sport_code = sport_code
        self.config = SPORT_CONFIGS[sport_code]
        self.elo_system = elo_system or ELOSystem(sport_code)
        self.rolling_windows = self.config.rolling_windows
    
    @abstractmethod
    def generate_sport_specific_features(
        self,
        context: GameContext,
    ) -> Dict[str, float]:
        """Generate sport-specific features (to be implemented by subclasses)"""
        pass
    
    def generate_base_features(
        self,
        context: GameContext,
    ) -> Dict[str, float]:
        """Generate common features applicable to all sports"""
        features = {}
        
        # ELO-based features
        features.update(self._generate_elo_features(context))
        
        # Recent form features
        features.update(self._generate_form_features(context))
        
        # Rest and schedule features
        features.update(self._generate_rest_features(context))
        
        # Head-to-head features
        features.update(self._generate_h2h_features(context))
        
        # Line movement features
        features.update(self._generate_line_movement_features(context))
        
        # Situational features
        features.update(self._generate_situational_features(context))
        
        return features
    
    def _generate_elo_features(self, context: GameContext) -> Dict[str, float]:
        """Generate ELO-based features"""
        home_elo = self.elo_system.get_rating(context.home_team_id)
        away_elo = self.elo_system.get_rating(context.away_team_id)
        
        home_prob, away_prob = self.elo_system.calculate_win_probability(
            context.home_team_id, context.away_team_id
        )
        
        return {
            'home_elo': home_elo,
            'away_elo': away_elo,
            'elo_diff': home_elo - away_elo,
            'elo_diff_with_home_adv': (
                home_elo + self.elo_system.home_advantage - away_elo
            ),
            'home_elo_win_prob': home_prob,
            'away_elo_win_prob': away_prob,
            'elo_rating_ratio': home_elo / max(away_elo, 1),
        }
    
    def _generate_form_features(self, context: GameContext) -> Dict[str, float]:
        """Generate recent form features"""
        features = {}
        
        for prefix, games in [
            ('home', context.home_recent_games),
            ('away', context.away_recent_games)
        ]:
            for window in [5, 10]:
                window_games = games[:window] if games else []
                
                if window_games:
                    wins = sum(1 for g in window_games if g.get('won', False))
                    margins = [g.get('margin', 0) for g in window_games]
                    
                    features[f'{prefix}_last_{window}_wins'] = wins
                    features[f'{prefix}_last_{window}_win_pct'] = wins / window
                    features[f'{prefix}_last_{window}_avg_margin'] = np.mean(margins)
                    features[f'{prefix}_last_{window}_std_margin'] = np.std(margins) if len(margins) > 1 else 0
                else:
                    features[f'{prefix}_last_{window}_wins'] = 0
                    features[f'{prefix}_last_{window}_win_pct'] = 0.5
                    features[f'{prefix}_last_{window}_avg_margin'] = 0
                    features[f'{prefix}_last_{window}_std_margin'] = 0
            
            # Streak features
            streak = self._calculate_streak(games)
            features[f'{prefix}_win_streak'] = max(streak, 0)
            features[f'{prefix}_lose_streak'] = abs(min(streak, 0))
            
            # Momentum score (exponentially weighted)
            momentum = self._calculate_momentum(games)
            features[f'{prefix}_momentum'] = momentum
            
            # ATS (Against The Spread) record
            ats_record = self._calculate_ats_record(games[:10])
            features[f'{prefix}_ats_last_10'] = ats_record
            
            # Over/Under tendency
            ou_record = self._calculate_ou_record(games[:10])
            features[f'{prefix}_over_last_10'] = ou_record
        
        # Combined form differential
        features['form_diff_5'] = (
            features['home_last_5_win_pct'] - features['away_last_5_win_pct']
        )
        features['momentum_diff'] = (
            features['home_momentum'] - features['away_momentum']
        )
        
        return features
    
    def _generate_rest_features(self, context: GameContext) -> Dict[str, float]:
        """Generate rest and fatigue features"""
        features = {}
        
        for prefix, games in [
            ('home', context.home_recent_games),
            ('away', context.away_recent_games)
        ]:
            if games:
                last_game = games[0]
                last_game_date = last_game.get('game_date')
                if last_game_date:
                    if isinstance(last_game_date, str):
                        last_game_date = datetime.fromisoformat(last_game_date)
                    rest_days = (context.game_date - last_game_date).days
                else:
                    rest_days = 7  # Default assumption
            else:
                rest_days = 7
            
            features[f'{prefix}_rest_days'] = rest_days
            features[f'{prefix}_back_to_back'] = 1 if rest_days <= 1 else 0
            
            # Games in last N days
            for days in [7, 14]:
                cutoff = context.game_date - timedelta(days=days)
                games_in_window = sum(
                    1 for g in games
                    if g.get('game_date') and 
                    (datetime.fromisoformat(g['game_date']) if isinstance(g['game_date'], str) else g['game_date']) >= cutoff
                )
                features[f'{prefix}_games_last_{days}'] = games_in_window
        
        # Rest advantage
        features['rest_advantage'] = (
            features['home_rest_days'] - features['away_rest_days']
        )
        features['both_back_to_back'] = (
            features['home_back_to_back'] * features['away_back_to_back']
        )
        features['home_rest_disadvantage'] = (
            1 if features['rest_advantage'] < 0 else 0
        )
        
        return features
    
    def _generate_h2h_features(self, context: GameContext) -> Dict[str, float]:
        """Generate head-to-head features"""
        features = {}
        h2h = context.h2h_games
        
        if h2h:
            # Home team's record vs away team
            home_wins = sum(
                1 for g in h2h 
                if (g.get('home_team_id') == context.home_team_id and g.get('home_won', False)) or
                   (g.get('away_team_id') == context.home_team_id and not g.get('home_won', True))
            )
            total_games = len(h2h)
            
            features['h2h_total_games'] = total_games
            features['h2h_home_wins'] = home_wins
            features['h2h_home_win_pct'] = home_wins / total_games if total_games > 0 else 0.5
            
            # Recent H2H (last 5)
            recent_h2h = h2h[:5]
            if recent_h2h:
                recent_wins = sum(
                    1 for g in recent_h2h
                    if (g.get('home_team_id') == context.home_team_id and g.get('home_won', False)) or
                       (g.get('away_team_id') == context.home_team_id and not g.get('home_won', True))
                )
                features['h2h_last_5_wins'] = recent_wins
            else:
                features['h2h_last_5_wins'] = 0
            
            # Average margin in H2H
            margins = []
            for g in h2h:
                if g.get('home_team_id') == context.home_team_id:
                    margins.append(g.get('home_score', 0) - g.get('away_score', 0))
                else:
                    margins.append(g.get('away_score', 0) - g.get('home_score', 0))
            features['h2h_avg_margin'] = np.mean(margins) if margins else 0
        else:
            features['h2h_total_games'] = 0
            features['h2h_home_wins'] = 0
            features['h2h_home_win_pct'] = 0.5
            features['h2h_last_5_wins'] = 0
            features['h2h_avg_margin'] = 0
        
        return features
    
    def _generate_line_movement_features(
        self,
        context: GameContext,
    ) -> Dict[str, float]:
        """Generate line movement features"""
        features = {}
        odds = context.odds_data
        
        # Spread features
        opening_spread = odds.get('opening_spread', 0)
        current_spread = odds.get('current_spread', 0)
        features['opening_spread'] = opening_spread
        features['current_spread'] = current_spread
        features['spread_movement'] = current_spread - opening_spread
        features['spread_moved_toward_home'] = (
            1 if features['spread_movement'] < 0 else 0
        )
        
        # Total features
        opening_total = odds.get('opening_total', 0)
        current_total = odds.get('current_total', 0)
        features['opening_total'] = opening_total
        features['current_total'] = current_total
        features['total_movement'] = current_total - opening_total
        
        # Steam move detection (significant movement in short time)
        features['steam_move_spread'] = odds.get('steam_move_spread', 0)
        features['steam_move_total'] = odds.get('steam_move_total', 0)
        
        # Reverse line movement
        public_pct_home = odds.get('public_bet_pct_home', 0.5)
        features['public_bet_pct_home'] = public_pct_home
        features['reverse_line_movement'] = (
            1 if (public_pct_home > 0.6 and features['spread_movement'] > 0) or
                 (public_pct_home < 0.4 and features['spread_movement'] < 0)
            else 0
        )
        
        # Moneyline features
        features['home_ml_odds'] = odds.get('home_ml', -110)
        features['away_ml_odds'] = odds.get('away_ml', -110)
        features['home_implied_prob'] = self._american_to_prob(
            features['home_ml_odds']
        )
        features['away_implied_prob'] = self._american_to_prob(
            features['away_ml_odds']
        )
        
        return features
    
    def _generate_situational_features(
        self,
        context: GameContext,
    ) -> Dict[str, float]:
        """Generate situational features"""
        features = {}
        
        features['is_home'] = 1  # Always 1 for home team perspective
        features['is_playoff'] = 1 if context.is_playoff else 0
        features['week'] = context.week or 0
        
        # Day of week (0=Monday)
        features['day_of_week'] = context.game_date.weekday()
        features['is_weekend'] = 1 if features['day_of_week'] >= 5 else 0
        
        # Month (for seasonal patterns)
        features['month'] = context.game_date.month
        
        # Prime time indicator (evening games)
        hour = context.game_date.hour
        features['is_prime_time'] = 1 if 19 <= hour <= 21 else 0
        features['is_early_game'] = 1 if hour < 15 else 0
        
        # Divisional/Conference games from team stats
        home_stats = context.home_team_stats
        away_stats = context.away_team_stats
        features['is_divisional'] = (
            1 if home_stats.get('division') == away_stats.get('division') else 0
        )
        features['is_conference'] = (
            1 if home_stats.get('conference') == away_stats.get('conference') else 0
        )
        
        return features
    
    def _calculate_streak(self, games: List[Dict]) -> int:
        """Calculate current win/loss streak (positive=wins, negative=losses)"""
        if not games:
            return 0
        
        streak = 0
        is_winning = games[0].get('won', False)
        
        for game in games:
            if game.get('won', False) == is_winning:
                streak += 1 if is_winning else -1
            else:
                break
        
        return streak
    
    def _calculate_momentum(
        self,
        games: List[Dict],
        decay: float = None,
    ) -> float:
        """Calculate exponentially weighted momentum score"""
        if not games:
            return 0.0
        
        decay = decay or default_ml_config.momentum_decay_factor
        momentum = 0.0
        weight = 1.0
        
        for game in games[:10]:  # Use last 10 games
            result = 1 if game.get('won', False) else -1
            margin_factor = min(abs(game.get('margin', 0)) / 20, 1.5)  # Cap effect
            momentum += weight * result * (1 + margin_factor * 0.2)
            weight *= decay
        
        return momentum
    
    def _calculate_ats_record(self, games: List[Dict]) -> float:
        """Calculate against-the-spread record"""
        if not games:
            return 0.5
        
        covered = sum(1 for g in games if g.get('covered_spread', False))
        return covered / len(games)
    
    def _calculate_ou_record(self, games: List[Dict]) -> float:
        """Calculate over/under record (proportion of overs)"""
        if not games:
            return 0.5
        
        overs = sum(1 for g in games if g.get('went_over', False))
        return overs / len(games)
    
    def _american_to_prob(self, odds: int) -> float:
        """Convert American odds to implied probability"""
        if odds == 0:
            return 0.5
        if odds > 0:
            return 100 / (odds + 100)
        else:
            return abs(odds) / (abs(odds) + 100)


class FootballFeatureGenerator(BaseFeatureGenerator):
    """Feature generator for football (NFL, NCAAF, CFL)"""
    
    def generate_sport_specific_features(
        self,
        context: GameContext,
    ) -> Dict[str, float]:
        """Generate football-specific features"""
        features = {}
        
        for prefix, stats in [
            ('home', context.home_team_stats),
            ('away', context.away_team_stats)
        ]:
            # Offensive efficiency
            features[f'{prefix}_ppg'] = stats.get('points_per_game', 0)
            features[f'{prefix}_ypg'] = stats.get('yards_per_game', 0)
            features[f'{prefix}_pass_ypg'] = stats.get('passing_yards_per_game', 0)
            features[f'{prefix}_rush_ypg'] = stats.get('rushing_yards_per_game', 0)
            features[f'{prefix}_top'] = stats.get('time_of_possession', 30)
            
            # Efficiency metrics
            features[f'{prefix}_third_down_pct'] = stats.get('third_down_pct', 0.4)
            features[f'{prefix}_red_zone_pct'] = stats.get('red_zone_pct', 0.5)
            features[f'{prefix}_turnover_margin'] = stats.get('turnover_margin', 0)
            
            # Defensive metrics
            features[f'{prefix}_ppg_allowed'] = stats.get('points_allowed_per_game', 0)
            features[f'{prefix}_ypg_allowed'] = stats.get('yards_allowed_per_game', 0)
            features[f'{prefix}_sacks'] = stats.get('sacks_per_game', 0)
            features[f'{prefix}_takeaways'] = stats.get('takeaways_per_game', 0)
            
            # Advanced metrics
            features[f'{prefix}_epa_per_play'] = stats.get('epa_per_play', 0)
            features[f'{prefix}_success_rate'] = stats.get('success_rate', 0.5)
        
        # Differentials
        features['ppg_diff'] = features['home_ppg'] - features['away_ppg']
        features['ypg_diff'] = features['home_ypg'] - features['away_ypg']
        features['defensive_diff'] = (
            features['away_ppg_allowed'] - features['home_ppg_allowed']
        )
        
        # Matchup features
        features['home_pass_vs_away_pass_def'] = (
            features['home_pass_ypg'] - 
            context.away_team_stats.get('passing_yards_allowed_per_game', 200)
        )
        features['home_rush_vs_away_rush_def'] = (
            features['home_rush_ypg'] - 
            context.away_team_stats.get('rushing_yards_allowed_per_game', 100)
        )
        
        # Weather impact (outdoor games)
        weather = context.weather_data
        if weather and not weather.get('is_dome', True):
            features['temperature'] = weather.get('temperature', 70)
            features['wind_speed'] = weather.get('wind_speed', 0)
            features['precipitation_prob'] = weather.get('precipitation_prob', 0)
            features['weather_impact'] = self._calculate_weather_impact(weather)
        else:
            features['temperature'] = 70
            features['wind_speed'] = 0
            features['precipitation_prob'] = 0
            features['weather_impact'] = 0
        
        return features
    
    def _calculate_weather_impact(self, weather: Dict) -> float:
        """Calculate weather impact score for football"""
        impact = 0.0
        
        # Temperature impact (extreme cold or heat)
        temp = weather.get('temperature', 70)
        if temp < 32:
            impact += (32 - temp) / 20  # Cold weather effect
        elif temp > 90:
            impact += (temp - 90) / 20  # Heat effect
        
        # Wind impact (affects passing)
        wind = weather.get('wind_speed', 0)
        if wind > 15:
            impact += (wind - 15) / 10
        
        # Precipitation impact
        precip = weather.get('precipitation_prob', 0)
        impact += precip * 0.5
        
        return min(impact, 2.0)  # Cap at 2.0


class BasketballFeatureGenerator(BaseFeatureGenerator):
    """Feature generator for basketball (NBA, NCAAB, WNBA)"""
    
    def generate_sport_specific_features(
        self,
        context: GameContext,
    ) -> Dict[str, float]:
        """Generate basketball-specific features"""
        features = {}
        
        for prefix, stats in [
            ('home', context.home_team_stats),
            ('away', context.away_team_stats)
        ]:
            # Offensive ratings
            features[f'{prefix}_offensive_rating'] = stats.get('offensive_rating', 110)
            features[f'{prefix}_defensive_rating'] = stats.get('defensive_rating', 110)
            features[f'{prefix}_net_rating'] = (
                features[f'{prefix}_offensive_rating'] - 
                features[f'{prefix}_defensive_rating']
            )
            features[f'{prefix}_pace'] = stats.get('pace', 100)
            
            # Shooting efficiency
            features[f'{prefix}_efg_pct'] = stats.get('effective_fg_pct', 0.50)
            features[f'{prefix}_ts_pct'] = stats.get('true_shooting_pct', 0.55)
            features[f'{prefix}_three_pt_pct'] = stats.get('three_point_pct', 0.35)
            features[f'{prefix}_three_pt_rate'] = stats.get('three_point_rate', 0.35)
            
            # Rebounding
            features[f'{prefix}_orb_pct'] = stats.get('offensive_rebound_pct', 0.25)
            features[f'{prefix}_drb_pct'] = stats.get('defensive_rebound_pct', 0.75)
            features[f'{prefix}_trb_pct'] = stats.get('total_rebound_pct', 0.50)
            
            # Ball movement
            features[f'{prefix}_ast_pct'] = stats.get('assist_pct', 0.60)
            features[f'{prefix}_tov_pct'] = stats.get('turnover_pct', 0.13)
            features[f'{prefix}_ast_to_ratio'] = (
                features[f'{prefix}_ast_pct'] / 
                max(features[f'{prefix}_tov_pct'], 0.01)
            )
            
            # Free throws
            features[f'{prefix}_ft_rate'] = stats.get('free_throw_rate', 0.25)
            features[f'{prefix}_ft_pct'] = stats.get('free_throw_pct', 0.75)
            
            # Defense
            features[f'{prefix}_stl_pct'] = stats.get('steal_pct', 0.08)
            features[f'{prefix}_blk_pct'] = stats.get('block_pct', 0.05)
            features[f'{prefix}_opp_efg_pct'] = stats.get('opponent_efg_pct', 0.50)
        
        # Pace-adjusted scoring
        avg_pace = (features['home_pace'] + features['away_pace']) / 2
        features['projected_pace'] = avg_pace
        features['projected_possessions'] = avg_pace * 0.96  # Game adjustment
        
        # Four factors differentials
        features['efg_diff'] = features['home_efg_pct'] - features['away_efg_pct']
        features['tov_diff'] = features['away_tov_pct'] - features['home_tov_pct']
        features['orb_diff'] = features['home_orb_pct'] - features['away_orb_pct']
        features['ft_rate_diff'] = features['home_ft_rate'] - features['away_ft_rate']
        
        # Net rating differential
        features['net_rating_diff'] = (
            features['home_net_rating'] - features['away_net_rating']
        )
        
        # Matchup-specific
        features['pace_mismatch'] = abs(
            features['home_pace'] - features['away_pace']
        )
        features['three_pt_battle'] = (
            features['home_three_pt_pct'] - features['away_opp_efg_pct']
        )
        
        return features


class HockeyFeatureGenerator(BaseFeatureGenerator):
    """Feature generator for hockey (NHL)"""
    
    def generate_sport_specific_features(
        self,
        context: GameContext,
    ) -> Dict[str, float]:
        """Generate hockey-specific features"""
        features = {}
        
        for prefix, stats in [
            ('home', context.home_team_stats),
            ('away', context.away_team_stats)
        ]:
            # Goals and scoring
            features[f'{prefix}_goals_per_game'] = stats.get('goals_per_game', 3.0)
            features[f'{prefix}_goals_against_per_game'] = stats.get('goals_against_per_game', 3.0)
            features[f'{prefix}_goal_diff'] = (
                features[f'{prefix}_goals_per_game'] - 
                features[f'{prefix}_goals_against_per_game']
            )
            
            # Shot metrics
            features[f'{prefix}_shots_per_game'] = stats.get('shots_per_game', 30)
            features[f'{prefix}_shots_against_per_game'] = stats.get('shots_against_per_game', 30)
            features[f'{prefix}_shooting_pct'] = stats.get('shooting_pct', 0.10)
            features[f'{prefix}_save_pct'] = stats.get('save_pct', 0.90)
            
            # Advanced metrics
            features[f'{prefix}_corsi_pct'] = stats.get('corsi_pct', 0.50)
            features[f'{prefix}_fenwick_pct'] = stats.get('fenwick_pct', 0.50)
            features[f'{prefix}_expected_goals_for'] = stats.get('expected_goals_for', 2.8)
            features[f'{prefix}_expected_goals_against'] = stats.get('expected_goals_against', 2.8)
            features[f'{prefix}_xg_diff'] = (
                features[f'{prefix}_expected_goals_for'] - 
                features[f'{prefix}_expected_goals_against']
            )
            
            # Special teams
            features[f'{prefix}_power_play_pct'] = stats.get('power_play_pct', 0.20)
            features[f'{prefix}_penalty_kill_pct'] = stats.get('penalty_kill_pct', 0.80)
            features[f'{prefix}_pp_opportunities'] = stats.get('pp_opportunities_per_game', 3)
            
            # Goaltending
            features[f'{prefix}_goalie_rating'] = stats.get('goalie_rating', 0.91)
            features[f'{prefix}_quality_starts_pct'] = stats.get('quality_starts_pct', 0.50)
        
        # Differentials
        features['goals_diff'] = (
            features['home_goals_per_game'] - features['away_goals_per_game']
        )
        features['corsi_diff'] = (
            features['home_corsi_pct'] - features['away_corsi_pct']
        )
        features['xg_differential'] = (
            features['home_xg_diff'] - features['away_xg_diff']
        )
        features['special_teams_diff'] = (
            (features['home_power_play_pct'] + features['home_penalty_kill_pct']) -
            (features['away_power_play_pct'] + features['away_penalty_kill_pct'])
        )
        
        return features


class BaseballFeatureGenerator(BaseFeatureGenerator):
    """Feature generator for baseball (MLB)"""
    
    def generate_sport_specific_features(
        self,
        context: GameContext,
    ) -> Dict[str, float]:
        """Generate baseball-specific features"""
        features = {}
        
        for prefix, stats in [
            ('home', context.home_team_stats),
            ('away', context.away_team_stats)
        ]:
            # Team batting
            features[f'{prefix}_runs_per_game'] = stats.get('runs_per_game', 4.5)
            features[f'{prefix}_batting_avg'] = stats.get('batting_avg', 0.250)
            features[f'{prefix}_obp'] = stats.get('on_base_pct', 0.320)
            features[f'{prefix}_slg'] = stats.get('slugging_pct', 0.400)
            features[f'{prefix}_ops'] = features[f'{prefix}_obp'] + features[f'{prefix}_slg']
            features[f'{prefix}_woba'] = stats.get('woba', 0.320)
            features[f'{prefix}_wrc_plus'] = stats.get('wrc_plus', 100)
            
            # Power and contact
            features[f'{prefix}_hr_per_game'] = stats.get('home_runs_per_game', 1.2)
            features[f'{prefix}_strikeout_pct'] = stats.get('strikeout_pct', 0.22)
            features[f'{prefix}_walk_pct'] = stats.get('walk_pct', 0.08)
            features[f'{prefix}_iso'] = features[f'{prefix}_slg'] - features[f'{prefix}_batting_avg']
            
            # Team pitching
            features[f'{prefix}_era'] = stats.get('era', 4.00)
            features[f'{prefix}_whip'] = stats.get('whip', 1.30)
            features[f'{prefix}_fip'] = stats.get('fip', 4.00)
            features[f'{prefix}_k_per_9'] = stats.get('strikeouts_per_9', 8.5)
            features[f'{prefix}_bb_per_9'] = stats.get('walks_per_9', 3.0)
            features[f'{prefix}_hr_per_9'] = stats.get('home_runs_per_9', 1.2)
            
            # Defense
            features[f'{prefix}_fielding_pct'] = stats.get('fielding_pct', 0.985)
            features[f'{prefix}_def_efficiency'] = stats.get('defensive_efficiency', 0.700)
            
            # Bullpen
            features[f'{prefix}_bullpen_era'] = stats.get('bullpen_era', 4.00)
            features[f'{prefix}_save_pct'] = stats.get('save_pct', 0.65)
        
        # Starting pitcher features
        sp_stats = context.home_team_stats.get('starting_pitcher', {})
        features['home_sp_era'] = sp_stats.get('era', 4.00)
        features['home_sp_whip'] = sp_stats.get('whip', 1.30)
        features['home_sp_k_per_9'] = sp_stats.get('k_per_9', 8.5)
        features['home_sp_war'] = sp_stats.get('war', 2.0)
        
        sp_stats = context.away_team_stats.get('starting_pitcher', {})
        features['away_sp_era'] = sp_stats.get('era', 4.00)
        features['away_sp_whip'] = sp_stats.get('whip', 1.30)
        features['away_sp_k_per_9'] = sp_stats.get('k_per_9', 8.5)
        features['away_sp_war'] = sp_stats.get('war', 2.0)
        
        # Differentials
        features['run_diff'] = (
            features['home_runs_per_game'] - features['away_runs_per_game']
        )
        features['ops_diff'] = features['home_ops'] - features['away_ops']
        features['pitching_diff'] = features['away_era'] - features['home_era']
        features['sp_era_diff'] = features['away_sp_era'] - features['home_sp_era']
        
        return features


class TennisFeatureGenerator(BaseFeatureGenerator):
    """Feature generator for tennis (ATP, WTA)"""
    
    def generate_sport_specific_features(
        self,
        context: GameContext,
    ) -> Dict[str, float]:
        """Generate tennis-specific features"""
        features = {}
        
        # In tennis, home/away maps to player_1/player_2
        for prefix, stats in [
            ('p1', context.home_team_stats),  # Player 1
            ('p2', context.away_team_stats)   # Player 2
        ]:
            # Rankings
            features[f'{prefix}_ranking'] = stats.get('ranking', 100)
            features[f'{prefix}_ranking_points'] = stats.get('ranking_points', 1000)
            
            # Serve stats
            features[f'{prefix}_first_serve_pct'] = stats.get('first_serve_pct', 0.60)
            features[f'{prefix}_first_serve_win_pct'] = stats.get('first_serve_win_pct', 0.70)
            features[f'{prefix}_second_serve_win_pct'] = stats.get('second_serve_win_pct', 0.50)
            features[f'{prefix}_ace_pct'] = stats.get('ace_pct', 0.08)
            features[f'{prefix}_double_fault_pct'] = stats.get('double_fault_pct', 0.03)
            features[f'{prefix}_service_games_won_pct'] = stats.get('service_games_won_pct', 0.80)
            
            # Return stats
            features[f'{prefix}_first_return_win_pct'] = stats.get('first_return_win_pct', 0.30)
            features[f'{prefix}_second_return_win_pct'] = stats.get('second_return_win_pct', 0.50)
            features[f'{prefix}_break_points_converted'] = stats.get('break_points_converted', 0.40)
            features[f'{prefix}_return_games_won_pct'] = stats.get('return_games_won_pct', 0.25)
            
            # Overall performance
            features[f'{prefix}_win_pct_ytd'] = stats.get('win_pct_ytd', 0.50)
            features[f'{prefix}_sets_won_pct'] = stats.get('sets_won_pct', 0.50)
            features[f'{prefix}_tiebreaks_won_pct'] = stats.get('tiebreaks_won_pct', 0.50)
            
            # Surface-specific
            surface = context.venue_id or 'hard'  # Reuse venue_id for surface
            features[f'{prefix}_surface_win_pct'] = stats.get(f'{surface}_win_pct', 0.50)
        
        # Ranking differential
        features['ranking_diff'] = features['p2_ranking'] - features['p1_ranking']
        features['ranking_points_diff'] = (
            features['p1_ranking_points'] - features['p2_ranking_points']
        )
        
        # Serve differential
        features['serve_rating_diff'] = (
            (features['p1_first_serve_win_pct'] * 0.6 + 
             features['p1_second_serve_win_pct'] * 0.4) -
            (features['p2_first_serve_win_pct'] * 0.6 + 
             features['p2_second_serve_win_pct'] * 0.4)
        )
        
        # Return differential
        features['return_rating_diff'] = (
            features['p1_return_games_won_pct'] - features['p2_return_games_won_pct']
        )
        
        return features


class FeatureEngineer:
    """
    Main feature engineering orchestrator.
    
    Coordinates feature generation across all sports using
    sport-specific generators.
    """
    
    def __init__(self, elo_manager: MultiSportELOManager = None):
        """Initialize feature engineer with sport generators"""
        self.elo_manager = elo_manager or MultiSportELOManager()
        self._generators: Dict[str, BaseFeatureGenerator] = {}
        
        # Initialize sport-specific generators
        for sport_code in SPORT_CONFIGS:
            elo_system = self.elo_manager.get_system(sport_code)
            
            if sport_code in ['NFL', 'NCAAF', 'CFL']:
                self._generators[sport_code] = FootballFeatureGenerator(
                    sport_code, elo_system
                )
            elif sport_code in ['NBA', 'NCAAB', 'WNBA']:
                self._generators[sport_code] = BasketballFeatureGenerator(
                    sport_code, elo_system
                )
            elif sport_code == 'NHL':
                self._generators[sport_code] = HockeyFeatureGenerator(
                    sport_code, elo_system
                )
            elif sport_code == 'MLB':
                self._generators[sport_code] = BaseballFeatureGenerator(
                    sport_code, elo_system
                )
            elif sport_code in ['ATP', 'WTA']:
                self._generators[sport_code] = TennisFeatureGenerator(
                    sport_code, elo_system
                )
        
        logger.info(f"Initialized feature engineer for {len(self._generators)} sports")
    
    def generate_features(self, context: GameContext) -> FeatureSet:
        """
        Generate all features for a game.
        
        Args:
            context: Game context with all required data
            
        Returns:
            FeatureSet with all computed features
        """
        sport_code = context.sport_code
        generator = self._generators.get(sport_code)
        
        if generator is None:
            raise ValueError(f"No generator for sport: {sport_code}")
        
        # Generate base features (common to all sports)
        features = generator.generate_base_features(context)
        
        # Generate sport-specific features
        sport_features = generator.generate_sport_specific_features(context)
        features.update(sport_features)
        
        # Add injury impact features
        injury_features = self._generate_injury_features(context)
        features.update(injury_features)
        
        # Sort feature names for consistent ordering
        feature_names = sorted(features.keys())
        
        return FeatureSet(
            game_id=context.game_id,
            sport_code=sport_code,
            computed_at=datetime.utcnow(),
            features=features,
            feature_names=feature_names,
        )
    
    def _generate_injury_features(self, context: GameContext) -> Dict[str, float]:
        """Generate injury-related features"""
        features = {}
        injury_data = context.injury_data
        
        for prefix in ['home', 'away']:
            team_injuries = injury_data.get(prefix, {})
            features[f'{prefix}_key_players_out'] = team_injuries.get('key_players_out', 0)
            features[f'{prefix}_injury_impact_score'] = team_injuries.get('impact_score', 0)
            features[f'{prefix}_star_player_out'] = team_injuries.get('star_out', 0)
        
        features['injury_advantage'] = (
            features['away_injury_impact_score'] - 
            features['home_injury_impact_score']
        )
        
        return features
    
    def generate_features_batch(
        self,
        contexts: List[GameContext],
    ) -> pd.DataFrame:
        """
        Generate features for multiple games efficiently.
        
        Args:
            contexts: List of game contexts
            
        Returns:
            DataFrame with all features
        """
        all_features = []
        
        for context in contexts:
            try:
                feature_set = self.generate_features(context)
                row = feature_set.features.copy()
                row['game_id'] = context.game_id
                row['sport_code'] = context.sport_code
                all_features.append(row)
            except Exception as e:
                logger.warning(f"Failed to generate features for {context.game_id}: {e}")
        
        if all_features:
            df = pd.DataFrame(all_features)
            return df
        
        return pd.DataFrame()
    
    def get_feature_names(self, sport_code: str) -> List[str]:
        """Get list of feature names for a sport"""
        generator = self._generators.get(sport_code)
        if not generator:
            return []
        
        # Generate dummy features to get names
        dummy_context = GameContext(
            game_id='dummy',
            sport_code=sport_code,
            home_team_id='home',
            away_team_id='away',
            game_date=datetime.utcnow(),
        )
        
        feature_set = self.generate_features(dummy_context)
        return feature_set.feature_names
    
    def get_feature_count(self, sport_code: str) -> int:
        """Get number of features for a sport"""
        return len(self.get_feature_names(sport_code))


# =============================================================================
# EXPORT ALIASES FOR BACKWARD COMPATIBILITY
# =============================================================================

__all__ = [
    # Main Classes
    "FeatureEngineer",
    "FeatureSet",
    "GameContext",
    # Feature Generators
    "BaseFeatureGenerator",
    "FootballFeatureGenerator",
    "BasketballFeatureGenerator",
    "HockeyFeatureGenerator",
    "BaseballFeatureGenerator",
    "TennisFeatureGenerator",
    # Re-exported from elo_rating for convenience
    "ELORating",
    "ELORatingSystem",
    "ELOSystem",
    "MultiSportELOManager",
]
