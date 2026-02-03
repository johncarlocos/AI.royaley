"""
ROYALEY - Comprehensive ML Feature Extraction Service
======================================================

Extracts features across 5 dimensions for professional sports betting ML:
1. TEAM FEATURES - Rolling stats, form, H2H, home/away splits
2. GAME CONTEXT - Rest days, travel, schedule spots
3. PLAYER FEATURES - Key player trends, availability
4. ODDS/MARKET - Line movements, CLV, sharp action
5. SITUATIONAL - Revenge, letdown, streaks

Each dimension can be tested independently:
    python -c "
    import asyncio
    from app.core.database import db_manager
    from app.services.master_data.ml_features import MLFeatureService
    
    async def test():
        await db_manager.initialize()
        async with db_manager.session() as session:
            svc = MLFeatureService(session)
            # Test one dimension at a time:
            result = await svc.test_team_features('NBA', limit=5)
            print(result)
    asyncio.run(test())
    "
"""

import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


# =============================================================================
# FEATURE DATACLASS - All features in one structure
# =============================================================================

@dataclass
class MLFeatureVector:
    """Complete feature vector for one game - all 5 dimensions."""
    
    # === IDENTIFIERS ===
    master_game_id: str
    sport_code: str
    scheduled_at: datetime
    season: Optional[int] = None
    home_team_id: Optional[str] = None
    away_team_id: Optional[str] = None
    home_team_name: Optional[str] = None
    away_team_name: Optional[str] = None
    
    # === TARGET VARIABLES ===
    home_score: Optional[int] = None
    away_score: Optional[int] = None
    home_win: Optional[int] = None  # 1=home win, 0=away win
    total_points: Optional[int] = None
    score_margin: Optional[int] = None  # home - away
    spread_result: Optional[int] = None  # 1=home covered, 0=away covered
    over_result: Optional[int] = None  # 1=over, 0=under
    
    # === 1. TEAM FEATURES (Rolling Performance) ===
    # Home team rolling stats
    home_wins_last5: Optional[int] = None
    home_wins_last10: Optional[int] = None
    home_win_pct_last10: Optional[float] = None
    home_avg_pts_last10: Optional[float] = None
    home_avg_pts_allowed_last10: Optional[float] = None
    home_avg_margin_last10: Optional[float] = None
    home_home_win_pct: Optional[float] = None  # Win % at home only
    home_ats_record_last10: Optional[float] = None  # ATS win %
    home_ou_over_pct_last10: Optional[float] = None  # Over %
    
    # Away team rolling stats
    away_wins_last5: Optional[int] = None
    away_wins_last10: Optional[int] = None
    away_win_pct_last10: Optional[float] = None
    away_avg_pts_last10: Optional[float] = None
    away_avg_pts_allowed_last10: Optional[float] = None
    away_avg_margin_last10: Optional[float] = None
    away_away_win_pct: Optional[float] = None  # Win % on road only
    away_ats_record_last10: Optional[float] = None
    away_ou_over_pct_last10: Optional[float] = None
    
    # Head-to-head
    h2h_home_wins_last5: Optional[int] = None
    h2h_home_avg_margin: Optional[float] = None
    h2h_total_avg: Optional[float] = None
    
    # Power ratings (derived)
    home_power_rating: Optional[float] = None
    away_power_rating: Optional[float] = None
    power_rating_diff: Optional[float] = None
    
    # === 2. GAME CONTEXT FEATURES ===
    home_rest_days: Optional[int] = None
    away_rest_days: Optional[int] = None
    rest_advantage: Optional[int] = None  # home - away
    home_is_back_to_back: Optional[bool] = None
    away_is_back_to_back: Optional[bool] = None
    home_3_in_4_nights: Optional[bool] = None
    away_3_in_4_nights: Optional[bool] = None
    is_divisional: Optional[bool] = None
    is_conference: Optional[bool] = None
    is_rivalry: Optional[bool] = None
    is_playoff: Optional[bool] = None
    is_neutral_site: Optional[bool] = None
    day_of_week: Optional[int] = None  # 0=Mon, 6=Sun
    is_night_game: Optional[bool] = None
    month: Optional[int] = None
    
    # === 3. PLAYER FEATURES ===
    home_star_player_pts_avg: Optional[float] = None
    away_star_player_pts_avg: Optional[float] = None
    home_top3_players_pts_avg: Optional[float] = None
    away_top3_players_pts_avg: Optional[float] = None
    home_injuries_out: Optional[int] = None
    away_injuries_out: Optional[int] = None
    home_injury_impact: Optional[float] = None
    away_injury_impact: Optional[float] = None
    home_starters_out: Optional[int] = None
    away_starters_out: Optional[int] = None
    
    # === 4. ODDS/MARKET FEATURES ===
    spread_open: Optional[float] = None
    spread_close: Optional[float] = None
    spread_movement: Optional[float] = None
    moneyline_home_open: Optional[int] = None
    moneyline_home_close: Optional[int] = None
    moneyline_away_close: Optional[int] = None
    total_open: Optional[float] = None
    total_close: Optional[float] = None
    total_movement: Optional[float] = None
    
    # Pinnacle (sharp benchmark)
    pinnacle_spread: Optional[float] = None
    pinnacle_ml_home: Optional[int] = None
    pinnacle_total: Optional[float] = None
    
    # Market indicators
    num_books: Optional[int] = None
    consensus_spread: Optional[float] = None
    consensus_total: Optional[float] = None
    implied_home_prob: Optional[float] = None
    no_vig_home_prob: Optional[float] = None
    
    # Public betting
    public_spread_home_pct: Optional[float] = None
    public_ml_home_pct: Optional[float] = None
    public_total_over_pct: Optional[float] = None
    public_money_home_pct: Optional[float] = None
    
    # Sharp indicators
    is_reverse_line_move: Optional[bool] = None
    sharp_action_indicator: Optional[float] = None
    steam_move: Optional[bool] = None
    
    # === 5. SITUATIONAL FEATURES ===
    home_streak: Optional[int] = None  # Positive=wins, negative=losses
    away_streak: Optional[int] = None
    home_is_revenge: Optional[bool] = None  # Lost to opponent last meeting
    away_is_revenge: Optional[bool] = None
    home_letdown_spot: Optional[bool] = None  # After big win
    away_letdown_spot: Optional[bool] = None
    home_lookahead_spot: Optional[bool] = None  # Big game coming
    away_lookahead_spot: Optional[bool] = None
    home_season_game_num: Optional[int] = None
    away_season_game_num: Optional[int] = None
    
    # === WEATHER (outdoor sports) ===
    temperature_f: Optional[float] = None
    wind_speed_mph: Optional[float] = None
    precipitation_pct: Optional[float] = None
    humidity_pct: Optional[float] = None
    is_dome: Optional[bool] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DataFrame."""
        return asdict(self)


# =============================================================================
# MAIN SERVICE CLASS
# =============================================================================

class MLFeatureService:
    """
    Comprehensive ML feature extraction service.
    
    Features are organized into testable modules:
    - Team features (rolling stats)
    - Game context features
    - Player features
    - Odds features
    - Situational features
    """
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    # =========================================================================
    # FULL EXTRACTION
    # =========================================================================
    
    async def extract_all_features(
        self,
        sport_code: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        season: Optional[int] = None,
        completed_only: bool = True,
        limit: Optional[int] = None,
    ) -> List[MLFeatureVector]:
        """
        Extract ALL features for games matching the filter.
        Returns list of MLFeatureVector ready for ML training.
        """
        games = await self._get_games(
            sport_code, start_date, end_date, season, completed_only, limit
        )
        
        logger.info(f"Extracting features for {len(games)} {sport_code} games...")
        
        features = []
        for i, g in enumerate(games):
            fv = MLFeatureVector(
                master_game_id=str(g['id']),
                sport_code=g['sport_code'],
                scheduled_at=g['scheduled_at'],
                season=g['season'],
                home_team_id=str(g['home_team_id']) if g['home_team_id'] else None,
                away_team_id=str(g['away_team_id']) if g['away_team_id'] else None,
                home_team_name=g['home_name'],
                away_team_name=g['away_name'],
                home_score=g['home_score'],
                away_score=g['away_score'],
                is_playoff=g.get('is_playoff'),
                is_neutral_site=g.get('is_neutral_site'),
            )
            
            # Compute basic targets (home_win, total_points, score_margin)
            self._compute_basic_targets(fv)
            
            # Extract all 5 dimensions + weather
            await self._extract_team_features(fv)
            await self._extract_game_context_features(fv)
            await self._extract_player_features(fv)
            await self._extract_odds_features(fv)
            await self._extract_situational_features(fv)
            await self._extract_weather_features(fv)
            
            # Compute betting targets AFTER odds extraction (need spread_close, total_close)
            self._compute_betting_targets(fv)
            
            # Extract season from scheduled_at if not set
            if fv.season is None and fv.scheduled_at:
                fv.season = self._calculate_season(fv.scheduled_at, sport_code)
            
            features.append(fv)
            
            if (i + 1) % 100 == 0:
                logger.info(f"  Processed {i + 1}/{len(games)} games")
        
        logger.info(f"✅ Extracted {len(features)} feature vectors")
        return features
    
    # =========================================================================
    # INDIVIDUAL TEST METHODS (test each dimension separately)
    # =========================================================================
    
    async def test_team_features(
        self, sport_code: str, limit: int = 5
    ) -> List[Dict]:
        """Test team features extraction only."""
        games = await self._get_games(sport_code, limit=limit)
        results = []
        
        for g in games:
            fv = MLFeatureVector(
                master_game_id=str(g['id']),
                sport_code=g['sport_code'],
                scheduled_at=g['scheduled_at'],
                home_team_id=str(g['home_team_id']) if g['home_team_id'] else None,
                away_team_id=str(g['away_team_id']) if g['away_team_id'] else None,
                home_team_name=g['home_name'],
                away_team_name=g['away_name'],
            )
            await self._extract_team_features(fv)
            
            results.append({
                'game': f"{fv.home_team_name} vs {fv.away_team_name}",
                'date': str(fv.scheduled_at.date()) if fv.scheduled_at else None,
                'home_win_pct_last10': fv.home_win_pct_last10,
                'away_win_pct_last10': fv.away_win_pct_last10,
                'home_avg_pts_last10': fv.home_avg_pts_last10,
                'away_avg_pts_last10': fv.away_avg_pts_last10,
                'home_avg_margin_last10': fv.home_avg_margin_last10,
                'h2h_home_wins_last5': fv.h2h_home_wins_last5,
                'power_rating_diff': fv.power_rating_diff,
            })
        
        return results
    
    async def test_game_context_features(
        self, sport_code: str, limit: int = 5
    ) -> List[Dict]:
        """Test game context features extraction only."""
        games = await self._get_games(sport_code, limit=limit)
        results = []
        
        for g in games:
            fv = MLFeatureVector(
                master_game_id=str(g['id']),
                sport_code=g['sport_code'],
                scheduled_at=g['scheduled_at'],
                home_team_id=str(g['home_team_id']) if g['home_team_id'] else None,
                away_team_id=str(g['away_team_id']) if g['away_team_id'] else None,
                home_team_name=g['home_name'],
                away_team_name=g['away_name'],
            )
            await self._extract_game_context_features(fv)
            
            results.append({
                'game': f"{fv.home_team_name} vs {fv.away_team_name}",
                'date': str(fv.scheduled_at.date()) if fv.scheduled_at else None,
                'home_rest_days': fv.home_rest_days,
                'away_rest_days': fv.away_rest_days,
                'rest_advantage': fv.rest_advantage,
                'home_b2b': fv.home_is_back_to_back,
                'away_b2b': fv.away_is_back_to_back,
                'day_of_week': fv.day_of_week,
                'is_night_game': fv.is_night_game,
            })
        
        return results
    
    async def test_player_features(
        self, sport_code: str, limit: int = 5
    ) -> List[Dict]:
        """Test player features extraction only."""
        games = await self._get_games(sport_code, limit=limit)
        results = []
        
        for g in games:
            fv = MLFeatureVector(
                master_game_id=str(g['id']),
                sport_code=g['sport_code'],
                scheduled_at=g['scheduled_at'],
                home_team_id=str(g['home_team_id']) if g['home_team_id'] else None,
                away_team_id=str(g['away_team_id']) if g['away_team_id'] else None,
                home_team_name=g['home_name'],
                away_team_name=g['away_name'],
            )
            await self._extract_player_features(fv)
            
            results.append({
                'game': f"{fv.home_team_name} vs {fv.away_team_name}",
                'date': str(fv.scheduled_at.date()) if fv.scheduled_at else None,
                'home_star_pts_avg': fv.home_star_player_pts_avg,
                'away_star_pts_avg': fv.away_star_player_pts_avg,
                'home_injuries_out': fv.home_injuries_out,
                'away_injuries_out': fv.away_injuries_out,
            })
        
        return results
    
    async def test_odds_features(
        self, sport_code: str, limit: int = 5
    ) -> List[Dict]:
        """Test odds/market features extraction only."""
        games = await self._get_games(sport_code, limit=limit)
        results = []
        
        for g in games:
            fv = MLFeatureVector(
                master_game_id=str(g['id']),
                sport_code=g['sport_code'],
                scheduled_at=g['scheduled_at'],
                home_team_name=g['home_name'],
                away_team_name=g['away_name'],
            )
            await self._extract_odds_features(fv)
            
            results.append({
                'game': f"{fv.home_team_name} vs {fv.away_team_name}",
                'date': str(fv.scheduled_at.date()) if fv.scheduled_at else None,
                'spread_open': fv.spread_open,
                'spread_close': fv.spread_close,
                'spread_movement': fv.spread_movement,
                'total_close': fv.total_close,
                'pinnacle_spread': fv.pinnacle_spread,
                'num_books': fv.num_books,
                'is_rlm': fv.is_reverse_line_move,
            })
        
        return results
    
    async def test_situational_features(
        self, sport_code: str, limit: int = 5
    ) -> List[Dict]:
        """Test situational features extraction only."""
        games = await self._get_games(sport_code, limit=limit)
        results = []
        
        for g in games:
            fv = MLFeatureVector(
                master_game_id=str(g['id']),
                sport_code=g['sport_code'],
                scheduled_at=g['scheduled_at'],
                home_team_id=str(g['home_team_id']) if g['home_team_id'] else None,
                away_team_id=str(g['away_team_id']) if g['away_team_id'] else None,
                home_team_name=g['home_name'],
                away_team_name=g['away_name'],
            )
            await self._extract_situational_features(fv)
            
            results.append({
                'game': f"{fv.home_team_name} vs {fv.away_team_name}",
                'date': str(fv.scheduled_at.date()) if fv.scheduled_at else None,
                'home_streak': fv.home_streak,
                'away_streak': fv.away_streak,
                'home_revenge': fv.home_is_revenge,
                'away_revenge': fv.away_is_revenge,
                'home_game_num': fv.home_season_game_num,
                'away_game_num': fv.away_season_game_num,
            })
        
        return results
    
    # =========================================================================
    # HELPER: GET GAMES
    # =========================================================================
    
    async def _get_games(
        self,
        sport_code: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        season: Optional[int] = None,
        completed_only: bool = True,
        limit: Optional[int] = None,
    ) -> List[Dict]:
        """Fetch games from master_games table."""
        conditions = ["mg.sport_code = :sport"]
        params: Dict[str, Any] = {"sport": sport_code}
        
        if completed_only:
            conditions.append("mg.home_score IS NOT NULL")
            conditions.append("mg.status = 'final'")
        if season:
            conditions.append("mg.season = :season")
            params["season"] = season
        if start_date:
            conditions.append("mg.scheduled_at >= :start")
            params["start"] = start_date
        if end_date:
            conditions.append("mg.scheduled_at <= :end")
            params["end"] = end_date
        
        where_clause = " AND ".join(conditions)
        limit_clause = f"LIMIT {limit}" if limit else ""
        
        result = await self.session.execute(text(f"""
            SELECT mg.id, mg.sport_code, mg.scheduled_at, mg.season,
                   mg.home_score, mg.away_score, mg.status,
                   mg.home_master_team_id as home_team_id,
                   mg.away_master_team_id as away_team_id,
                   mg.is_playoff, mg.is_neutral_site,
                   ht.canonical_name as home_name,
                   at_.canonical_name as away_name
            FROM master_games mg
            LEFT JOIN master_teams ht ON mg.home_master_team_id = ht.id
            LEFT JOIN master_teams at_ ON mg.away_master_team_id = at_.id
            WHERE {where_clause}
            ORDER BY mg.scheduled_at DESC
            {limit_clause}
        """), params)
        
        rows = result.fetchall()
        columns = ['id', 'sport_code', 'scheduled_at', 'season', 'home_score', 
                   'away_score', 'status', 'home_team_id', 'away_team_id',
                   'is_playoff', 'is_neutral_site', 'home_name', 'away_name']
        
        return [dict(zip(columns, row)) for row in rows]
    
    # =========================================================================
    # COMPUTE TARGETS
    # =========================================================================
    
    def _compute_basic_targets(self, fv: MLFeatureVector):
        """Compute basic target variables from scores (called before odds extraction)."""
        if fv.home_score is not None and fv.away_score is not None:
            fv.home_win = 1 if fv.home_score > fv.away_score else 0
            fv.total_points = fv.home_score + fv.away_score
            fv.score_margin = fv.home_score - fv.away_score
    
    def _compute_betting_targets(self, fv: MLFeatureVector):
        """Compute betting targets AFTER odds extraction (need spread_close, total_close)."""
        if fv.score_margin is None:
            return
        
        # Spread result: 1 if home covered, 0 if away covered
        # Home covers if: score_margin > -spread_close (spread is negative for favorites)
        # Example: Home -3.5, wins by 5 → margin=5, spread=-3.5 → 5 > 3.5 = True = 1 (covered)
        # Example: Home -3.5, wins by 2 → margin=2, spread=-3.5 → 2 > 3.5 = False = 0 (didn't cover)
        if fv.spread_close is not None:
            # spread_close is from home perspective (negative = home favored)
            home_cover_margin = fv.score_margin + fv.spread_close
            fv.spread_result = 1 if home_cover_margin > 0 else 0
        
        # Over/Under result: 1 if over hit, 0 if under hit
        if fv.total_points is not None and fv.total_close is not None:
            fv.over_result = 1 if fv.total_points > fv.total_close else 0
    
    def _calculate_season(self, game_date: datetime, sport_code: str) -> int:
        """Calculate season year from game date based on sport."""
        year = game_date.year
        month = game_date.month
        
        # Fall/Winter sports (NFL, NBA, NHL, NCAAF, NCAAB, CFL)
        # Season spans two years, labeled by start year
        if sport_code in ['NFL', 'NCAAF', 'CFL']:
            # Football: Aug-Feb, season is the year it started
            return year if month >= 8 else year - 1
        elif sport_code in ['NBA', 'NHL', 'NCAAB', 'WNBA']:
            # Basketball/Hockey: Oct-Jun, season is the year it started
            return year if month >= 10 else year - 1
        elif sport_code in ['MLB']:
            # Baseball: Mar-Oct, season is same year
            return year
        elif sport_code in ['ATP', 'WTA']:
            # Tennis: Jan-Nov, season is same year
            return year
        else:
            return year
    
    async def _compute_targets(self, fv: MLFeatureVector):
        """Legacy method - kept for backward compatibility."""
        self._compute_basic_targets(fv)
    
    # =========================================================================
    # 1. TEAM FEATURES
    # =========================================================================
    
    async def _extract_team_features(self, fv: MLFeatureVector):
        """Extract rolling team performance features."""
        if not fv.home_team_id or not fv.away_team_id:
            return
        
        # Home team rolling stats
        home_stats = await self._get_team_rolling_stats(
            fv.home_team_id, fv.scheduled_at, fv.sport_code
        )
        if home_stats:
            fv.home_wins_last5 = home_stats.get('wins_last5')
            fv.home_wins_last10 = home_stats.get('wins_last10')
            fv.home_win_pct_last10 = home_stats.get('win_pct_last10')
            fv.home_avg_pts_last10 = home_stats.get('avg_pts_last10')
            fv.home_avg_pts_allowed_last10 = home_stats.get('avg_pts_allowed_last10')
            fv.home_avg_margin_last10 = home_stats.get('avg_margin_last10')
            fv.home_home_win_pct = home_stats.get('home_win_pct')
            fv.home_ats_record_last10 = home_stats.get('ats_pct_last10')
            fv.home_ou_over_pct_last10 = home_stats.get('ou_over_pct_last10')
        
        # Away team rolling stats
        away_stats = await self._get_team_rolling_stats(
            fv.away_team_id, fv.scheduled_at, fv.sport_code
        )
        if away_stats:
            fv.away_wins_last5 = away_stats.get('wins_last5')
            fv.away_wins_last10 = away_stats.get('wins_last10')
            fv.away_win_pct_last10 = away_stats.get('win_pct_last10')
            fv.away_avg_pts_last10 = away_stats.get('avg_pts_last10')
            fv.away_avg_pts_allowed_last10 = away_stats.get('avg_pts_allowed_last10')
            fv.away_avg_margin_last10 = away_stats.get('avg_margin_last10')
            fv.away_away_win_pct = away_stats.get('away_win_pct')
            fv.away_ats_record_last10 = away_stats.get('ats_pct_last10')
            fv.away_ou_over_pct_last10 = away_stats.get('ou_over_pct_last10')
        
        # Head-to-head
        h2h = await self._get_head_to_head(
            fv.home_team_id, fv.away_team_id, fv.scheduled_at, fv.sport_code
        )
        if h2h:
            fv.h2h_home_wins_last5 = h2h.get('home_wins')
            fv.h2h_home_avg_margin = h2h.get('home_avg_margin')
            fv.h2h_total_avg = h2h.get('total_avg')
        
        # Power ratings (simple: margin + adjustment)
        if fv.home_avg_margin_last10 is not None:
            fv.home_power_rating = fv.home_avg_margin_last10
        if fv.away_avg_margin_last10 is not None:
            fv.away_power_rating = fv.away_avg_margin_last10
        if fv.home_power_rating is not None and fv.away_power_rating is not None:
            fv.power_rating_diff = round(fv.home_power_rating - fv.away_power_rating, 2)
    
    async def _get_team_rolling_stats(
        self, team_id: str, before_date: datetime, sport_code: str
    ) -> Optional[Dict]:
        """Get rolling stats for a team before a specific date."""
        result = await self.session.execute(text("""
            WITH team_games AS (
                SELECT 
                    mg.id,
                    mg.scheduled_at,
                    mg.home_score,
                    mg.away_score,
                    CASE WHEN mg.home_master_team_id = :tid THEN 'home' ELSE 'away' END as side,
                    CASE WHEN mg.home_master_team_id = :tid 
                         THEN mg.home_score ELSE mg.away_score END as team_pts,
                    CASE WHEN mg.home_master_team_id = :tid 
                         THEN mg.away_score ELSE mg.home_score END as opp_pts,
                    CASE WHEN (mg.home_master_team_id = :tid AND mg.home_score > mg.away_score)
                           OR (mg.away_master_team_id = :tid AND mg.away_score > mg.home_score)
                         THEN 1 ELSE 0 END as win
                FROM master_games mg
                WHERE (mg.home_master_team_id = :tid OR mg.away_master_team_id = :tid)
                  AND mg.scheduled_at < :before
                  AND mg.home_score IS NOT NULL
                  AND mg.sport_code = :sport
                ORDER BY mg.scheduled_at DESC
                LIMIT 20
            )
            SELECT 
                SUM(CASE WHEN rn <= 5 THEN win ELSE 0 END) as wins_last5,
                SUM(CASE WHEN rn <= 10 THEN win ELSE 0 END) as wins_last10,
                AVG(CASE WHEN rn <= 10 THEN win::float ELSE NULL END) as win_pct_last10,
                AVG(CASE WHEN rn <= 10 THEN team_pts::float ELSE NULL END) as avg_pts_last10,
                AVG(CASE WHEN rn <= 10 THEN opp_pts::float ELSE NULL END) as avg_pts_allowed_last10,
                AVG(CASE WHEN rn <= 10 THEN (team_pts - opp_pts)::float ELSE NULL END) as avg_margin_last10,
                AVG(CASE WHEN rn <= 10 AND side = 'home' THEN win::float ELSE NULL END) as home_win_pct,
                AVG(CASE WHEN rn <= 10 AND side = 'away' THEN win::float ELSE NULL END) as away_win_pct
            FROM (
                SELECT *, ROW_NUMBER() OVER (ORDER BY scheduled_at DESC) as rn
                FROM team_games
            ) t
        """), {
            "tid": team_id,
            "before": before_date,
            "sport": sport_code,
        })
        
        row = result.fetchone()
        if not row:
            return None
        
        return {
            'wins_last5': row[0],
            'wins_last10': row[1],
            'win_pct_last10': round(row[2], 3) if row[2] else None,
            'avg_pts_last10': round(row[3], 1) if row[3] else None,
            'avg_pts_allowed_last10': round(row[4], 1) if row[4] else None,
            'avg_margin_last10': round(row[5], 1) if row[5] else None,
            'home_win_pct': round(row[6], 3) if row[6] else None,
            'away_win_pct': round(row[7], 3) if row[7] else None,
        }
    
    async def _get_head_to_head(
        self, home_team_id: str, away_team_id: str, 
        before_date: datetime, sport_code: str
    ) -> Optional[Dict]:
        """Get head-to-head record between two teams."""
        result = await self.session.execute(text("""
            SELECT 
                SUM(CASE WHEN mg.home_master_team_id = :home_tid 
                         AND mg.home_score > mg.away_score THEN 1
                         WHEN mg.away_master_team_id = :home_tid 
                         AND mg.away_score > mg.home_score THEN 1
                         ELSE 0 END) as home_team_wins,
                AVG(CASE WHEN mg.home_master_team_id = :home_tid 
                         THEN mg.home_score - mg.away_score
                         ELSE mg.away_score - mg.home_score END) as home_avg_margin,
                AVG(mg.home_score + mg.away_score) as total_avg,
                COUNT(*) as games
            FROM master_games mg
            WHERE ((mg.home_master_team_id = :home_tid AND mg.away_master_team_id = :away_tid)
                OR (mg.home_master_team_id = :away_tid AND mg.away_master_team_id = :home_tid))
              AND mg.scheduled_at < :before
              AND mg.home_score IS NOT NULL
              AND mg.sport_code = :sport
            LIMIT 5
        """), {
            "home_tid": home_team_id,
            "away_tid": away_team_id,
            "before": before_date,
            "sport": sport_code,
        })
        
        row = result.fetchone()
        if not row or row[3] == 0:
            return None
        
        return {
            'home_wins': row[0],
            'home_avg_margin': round(row[1], 1) if row[1] else None,
            'total_avg': round(row[2], 1) if row[2] else None,
        }
    
    # =========================================================================
    # 2. GAME CONTEXT FEATURES
    # =========================================================================
    
    async def _extract_game_context_features(self, fv: MLFeatureVector):
        """Extract game context features (rest, schedule, etc.)."""
        if not fv.scheduled_at:
            return
        
        # Day/time features
        fv.day_of_week = fv.scheduled_at.weekday()
        fv.is_night_game = fv.scheduled_at.hour >= 18
        fv.month = fv.scheduled_at.month
        
        # Rest days for each team
        if fv.home_team_id:
            home_rest = await self._get_rest_days(
                fv.home_team_id, fv.scheduled_at, fv.sport_code
            )
            if home_rest:
                fv.home_rest_days = home_rest['rest_days']
                fv.home_is_back_to_back = home_rest['is_b2b']
                fv.home_3_in_4_nights = home_rest['is_3_in_4']
        
        if fv.away_team_id:
            away_rest = await self._get_rest_days(
                fv.away_team_id, fv.scheduled_at, fv.sport_code
            )
            if away_rest:
                fv.away_rest_days = away_rest['rest_days']
                fv.away_is_back_to_back = away_rest['is_b2b']
                fv.away_3_in_4_nights = away_rest['is_3_in_4']
        
        # Rest advantage
        if fv.home_rest_days is not None and fv.away_rest_days is not None:
            fv.rest_advantage = fv.home_rest_days - fv.away_rest_days
    
    async def _get_rest_days(
        self, team_id: str, game_date: datetime, sport_code: str
    ) -> Optional[Dict]:
        """Calculate rest days and schedule density for a team."""
        # Get last game
        result = await self.session.execute(text("""
            SELECT scheduled_at
            FROM master_games
            WHERE (home_master_team_id = :tid OR away_master_team_id = :tid)
              AND scheduled_at < :game_date
              AND home_score IS NOT NULL
              AND sport_code = :sport
            ORDER BY scheduled_at DESC
            LIMIT 1
        """), {
            "tid": team_id,
            "game_date": game_date,
            "sport": sport_code,
        })
        last_game = result.fetchone()
        
        if not last_game:
            return {'rest_days': 7, 'is_b2b': False, 'is_3_in_4': False}
        
        rest_days = (game_date.date() - last_game[0].date()).days
        is_b2b = rest_days <= 1
        
        # Check for 3 games in 4 nights
        result2 = await self.session.execute(text("""
            SELECT COUNT(*)
            FROM master_games
            WHERE (home_master_team_id = :tid OR away_master_team_id = :tid)
              AND scheduled_at >= :start_date
              AND scheduled_at < :game_date
              AND home_score IS NOT NULL
              AND sport_code = :sport
        """), {
            "tid": team_id,
            "start_date": game_date - timedelta(days=4),
            "game_date": game_date,
            "sport": sport_code,
        })
        games_in_4 = result2.scalar()
        is_3_in_4 = games_in_4 >= 2  # This game would be 3rd
        
        return {
            'rest_days': rest_days,
            'is_b2b': is_b2b,
            'is_3_in_4': is_3_in_4,
        }
    
    # =========================================================================
    # 3. PLAYER FEATURES
    # =========================================================================
    
    async def _extract_player_features(self, fv: MLFeatureVector):
        """Extract player-related features."""
        # Star player stats
        if fv.home_team_id:
            home_player = await self._get_star_player_stats(
                fv.home_team_id, fv.scheduled_at, fv.sport_code
            )
            if home_player:
                fv.home_star_player_pts_avg = home_player.get('star_pts_avg')
                fv.home_top3_players_pts_avg = home_player.get('top3_pts_avg')
        
        if fv.away_team_id:
            away_player = await self._get_star_player_stats(
                fv.away_team_id, fv.scheduled_at, fv.sport_code
            )
            if away_player:
                fv.away_star_player_pts_avg = away_player.get('star_pts_avg')
                fv.away_top3_players_pts_avg = away_player.get('top3_pts_avg')
        
        # Injuries
        await self._extract_injury_features(fv)
    
    async def _get_star_player_stats(
        self, team_id: str, before_date: datetime, sport_code: str
    ) -> Optional[Dict]:
        """Get star player performance for a team."""
        # Try master_player_stats first
        try:
            result = await self.session.execute(text("""
                WITH player_avgs AS (
                    SELECT 
                        mps.master_player_id,
                        AVG(mps.points) as avg_pts
                    FROM master_player_stats mps
                    JOIN master_games mg ON mps.master_game_id = mg.id
                    WHERE mps.master_team_id = :tid
                      AND mg.scheduled_at < :before
                      AND mg.scheduled_at > :cutoff
                      AND mg.sport_code = :sport
                    GROUP BY mps.master_player_id
                    HAVING COUNT(*) >= 3
                    ORDER BY AVG(mps.points) DESC
                    LIMIT 3
                )
                SELECT 
                    MAX(avg_pts) as star_pts,
                    AVG(avg_pts) as top3_pts
                FROM player_avgs
            """), {
                "tid": team_id,
                "before": before_date,
                "cutoff": before_date - timedelta(days=60),
                "sport": sport_code,
            })
            
            row = result.fetchone()
            if row and row[0] is not None:
                return {
                    'star_pts_avg': round(row[0], 1) if row[0] else None,
                    'top3_pts_avg': round(row[1], 1) if row[1] else None,
                }
        except Exception:
            # master_player_stats table may not exist or have different schema
            pass
        
        # No data available - player stats tables need to be populated
        return None
    
    async def _extract_injury_features(self, fv: MLFeatureVector):
        """Extract injury features from injuries table."""
        for side, team_id in [('home', fv.home_team_id), ('away', fv.away_team_id)]:
            if not team_id:
                continue
            
            result = await self.session.execute(text("""
                SELECT 
                    COUNT(*) FILTER (WHERE status IN ('Out', 'IR', 'Suspended')) as num_out,
                    COALESCE(SUM(impact_score) FILTER (WHERE status IN ('Out', 'IR', 'Suspended')), 0) as impact,
                    COUNT(*) FILTER (WHERE is_starter AND status IN ('Out', 'IR', 'Suspended')) as starters_out
                FROM injuries
                WHERE master_team_id = :tid
                  AND last_updated >= :cutoff
            """), {
                "tid": team_id,
                "cutoff": fv.scheduled_at - timedelta(days=7) if fv.scheduled_at else datetime.now() - timedelta(days=7),
            })
            
            row = result.fetchone()
            if row:
                if side == 'home':
                    fv.home_injuries_out = row[0] or 0
                    fv.home_injury_impact = float(row[1] or 0)
                    fv.home_starters_out = row[2] or 0
                else:
                    fv.away_injuries_out = row[0] or 0
                    fv.away_injury_impact = float(row[1] or 0)
                    fv.away_starters_out = row[2] or 0
    
    # =========================================================================
    # 4. ODDS/MARKET FEATURES
    # =========================================================================
    
    async def _extract_odds_features(self, fv: MLFeatureVector):
        """Extract odds and market features from master_odds."""
        # Query all relevant columns from master_odds
        # Schema: opening_line, closing_line (spread), opening_total, closing_total (total)
        #         opening_odds_home, closing_odds_home (moneyline)
        result = await self.session.execute(text("""
            SELECT 
                mo.bet_type,
                mo.sportsbook_key,
                mo.opening_line,
                mo.closing_line,
                mo.line_movement,
                mo.opening_odds_home,
                mo.closing_odds_home,
                mo.opening_odds_away,
                mo.closing_odds_away,
                mo.opening_total,
                mo.closing_total,
                mo.opening_over_odds,
                mo.closing_over_odds,
                mo.is_sharp
            FROM master_odds mo
            WHERE mo.master_game_id = :mgid
        """), {"mgid": fv.master_game_id})
        
        rows = result.fetchall()
        if not rows:
            return
        
        spreads = []
        totals = []
        books_seen = set()
        
        for r in rows:
            (bet_type, book_key, open_line, close_line, movement,
             open_odds_home, close_odds_home, open_odds_away, close_odds_away,
             open_total, close_total, open_over_odds, close_over_odds, is_sharp) = r
            
            books_seen.add(book_key)
            
            # Check if this is a sharp book (pinnacle, etc)
            is_sharp_book = is_sharp or (book_key and 'pinnacle' in book_key.lower())
            
            if bet_type == 'spread':
                if close_line is not None:
                    spreads.append(close_line)
                
                if fv.spread_open is None and open_line is not None:
                    fv.spread_open = open_line
                if close_line is not None:
                    fv.spread_close = close_line
                if movement is not None:
                    fv.spread_movement = movement
                
                if is_sharp_book and fv.pinnacle_spread is None:
                    fv.pinnacle_spread = close_line or open_line
            
            elif bet_type == 'moneyline':
                # Use opening_odds_home and closing_odds_home for moneyline
                if close_odds_home is not None:
                    if fv.moneyline_home_close is None:
                        fv.moneyline_home_close = close_odds_home
                    if fv.moneyline_away_close is None and close_odds_away is not None:
                        fv.moneyline_away_close = close_odds_away
                if open_odds_home is not None:
                    if fv.moneyline_home_open is None:
                        fv.moneyline_home_open = open_odds_home
                
                if is_sharp_book and fv.pinnacle_ml_home is None:
                    fv.pinnacle_ml_home = close_odds_home or open_odds_home
            
            elif bet_type == 'total':
                # Use opening_total and closing_total for totals
                actual_open = open_total if open_total is not None else open_line
                actual_close = close_total if close_total is not None else close_line
                
                if actual_close is not None:
                    totals.append(actual_close)
                
                if fv.total_open is None and actual_open is not None:
                    fv.total_open = actual_open
                if actual_close is not None:
                    fv.total_close = actual_close
                if movement is not None:
                    fv.total_movement = movement
                
                if is_sharp_book and fv.pinnacle_total is None:
                    fv.pinnacle_total = actual_close or actual_open
        
        fv.num_books = len(books_seen)
        
        if spreads:
            fv.consensus_spread = round(sum(spreads) / len(spreads), 2)
        if totals:
            fv.consensus_total = round(sum(totals) / len(totals), 2)
        
        # Implied probability from moneyline
        if fv.moneyline_home_close:
            ml = fv.moneyline_home_close
            if ml < 0:
                fv.implied_home_prob = round(abs(ml) / (abs(ml) + 100), 4)
            else:
                fv.implied_home_prob = round(100 / (ml + 100), 4)
        
        # No-vig probability
        if fv.moneyline_home_close and fv.moneyline_away_close:
            home_ml = fv.moneyline_home_close
            away_ml = fv.moneyline_away_close
            
            # Convert to implied probabilities
            home_imp = abs(home_ml) / (abs(home_ml) + 100) if home_ml < 0 else 100 / (home_ml + 100)
            away_imp = abs(away_ml) / (abs(away_ml) + 100) if away_ml < 0 else 100 / (away_ml + 100)
            
            # Remove vig
            total_imp = home_imp + away_imp
            if total_imp > 0:
                fv.no_vig_home_prob = round(home_imp / total_imp, 4)
        
        # Reverse line movement detection
        await self._detect_reverse_line_move(fv)
    
    async def _detect_reverse_line_move(self, fv: MLFeatureVector):
        """Detect reverse line movement and extract public betting data."""
        # Check public betting - extract all relevant fields
        result = await self.session.execute(text("""
            SELECT 
                spread_home_bet_pct,
                spread_home_money_pct,
                ml_home_bet_pct,
                ml_home_money_pct,
                total_over_bet_pct,
                total_over_money_pct,
                is_rlm_spread,
                is_rlm_total,
                is_steam_spread,
                is_sharp_spread
            FROM public_betting
            WHERE master_game_id = :mgid
            LIMIT 1
        """), {"mgid": fv.master_game_id})
        
        row = result.fetchone()
        if row:
            fv.public_spread_home_pct = row[0]
            fv.public_money_home_pct = row[1]
            fv.public_ml_home_pct = row[2]
            # public_money_ml_home_pct not in MLFeatureVector, skip row[3]
            fv.public_total_over_pct = row[4]
            # public_money_total_over_pct not in MLFeatureVector, skip row[5]
            
            # Use pre-calculated RLM if available
            if row[6] is not None:
                fv.is_reverse_line_move = row[6]
            
            # Steam move from database
            if row[8] is not None:
                fv.steam_move = row[8]
            
            # Sharp action indicator from database or calculate
            if row[9] is not None:
                fv.sharp_action_indicator = 1.0 if row[9] else 0.0
            elif fv.public_money_home_pct and fv.public_spread_home_pct:
                # Calculate money vs bets divergence
                fv.sharp_action_indicator = fv.public_money_home_pct - fv.public_spread_home_pct
        
        # Calculate RLM if not already set and we have the data
        if fv.is_reverse_line_move is None and fv.spread_movement and fv.public_spread_home_pct:
            # RLM: Line moves opposite to public betting
            # If public is on home (>55%) but line moved toward home (negative movement), that's RLM
            public_on_home = fv.public_spread_home_pct > 55
            line_moved_to_home = fv.spread_movement < -0.5
            line_moved_to_away = fv.spread_movement > 0.5
            
            fv.is_reverse_line_move = (
                (public_on_home and line_moved_to_home) or
                (not public_on_home and line_moved_to_away)
            )
    
    # =========================================================================
    # 5. SITUATIONAL FEATURES
    # =========================================================================
    
    async def _extract_situational_features(self, fv: MLFeatureVector):
        """Extract situational/spot features."""
        # Win/loss streaks
        if fv.home_team_id:
            fv.home_streak = await self._get_streak(
                fv.home_team_id, fv.scheduled_at, fv.sport_code
            )
            fv.home_season_game_num = await self._get_season_game_num(
                fv.home_team_id, fv.scheduled_at, fv.sport_code
            )
        
        if fv.away_team_id:
            fv.away_streak = await self._get_streak(
                fv.away_team_id, fv.scheduled_at, fv.sport_code
            )
            fv.away_season_game_num = await self._get_season_game_num(
                fv.away_team_id, fv.scheduled_at, fv.sport_code
            )
        
        # Revenge games
        if fv.home_team_id and fv.away_team_id:
            revenge = await self._check_revenge_game(
                fv.home_team_id, fv.away_team_id, fv.scheduled_at, fv.sport_code
            )
            if revenge:
                fv.home_is_revenge = revenge.get('home_revenge')
                fv.away_is_revenge = revenge.get('away_revenge')
        
        # Letdown spots (after big win)
        if fv.home_team_id:
            fv.home_letdown_spot = await self._check_letdown_spot(
                fv.home_team_id, fv.scheduled_at, fv.sport_code
            )
        if fv.away_team_id:
            fv.away_letdown_spot = await self._check_letdown_spot(
                fv.away_team_id, fv.scheduled_at, fv.sport_code
            )
    
    async def _get_streak(
        self, team_id: str, before_date: datetime, sport_code: str
    ) -> Optional[int]:
        """Get current win/loss streak (positive=wins, negative=losses)."""
        result = await self.session.execute(text("""
            SELECT 
                CASE WHEN home_master_team_id = :tid 
                     THEN CASE WHEN home_score > away_score THEN 1 ELSE -1 END
                     ELSE CASE WHEN away_score > home_score THEN 1 ELSE -1 END
                END as result
            FROM master_games
            WHERE (home_master_team_id = :tid OR away_master_team_id = :tid)
              AND scheduled_at < :before
              AND home_score IS NOT NULL
              AND sport_code = :sport
            ORDER BY scheduled_at DESC
            LIMIT 10
        """), {
            "tid": team_id,
            "before": before_date,
            "sport": sport_code,
        })
        
        rows = result.fetchall()
        if not rows:
            return 0
        
        streak = 0
        first_result = rows[0][0]
        
        for row in rows:
            if row[0] == first_result:
                streak += first_result
            else:
                break
        
        return streak
    
    async def _get_season_game_num(
        self, team_id: str, game_date: datetime, sport_code: str
    ) -> Optional[int]:
        """Get team's game number in the season."""
        # Approximate season start
        season_start = datetime(game_date.year, 1, 1)
        if game_date.month >= 9:  # Fall sports
            season_start = datetime(game_date.year, 9, 1)
        elif game_date.month <= 6:  # Continuing from prior year
            season_start = datetime(game_date.year - 1, 9, 1)
        
        result = await self.session.execute(text("""
            SELECT COUNT(*) + 1
            FROM master_games
            WHERE (home_master_team_id = :tid OR away_master_team_id = :tid)
              AND scheduled_at < :game_date
              AND scheduled_at >= :season_start
              AND home_score IS NOT NULL
              AND sport_code = :sport
        """), {
            "tid": team_id,
            "game_date": game_date,
            "season_start": season_start,
            "sport": sport_code,
        })
        
        return result.scalar()
    
    async def _check_revenge_game(
        self, home_team_id: str, away_team_id: str,
        game_date: datetime, sport_code: str
    ) -> Optional[Dict]:
        """Check if either team lost to the other in their last meeting."""
        result = await self.session.execute(text("""
            SELECT 
                home_master_team_id,
                away_master_team_id,
                home_score,
                away_score
            FROM master_games
            WHERE ((home_master_team_id = :home AND away_master_team_id = :away)
                OR (home_master_team_id = :away AND away_master_team_id = :home))
              AND scheduled_at < :before
              AND home_score IS NOT NULL
              AND sport_code = :sport
            ORDER BY scheduled_at DESC
            LIMIT 1
        """), {
            "home": home_team_id,
            "away": away_team_id,
            "before": game_date,
            "sport": sport_code,
        })
        
        row = result.fetchone()
        if not row:
            return None
        
        last_home_id, last_away_id, last_home_score, last_away_score = row
        
        # Who lost last time?
        last_home_won = last_home_score > last_away_score
        
        home_revenge = False
        away_revenge = False
        
        if str(last_home_id) == home_team_id:
            # Home team was home last time
            home_revenge = not last_home_won  # Lost last time
            away_revenge = last_home_won
        else:
            # Home team was away last time
            home_revenge = last_home_won  # Lost as away = home won
            away_revenge = not last_home_won
        
        return {
            'home_revenge': home_revenge,
            'away_revenge': away_revenge,
        }
    
    async def _check_letdown_spot(
        self, team_id: str, game_date: datetime, sport_code: str
    ) -> Optional[bool]:
        """Check if team is in a letdown spot (after big win)."""
        result = await self.session.execute(text("""
            SELECT 
                CASE WHEN home_master_team_id = :tid 
                     THEN home_score - away_score
                     ELSE away_score - home_score END as margin
            FROM master_games
            WHERE (home_master_team_id = :tid OR away_master_team_id = :tid)
              AND scheduled_at < :before
              AND home_score IS NOT NULL
              AND sport_code = :sport
            ORDER BY scheduled_at DESC
            LIMIT 1
        """), {
            "tid": team_id,
            "before": game_date,
            "sport": sport_code,
        })
        
        row = result.fetchone()
        if not row:
            return False
        
        # Big win = margin >= 15 points (adjust per sport)
        return row[0] >= 15
    
    # =========================================================================
    # 6. WEATHER FEATURES
    # =========================================================================
    
    async def _extract_weather_features(self, fv: MLFeatureVector):
        """Extract weather features for outdoor sports."""
        # Indoor sports - set is_dome and skip weather lookup
        indoor_sports = ['NBA', 'NHL', 'WNBA', 'NCAAB']
        if fv.sport_code in indoor_sports:
            fv.is_dome = True
            return
        
        # Try to get weather data from weather_data table
        try:
            result = await self.session.execute(text("""
                SELECT 
                    wd.temperature_f,
                    wd.wind_speed_mph,
                    wd.precipitation_pct,
                    wd.humidity_pct,
                    v.is_dome,
                    v.is_outdoor
                FROM weather_data wd
                LEFT JOIN venues v ON wd.venue_id = v.id
                WHERE wd.game_id IN (
                    SELECT gm.source_game_id 
                    FROM game_mappings gm 
                    WHERE gm.master_game_id = :mgid
                )
                LIMIT 1
            """), {"mgid": fv.master_game_id})
            
            row = result.fetchone()
            if row:
                fv.temperature_f = row[0]
                fv.wind_speed_mph = row[1]
                fv.precipitation_pct = row[2]
                fv.humidity_pct = row[3]
                if row[4] is not None:
                    fv.is_dome = row[4]
                elif row[5] is not None:
                    fv.is_dome = not row[5]
                return
        except Exception:
            # weather_data table may have different schema or not exist
            pass
        
        # Fallback: Try to get venue info from master_games
        try:
            result2 = await self.session.execute(text("""
                SELECT v.is_dome, v.is_outdoor
                FROM master_games mg
                LEFT JOIN venues v ON mg.venue_id = v.id
                WHERE mg.id = :mgid
            """), {"mgid": fv.master_game_id})
            
            row2 = result2.fetchone()
            if row2:
                if row2[0] is not None:
                    fv.is_dome = row2[0]
                elif row2[1] is not None:
                    fv.is_dome = not row2[1]
        except Exception:
            pass


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def extract_ml_features(
    session: AsyncSession,
    sport_code: str,
    **kwargs
) -> List[MLFeatureVector]:
    """Convenience function to extract all features."""
    service = MLFeatureService(session)
    return await service.extract_all_features(sport_code, **kwargs)


def features_to_dataframe(features: List[MLFeatureVector]):
    """Convert feature list to pandas DataFrame."""
    import pandas as pd
    rows = [f.to_dict() for f in features]
    df = pd.DataFrame(rows)
    
    # Organize columns
    id_cols = ['master_game_id', 'sport_code', 'scheduled_at', 'season',
               'home_team_id', 'away_team_id', 'home_team_name', 'away_team_name']
    target_cols = ['home_win', 'total_points', 'score_margin', 'spread_result', 
                   'over_result', 'home_score', 'away_score']
    
    existing_ids = [c for c in id_cols if c in df.columns]
    existing_targets = [c for c in target_cols if c in df.columns]
    feature_cols = [c for c in df.columns if c not in existing_ids + existing_targets]
    
    return df[existing_ids + existing_targets + sorted(feature_cols)]