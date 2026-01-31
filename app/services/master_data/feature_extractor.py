"""
ROYALEY - ML Feature Extraction from Master Data
Reads unified data through master_games and produces clean feature vectors
ready for model training and prediction.

This replaces the old feature engineering that couldn't aggregate across sources.

Usage:
    extractor = MasterFeatureExtractor(session)
    features = await extractor.build_features("NBA", season=2025)
    df = features.to_dataframe()  # Ready for H2O / AutoGluon
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


@dataclass
class GameFeatureVector:
    """One row of features for a single master_game."""
    master_game_id: str
    sport_code: str
    scheduled_at: datetime
    season: Optional[int]

    # Target variables
    home_score: Optional[int] = None
    away_score: Optional[int] = None
    home_win: Optional[int] = None  # 1/0/None
    total_points: Optional[int] = None
    score_margin: Optional[int] = None  # home - away

    # Team identifiers
    home_team: Optional[str] = None
    away_team: Optional[str] = None

    # Odds features (~15)
    spread_open: Optional[float] = None
    spread_close: Optional[float] = None
    spread_movement: Optional[float] = None
    moneyline_home: Optional[int] = None
    moneyline_away: Optional[int] = None
    total_open: Optional[float] = None
    total_close: Optional[float] = None
    total_movement: Optional[float] = None
    pinnacle_spread: Optional[float] = None
    pinnacle_ml_home: Optional[int] = None
    pinnacle_total: Optional[float] = None
    num_books_with_odds: Optional[int] = None
    consensus_spread: Optional[float] = None
    consensus_total: Optional[float] = None
    implied_prob_home: Optional[float] = None

    # Public betting features (~10)
    public_spread_home_pct: Optional[float] = None
    public_ml_home_pct: Optional[float] = None
    public_total_over_pct: Optional[float] = None
    public_money_spread_home_pct: Optional[float] = None
    sharp_action_spread: Optional[bool] = None
    is_rlm_spread: Optional[bool] = None

    # Weather features (~5)
    temperature_f: Optional[float] = None
    wind_speed_mph: Optional[float] = None
    precipitation_pct: Optional[float] = None
    is_dome: Optional[bool] = None
    humidity_pct: Optional[float] = None

    # Injury features (~5)
    home_injuries_out: Optional[int] = None
    away_injuries_out: Optional[int] = None
    home_injury_impact: Optional[float] = None
    away_injury_impact: Optional[float] = None
    home_starter_out: Optional[int] = None
    away_starter_out: Optional[int] = None

    # All extra features stored as dict
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Flatten to dictionary for DataFrame creation."""
        d = {k: v for k, v in self.__dict__.items() if k != "extra"}
        d.update(self.extra)
        return d


class MasterFeatureExtractor:
    """
    Extracts ML-ready features from the unified master data layer.

    Key advantage: queries through master_game_id bypass all duplication
    and pull from ALL sources simultaneously.
    """

    def __init__(self, session: AsyncSession):
        self.session = session

    async def build_features(
        self,
        sport_code: str,
        season: Optional[int] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        completed_only: bool = True,
    ) -> List[GameFeatureVector]:
        """
        Build feature vectors for all games matching the filter.

        Args:
            sport_code: NFL, NBA, etc.
            season: Season year filter
            start_date: Start of date range
            end_date: End of date range
            completed_only: Only include games with final scores

        Returns:
            List of GameFeatureVector, one per master_game
        """
        # Get eligible master games
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

        games_result = await self.session.execute(text(f"""
            SELECT mg.id, mg.sport_code, mg.scheduled_at, mg.season,
                   mg.home_score, mg.away_score, mg.status,
                   mg.home_master_team_id, mg.away_master_team_id,
                   mg.venue_id, mg.is_playoff, mg.is_neutral_site,
                   ht.canonical_name as home_name,
                   at_.canonical_name as away_name
            FROM master_games mg
            LEFT JOIN master_teams ht ON mg.home_master_team_id = ht.id
            LEFT JOIN master_teams at_ ON mg.away_master_team_id = at_.id
            WHERE {where_clause}
            ORDER BY mg.scheduled_at
        """), params)

        games = games_result.fetchall()
        logger.info(f"Building features for {len(games)} {sport_code} games")

        features = []
        for g in games:
            fv = GameFeatureVector(
                master_game_id=str(g[0]),
                sport_code=g[1],
                scheduled_at=g[2],
                season=g[3],
                home_score=g[4],
                away_score=g[5],
                home_team=g[12],
                away_team=g[13],
            )

            # Compute target variables
            if fv.home_score is not None and fv.away_score is not None:
                fv.home_win = 1 if fv.home_score > fv.away_score else 0
                fv.total_points = fv.home_score + fv.away_score
                fv.score_margin = fv.home_score - fv.away_score

            fv.extra["is_playoff"] = g[10]
            fv.extra["is_neutral_site"] = g[11]

            # Attach odds features
            await self._attach_odds_features(fv)

            # Attach public betting features
            await self._attach_public_betting_features(fv)

            # Attach weather features
            await self._attach_weather_features(fv, g[9])  # venue_id

            # Attach injury features
            await self._attach_injury_features(fv, str(g[7]) if g[7] else None, str(g[8]) if g[8] else None)

            features.append(fv)

        logger.info(f"Built {len(features)} feature vectors")
        return features

    async def _attach_odds_features(self, fv: GameFeatureVector):
        """Pull odds features for a master game — aggregated across ALL sportsbooks."""
        result = await self.session.execute(text("""
            SELECT o.bet_type, o.home_line, o.away_line,
                   o.home_odds, o.away_odds, o.total,
                   o.over_odds, o.under_odds, o.is_opening,
                   o.sportsbook_key, o.recorded_at,
                   s.is_sharp, s.priority
            FROM odds o
            LEFT JOIN sportsbooks s ON o.sportsbook_id = s.id
            WHERE o.master_game_id = :mgid
            ORDER BY o.recorded_at
        """), {"mgid": fv.master_game_id})
        rows = result.fetchall()

        if not rows:
            return

        books_seen = set()
        spreads = []
        totals = []
        pinnacle_spread = None
        pinnacle_ml_home = None
        pinnacle_total = None

        for r in rows:
            bet_type = r[0]
            book_key = r[9]
            is_sharp = r[11]
            books_seen.add(book_key)

            if bet_type == "spread":
                spreads.append({
                    "line": r[1], "is_opening": r[8],
                    "home_odds": r[3], "away_odds": r[4],
                })
                if is_sharp and pinnacle_spread is None:
                    pinnacle_spread = r[1]
            elif bet_type == "moneyline":
                if fv.moneyline_home is None:
                    fv.moneyline_home = r[3]
                    fv.moneyline_away = r[4]
                if is_sharp and pinnacle_ml_home is None:
                    pinnacle_ml_home = r[3]
            elif bet_type == "total":
                totals.append({"total": r[5], "is_opening": r[8]})
                if is_sharp and pinnacle_total is None:
                    pinnacle_total = r[5]

        # Compute spread features
        if spreads:
            openers = [s["line"] for s in spreads if s["is_opening"] and s["line"] is not None]
            closers = [s["line"] for s in spreads if not s["is_opening"] and s["line"] is not None]
            all_lines = [s["line"] for s in spreads if s["line"] is not None]

            fv.spread_open = openers[0] if openers else (all_lines[0] if all_lines else None)
            fv.spread_close = closers[-1] if closers else (all_lines[-1] if all_lines else None)
            if fv.spread_open is not None and fv.spread_close is not None:
                fv.spread_movement = fv.spread_close - fv.spread_open
            if all_lines:
                fv.consensus_spread = sum(all_lines) / len(all_lines)

        # Compute total features
        if totals:
            open_totals = [t["total"] for t in totals if t["is_opening"] and t["total"] is not None]
            close_totals = [t["total"] for t in totals if not t["is_opening"] and t["total"] is not None]
            all_totals = [t["total"] for t in totals if t["total"] is not None]

            fv.total_open = open_totals[0] if open_totals else (all_totals[0] if all_totals else None)
            fv.total_close = close_totals[-1] if close_totals else (all_totals[-1] if all_totals else None)
            if fv.total_open is not None and fv.total_close is not None:
                fv.total_movement = fv.total_close - fv.total_open
            if all_totals:
                fv.consensus_total = sum(all_totals) / len(all_totals)

        fv.pinnacle_spread = pinnacle_spread
        fv.pinnacle_ml_home = pinnacle_ml_home
        fv.pinnacle_total = pinnacle_total
        fv.num_books_with_odds = len(books_seen)

        # Implied probability from moneyline
        if fv.moneyline_home and fv.moneyline_home != 0:
            if fv.moneyline_home < 0:
                fv.implied_prob_home = abs(fv.moneyline_home) / (abs(fv.moneyline_home) + 100)
            else:
                fv.implied_prob_home = 100 / (fv.moneyline_home + 100)

    async def _attach_public_betting_features(self, fv: GameFeatureVector):
        """Pull public betting percentages."""
        result = await self.session.execute(text("""
            SELECT spread_home_bet_pct, ml_home_bet_pct, total_over_bet_pct,
                   spread_home_money_pct, is_sharp_spread, is_rlm_spread
            FROM public_betting
            WHERE master_game_id = :mgid
            LIMIT 1
        """), {"mgid": fv.master_game_id})
        row = result.fetchone()

        if row:
            fv.public_spread_home_pct = row[0]
            fv.public_ml_home_pct = row[1]
            fv.public_total_over_pct = row[2]
            fv.public_money_spread_home_pct = row[3]
            fv.sharp_action_spread = row[4]
            fv.is_rlm_spread = row[5]

    async def _attach_weather_features(self, fv: GameFeatureVector, venue_id: Optional[str]):
        """Pull weather data for outdoor games."""
        result = await self.session.execute(text("""
            SELECT wd.temperature_f, wd.wind_speed_mph, wd.precipitation_pct,
                   wd.is_dome, wd.humidity_pct
            FROM weather_data wd
            JOIN games g ON wd.game_id = g.id
            WHERE g.master_game_id = :mgid
            LIMIT 1
        """), {"mgid": fv.master_game_id})
        row = result.fetchone()

        if row:
            fv.temperature_f = row[0]
            fv.wind_speed_mph = row[1]
            fv.precipitation_pct = row[2]
            fv.is_dome = row[3]
            fv.humidity_pct = row[4]

    async def _attach_injury_features(
        self, fv: GameFeatureVector,
        home_team_id: Optional[str], away_team_id: Optional[str]
    ):
        """Pull injury counts for both teams near game time."""
        if not home_team_id or not away_team_id:
            return

        for side, mt_id in [("home", home_team_id), ("away", away_team_id)]:
            result = await self.session.execute(text("""
                SELECT COUNT(*) FILTER (WHERE i.status IN ('Out', 'IR', 'Suspended')) as num_out,
                       SUM(i.impact_score) FILTER (WHERE i.status IN ('Out', 'IR', 'Suspended')) as total_impact,
                       COUNT(*) FILTER (WHERE i.is_starter AND i.status IN ('Out', 'IR', 'Suspended')) as starters_out
                FROM injuries i
                WHERE (i.master_team_id = :mtid OR i.team_id IN (
                    SELECT source_team_db_id FROM team_mappings WHERE master_team_id = :mtid
                ))
                AND i.last_updated >= :cutoff
            """), {
                "mtid": mt_id,
                "cutoff": fv.scheduled_at - timedelta(days=7),
            })
            row = result.fetchone()

            if row:
                if side == "home":
                    fv.home_injuries_out = row[0] or 0
                    fv.home_injury_impact = float(row[1] or 0)
                    fv.home_starter_out = row[2] or 0
                else:
                    fv.away_injuries_out = row[0] or 0
                    fv.away_injury_impact = float(row[1] or 0)
                    fv.away_starter_out = row[2] or 0

    @staticmethod
    def to_dataframe(features: List[GameFeatureVector]):
        """Convert feature list to pandas DataFrame (import pandas lazily)."""
        import pandas as pd
        rows = [f.to_dict() for f in features]
        df = pd.DataFrame(rows)
        # Sort columns: targets first, then features
        target_cols = ["home_win", "total_points", "score_margin", "home_score", "away_score"]
        id_cols = ["master_game_id", "sport_code", "scheduled_at", "season", "home_team", "away_team"]
        existing_targets = [c for c in target_cols if c in df.columns]
        existing_ids = [c for c in id_cols if c in df.columns]
        other_cols = [c for c in df.columns if c not in existing_targets + existing_ids]
        df = df[existing_ids + existing_targets + sorted(other_cols)]
        return df
