"""
ROYALEY - ML Feature Extraction from Master Data
Reads unified data through master_games + master_odds and produces clean
feature vectors ready for model training and prediction.

TWO MODES:
  1. From master_odds (live): Builds features on-the-fly from master_odds table
  2. From ml_training_dataset (fast): Reads pre-computed features from materialized table

Usage:
    extractor = MasterFeatureExtractor(session)

    # Mode 1: Build from master_odds (accurate, slower)
    features = await extractor.build_features("NBA", season=2025)
    df = MasterFeatureExtractor.to_dataframe(features)

    # Mode 2: Read from materialized table (fast, for training)
    df = await extractor.load_training_data("NFL", season=2024)
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

    # Odds features (~16) — from master_odds
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
    no_vig_prob_home: Optional[float] = None

    # Public betting features (~6)
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

    # Injury features (~6)
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

    Key advantage over old system:
    - Reads from master_odds (deduplicated, with open/close/movement)
    - No more averaging duplicate odds from multiple collectors
    - Pinnacle lines explicitly separated from consensus
    """

    def __init__(self, session: AsyncSession):
        self.session = session

    # =========================================================================
    # MODE 1: Build features on-the-fly from master_odds
    # =========================================================================

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
        Reads from master_odds (deduplicated) rather than raw odds.
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

            # Attach features from master data tables
            await self._attach_odds_features(fv)
            await self._attach_public_betting_features(fv)
            await self._attach_weather_features(fv, g[9])  # venue_id
            await self._attach_injury_features(fv, str(g[7]) if g[7] else None, str(g[8]) if g[8] else None)

            features.append(fv)

        logger.info(f"Built {len(features)} feature vectors")
        return features

    async def _attach_odds_features(self, fv: GameFeatureVector):
        """
        Pull odds features from master_odds — one clean row per (game x book x bet_type).
        No more duplicates from multiple collectors.
        """
        result = await self.session.execute(text("""
            SELECT mo.sportsbook_key, mo.bet_type,
                   mo.opening_line, mo.closing_line,
                   mo.opening_odds_home, mo.opening_odds_away,
                   mo.closing_odds_home, mo.closing_odds_away,
                   mo.opening_total, mo.closing_total,
                   mo.line_movement, mo.no_vig_prob_home,
                   mo.is_sharp
            FROM master_odds mo
            WHERE mo.master_game_id = :mgid
              AND mo.period = 'full'
            ORDER BY mo.is_sharp DESC, mo.sportsbook_key
        """), {"mgid": fv.master_game_id})
        rows = result.fetchall()

        if not rows:
            fv.num_books_with_odds = 0
            return

        books_seen = set()
        all_spreads = []
        all_totals = []

        for r in rows:
            book_key = r[0]
            bet_type = r[1]
            is_sharp = r[12]
            books_seen.add(book_key)

            if bet_type == "spread":
                closing = r[3]  # closing_line
                opening = r[2]  # opening_line
                if closing is not None:
                    all_spreads.append(closing)
                elif opening is not None:
                    all_spreads.append(opening)

                # First spread found (sharp books first due to ORDER BY)
                if fv.spread_open is None and opening is not None:
                    fv.spread_open = opening
                if closing is not None:
                    fv.spread_close = closing
                if r[10] is not None and fv.spread_movement is None:
                    fv.spread_movement = r[10]

                if is_sharp and fv.pinnacle_spread is None:
                    fv.pinnacle_spread = closing if closing else opening

            elif bet_type == "moneyline":
                c_home = r[6]  # closing_odds_home
                c_away = r[7]
                o_home = r[4]  # opening_odds_home
                o_away = r[5]

                if fv.moneyline_home is None:
                    fv.moneyline_home = c_home or o_home
                    fv.moneyline_away = c_away or o_away

                if is_sharp and fv.pinnacle_ml_home is None:
                    fv.pinnacle_ml_home = c_home or o_home

                if fv.no_vig_prob_home is None and r[11] is not None:
                    fv.no_vig_prob_home = r[11]

            elif bet_type == "total":
                c_total = r[9]  # closing_total
                o_total = r[8]  # opening_total
                if c_total is not None:
                    all_totals.append(c_total)
                elif o_total is not None:
                    all_totals.append(o_total)

                if fv.total_open is None and o_total is not None:
                    fv.total_open = o_total
                if c_total is not None:
                    fv.total_close = c_total
                if r[10] is not None and fv.total_movement is None:
                    fv.total_movement = r[10]

                if is_sharp and fv.pinnacle_total is None:
                    fv.pinnacle_total = c_total or o_total

        fv.num_books_with_odds = len(books_seen)

        # Consensus (average across all books)
        if all_spreads:
            fv.consensus_spread = round(sum(all_spreads) / len(all_spreads), 2)
        if all_totals:
            fv.consensus_total = round(sum(all_totals) / len(all_totals), 2)

        # Implied probability
        if fv.moneyline_home and fv.moneyline_home != 0:
            if fv.moneyline_home < 0:
                fv.implied_prob_home = round(abs(fv.moneyline_home) / (abs(fv.moneyline_home) + 100), 6)
            else:
                fv.implied_prob_home = round(100 / (fv.moneyline_home + 100), 6)

    async def _attach_public_betting_features(self, fv: GameFeatureVector):
        """Pull public betting percentages."""
        result = await self.session.execute(text("""
            SELECT spread_home_bet_pct, ml_home_bet_pct, total_over_bet_pct,
                   spread_home_money_pct
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
            # Sharp action: money% significantly higher than bet%
            if row[3] is not None and row[0] is not None:
                fv.sharp_action_spread = (row[3] - row[0]) > 10

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
                       COALESCE(SUM(i.impact_score) FILTER (WHERE i.status IN ('Out', 'IR', 'Suspended')), 0) as total_impact,
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

    # =========================================================================
    # MODE 2: Load from materialized ml_training_dataset (fast)
    # =========================================================================

    async def load_training_data(
        self,
        sport_code: str,
        season: Optional[int] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ):
        """
        Fast path: load pre-computed features from ml_training_dataset.
        Returns pandas DataFrame ready for H2O / AutoGluon.

        Prerequisite: Run `python -m scripts.build_ml_training_data` first.
        """
        import pandas as pd

        conditions = ["sport_code = :sport"]
        params: Dict[str, Any] = {"sport": sport_code}

        if season:
            conditions.append("season = :season")
            params["season"] = season
        if start_date:
            conditions.append("scheduled_at >= :start")
            params["start"] = start_date
        if end_date:
            conditions.append("scheduled_at <= :end")
            params["end"] = end_date

        where_clause = " AND ".join(conditions)

        result = await self.session.execute(text(f"""
            SELECT master_game_id, sport_code, season, scheduled_at,
                   home_team, away_team,
                   home_score, away_score, home_win, total_points, score_margin,
                   spread_open, spread_close, spread_movement,
                   moneyline_home, moneyline_away,
                   total_open, total_close, total_movement,
                   pinnacle_spread, pinnacle_ml_home, pinnacle_total,
                   num_books_with_odds, consensus_spread, consensus_total,
                   implied_prob_home, no_vig_prob_home,
                   public_spread_home_pct, public_ml_home_pct, public_total_over_pct,
                   public_money_spread_home_pct, sharp_action_indicator, is_rlm_spread,
                   temperature_f, wind_speed_mph, precipitation_pct, is_dome, humidity_pct,
                   home_injuries_out, away_injuries_out, home_injury_impact, away_injury_impact,
                   home_starter_out, away_starter_out,
                   is_playoff, is_neutral_site
            FROM ml_training_dataset
            WHERE {where_clause}
            ORDER BY scheduled_at
        """), params)

        columns = [
            "master_game_id", "sport_code", "season", "scheduled_at",
            "home_team", "away_team",
            "home_score", "away_score", "home_win", "total_points", "score_margin",
            "spread_open", "spread_close", "spread_movement",
            "moneyline_home", "moneyline_away",
            "total_open", "total_close", "total_movement",
            "pinnacle_spread", "pinnacle_ml_home", "pinnacle_total",
            "num_books_with_odds", "consensus_spread", "consensus_total",
            "implied_prob_home", "no_vig_prob_home",
            "public_spread_home_pct", "public_ml_home_pct", "public_total_over_pct",
            "public_money_spread_home_pct", "sharp_action_indicator", "is_rlm_spread",
            "temperature_f", "wind_speed_mph", "precipitation_pct", "is_dome", "humidity_pct",
            "home_injuries_out", "away_injuries_out", "home_injury_impact", "away_injury_impact",
            "home_starter_out", "away_starter_out",
            "is_playoff", "is_neutral_site",
        ]

        rows = result.fetchall()
        df = pd.DataFrame(rows, columns=columns)
        logger.info(f"Loaded {len(df)} training rows for {sport_code}" +
                    (f" season={season}" if season else ""))
        return df

    # =========================================================================
    # UTILITIES
    # =========================================================================

    @staticmethod
    def to_dataframe(features: List[GameFeatureVector]):
        """Convert feature list to pandas DataFrame (import pandas lazily)."""
        import pandas as pd
        rows = [f.to_dict() for f in features]
        df = pd.DataFrame(rows)
        # Sort columns: IDs first, targets, then features alphabetically
        target_cols = ["home_win", "total_points", "score_margin", "home_score", "away_score"]
        id_cols = ["master_game_id", "sport_code", "scheduled_at", "season", "home_team", "away_team"]
        existing_targets = [c for c in target_cols if c in df.columns]
        existing_ids = [c for c in id_cols if c in df.columns]
        other_cols = [c for c in df.columns if c not in existing_targets + existing_ids]
        df = df[existing_ids + existing_targets + sorted(other_cols)]
        return df