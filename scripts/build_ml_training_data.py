"""
ROYALEY - Build ML Training Dataset
Materializes features from master_odds + public_betting + weather + injuries
into the flat ml_training_dataset table, ready for H2O / AutoGluon.

Run: python -m scripts.build_ml_training_data
     python -m scripts.build_ml_training_data --sport NFL --season 2024
     python -m scripts.build_ml_training_data --export csv

Flow:
  master_games → join master_odds (aggregated across books)
               → join public_betting
               → join weather_data
               → join injuries
               → INSERT into ml_training_dataset

Output: One row per master_game with ~45 feature columns.
"""

import argparse
import asyncio
import csv
import logging
import sys
import os
from datetime import datetime, timedelta
from uuid import uuid4

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text
from app.core.database import db_manager

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# HELPER: De-vig for implied probability
# =============================================================================

def american_to_implied(odds):
    """Convert American odds to implied probability."""
    if not odds or odds == 0:
        return None
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    return 100 / (odds + 100)


# =============================================================================
# MAIN BUILD
# =============================================================================

async def build_training_data(sport_filter=None, season_filter=None, completed_only=True):
    """Build or rebuild the ml_training_dataset table."""
    await db_manager.initialize()

    async with db_manager.async_session() as session:

        # ── Get eligible master games ──
        conditions = []
        params = {}

        if completed_only:
            conditions.append("mg.home_score IS NOT NULL")
            conditions.append("mg.status = 'final'")
        if sport_filter:
            conditions.append("mg.sport_code = :sport")
            params["sport"] = sport_filter
        if season_filter:
            conditions.append("mg.season = :season")
            params["season"] = season_filter

        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""

        count_result = await session.execute(text(f"""
            SELECT COUNT(*) FROM master_games mg {where_clause}
        """), params)
        total = count_result.scalar()
        logger.info(f"📊 Master games to process: {total:,}")

        if total == 0:
            logger.warning("⚠️  No master games match filters. Nothing to build.")
            await db_manager.close()
            return

        # ── Process in batches ──
        batch_size = 1000
        offset = 0
        total_built = 0
        total_with_odds = 0
        total_with_weather = 0
        total_with_injuries = 0
        total_with_betting = 0

        while offset < total:
            games_result = await session.execute(text(f"""
                SELECT mg.id, mg.sport_code, mg.season, mg.scheduled_at,
                       mg.home_score, mg.away_score, mg.status,
                       mg.home_master_team_id, mg.away_master_team_id,
                       mg.venue_id, mg.is_playoff, mg.is_neutral_site,
                       ht.canonical_name as home_name,
                       at_.canonical_name as away_name
                FROM master_games mg
                LEFT JOIN master_teams ht ON mg.home_master_team_id = ht.id
                LEFT JOIN master_teams at_ ON mg.away_master_team_id = at_.id
                {where_clause}
                ORDER BY mg.scheduled_at
                LIMIT :lim OFFSET :off
            """), {**params, "lim": batch_size, "off": offset})

            games = games_result.fetchall()
            if not games:
                break

            for g in games:
                mgid = str(g[0])
                sport_code = g[1]
                season = g[2]
                scheduled_at = g[3]
                home_score = g[4]
                away_score = g[5]
                home_name = g[12]
                away_name = g[13]
                home_mt_id = str(g[7]) if g[7] else None
                away_mt_id = str(g[8]) if g[8] else None
                venue_id = g[9]
                is_playoff = g[10]
                is_neutral = g[11]

                # Targets
                home_win = None
                total_points = None
                score_margin = None
                if home_score is not None and away_score is not None:
                    home_win = 1 if home_score > away_score else 0
                    total_points = home_score + away_score
                    score_margin = home_score - away_score

                # ── ODDS from master_odds (aggregated) ──
                odds_features = await _get_odds_features(session, mgid)
                if odds_features["num_books_with_odds"]:
                    total_with_odds += 1

                # ── PUBLIC BETTING ──
                betting = await _get_betting_features(session, mgid)
                if betting["public_spread_home_pct"] is not None:
                    total_with_betting += 1

                # ── WEATHER ──
                weather = await _get_weather_features(session, mgid)
                if weather["temperature_f"] is not None:
                    total_with_weather += 1

                # ── INJURIES ──
                injuries = await _get_injury_features(session, home_mt_id, away_mt_id, scheduled_at)
                if injuries["home_injuries_out"] is not None:
                    total_with_injuries += 1

                # ── UPSERT into ml_training_dataset ──
                await session.execute(text("""
                    INSERT INTO ml_training_dataset (
                        id, master_game_id, sport_code, season, scheduled_at,
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
                        is_playoff, is_neutral_site,
                        feature_version, computed_at
                    ) VALUES (
                        :id, :mgid, :sport, :season, :sched,
                        :ht, :at,
                        :hs, :as_, :hw, :tp, :sm,
                        :so, :sc, :smv,
                        :mlh, :mla,
                        :to_, :tc, :tmv,
                        :ps, :pmh, :pt,
                        :nb, :cs, :ct,
                        :iph, :nvp,
                        :psh, :pmml, :pto,
                        :pmsp, :sai, :rlm,
                        :tf, :ws, :pp, :dome, :hp,
                        :hio, :aio, :hii, :aii,
                        :hso, :aso,
                        :isp, :isn,
                        '2.0', NOW()
                    )
                    ON CONFLICT (master_game_id) DO UPDATE SET
                        home_score = EXCLUDED.home_score,
                        away_score = EXCLUDED.away_score,
                        home_win = EXCLUDED.home_win,
                        total_points = EXCLUDED.total_points,
                        score_margin = EXCLUDED.score_margin,
                        spread_open = EXCLUDED.spread_open,
                        spread_close = EXCLUDED.spread_close,
                        spread_movement = EXCLUDED.spread_movement,
                        moneyline_home = EXCLUDED.moneyline_home,
                        moneyline_away = EXCLUDED.moneyline_away,
                        total_open = EXCLUDED.total_open,
                        total_close = EXCLUDED.total_close,
                        total_movement = EXCLUDED.total_movement,
                        pinnacle_spread = EXCLUDED.pinnacle_spread,
                        pinnacle_ml_home = EXCLUDED.pinnacle_ml_home,
                        pinnacle_total = EXCLUDED.pinnacle_total,
                        num_books_with_odds = EXCLUDED.num_books_with_odds,
                        consensus_spread = EXCLUDED.consensus_spread,
                        consensus_total = EXCLUDED.consensus_total,
                        implied_prob_home = EXCLUDED.implied_prob_home,
                        no_vig_prob_home = EXCLUDED.no_vig_prob_home,
                        public_spread_home_pct = EXCLUDED.public_spread_home_pct,
                        public_ml_home_pct = EXCLUDED.public_ml_home_pct,
                        public_total_over_pct = EXCLUDED.public_total_over_pct,
                        public_money_spread_home_pct = EXCLUDED.public_money_spread_home_pct,
                        sharp_action_indicator = EXCLUDED.sharp_action_indicator,
                        is_rlm_spread = EXCLUDED.is_rlm_spread,
                        temperature_f = EXCLUDED.temperature_f,
                        wind_speed_mph = EXCLUDED.wind_speed_mph,
                        precipitation_pct = EXCLUDED.precipitation_pct,
                        is_dome = EXCLUDED.is_dome,
                        humidity_pct = EXCLUDED.humidity_pct,
                        home_injuries_out = EXCLUDED.home_injuries_out,
                        away_injuries_out = EXCLUDED.away_injuries_out,
                        home_injury_impact = EXCLUDED.home_injury_impact,
                        away_injury_impact = EXCLUDED.away_injury_impact,
                        home_starter_out = EXCLUDED.home_starter_out,
                        away_starter_out = EXCLUDED.away_starter_out,
                        is_playoff = EXCLUDED.is_playoff,
                        is_neutral_site = EXCLUDED.is_neutral_site,
                        computed_at = NOW()
                """), {
                    "id": str(uuid4()), "mgid": mgid, "sport": sport_code,
                    "season": season, "sched": scheduled_at,
                    "ht": home_name, "at": away_name,
                    "hs": home_score, "as_": away_score, "hw": home_win,
                    "tp": total_points, "sm": score_margin,
                    **odds_features, **betting, **weather, **injuries,
                    "isp": is_playoff, "isn": is_neutral,
                })

                total_built += 1

            await session.commit()
            offset += batch_size
            logger.info(f"   ... built {total_built:,}/{total:,} rows")

        # ── Final report ──
        logger.info("")
        logger.info("=" * 60)
        logger.info("✅ ML TRAINING DATASET BUILD COMPLETE")
        logger.info(f"   Total rows built:      {total_built:>10,}")
        logger.info(f"   With odds:             {total_with_odds:>10,} ({_pct(total_with_odds, total_built)})")
        logger.info(f"   With public betting:   {total_with_betting:>10,} ({_pct(total_with_betting, total_built)})")
        logger.info(f"   With weather:          {total_with_weather:>10,} ({_pct(total_with_weather, total_built)})")
        logger.info(f"   With injuries:         {total_with_injuries:>10,} ({_pct(total_with_injuries, total_built)})")
        logger.info("")
        ml_ready = total_with_odds  # games with odds are trainable
        logger.info(f"   🏆 ML-TRAINABLE ROWS:  {ml_ready:>10,}")
        logger.info("=" * 60)

    await db_manager.close()
    return total_built


def _pct(part, total):
    if total == 0:
        return "0.0%"
    return f"{100 * part / total:.1f}%"


# =============================================================================
# FEATURE EXTRACTION HELPERS (read from master_odds)
# =============================================================================

async def _get_odds_features(session, master_game_id: str) -> dict:
    """Aggregate odds features from master_odds for one game."""
    result = await session.execute(text("""
        SELECT mo.sportsbook_key, mo.bet_type,
               mo.opening_line, mo.closing_line,
               mo.opening_odds_home, mo.opening_odds_away,
               mo.closing_odds_home, mo.closing_odds_away,
               mo.opening_total, mo.closing_total,
               mo.line_movement, mo.no_vig_prob_home,
               mo.is_sharp
        FROM master_odds mo
        WHERE mo.master_game_id = :mgid AND mo.period = 'full'
        ORDER BY mo.is_sharp DESC
    """), {"mgid": master_game_id})
    rows = result.fetchall()

    f = {
        "so": None, "sc": None, "smv": None,
        "mlh": None, "mla": None,
        "to_": None, "tc": None, "tmv": None,
        "ps": None, "pmh": None, "pt": None,
        "nb": None, "cs": None, "ct": None,
        "iph": None, "nvp": None,
    }

    if not rows:
        f["nb"] = 0
        return f

    books_seen = set()
    all_spreads = []
    all_totals = []

    for r in rows:
        book, bt = r[0], r[1]
        books_seen.add(book)
        is_sharp = r[12]

        if bt == "spread":
            if r[2] is not None:
                all_spreads.append(r[2])  # opening_line
            if r[3] is not None:
                all_spreads.append(r[3])  # closing_line

            # Use first spread found (sharp books come first due to ORDER BY)
            if f["so"] is None and r[2] is not None:
                f["so"] = r[2]
            if r[3] is not None:
                f["sc"] = r[3]  # keep updating to last closing
            if r[10] is not None and f["smv"] is None:
                f["smv"] = r[10]

            if is_sharp and f["ps"] is None:
                f["ps"] = r[3] if r[3] is not None else r[2]

        elif bt == "moneyline":
            if f["mlh"] is None:
                f["mlh"] = r[6] if r[6] else r[4]  # closing or opening
                f["mla"] = r[7] if r[7] else r[5]

            if is_sharp and f["pmh"] is None:
                f["pmh"] = r[6] if r[6] else r[4]

            if f["nvp"] is None and r[11] is not None:
                f["nvp"] = r[11]
            elif f["nvp"] is None and f["mlh"]:
                f["nvp"] = american_to_implied(f["mlh"])

        elif bt == "total":
            if r[8] is not None:
                all_totals.append(r[8])
            if r[9] is not None:
                all_totals.append(r[9])

            if f["to_"] is None and r[8] is not None:
                f["to_"] = r[8]
            if r[9] is not None:
                f["tc"] = r[9]
            if r[10] is not None and f["tmv"] is None:
                f["tmv"] = r[10]

            if is_sharp and f["pt"] is None:
                f["pt"] = r[9] if r[9] is not None else r[8]

    f["nb"] = len(books_seen)

    # Consensus lines (average across all books)
    if all_spreads:
        f["cs"] = round(sum(all_spreads) / len(all_spreads), 2)
    if all_totals:
        f["ct"] = round(sum(all_totals) / len(all_totals), 2)

    # Implied probability from moneyline
    if f["mlh"] and not f["iph"]:
        f["iph"] = american_to_implied(f["mlh"])
    if f["iph"]:
        f["iph"] = round(f["iph"], 6)
    if f["nvp"]:
        f["nvp"] = round(f["nvp"], 6)

    return f


async def _get_betting_features(session, master_game_id: str) -> dict:
    """Pull public betting features."""
    result = await session.execute(text("""
        SELECT spread_home_bet_pct, ml_home_bet_pct, total_over_bet_pct,
               spread_home_money_pct
        FROM public_betting
        WHERE master_game_id = :mgid
        LIMIT 1
    """), {"mgid": master_game_id})
    row = result.fetchone()

    f = {"psh": None, "pmml": None, "pto": None, "pmsp": None, "sai": None, "rlm": None}
    if row:
        f["psh"] = row[0]
        f["pmml"] = row[1]
        f["pto"] = row[2]
        f["pmsp"] = row[3]
        # Detect sharp action: money% > bet% by 10+ points indicates sharp money
        if f["pmsp"] is not None and f["psh"] is not None:
            f["sai"] = (f["pmsp"] - f["psh"]) > 10
        # RLM: line moves against public (public on home but line moves away)
        f["rlm"] = False
    return f


async def _get_weather_features(session, master_game_id: str) -> dict:
    """Pull weather data via games.master_game_id."""
    result = await session.execute(text("""
        SELECT wd.temperature_f, wd.wind_speed_mph, wd.precipitation_pct,
               wd.is_dome, wd.humidity_pct
        FROM weather_data wd
        JOIN games g ON wd.game_id = g.id
        WHERE g.master_game_id = :mgid
        LIMIT 1
    """), {"mgid": master_game_id})
    row = result.fetchone()

    f = {"tf": None, "ws": None, "pp": None, "dome": None, "hp": None}
    if row:
        f["tf"] = row[0]
        f["ws"] = row[1]
        f["pp"] = row[2]
        f["dome"] = row[3]
        f["hp"] = row[4]
    return f


async def _get_injury_features(session, home_mt_id, away_mt_id, scheduled_at) -> dict:
    """Pull injury counts for both teams."""
    f = {"hio": None, "aio": None, "hii": None, "aii": None, "hso": None, "aso": None}

    if not home_mt_id or not away_mt_id:
        return f

    cutoff = scheduled_at - timedelta(days=7) if scheduled_at else None
    if not cutoff:
        return f

    for side, mt_id in [("h", home_mt_id), ("a", away_mt_id)]:
        try:
            result = await session.execute(text("""
                SELECT COUNT(*) FILTER (WHERE i.status IN ('Out', 'IR', 'Suspended')) as num_out,
                       COALESCE(SUM(i.impact_score) FILTER (WHERE i.status IN ('Out', 'IR', 'Suspended')), 0) as total_impact,
                       COUNT(*) FILTER (WHERE i.is_starter AND i.status IN ('Out', 'IR', 'Suspended')) as starters_out
                FROM injuries i
                WHERE (i.master_team_id = :mtid OR i.team_id IN (
                    SELECT source_team_db_id FROM team_mappings WHERE master_team_id = :mtid
                ))
                AND i.last_updated >= :cutoff
            """), {"mtid": mt_id, "cutoff": cutoff})
            row = result.fetchone()
            if row:
                if side == "h":
                    f["hio"] = row[0] or 0
                    f["hii"] = float(row[1] or 0)
                    f["hso"] = row[2] or 0
                else:
                    f["aio"] = row[0] or 0
                    f["aii"] = float(row[1] or 0)
                    f["aso"] = row[2] or 0
        except Exception as e:
            logger.debug(f"Injury query failed for {mt_id}: {e}")

    return f


# =============================================================================
# EXPORT TO CSV
# =============================================================================

async def export_to_csv(output_path: str, sport_filter=None, season_filter=None):
    """Export ml_training_dataset to CSV for H2O / AutoGluon."""
    await db_manager.initialize()

    async with db_manager.async_session() as session:
        conditions = []
        params = {}
        if sport_filter:
            conditions.append("sport_code = :sport")
            params["sport"] = sport_filter
        if season_filter:
            conditions.append("season = :season")
            params["season"] = season_filter

        where = "WHERE " + " AND ".join(conditions) if conditions else ""

        result = await session.execute(text(f"""
            SELECT sport_code, season, scheduled_at, home_team, away_team,
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
            {where}
            ORDER BY scheduled_at
        """), params)
        rows = result.fetchall()

        headers = [
            "sport_code", "season", "scheduled_at", "home_team", "away_team",
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

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for row in rows:
                writer.writerow(list(row))

        logger.info(f"✅ Exported {len(rows):,} rows to {output_path}")

    await db_manager.close()


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Build ML Training Dataset")
    parser.add_argument("--sport", type=str, help="Filter by sport code (e.g. NFL)")
    parser.add_argument("--season", type=int, help="Filter by season year")
    parser.add_argument("--export", type=str, choices=["csv"], help="Export format")
    parser.add_argument("--output", type=str, default="ml_training_data.csv", help="Export file path")
    parser.add_argument("--include-scheduled", action="store_true", help="Include non-completed games")

    args = parser.parse_args()

    if args.export == "csv":
        asyncio.run(export_to_csv(args.output, args.sport, args.season))
    else:
        asyncio.run(build_training_data(
            sport_filter=args.sport,
            season_filter=args.season,
            completed_only=not args.include_scheduled,
        ))


if __name__ == "__main__":
    main()
