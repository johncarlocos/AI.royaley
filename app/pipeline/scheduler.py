"""
ROYALEY - Prediction Pipeline Scheduler
Runs as a background service with 3 automated jobs:

  1. ODDS REFRESH    - Fetches latest odds → updates upcoming_odds → "Circa." column changes
  2. CLOSING CAPTURE - Snapshots lines before game starts → enables CLV calculation
  3. GAME GRADING    - Fetches scores → grades W/L/P → calculates CLV + profit/loss

Budget: Free tier = 500 requests/month. Each sport × 3 markets = 3 requests.
Default schedule uses ~350-400 requests/month with 2-3 active sports.

Usage:
    docker exec royaley_api python -m app.pipeline.scheduler
    docker exec royaley_api python -m app.pipeline.scheduler --once    # Run all jobs once and exit
    docker exec royaley_api python -m app.pipeline.scheduler --status  # Show API quota + job stats
"""

import asyncio
import argparse
import logging
import sys
from datetime import datetime, timezone, timedelta
from typing import Optional, List

import httpx
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from app.core.config import settings, ODDS_API_SPORT_KEYS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("royaley.scheduler")


# =============================================================================
# JOB 1: ODDS REFRESH - Updates upcoming_odds with latest market lines
# =============================================================================

async def refresh_odds(db: AsyncSession, api_key: str) -> dict:
    """
    Fetch latest odds for sports with upcoming games.
    Only fetches sports that have games in the next 48 hours → saves API quota.
    Returns stats dict.
    """
    stats = {"sports_fetched": 0, "odds_updated": 0, "api_requests": 0}

    # Find which sports have upcoming games
    active_sports = await db.execute(text("""
        SELECT DISTINCT s.code
        FROM upcoming_games ug
        JOIN sports s ON ug.sport_id = s.id
        WHERE ug.status = 'scheduled'
          AND ug.scheduled_at >= NOW()
          AND ug.scheduled_at <= NOW() + INTERVAL '48 hours'
    """))
    sport_codes = [row[0] for row in active_sports.fetchall()]

    if not sport_codes:
        logger.info("  No sports with upcoming games in next 48h, skipping odds refresh")
        return stats

    logger.info(f"  Active sports with upcoming games: {sport_codes}")

    markets = ["h2h", "spreads", "totals"]
    sharp_books = {"pinnacle", "pinnacle_alt"}

    async with httpx.AsyncClient(timeout=30.0) as client:
        for sport_code in sport_codes:
            api_sport_key = ODDS_API_SPORT_KEYS.get(sport_code)
            if not api_sport_key:
                continue

            # Get sport_id
            sport_row = await db.execute(
                text("SELECT id FROM sports WHERE code = :code"),
                {"code": sport_code},
            )
            sport = sport_row.fetchone()
            if not sport:
                continue
            sport_id = sport[0]

            # Fetch all 3 markets
            all_events = {}
            for market in markets:
                try:
                    resp = await client.get(
                        f"https://api.the-odds-api.com/v4/sports/{api_sport_key}/odds",
                        params={
                            "apiKey": api_key,
                            "regions": "us,us2",
                            "markets": market,
                            "oddsFormat": "american",
                        },
                    )
                    stats["api_requests"] += 1

                    remaining = int(resp.headers.get("x-requests-remaining", 500))
                    if remaining < 20:
                        logger.warning(f"  ⚠️  Low API quota: {remaining} requests remaining")

                    if resp.status_code == 429:
                        logger.error("  Rate limited! Stopping odds refresh.")
                        return stats
                    if resp.status_code == 401:
                        logger.error("  Invalid API key!")
                        return stats

                    resp.raise_for_status()
                    events = resp.json()

                    for event in events:
                        eid = event["id"]
                        if eid not in all_events:
                            all_events[eid] = {"id": eid, "bookmakers": []}
                        all_events[eid]["bookmakers"].extend(event.get("bookmakers", []))

                    logger.info(f"    {sport_code}/{market}: {len(events)} events (quota left: {remaining})")

                except Exception as e:
                    logger.error(f"    Error fetching {sport_code}/{market}: {e}")

            # Update odds for each game
            for eid, event_data in all_events.items():
                # Find the upcoming_game by external_id
                game_row = await db.execute(
                    text("SELECT id FROM upcoming_games WHERE external_id = :eid"),
                    {"eid": eid},
                )
                game = game_row.fetchone()
                if not game:
                    continue

                game_id = game[0]

                # Deduplicate bookmakers
                seen = set()
                for bm in event_data["bookmakers"]:
                    for mkt in bm.get("markets", []):
                        key = (bm["key"], mkt["key"])
                        if key in seen:
                            continue
                        seen.add(key)

                        bet_type_map = {"h2h": "moneyline", "spreads": "spread", "totals": "total"}
                        bet_type = bet_type_map.get(mkt["key"], mkt["key"])
                        outcomes = mkt.get("outcomes", [])

                        home_line = away_line = home_odds = away_odds = None
                        total_val = over_odds = under_odds = None
                        home_ml = away_ml = None

                        for outcome in outcomes:
                            name = outcome.get("name", "")
                            price = outcome.get("price")
                            point = outcome.get("point")

                            if bet_type == "spread":
                                if name == outcomes[0].get("name"):
                                    home_line = point
                                    home_odds = price
                                    away_line = -point if point else None
                                else:
                                    away_line = point
                                    away_odds = price
                                    if home_line is None:
                                        home_line = -point if point else None
                            elif bet_type == "total":
                                if name.lower() == "over":
                                    total_val = point
                                    over_odds = price
                                elif name.lower() == "under":
                                    total_val = point
                                    under_odds = price
                            elif bet_type == "moneyline":
                                if name == outcomes[0].get("name"):
                                    home_ml = price
                                else:
                                    away_ml = price

                        try:
                            await db.execute(text("""
                                INSERT INTO upcoming_odds
                                    (id, upcoming_game_id, sportsbook_key, sportsbook_name, is_sharp,
                                     bet_type, home_line, away_line, home_odds, away_odds,
                                     total, over_odds, under_odds, home_ml, away_ml,
                                     source, recorded_at, updated_at)
                                VALUES
                                    (gen_random_uuid(), :game_id, :book_key, :book_name, :is_sharp,
                                     :bet_type, :home_line, :away_line, :home_odds, :away_odds,
                                     :total, :over_odds, :under_odds, :home_ml, :away_ml,
                                     'odds_api', NOW(), NOW())
                                ON CONFLICT (upcoming_game_id, sportsbook_key, bet_type) DO UPDATE SET
                                    home_line = EXCLUDED.home_line,
                                    away_line = EXCLUDED.away_line,
                                    home_odds = EXCLUDED.home_odds,
                                    away_odds = EXCLUDED.away_odds,
                                    total = EXCLUDED.total,
                                    over_odds = EXCLUDED.over_odds,
                                    under_odds = EXCLUDED.under_odds,
                                    home_ml = EXCLUDED.home_ml,
                                    away_ml = EXCLUDED.away_ml,
                                    updated_at = NOW()
                            """), {
                                "game_id": game_id,
                                "book_key": bm["key"],
                                "book_name": bm.get("title", bm["key"]),
                                "is_sharp": bm["key"] in sharp_books,
                                "bet_type": bet_type,
                                "home_line": home_line,
                                "away_line": away_line,
                                "home_odds": home_odds,
                                "away_odds": away_odds,
                                "total": total_val,
                                "over_odds": over_odds,
                                "under_odds": under_odds,
                                "home_ml": home_ml,
                                "away_ml": away_ml,
                            })
                            stats["odds_updated"] += 1
                        except Exception as e:
                            await db.rollback()
                            logger.debug(f"    Odds upsert error: {e}")

            stats["sports_fetched"] += 1
            await db.commit()

    return stats


# =============================================================================
# JOB 2: CLOSING LINE CAPTURE - Snapshot lines before game starts
# =============================================================================

async def capture_closing_lines(db: AsyncSession) -> int:
    """
    For games starting within 30 minutes that don't yet have closing lines,
    snapshot the current consensus into prediction_results as the closing line.
    Returns count of predictions with closing lines captured.
    """
    # Find predictions for games about to start (within 30 min) that haven't been closed yet
    rows = await db.execute(text("""
        SELECT
            p.id as pred_id,
            p.bet_type,
            p.predicted_side,
            p.line_at_prediction,
            p.odds_at_prediction,
            p.home_line_open,
            p.away_line_open,
            p.total_open,
            p.home_ml_open,
            p.away_ml_open,
            -- Current consensus (Pinnacle preferred)
            COALESCE(MAX(CASE WHEN uo.sportsbook_key = 'pinnacle' THEN uo.home_line END), AVG(uo.home_line)) as curr_home_line,
            COALESCE(MAX(CASE WHEN uo.sportsbook_key = 'pinnacle' THEN uo.away_line END), AVG(uo.away_line)) as curr_away_line,
            COALESCE(MAX(CASE WHEN uo.sportsbook_key = 'pinnacle' THEN uo.home_odds END), AVG(uo.home_odds)) as curr_home_odds,
            COALESCE(MAX(CASE WHEN uo.sportsbook_key = 'pinnacle' THEN uo.away_odds END), AVG(uo.away_odds)) as curr_away_odds,
            COALESCE(MAX(CASE WHEN uo.sportsbook_key = 'pinnacle' THEN uo.total END), AVG(uo.total)) as curr_total,
            COALESCE(MAX(CASE WHEN uo.sportsbook_key = 'pinnacle' THEN uo.over_odds END), AVG(uo.over_odds)) as curr_over_odds,
            COALESCE(MAX(CASE WHEN uo.sportsbook_key = 'pinnacle' THEN uo.under_odds END), AVG(uo.under_odds)) as curr_under_odds,
            COALESCE(MAX(CASE WHEN uo.sportsbook_key = 'pinnacle' THEN uo.home_ml END), AVG(uo.home_ml)) as curr_home_ml,
            COALESCE(MAX(CASE WHEN uo.sportsbook_key = 'pinnacle' THEN uo.away_ml END), AVG(uo.away_ml)) as curr_away_ml
        FROM predictions p
        JOIN upcoming_games ug ON p.upcoming_game_id = ug.id
        LEFT JOIN upcoming_odds uo ON uo.upcoming_game_id = ug.id AND uo.bet_type = p.bet_type
        LEFT JOIN prediction_results pr ON pr.prediction_id = p.id
        WHERE ug.scheduled_at BETWEEN NOW() AND NOW() + INTERVAL '30 minutes'
          AND pr.id IS NULL  -- No result yet (not already captured)
        GROUP BY p.id, p.bet_type, p.predicted_side, p.line_at_prediction,
                 p.odds_at_prediction, p.home_line_open, p.away_line_open,
                 p.total_open, p.home_ml_open, p.away_ml_open
    """))

    count = 0
    for row in rows.fetchall():
        # Determine closing line and odds for the predicted side
        closing_line = None
        closing_odds = None

        if row.bet_type == "spread":
            if row.predicted_side == "home":
                closing_line = round((row.curr_home_line or 0) * 2) / 2
                closing_odds = int(round(row.curr_home_odds or -110))
            else:
                closing_line = round((row.curr_away_line or 0) * 2) / 2
                closing_odds = int(round(row.curr_away_odds or -110))
        elif row.bet_type == "total":
            closing_line = round((row.curr_total or 0) * 2) / 2
            if row.predicted_side == "over":
                closing_odds = int(round(row.curr_over_odds or -110))
            else:
                closing_odds = int(round(row.curr_under_odds or -110))
        elif row.bet_type == "moneyline":
            if row.predicted_side == "home":
                closing_odds = int(round(row.curr_home_ml or -110))
            else:
                closing_odds = int(round(row.curr_away_ml or -110))

        # Create prediction_result with closing line (result=pending, will be graded later)
        try:
            await db.execute(text("""
                INSERT INTO prediction_results
                    (id, prediction_id, actual_result, closing_line, closing_odds, graded_at)
                VALUES
                    (gen_random_uuid(), :pred_id, 'pending', :cl, :co, NOW())
                ON CONFLICT (prediction_id) DO NOTHING
            """), {
                "pred_id": row.pred_id,
                "cl": closing_line,
                "co": closing_odds,
            })
            count += 1
        except Exception as e:
            logger.debug(f"    Closing capture error: {e}")

    if count > 0:
        await db.commit()
        logger.info(f"  Captured closing lines for {count} predictions")

    return count


# =============================================================================
# JOB 3: GAME GRADING - Fetch scores, grade predictions, calculate CLV
# =============================================================================

async def grade_predictions(db: AsyncSession, api_key: str) -> dict:
    """
    For games that have finished:
    1. Fetch scores from Odds API
    2. Update upcoming_games with final scores
    3. Grade each prediction (win/loss/push)
    4. Calculate CLV
    Returns stats dict.
    """
    stats = {"games_graded": 0, "predictions_graded": 0, "api_requests": 0}

    # Find sports with ungraded games (game time has passed but no score yet)
    ungraded_sports = await db.execute(text("""
        SELECT DISTINCT s.code
        FROM upcoming_games ug
        JOIN sports s ON ug.sport_id = s.id
        WHERE ug.status = 'scheduled'
          AND ug.scheduled_at < NOW() - INTERVAL '3 hours'
    """))
    sport_codes = [row[0] for row in ungraded_sports.fetchall()]

    if not sport_codes:
        return stats

    logger.info(f"  Sports with ungraded games: {sport_codes}")

    # Fetch scores from Odds API
    async with httpx.AsyncClient(timeout=30.0) as client:
        for sport_code in sport_codes:
            api_sport_key = ODDS_API_SPORT_KEYS.get(sport_code)
            if not api_sport_key:
                continue

            try:
                resp = await client.get(
                    f"https://api.the-odds-api.com/v4/sports/{api_sport_key}/scores",
                    params={
                        "apiKey": api_key,
                        "daysFrom": 3,
                    },
                )
                stats["api_requests"] += 1

                if resp.status_code != 200:
                    logger.error(f"    Scores API error: {resp.status_code}")
                    continue

                scores = resp.json()
                logger.info(f"    {sport_code}: {len(scores)} score results")

                for game in scores:
                    if not game.get("completed"):
                        continue

                    ext_id = game["id"]
                    game_scores = game.get("scores", [])
                    if not game_scores or len(game_scores) < 2:
                        continue

                    # Parse scores
                    home_name = game.get("home_team", "")
                    home_score = away_score = None
                    for s in game_scores:
                        if s["name"] == home_name:
                            home_score = int(s["score"]) if s.get("score") else None
                        else:
                            away_score = int(s["score"]) if s.get("score") else None

                    if home_score is None or away_score is None:
                        continue

                    # Update upcoming_game with scores
                    result = await db.execute(text("""
                        UPDATE upcoming_games
                        SET home_score = :hs, away_score = :as_score,
                            status = 'completed', updated_at = NOW()
                        WHERE external_id = :eid AND status = 'scheduled'
                        RETURNING id
                    """), {"hs": home_score, "as_score": away_score, "eid": ext_id})

                    updated = result.fetchone()
                    if not updated:
                        continue

                    game_id = updated[0]
                    stats["games_graded"] += 1

                    # Grade each prediction for this game
                    preds = await db.execute(text("""
                        SELECT p.id, p.bet_type, p.predicted_side,
                               p.line_at_prediction, p.odds_at_prediction,
                               p.home_line_open, p.away_line_open, p.total_open,
                               p.home_ml_open, p.away_ml_open,
                               pr.closing_line, pr.closing_odds
                        FROM predictions p
                        LEFT JOIN prediction_results pr ON pr.prediction_id = p.id
                        WHERE p.upcoming_game_id = :gid
                    """), {"gid": game_id})

                    for pred in preds.fetchall():
                        result_val = _grade_single(
                            pred.bet_type, pred.predicted_side,
                            pred.line_at_prediction,
                            home_score, away_score
                        )

                        # Calculate CLV
                        clv = _calculate_clv(
                            pred.bet_type, pred.predicted_side,
                            pred.line_at_prediction, pred.odds_at_prediction,
                            pred.closing_line, pred.closing_odds,
                            pred.home_line_open, pred.away_line_open,
                            pred.total_open, pred.home_ml_open, pred.away_ml_open
                        )

                        # Calculate profit/loss (flat $100 bet for tracking)
                        pnl = _calculate_pnl(result_val, pred.odds_at_prediction)

                        # Upsert prediction_result
                        try:
                            await db.execute(text("""
                                INSERT INTO prediction_results
                                    (id, prediction_id, actual_result, closing_line,
                                     closing_odds, clv, profit_loss, graded_at)
                                VALUES
                                    (gen_random_uuid(), :pid, :result, :cl, :co, :clv, :pnl, NOW())
                                ON CONFLICT (prediction_id) DO UPDATE SET
                                    actual_result = EXCLUDED.actual_result,
                                    clv = EXCLUDED.clv,
                                    profit_loss = EXCLUDED.profit_loss,
                                    graded_at = NOW()
                            """), {
                                "pid": pred.id,
                                "result": result_val,
                                "cl": pred.closing_line,
                                "co": pred.closing_odds,
                                "clv": clv,
                                "pnl": pnl,
                            })
                            stats["predictions_graded"] += 1
                        except Exception as e:
                            await db.rollback()
                            logger.error(f"    Grade error for {pred.id}: {e}")

            except Exception as e:
                logger.error(f"    Error fetching scores for {sport_code}: {e}")

    if stats["games_graded"] > 0:
        await db.commit()

    return stats


def _grade_single(
    bet_type: str, predicted_side: str, line: Optional[float],
    home_score: int, away_score: int,
) -> str:
    """Grade a single prediction. Returns 'win', 'loss', or 'push'."""
    score_diff = home_score - away_score  # Positive = home won
    total_points = home_score + away_score

    if bet_type == "spread":
        if line is None:
            return "void"
        # line is from the predicted side's perspective
        if predicted_side == "home":
            adjusted = score_diff + line  # home_score - away_score + home_spread
        else:  # away
            adjusted = -score_diff + line  # away effectively
        if adjusted > 0:
            return "win"
        elif adjusted < 0:
            return "loss"
        else:
            return "push"

    elif bet_type == "total":
        if line is None:
            return "void"
        if predicted_side == "over":
            if total_points > line:
                return "win"
            elif total_points < line:
                return "loss"
            else:
                return "push"
        else:  # under
            if total_points < line:
                return "win"
            elif total_points > line:
                return "loss"
            else:
                return "push"

    elif bet_type == "moneyline":
        if predicted_side == "home":
            return "win" if score_diff > 0 else ("loss" if score_diff < 0 else "push")
        else:
            return "win" if score_diff < 0 else ("loss" if score_diff > 0 else "push")

    return "void"


def _calculate_clv(
    bet_type: str, predicted_side: str,
    open_line: Optional[float], open_odds: Optional[int],
    close_line: Optional[float], close_odds: Optional[int],
    home_line_open: Optional[float], away_line_open: Optional[float],
    total_open: Optional[float],
    home_ml_open: Optional[int], away_ml_open: Optional[int],
) -> Optional[float]:
    """
    Calculate Closing Line Value (CLV).
    Positive CLV = we got a better number than the closing line (good).

    Spread/Total CLV: difference in points (e.g., got +3.5, closed at +3.0 → CLV = +0.5)
    Moneyline CLV: difference in implied probability percentage
    """
    if close_line is None and close_odds is None:
        return None

    if bet_type == "spread":
        if open_line is not None and close_line is not None:
            # For spread: getting more points = better for the bettor
            return round(open_line - close_line, 1)

    elif bet_type == "total":
        if open_line is not None and close_line is not None:
            if predicted_side == "over":
                # For over: lower total at open = better (easier to go over)
                return round(close_line - open_line, 1)
            else:
                # For under: higher total at open = better (easier to go under)
                return round(open_line - close_line, 1)

    elif bet_type == "moneyline":
        if open_odds is not None and close_odds is not None:
            open_prob = _implied_prob(open_odds)
            close_prob = _implied_prob(close_odds)
            # If closing implied prob is higher, we got value (positive CLV)
            return round((close_prob - open_prob) * 100, 1)

    return None


def _calculate_pnl(result: str, odds: Optional[int]) -> Optional[float]:
    """Calculate profit/loss on a flat $100 bet."""
    if odds is None or result == "void":
        return None
    if result == "push":
        return 0.0
    if result == "win":
        if odds > 0:
            return round(odds, 2)  # e.g., +150 → win $150
        else:
            return round(100 / abs(odds) * 100, 2)  # e.g., -150 → win $66.67
    if result == "loss":
        return -100.0
    return None


def _implied_prob(american_odds: int) -> float:
    """Convert American odds to implied probability."""
    if american_odds > 0:
        return 100 / (american_odds + 100)
    else:
        return abs(american_odds) / (abs(american_odds) + 100)


# =============================================================================
# SCHEDULER LOOP
# =============================================================================

async def run_scheduler():
    """
    Main scheduler loop. Runs 3 jobs on different schedules:
      - Odds refresh: every 8 hours (3x/day)
      - Closing capture: every 15 minutes
      - Game grading: every 30 minutes
    """
    logger.info("=" * 60)
    logger.info("ROYALEY Prediction Scheduler Started")
    logger.info("=" * 60)
    logger.info("Schedule:")
    logger.info("  Odds refresh:    every 8 hours")
    logger.info("  Closing capture: every 15 minutes")
    logger.info("  Game grading:    every 30 minutes")
    logger.info("")

    engine = create_async_engine(settings.DATABASE_URL, echo=False)
    async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    api_key = settings.ODDS_API_KEY

    last_odds_refresh = datetime.min
    last_grading = datetime.min

    ODDS_INTERVAL = timedelta(hours=8)
    CLOSING_INTERVAL = timedelta(minutes=15)
    GRADING_INTERVAL = timedelta(minutes=30)

    cycle = 0
    while True:
        cycle += 1
        now = datetime.utcnow()

        # --- Closing line capture (most frequent - every 15 min) ---
        logger.info(f"[Cycle {cycle}] Checking for closing line captures...")
        try:
            async with async_session() as db:
                closed = await capture_closing_lines(db)
                if closed:
                    logger.info(f"  ✅ Captured {closed} closing lines")
        except Exception as e:
            logger.error(f"  Closing capture error: {e}")

        # --- Game grading (every 30 min) ---
        if now - last_grading >= GRADING_INTERVAL:
            logger.info(f"[Cycle {cycle}] Running game grading...")
            try:
                async with async_session() as db:
                    grade_stats = await grade_predictions(db, api_key)
                    if grade_stats["games_graded"] > 0:
                        logger.info(
                            f"  ✅ Graded {grade_stats['games_graded']} games, "
                            f"{grade_stats['predictions_graded']} predictions "
                            f"(API: {grade_stats['api_requests']} requests)"
                        )
            except Exception as e:
                logger.error(f"  Grading error: {e}")
            last_grading = now

        # --- Odds refresh (every 8 hours) ---
        if now - last_odds_refresh >= ODDS_INTERVAL:
            logger.info(f"[Cycle {cycle}] Running odds refresh...")
            try:
                async with async_session() as db:
                    odds_stats = await refresh_odds(db, api_key)
                    logger.info(
                        f"  ✅ Refreshed {odds_stats['sports_fetched']} sports, "
                        f"{odds_stats['odds_updated']} odds updated "
                        f"(API: {odds_stats['api_requests']} requests)"
                    )
            except Exception as e:
                logger.error(f"  Odds refresh error: {e}")
            last_odds_refresh = now

        # Sleep 15 minutes between cycles
        logger.info(f"[Cycle {cycle}] Sleeping 15 minutes...")
        await asyncio.sleep(900)  # 15 minutes


async def run_once():
    """Run all 3 jobs once and exit. Useful for testing or manual runs."""
    logger.info("=" * 60)
    logger.info("ROYALEY Scheduler - Single Run")
    logger.info("=" * 60)

    engine = create_async_engine(settings.DATABASE_URL, echo=False)
    async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    api_key = settings.ODDS_API_KEY

    # Job 1: Odds refresh (separate session)
    logger.info("\n1️⃣  Refreshing odds...")
    try:
        async with async_session() as db:
            odds_stats = await refresh_odds(db, api_key)
            logger.info(f"   Sports: {odds_stats['sports_fetched']}, "
                         f"Odds updated: {odds_stats['odds_updated']}, "
                         f"API requests: {odds_stats['api_requests']}")
    except Exception as e:
        logger.error(f"   Odds refresh error: {e}")

    # Job 2: Closing line capture (separate session)
    logger.info("\n2️⃣  Capturing closing lines...")
    try:
        async with async_session() as db:
            closed = await capture_closing_lines(db)
            logger.info(f"   Closing lines captured: {closed}")
    except Exception as e:
        logger.error(f"   Closing capture error: {e}")

    # Job 3: Game grading (separate session)
    logger.info("\n3️⃣  Grading predictions...")
    try:
        async with async_session() as db:
            grade_stats = await grade_predictions(db, api_key)
            logger.info(f"   Games graded: {grade_stats['games_graded']}, "
                         f"Predictions graded: {grade_stats['predictions_graded']}, "
                         f"API requests: {grade_stats['api_requests']}")
    except Exception as e:
        logger.error(f"   Grading error: {e}")

    await engine.dispose()
    logger.info("\n✅ Done!")


async def show_status():
    """Show current scheduler status and API quota."""
    engine = create_async_engine(settings.DATABASE_URL, echo=False)
    async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as db:
        # Prediction stats
        total = (await db.execute(text("SELECT COUNT(*) FROM predictions"))).scalar() or 0
        graded = (await db.execute(text("SELECT COUNT(*) FROM prediction_results WHERE actual_result != 'pending'"))).scalar() or 0
        with_closing = (await db.execute(text("SELECT COUNT(*) FROM prediction_results WHERE closing_line IS NOT NULL"))).scalar() or 0
        with_clv = (await db.execute(text("SELECT COUNT(*) FROM prediction_results WHERE clv IS NOT NULL"))).scalar() or 0

        # Win/loss
        wins = (await db.execute(text("SELECT COUNT(*) FROM prediction_results WHERE actual_result = 'win'"))).scalar() or 0
        losses = (await db.execute(text("SELECT COUNT(*) FROM prediction_results WHERE actual_result = 'loss'"))).scalar() or 0
        pushes = (await db.execute(text("SELECT COUNT(*) FROM prediction_results WHERE actual_result = 'push'"))).scalar() or 0

        # Upcoming games
        upcoming = (await db.execute(text("""
            SELECT COUNT(*) FROM upcoming_games
            WHERE status = 'scheduled' AND scheduled_at >= NOW()
        """))).scalar() or 0

        # Active sports
        active = await db.execute(text("""
            SELECT s.code, COUNT(ug.id) as game_count
            FROM upcoming_games ug
            JOIN sports s ON ug.sport_id = s.id
            WHERE ug.status = 'scheduled' AND ug.scheduled_at >= NOW()
            GROUP BY s.code
        """))

    await engine.dispose()

    print("\n" + "=" * 50)
    print("ROYALEY Scheduler Status")
    print("=" * 50)
    print(f"  Total predictions:    {total}")
    print(f"  Graded:               {graded} ({wins}W / {losses}L / {pushes}P)")
    print(f"  With closing lines:   {with_closing}")
    print(f"  With CLV:             {with_clv}")
    print(f"  Win rate:             {round(wins/(wins+losses)*100, 1) if (wins+losses) > 0 else 0}%")
    print(f"  Upcoming games:       {upcoming}")
    print(f"\n  Active sports:")
    for row in active.fetchall():
        print(f"    {row.code}: {row.game_count} games")
    print("=" * 50)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="ROYALEY Prediction Scheduler")
    parser.add_argument("--once", action="store_true", help="Run all jobs once and exit")
    parser.add_argument("--status", action="store_true", help="Show current status")
    args = parser.parse_args()

    if args.status:
        asyncio.run(show_status())
    elif args.once:
        asyncio.run(run_once())
    else:
        asyncio.run(run_scheduler())


if __name__ == "__main__":
    main()