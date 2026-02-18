"""
ROYALEY - Live Scores Pipeline
Fetches live and final scores from The Odds API → updates upcoming_games.

The Odds API /scores endpoint returns:
  - Live games with current scores
  - Completed games (with daysFrom param) with final scores
  - Cost: 1 request per sport (live only), 2 per sport (with completed)

Usage:
    docker exec royaley_api python -m app.pipeline.fetch_scores
    docker exec royaley_api python -m app.pipeline.fetch_scores --sport NBA
"""

import asyncio
import argparse
import logging
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional

import httpx
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from app.core.config import settings, ODDS_API_SPORT_KEYS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("royaley.scores")


# =============================================================================
# RESOLVE TENNIS TOURNAMENT KEYS (same logic as scheduler.py)
# =============================================================================

async def resolve_tennis_keys(api_key: str) -> dict:
    """Discover active ATP/WTA tournament keys from Odds API."""
    tennis_keys = {}
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                "https://api.the-odds-api.com/v4/sports",
                params={"apiKey": api_key},
            )
            if resp.status_code == 200:
                for s in resp.json():
                    key = s.get("key", "")
                    if key.startswith("tennis_atp_") and s.get("active"):
                        tennis_keys["ATP"] = key
                    elif key.startswith("tennis_wta_") and s.get("active"):
                        tennis_keys["WTA"] = key
                if tennis_keys:
                    logger.info(f"  Active tennis tournaments: {tennis_keys}")
    except Exception as e:
        logger.warning(f"  Failed to discover tennis keys: {e}")
    return tennis_keys


# =============================================================================
# SCORES API CLIENT
# =============================================================================

class ScoresAPIClient:
    """Fetches live scores from The Odds API."""

    BASE_URL = "https://api.the-odds-api.com/v4"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.requests_used = 0
        self.requests_remaining = 500

    async def get_scores(
        self,
        sport_key: str,
        days_from: int = 1,
    ) -> List[dict]:
        """
        Fetch scores for a sport.

        Args:
            sport_key: e.g. 'basketball_nba'
            days_from: 1-3, include completed games from past N days.
                       0 = live + upcoming only (cost: 1 request)
                       1-3 = also completed (cost: 2 requests)
        """
        params = {
            "apiKey": self.api_key,
        }
        if days_from > 0:
            params["daysFrom"] = min(days_from, 3)

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                f"{self.BASE_URL}/sports/{sport_key}/scores",
                params=params,
            )
            self._track_usage(resp)

            if resp.status_code != 200:
                logger.error(f"Scores API {resp.status_code}: {resp.text[:200]}")
                return []

            return resp.json()

    def _track_usage(self, resp):
        self.requests_used = int(resp.headers.get("x-requests-used", self.requests_used))
        self.requests_remaining = int(resp.headers.get("x-requests-remaining", self.requests_remaining))


# =============================================================================
# SCORE UPDATER
# =============================================================================

async def update_scores_in_db(
    db: AsyncSession,
    scores_data: List[dict],
    sport_code: str,
) -> dict:
    """
    Update upcoming_games with score data from The Odds API.

    The API returns:
    {
        "id": "abc123...",
        "sport_key": "basketball_nba",
        "home_team": "Los Angeles Lakers",
        "away_team": "Boston Celtics",
        "commence_time": "2026-02-18T00:30:00Z",
        "completed": false,
        "scores": [
            {"name": "Los Angeles Lakers", "score": "52"},
            {"name": "Boston Celtics", "score": "58"}
        ],
        "last_update": "2026-02-18T01:15:00Z"
    }
    """
    stats = {"updated": 0, "completed": 0, "not_found": 0, "no_scores": 0}

    for event in scores_data:
        event_id = event.get("id")
        if not event_id:
            continue

        scores = event.get("scores")
        completed = event.get("completed", False)
        home_team = event.get("home_team", "")
        away_team = event.get("away_team", "")

        # Parse scores
        home_score = None
        away_score = None

        if scores and len(scores) >= 2:
            for s in scores:
                score_name = s.get("name", "")
                score_val = s.get("score")
                if score_val is not None:
                    try:
                        score_int = int(score_val)
                    except (ValueError, TypeError):
                        score_int = None

                    if score_name == home_team:
                        home_score = score_int
                    elif score_name == away_team:
                        away_score = score_int

        if home_score is None and away_score is None:
            stats["no_scores"] += 1
            continue

        # Determine status
        if completed:
            new_status = "final"
        elif home_score is not None:
            new_status = "in_progress"
        else:
            new_status = "scheduled"

        # Update upcoming_games by external_id
        result = await db.execute(
            text("""
                UPDATE upcoming_games
                SET home_score = :home_score,
                    away_score = :away_score,
                    status = :status,
                    completed = :completed,
                    last_score_update = NOW(),
                    updated_at = NOW()
                WHERE external_id = :ext_id
                RETURNING id
            """),
            {
                "home_score": home_score,
                "away_score": away_score,
                "status": new_status,
                "completed": completed,
                "ext_id": event_id,
            },
        )
        row = result.fetchone()

        # Fallback: match by team names + sport + time (for tennis & name mismatches)
        if not row and home_team and away_team:
            commence_time = event.get("commence_time", "")
            if commence_time:
                # asyncpg needs a real datetime, not a string
                from datetime import datetime as _dt
                try:
                    ct_dt = _dt.fromisoformat(commence_time.replace("Z", "+00:00"))
                except (ValueError, AttributeError):
                    ct_dt = None

                if ct_dt:
                    try:
                        result = await db.execute(
                            text("""
                                UPDATE upcoming_games
                                SET home_score = :home_score,
                                    away_score = :away_score,
                                    status = :status,
                                    completed = :completed,
                                    last_score_update = NOW(),
                                    updated_at = NOW()
                                WHERE id = (
                                    SELECT ug.id FROM upcoming_games ug
                                    JOIN sports s ON ug.sport_id = s.id
                                    WHERE s.code = :sport_code
                                      AND ug.status IN ('scheduled', 'in_progress')
                                      AND (
                                        (ug.home_team_name = :home AND ug.away_team_name = :away)
                                        OR (LOWER(ug.home_team_name) = LOWER(:home) AND LOWER(ug.away_team_name) = LOWER(:away))
                                        OR (ug.home_team_name = :away AND ug.away_team_name = :home)
                                        OR (ug.home_team_name ILIKE '%' || :home_last || '%'
                                            AND ug.away_team_name ILIKE '%' || :away_last || '%')
                                      )
                                      AND ABS(EXTRACT(EPOCH FROM (ug.scheduled_at - :ct))) < 7200
                                    LIMIT 1
                                )
                                RETURNING id
                            """),
                            {
                                "home_score": home_score,
                                "away_score": away_score,
                                "status": new_status,
                                "completed": completed,
                                "sport_code": sport_code,
                                "home": home_team,
                                "away": away_team,
                                "home_last": home_team.split()[-1] if home_team.split() else home_team,
                                "away_last": away_team.split()[-1] if away_team.split() else away_team,
                                "ct": ct_dt,
                            },
                        )
                        row = result.fetchone()
                        if row:
                            logger.info(f"    Matched by team name fallback: {home_team} vs {away_team}")
                    except Exception as e:
                        logger.debug(f"    Team name fallback error: {e}")

        if row:
            if completed:
                stats["completed"] += 1
            else:
                stats["updated"] += 1
        else:
            stats["not_found"] += 1

    await db.commit()
    return stats


# =============================================================================
# MAIN PIPELINE
# =============================================================================

async def run_scores_pipeline(
    sports: Optional[List[str]] = None,
    days_from: int = 1,
    tournament_override: Optional[str] = None,
):
    """
    Main pipeline: fetch scores → update DB.

    Args:
        sports: Sport codes to fetch. None = all active.
        days_from: Include completed games from past N days (0-3).
        tournament_override: Force a specific tournament key (e.g. tennis_wta_qatar_open).
    """
    logger.info("=" * 50)
    logger.info("ROYALEY Live Scores Pipeline")
    logger.info("=" * 50)

    api_client = ScoresAPIClient(api_key=settings.ODDS_API_KEY)

    engine = create_async_engine(
        settings.DATABASE_URL.replace("+asyncpg", "+asyncpg"),
        echo=False,
    )
    async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    # Determine sports
    if tournament_override and sports:
        # Direct override: use the given tournament key for the specified sport
        sport_list = {s.upper(): tournament_override for s in sports}
        logger.info(f"Tournament override: {tournament_override}")
    elif sports:
        sport_list = {s.upper(): ODDS_API_SPORT_KEYS[s.upper()]
                      for s in sports if s.upper() in ODDS_API_SPORT_KEYS}
        # Also handle tennis if explicitly requested
        for s in sports:
            if s.upper() in ('ATP', 'WTA') and s.upper() not in sport_list:
                sport_list[s.upper()] = None  # Will be resolved below
    else:
        sport_list = dict(ODDS_API_SPORT_KEYS)
        # Add tennis placeholders (keys resolved dynamically below)
        sport_list["ATP"] = None
        sport_list["WTA"] = None

    # Resolve tennis tournament keys dynamically (skip if override provided)
    if not tournament_override:
        tennis_keys = await resolve_tennis_keys(settings.ODDS_API_KEY)
        for code in ["ATP", "WTA"]:
            if code in sport_list:
                if code in tennis_keys:
                    sport_list[code] = tennis_keys[code]
                else:
                    del sport_list[code]  # No active tournament, skip

    logger.info(f"Sports: {list(sport_list.keys())}")
    logger.info(f"Days from: {days_from}")

    total_updated = 0
    total_completed = 0

    async with async_session() as db:
        for sport_code, api_key in sport_list.items():
            scores = await api_client.get_scores(api_key, days_from=days_from)

            if not scores:
                continue

            # Count live vs completed
            live_count = sum(1 for s in scores if not s.get("completed") and s.get("scores"))
            final_count = sum(1 for s in scores if s.get("completed"))

            if live_count == 0 and final_count == 0:
                continue

            stats = await update_scores_in_db(db, scores, sport_code)

            logger.info(
                f"  {sport_code}: {live_count} live, {final_count} final | "
                f"DB: {stats['updated']} updated, {stats['completed']} completed, "
                f"{stats['not_found']} not found"
            )

            total_updated += stats["updated"]
            total_completed += stats["completed"]

    logger.info(f"DONE: {total_updated} live updates, {total_completed} completed | "
                f"API: {api_client.requests_remaining} remaining")

    await engine.dispose()


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="ROYALEY Live Scores Pipeline")
    parser.add_argument("--sport", type=str, help="Specific sport (e.g. NBA, WTA)")
    parser.add_argument("--days-from", type=int, default=1, help="Include completed games from past N days (0-3)")
    parser.add_argument("--tournament", type=str, help="Override tournament key (e.g. tennis_wta_qatar_open)")
    args = parser.parse_args()

    sports = [args.sport] if args.sport else None
    asyncio.run(run_scores_pipeline(sports=sports, days_from=args.days_from, tournament_override=args.tournament))


if __name__ == "__main__":
    main()