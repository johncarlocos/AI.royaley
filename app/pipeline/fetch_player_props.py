"""
ROYALEY - Player Props Pipeline
Fetches player prop lines from The Odds API → calculates edges → writes to DB.

Usage:
    # Fetch all in-season sports
    python -m app.pipeline.fetch_player_props

    # Fetch specific sport
    python -m app.pipeline.fetch_player_props --sport NBA

    # Dry run (show API calls without writing)
    python -m app.pipeline.fetch_player_props --dry-run

Runs inside the API container:
    docker exec royaley_api python -m app.pipeline.fetch_player_props
    docker exec royaley_api python -m app.pipeline.fetch_player_props --sport NBA
"""

import asyncio
import argparse
import hashlib
import logging
import math
import sys
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

import httpx
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from app.core.config import settings, ODDS_API_SPORT_KEYS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("royaley.player_props")


# =============================================================================
# CONFIGURATION
# =============================================================================

# Player prop markets per sport (The Odds API market keys)
# Each market costs 1 API request per region per event
PROP_MARKETS = {
    "NBA": [
        "player_points",
        "player_rebounds",
        "player_assists",
        "player_threes",
    ],
    "NCAAB": [
        "player_points",
        "player_rebounds",
        "player_assists",
    ],
    "NFL": [
        "player_pass_yds",
        "player_rush_yds",
        "player_reception_yds",
        "player_anytime_td",
    ],
    "NHL": [
        "player_points",
        "player_shots_on_goal",
    ],
    "MLB": [
        "pitcher_strikeouts",
        "batter_total_bases",
        "batter_hits",
    ],
    "WNBA": [
        "player_points",
        "player_rebounds",
        "player_assists",
    ],
}

# Map Odds API market keys → our internal prop_type codes
MARKET_TO_PROP_TYPE = {
    # Basketball
    "player_points": "points",
    "player_rebounds": "rebounds",
    "player_assists": "assists",
    "player_threes": "threes",
    "player_blocks": "blocks",
    "player_steals": "steals",
    "player_points_rebounds_assists": "pra",
    "player_double_double": "dd",
    # Football
    "player_pass_yds": "pass_yds",
    "player_pass_tds": "pass_tds",
    "player_pass_completions": "pass_comp",
    "player_rush_yds": "rush_yds",
    "player_rush_attempts": "rush_att",
    "player_reception_yds": "rec_yds",
    "player_receptions": "rec",
    "player_anytime_td": "atd",
    "player_first_td": "ftd",
    # Hockey
    "player_shots_on_goal": "sog",
    # (NHL "player_points" is goals+assists, maps to "points")
    # Baseball
    "pitcher_strikeouts": "strikeouts",
    "batter_total_bases": "total_bases",
    "batter_hits": "hits",
    "batter_home_runs": "home_runs",
    "batter_rbis": "rbis",
}

# Preferred bookmakers in order (sharpest first)
SHARP_BOOKS = ["pinnacle", "betcris", "bookmaker"]
SOFT_BOOKS = ["draftkings", "fanduel", "betmgm", "caesars", "pointsbetus"]

# Tier thresholds (calibrated)
TIER_A = 0.58
TIER_B = 0.55
TIER_C = 0.52


# =============================================================================
# ODDS API CLIENT - EVENT ODDS
# =============================================================================

class PropsAPIClient:
    """Fetches player prop odds from The Odds API event-odds endpoint."""

    BASE_URL = "https://api.the-odds-api.com/v4"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.requests_used = 0
        self.requests_remaining = 500

    async def get_events(self, sport_key: str) -> List[dict]:
        """Get list of upcoming events (games) for a sport."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                f"{self.BASE_URL}/sports/{sport_key}/events",
                params={"apiKey": self.api_key},
            )
            self._track_usage(resp)
            if resp.status_code != 200:
                logger.error(f"Events API {resp.status_code}: {resp.text[:200]}")
                return []
            return resp.json()

    async def get_event_props(
        self,
        sport_key: str,
        event_id: str,
        markets: List[str],
    ) -> Optional[dict]:
        """
        Fetch player prop odds for a single event.
        Cost: 1 request per market per region.
        """
        markets_str = ",".join(markets)
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                f"{self.BASE_URL}/sports/{sport_key}/events/{event_id}/odds",
                params={
                    "apiKey": self.api_key,
                    "regions": "us,us2",
                    "markets": markets_str,
                    "oddsFormat": "american",
                },
            )
            self._track_usage(resp)

            if resp.status_code == 404:
                # Event may have concluded or props not available yet
                return None
            if resp.status_code == 422:
                logger.warning(f"  Markets not available for event {event_id}: {resp.text[:100]}")
                return None
            if resp.status_code == 429:
                logger.error(f"Rate limited! Used: {self.requests_used}")
                return None
            if resp.status_code != 200:
                logger.error(f"Event odds API {resp.status_code}: {resp.text[:200]}")
                return None

            return resp.json()

    def _track_usage(self, resp):
        self.requests_used = int(resp.headers.get("x-requests-used", self.requests_used))
        self.requests_remaining = int(resp.headers.get("x-requests-remaining", self.requests_remaining))


# =============================================================================
# EDGE CALCULATOR
# =============================================================================

def american_to_implied(odds: int) -> float:
    """Convert American odds to implied probability."""
    if odds >= 100:
        return 100.0 / (odds + 100.0)
    else:
        return abs(odds) / (abs(odds) + 100.0)


def calculate_edge_from_books(outcomes_by_book: Dict[str, dict]) -> dict:
    """
    Calculate edge by comparing across bookmakers.

    For each player+prop, we get Over/Under odds from multiple books.
    We compute the consensus probability and detect edges.

    Returns dict with:
      - consensus_prob: average implied probability across books
      - best_over_odds, best_under_odds
      - sharp_prob: probability from sharpest book (Pinnacle preferred)
      - edge_pct: estimated edge in percentage
      - predicted_side: 'over' or 'under'
    """
    over_probs = []
    under_probs = []
    best_over = -9999
    best_under = -9999
    sharp_over_prob = None
    sharp_under_prob = None

    for book_key, data in outcomes_by_book.items():
        over_odds = data.get("over_odds")
        under_odds = data.get("under_odds")

        if over_odds is not None:
            op = american_to_implied(over_odds)
            over_probs.append(op)
            if over_odds > best_over:
                best_over = over_odds

        if under_odds is not None:
            up = american_to_implied(under_odds)
            under_probs.append(up)
            if under_odds > best_under:
                best_under = under_odds

        # Track sharp book probability
        if book_key in SHARP_BOOKS:
            if over_odds is not None:
                sharp_over_prob = american_to_implied(over_odds)
            if under_odds is not None:
                sharp_under_prob = american_to_implied(under_odds)

    if not over_probs and not under_probs:
        return None

    # Consensus = average implied probability (remove vig by normalizing)
    avg_over = sum(over_probs) / len(over_probs) if over_probs else 0.50
    avg_under = sum(under_probs) / len(under_probs) if under_probs else 0.50

    # Remove vig: normalize so Over + Under = 1.0
    total = avg_over + avg_under
    if total > 0:
        fair_over = avg_over / total
        fair_under = avg_under / total
    else:
        fair_over = 0.50
        fair_under = 0.50

    # Use sharp book as truth if available, otherwise consensus
    if sharp_over_prob and sharp_under_prob:
        sharp_total = sharp_over_prob + sharp_under_prob
        true_over = sharp_over_prob / sharp_total if sharp_total > 0 else fair_over
        true_under = sharp_under_prob / sharp_total if sharp_total > 0 else fair_under
    else:
        true_over = fair_over
        true_under = fair_under

    # Which side has edge?
    # Edge = true probability - best available implied probability
    best_over_implied = american_to_implied(best_over) if best_over > -9999 else 0.50
    best_under_implied = american_to_implied(best_under) if best_under > -9999 else 0.50

    over_edge = true_over - best_over_implied
    under_edge = true_under - best_under_implied

    if over_edge >= under_edge:
        predicted_side = "over"
        edge_pct = over_edge * 100
        probability = true_over
    else:
        predicted_side = "under"
        edge_pct = under_edge * 100
        probability = true_under

    # Minimum edge threshold
    if edge_pct < 0.5:
        edge_pct = abs(probability - 0.50) * 100  # fallback to deviation from 50%

    return {
        "predicted_side": predicted_side,
        "probability": round(probability, 4),
        "edge_pct": round(max(edge_pct, 0), 2),
        "best_over_odds": best_over if best_over > -9999 else None,
        "best_under_odds": best_under if best_under > -9999 else None,
        "num_books": len(outcomes_by_book),
    }


def assign_tier(probability: float) -> str:
    """Assign signal tier based on calibrated thresholds."""
    if probability >= TIER_A:
        return "A"
    elif probability >= TIER_B:
        return "B"
    elif probability >= TIER_C:
        return "C"
    return "D"


# =============================================================================
# RESPONSE PARSER
# =============================================================================

def parse_event_props(event_data: dict) -> List[dict]:
    """
    Parse The Odds API event-odds response into flat prop records.

    The API returns:
    {
      "id": "event123",
      "bookmakers": [
        {
          "key": "draftkings",
          "markets": [
            {
              "key": "player_points",
              "outcomes": [
                {"name": "Over", "description": "LeBron James", "price": -115, "point": 26.5},
                {"name": "Under", "description": "LeBron James", "price": -105, "point": 26.5},
                ...
              ]
            }
          ]
        }
      ]
    }
    """
    if not event_data:
        return []

    # Collect all outcomes grouped by (player, market, line)
    # Key: (player_name, market_key, line)
    # Value: {book_key: {over_odds, under_odds}}
    grouped: Dict[Tuple[str, str, float], Dict[str, dict]] = {}

    for bookmaker in event_data.get("bookmakers", []):
        book_key = bookmaker.get("key", "unknown")

        for market in bookmaker.get("markets", []):
            market_key = market.get("key", "")

            for outcome in market.get("outcomes", []):
                player_name = outcome.get("description", "")
                if not player_name:
                    continue

                side = outcome.get("name", "").lower()  # "Over" or "Under"
                price = outcome.get("price")
                line = outcome.get("point")

                if price is None:
                    continue

                # For anytime TD and similar, line may be 0.5
                if line is None:
                    line = 0.5

                key = (player_name, market_key, line)

                if key not in grouped:
                    grouped[key] = {}
                if book_key not in grouped[key]:
                    grouped[key][book_key] = {}

                if side == "over":
                    grouped[key][book_key]["over_odds"] = price
                elif side == "under":
                    grouped[key][book_key]["under_odds"] = price

    # Now calculate edge for each grouped prop
    results = []
    for (player_name, market_key, line), books_data in grouped.items():
        edge_result = calculate_edge_from_books(books_data)
        if not edge_result:
            continue

        prop_type = MARKET_TO_PROP_TYPE.get(market_key, market_key)
        tier = assign_tier(edge_result["probability"])

        results.append({
            "player_name": player_name,
            "market_key": market_key,
            "prop_type": prop_type,
            "line": line,
            "over_odds": edge_result["best_over_odds"],
            "under_odds": edge_result["best_under_odds"],
            "predicted_side": edge_result["predicted_side"],
            "probability": edge_result["probability"],
            "edge_pct": edge_result["edge_pct"],
            "tier": tier,
            "num_books": edge_result["num_books"],
            "predicted_value": round(line + (0.5 if edge_result["predicted_side"] == "over" else -0.5), 1),
        })

    return results


# =============================================================================
# DATABASE OPERATIONS
# =============================================================================

async def ensure_game_exists(
    db: AsyncSession,
    event_id: str,
    sport_id,
    home_team: str,
    away_team: str,
    commence_time: datetime,
) -> Optional[str]:
    """
    Ensure game exists in the `games` table (player_props FK target).
    Returns game UUID.
    """
    # Check if game already exists by external_id
    result = await db.execute(
        text("SELECT id FROM games WHERE external_id = :ext_id"),
        {"ext_id": event_id},
    )
    row = result.fetchone()
    if row:
        return str(row[0])

    # Get or create team IDs
    home_team_id = await _get_or_create_team(db, home_team, sport_id)
    away_team_id = await _get_or_create_team(db, away_team, sport_id)

    # Insert into games table
    result = await db.execute(
        text("""
            INSERT INTO games
                (id, sport_id, external_id, home_team_id, away_team_id,
                 scheduled_at, status, created_at, updated_at)
            VALUES
                (gen_random_uuid(), :sport_id, :ext_id, :home_id, :away_id,
                 :scheduled, 'scheduled', NOW(), NOW())
            ON CONFLICT (external_id) DO UPDATE SET
                scheduled_at = EXCLUDED.scheduled_at,
                updated_at = NOW()
            RETURNING id
        """),
        {
            "sport_id": sport_id,
            "ext_id": event_id,
            "home_id": home_team_id,
            "away_id": away_team_id,
            "scheduled": commence_time,
        },
    )
    row = result.fetchone()
    return str(row[0]) if row else None


async def _get_or_create_team(db: AsyncSession, team_name: str, sport_id) -> str:
    """Get or create a team, return its UUID."""
    # Generate stable abbreviation from team name
    parts = team_name.split()
    if len(parts) >= 2:
        abbr = parts[-1][:3].upper()
    else:
        abbr = team_name[:3].upper()

    # Check existing
    result = await db.execute(
        text("SELECT id FROM teams WHERE name = :name AND sport_id = :sport_id"),
        {"name": team_name, "sport_id": sport_id},
    )
    row = result.fetchone()
    if row:
        return str(row[0])

    # Create
    ext_id = f"props_{team_name.lower().replace(' ', '_')}"
    result = await db.execute(
        text("""
            INSERT INTO teams (id, sport_id, external_id, name, abbreviation, is_active, created_at, updated_at)
            VALUES (gen_random_uuid(), :sport_id, :ext_id, :name, :abbr, true, NOW(), NOW())
            ON CONFLICT (sport_id, name) DO UPDATE SET updated_at = NOW()
            RETURNING id
        """),
        {
            "sport_id": sport_id,
            "ext_id": ext_id,
            "name": team_name,
            "abbr": abbr,
        },
    )
    row = result.fetchone()
    return str(row[0]) if row else str(uuid4())


async def get_or_create_player(
    db: AsyncSession,
    player_name: str,
    team_name: str,
    sport_id,
) -> str:
    """Get or create a player, return UUID."""
    # Find by name (may have team changes, so just match name)
    result = await db.execute(
        text("""
            SELECT p.id FROM players p
            JOIN teams t ON t.id = p.team_id
            WHERE p.name = :name AND t.sport_id = :sport_id
            LIMIT 1
        """),
        {"name": player_name, "sport_id": sport_id},
    )
    row = result.fetchone()
    if row:
        return str(row[0])

    # Get team ID
    team_result = await db.execute(
        text("SELECT id FROM teams WHERE name = :name AND sport_id = :sport_id LIMIT 1"),
        {"name": team_name, "sport_id": sport_id},
    )
    team_row = team_result.fetchone()
    team_id = str(team_row[0]) if team_row else None

    # Create player
    ext_id = f"props_{player_name.lower().replace(' ', '_')}"
    result = await db.execute(
        text("""
            INSERT INTO players (id, team_id, external_id, name, status, is_active, created_at, updated_at)
            VALUES (gen_random_uuid(), :team_id, :ext_id, :name, 'active', true, NOW(), NOW())
            RETURNING id
        """),
        {
            "team_id": team_id,
            "ext_id": ext_id,
            "name": player_name,
        },
    )
    row = result.fetchone()
    return str(row[0]) if row else str(uuid4())


async def write_player_props(
    db: AsyncSession,
    game_id: str,
    sport_id,
    home_team: str,
    away_team: str,
    props: List[dict],
) -> int:
    """
    Write parsed player props to the player_props table.
    Deletes existing pending props for this game first (refresh with latest odds).
    Returns count of props written.
    """
    # Delete old pending props for this game (keep graded ones)
    await db.execute(
        text("DELETE FROM player_props WHERE game_id = :gid AND result = 'pending'"),
        {"gid": game_id},
    )

    written = 0

    for prop in props:
        try:
            # Determine which team the player is on (best guess from context)
            # The Odds API doesn't tell us team directly, so we'll assign to home team
            # This will be overwritten once we have real roster data
            player_id = await get_or_create_player(
                db, prop["player_name"], home_team, sport_id
            )

            # Insert prop
            await db.execute(
                text("""
                    INSERT INTO player_props
                        (id, game_id, player_id, prop_type, line,
                         over_odds, under_odds, predicted_value,
                         predicted_side, probability, signal_tier,
                         result, created_at)
                    VALUES
                        (gen_random_uuid(), :game_id, :player_id, :prop_type, :line,
                         :over_odds, :under_odds, :predicted_value,
                         :predicted_side, :probability, :tier::signaltier,
                         'pending', NOW())
                """),
                {
                    "game_id": game_id,
                    "player_id": player_id,
                    "prop_type": prop["prop_type"],
                    "line": prop["line"],
                    "over_odds": prop["over_odds"],
                    "under_odds": prop["under_odds"],
                    "predicted_value": prop["predicted_value"],
                    "predicted_side": prop["predicted_side"],
                    "probability": prop["probability"],
                    "tier": prop["tier"],
                },
            )
            written += 1

        except Exception as e:
            logger.warning(f"  Failed to write prop {prop['player_name']} {prop['prop_type']}: {e}")

    return written


# =============================================================================
# MAIN PIPELINE
# =============================================================================

async def run_player_props_pipeline(
    sports: Optional[List[str]] = None,
    dry_run: bool = False,
    max_events_per_sport: int = 10,
):
    """
    Main pipeline: fetch prop lines → calculate edges → write to DB.

    Args:
        sports: List of sport codes to fetch (e.g. ['NBA', 'NHL']). None = all in-season.
        dry_run: If True, show what would happen without DB writes.
        max_events_per_sport: Limit events to conserve API quota.
    """
    logger.info("=" * 60)
    logger.info("ROYALEY Player Props Pipeline")
    logger.info("=" * 60)

    api_client = PropsAPIClient(api_key=settings.ODDS_API_KEY)

    engine = create_async_engine(
        settings.DATABASE_URL.replace("+asyncpg", "+asyncpg"),
        echo=False,
    )
    async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    # Determine which sports have prop markets
    if sports:
        sport_list = [s.upper() for s in sports]
    else:
        sport_list = list(PROP_MARKETS.keys())

    # Filter to sports we have API keys for
    sport_keys = {}
    for code in sport_list:
        if code in ODDS_API_SPORT_KEYS and code in PROP_MARKETS:
            sport_keys[code] = ODDS_API_SPORT_KEYS[code]

    if not sport_keys:
        logger.error("No valid sports with prop markets configured")
        return

    logger.info(f"Sports: {list(sport_keys.keys())}")
    logger.info(f"API quota remaining: {api_client.requests_remaining}")
    logger.info(f"Max events per sport: {max_events_per_sport}")
    logger.info(f"Dry run: {dry_run}")
    logger.info("")

    total_props = 0
    total_events = 0

    async with async_session() as db:
        for sport_code, api_sport_key in sport_keys.items():
            markets = PROP_MARKETS.get(sport_code, [])
            if not markets:
                continue

            logger.info(f"{'─' * 50}")
            logger.info(f"📊 {sport_code} - Markets: {markets}")

            # Get sport_id from DB
            sport_row = await db.execute(
                text("SELECT id FROM sports WHERE code = :code"),
                {"code": sport_code},
            )
            sport = sport_row.fetchone()
            if not sport:
                logger.warning(f"  Sport {sport_code} not in DB, skipping")
                continue
            sport_id = sport[0]

            # Step 1: Get upcoming events
            events = await api_client.get_events(api_sport_key)
            if not events:
                logger.info(f"  No upcoming events for {sport_code}")
                continue

            # Filter to upcoming events (include games that started within last 2 hours for live props)
            now = datetime.utcnow()
            cutoff = now - timedelta(hours=2)
            future_events = []
            past_events = 0
            for ev in events:
                try:
                    ct = datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00")).replace(tzinfo=None)
                    if ct > cutoff:
                        ev["_commence_dt"] = ct
                        future_events.append(ev)
                    else:
                        past_events += 1
                except Exception:
                    pass

            # Sort by soonest first, limit
            future_events.sort(key=lambda e: e["_commence_dt"])
            future_events = future_events[:max_events_per_sport]

            logger.info(f"  {len(future_events)} upcoming events (of {len(events)} total, {past_events} already past)")

            # Step 2: For each event, fetch prop odds
            for ev in future_events:
                event_id = ev["id"]
                home_team = ev.get("home_team", "Unknown")
                away_team = ev.get("away_team", "Unknown")
                commence = ev["_commence_dt"]

                logger.info(f"  🏟  {away_team} @ {home_team} ({commence.strftime('%m/%d %I:%M %p')})")

                if dry_run:
                    logger.info(f"     [DRY RUN] Would fetch {len(markets)} markets")
                    continue

                # Check API quota
                if api_client.requests_remaining < len(markets) + 5:
                    logger.error(f"  ⚠️  Low API quota ({api_client.requests_remaining}), stopping")
                    break

                # Fetch props for this event
                event_data = await api_client.get_event_props(
                    api_sport_key, event_id, markets
                )

                if not event_data:
                    logger.info(f"     No props available yet")
                    continue

                # Debug: show what bookmakers returned
                bookmakers = event_data.get("bookmakers", [])
                if not bookmakers:
                    logger.info(f"     No bookmakers returned props (too early or small market)")
                    continue

                total_outcomes = 0
                for bm in bookmakers:
                    for mk in bm.get("markets", []):
                        total_outcomes += len(mk.get("outcomes", []))

                logger.info(
                    f"     📚 {len(bookmakers)} bookmakers, "
                    f"{sum(len(bm.get('markets',[])) for bm in bookmakers)} markets, "
                    f"{total_outcomes} outcomes"
                )

                # Step 3: Parse outcomes into prop records
                parsed_props = parse_event_props(event_data)
                if not parsed_props:
                    logger.info(f"     ⚠️ Outcomes found but no valid Over/Under pairs parsed")
                    continue

                # Step 4: Ensure game exists in DB
                game_id = await ensure_game_exists(
                    db, event_id, sport_id, home_team, away_team, commence
                )
                if not game_id:
                    logger.warning(f"     Could not create game record")
                    continue

                # Step 5: Write to player_props table
                written = await write_player_props(
                    db, game_id, sport_id, home_team, away_team, parsed_props
                )
                await db.commit()

                # Stats
                tier_counts = {}
                for p in parsed_props:
                    t = p["tier"]
                    tier_counts[t] = tier_counts.get(t, 0) + 1

                tier_str = " ".join(f"{t}:{c}" for t, c in sorted(tier_counts.items()))
                logger.info(
                    f"     ✅ {written} props written | "
                    f"Tiers: {tier_str} | "
                    f"Books: {parsed_props[0]['num_books'] if parsed_props else 0} avg | "
                    f"API: {api_client.requests_remaining} left"
                )

                total_props += written
                total_events += 1

                # Small delay to be nice to API
                await asyncio.sleep(0.5)

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"COMPLETE: {total_props} props across {total_events} events")
    logger.info(f"API requests remaining: {api_client.requests_remaining}")
    logger.info("=" * 60)

    await engine.dispose()


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="ROYALEY Player Props Pipeline")
    parser.add_argument("--sport", type=str, help="Specific sport (e.g. NBA, NHL)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would happen")
    parser.add_argument("--max-events", type=int, default=10, help="Max events per sport")
    args = parser.parse_args()

    sports = [args.sport] if args.sport else None

    asyncio.run(
        run_player_props_pipeline(
            sports=sports,
            dry_run=args.dry_run,
            max_events_per_sport=args.max_events,
        )
    )


if __name__ == "__main__":
    main()