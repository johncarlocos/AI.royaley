"""
ROYALEY - Live Prediction Pipeline
Fetches upcoming games from Odds API → saves to DB → generates predictions.

Usage:
    # Fetch all sports
    python -m app.pipeline.fetch_games
    
    # Fetch specific sport
    python -m app.pipeline.fetch_games --sport NBA
    
    # Fetch and generate predictions
    python -m app.pipeline.fetch_games --predict
    
    # Dry run (print what would be fetched)
    python -m app.pipeline.fetch_games --dry-run

Runs inside the API container:
    docker exec royaley_api python -m app.pipeline.fetch_games
    docker exec royaley_api python -m app.pipeline.fetch_games --sport NBA --predict
"""

import asyncio
import argparse
import hashlib
import logging
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from uuid import UUID, uuid4

import httpx
from sqlalchemy import select, text, and_
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.core.config import settings, ODDS_API_SPORT_KEYS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("royaley.pipeline")


# =============================================================================
# ODDS API CLIENT
# =============================================================================

class OddsAPIClient:
    """Fetches upcoming games and odds from The Odds API."""
    
    BASE_URL = "https://api.the-odds-api.com/v4"
    MARKETS = ["h2h", "spreads", "totals"]
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.requests_used = 0
        self.requests_remaining = 500
    
    async def fetch_sport_odds(
        self,
        sport_key: str,
        regions: str = "us,us2",
        odds_format: str = "american",
    ) -> List[dict]:
        """
        Fetch upcoming games with odds for a sport.
        
        Each API call uses ~3 requests from quota (1 per market).
        """
        all_events = {}
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            for market in self.MARKETS:
                try:
                    resp = await client.get(
                        f"{self.BASE_URL}/sports/{sport_key}/odds",
                        params={
                            "apiKey": self.api_key,
                            "regions": regions,
                            "markets": market,
                            "oddsFormat": odds_format,
                        },
                    )
                    
                    # Track rate limits from headers
                    self.requests_used = int(resp.headers.get("x-requests-used", 0))
                    self.requests_remaining = int(resp.headers.get("x-requests-remaining", 500))
                    
                    if resp.status_code == 401:
                        logger.error("Invalid Odds API key")
                        return []
                    if resp.status_code == 429:
                        logger.error(f"Rate limited! Used: {self.requests_used}, Remaining: {self.requests_remaining}")
                        return list(all_events.values())
                    
                    resp.raise_for_status()
                    events = resp.json()
                    
                    # Merge events (same game appears in each market call)
                    for event in events:
                        eid = event["id"]
                        if eid not in all_events:
                            all_events[eid] = {
                                "id": eid,
                                "sport_key": event.get("sport_key", sport_key),
                                "sport_title": event.get("sport_title", ""),
                                "commence_time": event["commence_time"],
                                "home_team": event["home_team"],
                                "away_team": event["away_team"],
                                "bookmakers": [],
                            }
                        # Append bookmakers with this market's data
                        all_events[eid]["bookmakers"].extend(event.get("bookmakers", []))
                    
                    logger.info(
                        f"  {sport_key}/{market}: {len(events)} events "
                        f"(quota: {self.requests_used}/{self.requests_used + self.requests_remaining})"
                    )
                    
                except httpx.HTTPError as e:
                    logger.error(f"HTTP error fetching {sport_key}/{market}: {e}")
                except Exception as e:
                    logger.error(f"Error fetching {sport_key}/{market}: {e}")
        
        return list(all_events.values())


# =============================================================================
# DATABASE OPERATIONS
# =============================================================================

async def get_or_create_team(
    db: AsyncSession,
    team_name: str,
    sport_id: UUID,
) -> UUID:
    """Find existing team or create a new one in the shared teams table."""
    
    # Try exact match first
    result = await db.execute(
        select(text("id")).select_from(text("teams")).where(
            and_(
                text("sport_id = :sport_id"),
                text("name = :name"),
            )
        ),
        {"sport_id": sport_id, "name": team_name},
    )
    row = result.fetchone()
    if row:
        return row[0]
    
    # Try abbreviation or partial match
    result = await db.execute(
        text("""
            SELECT id FROM teams 
            WHERE sport_id = :sport_id 
            AND (name ILIKE :pattern OR :name ILIKE '%' || abbreviation || '%')
            LIMIT 1
        """),
        {"sport_id": sport_id, "name": team_name, "pattern": f"%{team_name}%"},
    )
    row = result.fetchone()
    if row:
        return row[0]
    
    # Create new team
    team_id = uuid4()
    # Generate abbreviation from team name (last word, first 3 chars)
    parts = team_name.split()
    abbrev = parts[-1][:3].upper() if parts else team_name[:3].upper()
    
    await db.execute(
        text("""
            INSERT INTO teams (id, sport_id, external_id, name, abbreviation, elo_rating, is_active, created_at, updated_at)
            VALUES (:id, :sport_id, :ext_id, :name, :abbrev, 1500.0, true, NOW(), NOW())
            ON CONFLICT (id) DO NOTHING
        """),
        {
            "id": team_id,
            "sport_id": sport_id,
            "ext_id": f"odds_api_{team_name.lower().replace(' ', '_')}",
            "name": team_name,
            "abbrev": abbrev,
        },
    )
    logger.info(f"  Created new team: {team_name} ({abbrev})")
    return team_id


async def save_upcoming_game(
    db: AsyncSession,
    event: dict,
    sport_id: UUID,
    sport_code: str,
) -> Optional[UUID]:
    """Save or update an upcoming game. Returns the game UUID."""
    
    external_id = event["id"]
    home_team_name = event["home_team"]
    away_team_name = event["away_team"]
    commence_time = datetime.fromisoformat(event["commence_time"].replace("Z", "+00:00")).replace(tzinfo=None)
    
    # Skip games that already started
    if commence_time < datetime.utcnow():
        return None
    
    # Get or create team IDs
    home_team_id = await get_or_create_team(db, home_team_name, sport_id)
    away_team_id = await get_or_create_team(db, away_team_name, sport_id)
    
    # Upsert upcoming game
    result = await db.execute(
        text("""
            INSERT INTO upcoming_games 
                (id, sport_id, external_id, home_team_id, away_team_id, 
                 home_team_name, away_team_name, scheduled_at, status, source, created_at, updated_at)
            VALUES 
                (gen_random_uuid(), :sport_id, :ext_id, :home_id, :away_id,
                 :home_name, :away_name, :scheduled, 'scheduled', 'odds_api', NOW(), NOW())
            ON CONFLICT (external_id) DO UPDATE SET
                home_team_name = EXCLUDED.home_team_name,
                away_team_name = EXCLUDED.away_team_name,
                scheduled_at = EXCLUDED.scheduled_at,
                updated_at = NOW()
            RETURNING id
        """),
        {
            "sport_id": sport_id,
            "ext_id": external_id,
            "home_id": home_team_id,
            "away_id": away_team_id,
            "home_name": home_team_name,
            "away_name": away_team_name,
            "scheduled": commence_time,
        },
    )
    game_row = result.fetchone()
    return game_row[0] if game_row else None


async def save_upcoming_odds(
    db: AsyncSession,
    upcoming_game_id: UUID,
    bookmakers: List[dict],
) -> int:
    """Save odds from all bookmakers for an upcoming game. Returns count saved."""
    
    # Deduplicate bookmakers by key+market (API returns duplicates across market calls)
    seen = set()
    unique_bookmakers = []
    for bm in bookmakers:
        for market in bm.get("markets", []):
            key = (bm["key"], market["key"])
            if key not in seen:
                seen.add(key)
                unique_bookmakers.append((bm, market))
    
    count = 0
    sharp_books = {"pinnacle", "pinnacle_alt"}
    
    for bm, market in unique_bookmakers:
        book_key = bm["key"]
        book_name = bm.get("title", book_key)
        market_key = market["key"]  # h2h, spreads, totals
        outcomes = market.get("outcomes", [])
        
        # Map Odds API market keys to our bet_type
        bet_type_map = {"h2h": "moneyline", "spreads": "spread", "totals": "total"}
        bet_type = bet_type_map.get(market_key, market_key)
        
        # Parse outcomes
        home_line = away_line = home_odds = away_odds = None
        total = over_odds = under_odds = None
        home_ml = away_ml = None
        
        for outcome in outcomes:
            name = outcome.get("name", "")
            price = outcome.get("price")
            point = outcome.get("point")
            
            if bet_type == "spread":
                # Spread: outcomes have point (e.g., -3.5) and price (e.g., -110)
                if name == outcomes[0].get("name"):  # First outcome = usually home
                    home_line = point
                    home_odds = price
                    away_line = -point if point else None
                else:
                    away_line = point
                    away_odds = price
                    if home_line is None:
                        home_line = -point if point else None
                # Fix: Odds API may not always put home first
                # Check by matching team names if available
                
            elif bet_type == "total":
                if name.lower() == "over":
                    total = point
                    over_odds = price
                elif name.lower() == "under":
                    total = point  # Same total for both
                    under_odds = price
                    
            elif bet_type == "moneyline":
                if name == outcomes[0].get("name"):
                    home_ml = price
                else:
                    away_ml = price
        
        # Upsert odds
        try:
            await db.execute(
                text("""
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
                """),
                {
                    "game_id": upcoming_game_id,
                    "book_key": book_key,
                    "book_name": book_name,
                    "is_sharp": book_key in sharp_books,
                    "bet_type": bet_type,
                    "home_line": home_line,
                    "away_line": away_line,
                    "home_odds": home_odds,
                    "away_odds": away_odds,
                    "total": total,
                    "over_odds": over_odds,
                    "under_odds": under_odds,
                    "home_ml": home_ml,
                    "away_ml": away_ml,
                },
            )
            count += 1
        except Exception as e:
            logger.debug(f"  Odds upsert error for {book_key}/{bet_type}: {e}")
            await db.rollback()
    
    return count


# =============================================================================
# PREDICTION GENERATION (SIMPLIFIED)
# =============================================================================

async def generate_predictions_for_game(
    db: AsyncSession,
    upcoming_game_id: UUID,
    sport_code: str,
) -> int:
    """
    Generate predictions for an upcoming game using consensus odds.
    
    For now, uses implied probability from Pinnacle/consensus lines
    to create predictions. When H2O models are loaded, this will
    use the full prediction engine instead.
    
    Returns count of predictions created.
    """
    
    # Get consensus odds (prefer Pinnacle, fall back to average)
    odds_rows = await db.execute(
        text("""
            SELECT bet_type, 
                   AVG(home_line) as avg_home_line,
                   AVG(home_odds) as avg_home_odds,
                   AVG(away_odds) as avg_away_odds,
                   AVG(total) as avg_total,
                   AVG(over_odds) as avg_over_odds,
                   AVG(under_odds) as avg_under_odds,
                   AVG(home_ml) as avg_home_ml,
                   AVG(away_ml) as avg_away_ml,
                   -- Pinnacle odds (sharp reference)
                   MAX(CASE WHEN sportsbook_key = 'pinnacle' THEN home_line END) as pin_home_line,
                   MAX(CASE WHEN sportsbook_key = 'pinnacle' THEN home_odds END) as pin_home_odds,
                   MAX(CASE WHEN sportsbook_key = 'pinnacle' THEN away_odds END) as pin_away_odds,
                   MAX(CASE WHEN sportsbook_key = 'pinnacle' THEN total END) as pin_total,
                   MAX(CASE WHEN sportsbook_key = 'pinnacle' THEN over_odds END) as pin_over_odds,
                   MAX(CASE WHEN sportsbook_key = 'pinnacle' THEN under_odds END) as pin_under_odds,
                   MAX(CASE WHEN sportsbook_key = 'pinnacle' THEN home_ml END) as pin_home_ml,
                   MAX(CASE WHEN sportsbook_key = 'pinnacle' THEN away_ml END) as pin_away_ml
            FROM upcoming_odds
            WHERE upcoming_game_id = :game_id
            GROUP BY bet_type
        """),
        {"game_id": upcoming_game_id},
    )
    
    count = 0
    for row in odds_rows.fetchall():
        bet_type = row.bet_type
        
        predictions_to_make = []
        
        if bet_type == "spread":
            line = row.pin_home_line or row.avg_home_line
            home_price = row.pin_home_odds or row.avg_home_odds
            away_price = row.pin_away_odds or row.avg_away_odds
            if line is not None and home_price is not None:
                home_prob = _implied_prob(home_price)
                away_prob = _implied_prob(away_price) if away_price else (1 - home_prob)
                # Predict the side with higher implied probability
                if home_prob >= away_prob:
                    predictions_to_make.append(("home", home_prob, line, home_price))
                else:
                    predictions_to_make.append(("away", away_prob, -line if line else None, away_price))
                    
        elif bet_type == "total":
            total = row.pin_total or row.avg_total
            over_price = row.pin_over_odds or row.avg_over_odds
            under_price = row.pin_under_odds or row.avg_under_odds
            if total is not None and over_price is not None:
                over_prob = _implied_prob(over_price)
                under_prob = _implied_prob(under_price) if under_price else (1 - over_prob)
                if over_prob >= under_prob:
                    predictions_to_make.append(("over", over_prob, total, over_price))
                else:
                    predictions_to_make.append(("under", under_prob, total, under_price))
                    
        elif bet_type == "moneyline":
            home_ml = row.pin_home_ml or row.avg_home_ml
            away_ml = row.pin_away_ml or row.avg_away_ml
            if home_ml is not None and away_ml is not None:
                home_prob = _implied_prob(home_ml)
                away_prob = _implied_prob(away_ml)
                # Normalize (remove vig)
                total_prob = home_prob + away_prob
                home_prob_fair = home_prob / total_prob if total_prob > 0 else 0.5
                away_prob_fair = away_prob / total_prob if total_prob > 0 else 0.5
                if home_prob_fair >= away_prob_fair:
                    predictions_to_make.append(("home", home_prob_fair, None, home_ml))
                else:
                    predictions_to_make.append(("away", away_prob_fair, None, away_ml))
        
        # Build opening snapshot from consensus (both sides, for all 4 frontend columns)
        open_home_line = None
        open_away_line = None
        open_home_odds = None
        open_away_odds = None
        open_total = None
        open_over_odds = None
        open_under_odds = None
        open_home_ml = None
        open_away_ml = None

        if bet_type == "spread":
            open_home_line = row.pin_home_line or row.avg_home_line
            open_away_line = -open_home_line if open_home_line is not None else None
            open_home_odds = int(row.pin_home_odds or row.avg_home_odds) if (row.pin_home_odds or row.avg_home_odds) else None
            open_away_odds = int(row.pin_away_odds or row.avg_away_odds) if (row.pin_away_odds or row.avg_away_odds) else None
        elif bet_type == "total":
            open_total = row.pin_total or row.avg_total
            open_over_odds = int(row.pin_over_odds or row.avg_over_odds) if (row.pin_over_odds or row.avg_over_odds) else None
            open_under_odds = int(row.pin_under_odds or row.avg_under_odds) if (row.pin_under_odds or row.avg_under_odds) else None
        elif bet_type == "moneyline":
            open_home_ml = int(row.pin_home_ml or row.avg_home_ml) if (row.pin_home_ml or row.avg_home_ml) else None
            open_away_ml = int(row.pin_away_ml or row.avg_away_ml) if (row.pin_away_ml or row.avg_away_ml) else None

        for predicted_side, probability, line_val, odds_val in predictions_to_make:
            # Calculate edge (model prob vs market implied, simplified for now)
            market_prob = probability  # Will be replaced with actual model output
            edge = 0.0  # No edge when using market-implied (placeholder)
            
            # Signal tier based on probability
            if probability >= 0.65:
                tier = "A"
            elif probability >= 0.60:
                tier = "B"
            elif probability >= 0.55:
                tier = "C"
            else:
                tier = "D"
            
            # Kelly fraction (simplified)
            if edge > 0 and odds_val:
                decimal_odds = _american_to_decimal(odds_val)
                kelly = (probability * decimal_odds - 1) / (decimal_odds - 1) if decimal_odds > 1 else 0
                kelly = max(0, min(kelly * 0.25, 0.02))  # Quarter-Kelly, max 2%
            else:
                kelly = 0.0
            
            # Prediction hash for integrity
            hash_input = f"{upcoming_game_id}:{bet_type}:{predicted_side}:{probability:.6f}:{datetime.utcnow().isoformat()}"
            pred_hash = hashlib.sha256(hash_input.encode()).hexdigest()
            
            try:
                await db.execute(
                    text("""
                        INSERT INTO predictions 
                            (id, upcoming_game_id, bet_type, predicted_side, probability,
                             line_at_prediction, odds_at_prediction, edge, signal_tier,
                             kelly_fraction, prediction_hash,
                             home_line_open, away_line_open, home_odds_open, away_odds_open,
                             total_open, over_odds_open, under_odds_open,
                             home_ml_open, away_ml_open,
                             created_at)
                        VALUES 
                            (gen_random_uuid(), :game_id, :bet_type, :side, :prob,
                             :line, :odds, :edge, :tier,
                             :kelly, :hash,
                             :home_line_open, :away_line_open, :home_odds_open, :away_odds_open,
                             :total_open, :over_odds_open, :under_odds_open,
                             :home_ml_open, :away_ml_open,
                             NOW())
                    """),
                    {
                        "game_id": upcoming_game_id,
                        "bet_type": bet_type,
                        "side": predicted_side,
                        "prob": round(probability, 6),
                        "line": line_val,
                        "odds": int(odds_val) if odds_val else None,
                        "edge": round(edge, 6),
                        "tier": tier,
                        "kelly": round(kelly, 6),
                        "hash": pred_hash,
                        "home_line_open": open_home_line,
                        "away_line_open": open_away_line,
                        "home_odds_open": open_home_odds,
                        "away_odds_open": open_away_odds,
                        "total_open": open_total,
                        "over_odds_open": open_over_odds,
                        "under_odds_open": open_under_odds,
                        "home_ml_open": open_home_ml,
                        "away_ml_open": open_away_ml,
                    },
                )
                count += 1
            except Exception as e:
                logger.error(f"  Error saving prediction: {e}")
                await db.rollback()
    
    return count


def _implied_prob(american_odds: int) -> float:
    """Convert American odds to implied probability."""
    if american_odds is None:
        return 0.5
    if american_odds > 0:
        return 100 / (american_odds + 100)
    else:
        return abs(american_odds) / (abs(american_odds) + 100)


def _american_to_decimal(american_odds: int) -> float:
    """Convert American odds to decimal odds."""
    if american_odds > 0:
        return (american_odds / 100) + 1
    else:
        return (100 / abs(american_odds)) + 1


# =============================================================================
# MAIN PIPELINE
# =============================================================================

async def run_pipeline(
    sports: Optional[List[str]] = None,
    generate_predictions: bool = False,
    dry_run: bool = False,
):
    """
    Main pipeline: Fetch games → Save to DB → (optionally) Generate predictions.
    
    Args:
        sports: List of sport codes to fetch (None = all active)
        generate_predictions: Whether to run prediction engine
        dry_run: Print what would happen without DB changes
    """
    
    logger.info("=" * 60)
    logger.info("ROYALEY Live Prediction Pipeline")
    logger.info("=" * 60)
    
    # Initialize
    api_client = OddsAPIClient(api_key=settings.ODDS_API_KEY)
    
    engine = create_async_engine(
        settings.DATABASE_URL.replace("+asyncpg", "+asyncpg"),
        echo=False,
    )
    async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    # Determine which sports to fetch
    sport_keys = ODDS_API_SPORT_KEYS
    if sports:
        sport_keys = {k: v for k, v in sport_keys.items() if k in [s.upper() for s in sports]}
    
    if not sport_keys:
        logger.error("No valid sports specified")
        return
    
    logger.info(f"Sports to fetch: {list(sport_keys.keys())}")
    logger.info(f"Odds API quota: {api_client.requests_remaining} requests remaining")
    logger.info(f"Generate predictions: {generate_predictions}")
    logger.info("")
    
    total_games = 0
    total_odds = 0
    total_predictions = 0
    
    async with async_session() as db:
        for sport_code, api_key in sport_keys.items():
            logger.info(f"{'─' * 40}")
            logger.info(f"Fetching {sport_code} ({api_key})...")
            
            # Get sport_id from DB
            sport_row = await db.execute(
                text("SELECT id FROM sports WHERE code = :code"),
                {"code": sport_code},
            )
            sport = sport_row.fetchone()
            if not sport:
                logger.warning(f"  Sport {sport_code} not found in DB, skipping")
                continue
            sport_id = sport[0]
            
            # Fetch from Odds API
            events = await api_client.fetch_sport_odds(api_key)
            
            if not events:
                logger.info(f"  No upcoming events for {sport_code}")
                continue
            
            logger.info(f"  Found {len(events)} upcoming events")
            
            if dry_run:
                for e in events:
                    logger.info(f"    {e['away_team']} @ {e['home_team']} — {e['commence_time']}")
                continue
            
            # Save each event
            for event in events:
                # Save game
                game_id = await save_upcoming_game(db, event, sport_id, sport_code)
                if not game_id:
                    continue  # Skipped (already started)
                
                total_games += 1
                
                # Save odds
                odds_count = await save_upcoming_odds(db, game_id, event.get("bookmakers", []))
                total_odds += odds_count
                
                # Generate predictions
                if generate_predictions:
                    pred_count = await generate_predictions_for_game(db, game_id, sport_code)
                    total_predictions += pred_count
                
                logger.info(
                    f"    {event['away_team']} @ {event['home_team']} — "
                    f"{odds_count} odds"
                    f"{f', {pred_count} predictions' if generate_predictions else ''}"
                )
            
            await db.commit()
            
            # Check quota
            if api_client.requests_remaining < 10:
                logger.warning(f"  Low quota ({api_client.requests_remaining} remaining), stopping")
                break
    
    await engine.dispose()
    
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"Pipeline Complete")
    logger.info(f"  Games saved:       {total_games}")
    logger.info(f"  Odds records:      {total_odds}")
    logger.info(f"  Predictions made:  {total_predictions}")
    logger.info(f"  API quota left:    {api_client.requests_remaining}")
    logger.info("=" * 60)


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="ROYALEY Live Prediction Pipeline")
    parser.add_argument("--sport", type=str, help="Specific sport code (e.g., NBA, NFL)")
    parser.add_argument("--predict", action="store_true", help="Generate predictions after fetching")
    parser.add_argument("--dry-run", action="store_true", help="Print without saving to DB")
    args = parser.parse_args()
    
    sports = [args.sport] if args.sport else None
    
    asyncio.run(run_pipeline(
        sports=sports,
        generate_predictions=args.predict,
        dry_run=args.dry_run,
    ))


if __name__ == "__main__":
    main()