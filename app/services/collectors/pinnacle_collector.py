"""
ROYALEY - Pinnacle Odds Collector (RapidAPI)
Phase 1: Data Collection Services

Collects sharp odds from Pinnacle via RapidAPI (tipsters provider).
Pinnacle is the sharpest sportsbook - their closing lines are the benchmark for CLV calculation.

API Provider: tipsters via RapidAPI
Base URL: https://pinnacle-odds.p.rapidapi.com
Host: pinnacle-odds.p.rapidapi.com

Endpoints:
- /kit/v1/sports - List all sports
- /kit/v1/leagues?sport_id=X - List leagues for a sport
- /kit/v1/markets?sport_id=X&is_have_odds=true - Get odds/markets
- /kit/v1/details?event_id=X - Event details
- /kit/v1/archive?sport_id=X&page_num=1 - Historical events
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models import Game, Odds, Sportsbook, Sport, Team, GameStatus, ClosingLine, OddsMovement
from app.services.collectors.base_collector import (
    BaseCollector,
    CollectorResult,
    collector_manager,
)

logger = logging.getLogger(__name__)


# Pinnacle sport ID mapping (from their API)
# Discovered via /kit/v1/sports endpoint
PINNACLE_SPORT_IDS = {
    "NFL": 7,            # American Football
    "NCAAF": 7,          # American Football (same sport, different leagues)
    "CFL": 7,            # American Football (same sport, different leagues)
    "NBA": 3,            # Basketball
    "NCAAB": 3,          # Basketball (same sport, different leagues)
    "WNBA": 3,           # Basketball (same sport, different leagues)
    "NHL": 4,            # Hockey
    "MLB": 9,            # Baseball
    "ATP": 2,            # Tennis
    "WTA": 2,            # Tennis (same sport, different leagues)
}

# League IDs for filtering (you may need to get these from /kit/v1/leagues endpoint)
PINNACLE_LEAGUE_NAMES = {
    "NFL": ["NFL", "National Football League"],
    "NCAAF": ["NCAA", "College Football", "NCAAF"],
    "CFL": ["CFL", "Canadian Football"],
    "NBA": ["NBA", "National Basketball Association"],
    "NCAAB": ["NCAA", "College Basketball", "NCAAB"],
    "WNBA": ["WNBA"],
    "NHL": ["NHL", "National Hockey League"],
    "MLB": ["MLB", "Major League Baseball"],
    "ATP": ["ATP"],
    "WTA": ["WTA"],
}


class PinnacleCollector(BaseCollector):
    """
    Collector for Pinnacle Odds via RapidAPI (tipsters provider).
    
    Pinnacle is considered the sharpest sportsbook globally.
    Their closing lines are the industry benchmark for CLV (Closing Line Value).
    """
    
    def __init__(self):
        # CORRECT RapidAPI endpoint
        super().__init__(
            name="pinnacle",
            base_url="https://pinnacle-odds.p.rapidapi.com",
            rate_limit=100,
            rate_window=60,
            timeout=30.0,
            max_retries=3,
        )
        self.api_key = settings.RAPIDAPI_KEY
        self._sports_cache: Dict[int, str] = {}
        self._leagues_cache: Dict[int, List[Dict]] = {}
    
    def _get_headers(self) -> Dict[str, str]:
        """Get RapidAPI headers with CORRECT host."""
        return {
            "Accept": "application/json",
            "X-RapidAPI-Key": self.api_key,
            "X-RapidAPI-Host": "pinnacle-odds.p.rapidapi.com",  # CORRECT HOST
        }
    
    async def collect(
        self,
        sport_code: str = None,
        include_live: bool = False,
        **kwargs,
    ) -> CollectorResult:
        """
        Collect Pinnacle odds for specified sports.
        
        Args:
            sport_code: Optional sport code (NFL, NBA, etc.)
            include_live: Include live/in-play odds
            
        Returns:
            CollectorResult with Pinnacle odds data
        """
        if not self.api_key:
            return CollectorResult(
                success=False,
                error="RapidAPI key not configured. Set RAPIDAPI_KEY in environment.",
            )
        
        sports_to_collect = (
            [sport_code] if sport_code 
            else list(PINNACLE_SPORT_IDS.keys())
        )
        
        all_odds = []
        errors = []
        successful_sports = []
        
        logger.info(f"[Pinnacle] 🎯 Starting collection for {len(sports_to_collect)} sport(s)")
        print(f"[Pinnacle] 🎯 Starting Pinnacle odds collection...")
        print(f"[Pinnacle] Sports: {', '.join(sports_to_collect)}")
        
        for sport in sports_to_collect:
            try:
                logger.info(f"[Pinnacle] Collecting {sport} odds...")
                print(f"[Pinnacle] 📊 Fetching {sport} odds from Pinnacle...")
                
                odds_data = await self._collect_sport_odds(sport, include_live)
                all_odds.extend(odds_data)
                successful_sports.append(sport)
                
                logger.info(f"[Pinnacle] ✅ Collected {len(odds_data)} odds records for {sport}")
                print(f"[Pinnacle] ✅ {sport}: {len(odds_data)} odds records")
                
            except Exception as e:
                logger.error(f"[Pinnacle] ❌ Error collecting {sport}: {e}")
                print(f"[Pinnacle] ❌ {sport} error: {e}")
                errors.append(f"{sport}: {str(e)}")
        
        logger.info(f"[Pinnacle] Collection complete: {len(all_odds)} total records")
        print(f"[Pinnacle] 🏁 Total: {len(all_odds)} odds records from {len(successful_sports)} sports")
        
        return CollectorResult(
            success=len(successful_sports) > 0,
            data=all_odds,
            records_count=len(all_odds),
            error="; ".join(errors) if errors else None,
            metadata={
                "source": "pinnacle",
                "sports_collected": sports_to_collect,
                "successful_sports": successful_sports,
                "failed_sports": [s for s in sports_to_collect if s not in successful_sports],
                "include_live": include_live,
            },
        )
    
    async def _collect_sport_odds(
        self,
        sport_code: str,
        include_live: bool = False,
    ) -> List[Dict[str, Any]]:
        """Collect odds for a single sport from Pinnacle."""
        sport_id = PINNACLE_SPORT_IDS.get(sport_code)
        if not sport_id:
            logger.warning(f"[Pinnacle] Unknown sport code: {sport_code}")
            return []
        
        # Use the markets endpoint with is_have_odds=true
        event_type = "prematch" if not include_live else "live"
        
        params = {
            "sport_id": sport_id,
            "is_have_odds": "true",
            "event_type": event_type,
        }
        
        logger.info(f"[Pinnacle] Fetching markets for sport_id={sport_id}")
        print(f"[Pinnacle] 🔍 Endpoint: /kit/v1/markets?sport_id={sport_id}&is_have_odds=true")
        
        data = await self.get("/kit/v1/markets", params=params)
        
        if not data:
            logger.warning(f"[Pinnacle] No data returned for {sport_code}")
            return []
        
        return self._parse_markets_response(data, sport_code)
    
    def _parse_markets_response(
        self,
        data: Any,
        sport_code: str,
    ) -> List[Dict[str, Any]]:
        """
        Parse Pinnacle markets response into normalized odds records.
        
        The /kit/v1/markets endpoint returns events with their odds.
        """
        parsed_odds = []
        
        # Handle different response structures
        events = []
        
        if isinstance(data, list):
            events = data
        elif isinstance(data, dict):
            # Could be wrapped in 'events' or 'data' key
            events = data.get("events", data.get("data", data.get("markets", [])))
            if not events and "leagues" in data:
                # Nested in leagues
                for league in data["leagues"]:
                    events.extend(league.get("events", []))
        
        logger.info(f"[Pinnacle] Parsing {len(events)} events")
        
        for event in events:
            try:
                event_odds = self._parse_event(event, sport_code)
                parsed_odds.extend(event_odds)
            except Exception as e:
                logger.warning(f"[Pinnacle] Error parsing event: {e}")
                continue
        
        return parsed_odds
    
    def _parse_event(
        self,
        event: Dict[str, Any],
        sport_code: str,
    ) -> List[Dict[str, Any]]:
        """Parse a single event into odds records."""
        records = []
        
        # Extract event info - handle different field names
        event_id = str(event.get("event_id", event.get("id", "")))
        
        # Teams can be in different formats
        home_team = (
            event.get("home", "") or 
            event.get("home_team", "") or 
            event.get("teams", {}).get("home", {}).get("name", "") or
            event.get("participants", [{}])[0].get("name", "") if event.get("participants") else ""
        )
        away_team = (
            event.get("away", "") or 
            event.get("away_team", "") or 
            event.get("teams", {}).get("away", {}).get("name", "") or
            event.get("participants", [{}])[1].get("name", "") if len(event.get("participants", [])) > 1 else ""
        )
        
        # Start time
        start_time = event.get("starts", event.get("start_time", event.get("commence_time", "")))
        
        # League info for filtering
        league_name = event.get("league_name", event.get("league", {}).get("name", ""))
        
        if not event_id or not home_team or not away_team:
            return records
        
        # Filter by league if needed
        target_leagues = PINNACLE_LEAGUE_NAMES.get(sport_code, [])
        if target_leagues and league_name:
            if not any(tl.lower() in league_name.lower() for tl in target_leagues):
                # Skip events from other leagues
                return records
        
        # Parse odds from periods
        periods = event.get("periods", event.get("odds", []))
        
        if isinstance(periods, dict):
            periods = [periods]
        
        for period in periods:
            # Only get full game odds (period 0 or period_number 0)
            period_num = period.get("period_number", period.get("number", period.get("period", 0)))
            if period_num != 0:
                continue
            
            # Moneyline
            moneyline = period.get("moneyline", period.get("money_line", {}))
            if moneyline:
                home_ml = moneyline.get("home", moneyline.get("homePrice"))
                away_ml = moneyline.get("away", moneyline.get("awayPrice"))
                
                if home_ml is not None or away_ml is not None:
                    records.append({
                        "sport_code": sport_code,
                        "external_id": event_id,
                        "home_team": home_team,
                        "away_team": away_team,
                        "commence_time": start_time,
                        "league_name": league_name,
                        "sportsbook_key": "pinnacle",
                        "sportsbook_name": "Pinnacle",
                        "bet_type": "moneyline",
                        "home_odds": self._convert_to_american(home_ml),
                        "away_odds": self._convert_to_american(away_ml),
                        "home_line": None,
                        "away_line": None,
                        "total": None,
                        "over_odds": None,
                        "under_odds": None,
                        "is_pinnacle": True,
                        "recorded_at": datetime.utcnow().isoformat(),
                    })
            
            # Spread
            spread = period.get("spread", period.get("spreads", period.get("handicap", {})))
            if isinstance(spread, list) and spread:
                spread = spread[0]
            
            if spread:
                hdp = spread.get("hdp", spread.get("handicap", spread.get("home_spread")))
                home_price = spread.get("home", spread.get("homePrice"))
                away_price = spread.get("away", spread.get("awayPrice"))
                
                if hdp is not None:
                    records.append({
                        "sport_code": sport_code,
                        "external_id": event_id,
                        "home_team": home_team,
                        "away_team": away_team,
                        "commence_time": start_time,
                        "league_name": league_name,
                        "sportsbook_key": "pinnacle",
                        "sportsbook_name": "Pinnacle",
                        "bet_type": "spread",
                        "home_odds": self._convert_to_american(home_price),
                        "away_odds": self._convert_to_american(away_price),
                        "home_line": float(hdp) if hdp is not None else None,
                        "away_line": -float(hdp) if hdp is not None else None,
                        "total": None,
                        "over_odds": None,
                        "under_odds": None,
                        "is_pinnacle": True,
                        "recorded_at": datetime.utcnow().isoformat(),
                    })
            
            # Total
            total = period.get("total", period.get("totals", period.get("over_under", {})))
            if isinstance(total, list) and total:
                total = total[0]
            
            if total:
                points = total.get("points", total.get("hdp", total.get("line")))
                over_price = total.get("over", total.get("overPrice"))
                under_price = total.get("under", total.get("underPrice"))
                
                if points is not None:
                    records.append({
                        "sport_code": sport_code,
                        "external_id": event_id,
                        "home_team": home_team,
                        "away_team": away_team,
                        "commence_time": start_time,
                        "league_name": league_name,
                        "sportsbook_key": "pinnacle",
                        "sportsbook_name": "Pinnacle",
                        "bet_type": "total",
                        "home_odds": None,
                        "away_odds": None,
                        "home_line": None,
                        "away_line": None,
                        "total": float(points) if points is not None else None,
                        "over_odds": self._convert_to_american(over_price),
                        "under_odds": self._convert_to_american(under_price),
                        "is_pinnacle": True,
                        "recorded_at": datetime.utcnow().isoformat(),
                    })
        
        # Also check for direct odds fields (alternative structure)
        if not records:
            # Try parsing direct odds fields
            for bet_type, fields in [
                ("moneyline", ["moneyline", "ml", "money_line"]),
                ("spread", ["spread", "spreads", "handicap"]),
                ("total", ["total", "totals", "over_under"]),
            ]:
                for field in fields:
                    if field in event and event[field]:
                        odds_data = event[field]
                        record = self._create_odds_record(
                            odds_data, bet_type, event_id, 
                            home_team, away_team, start_time, 
                            league_name, sport_code
                        )
                        if record:
                            records.append(record)
                        break
        
        return records
    
    def _create_odds_record(
        self,
        odds_data: Any,
        bet_type: str,
        event_id: str,
        home_team: str,
        away_team: str,
        start_time: str,
        league_name: str,
        sport_code: str,
    ) -> Optional[Dict[str, Any]]:
        """Create odds record from direct odds data."""
        if isinstance(odds_data, list) and odds_data:
            odds_data = odds_data[0]
        
        if not isinstance(odds_data, dict):
            return None
        
        record = {
            "sport_code": sport_code,
            "external_id": event_id,
            "home_team": home_team,
            "away_team": away_team,
            "commence_time": start_time,
            "league_name": league_name,
            "sportsbook_key": "pinnacle",
            "sportsbook_name": "Pinnacle",
            "bet_type": bet_type,
            "home_odds": None,
            "away_odds": None,
            "home_line": None,
            "away_line": None,
            "total": None,
            "over_odds": None,
            "under_odds": None,
            "is_pinnacle": True,
            "recorded_at": datetime.utcnow().isoformat(),
        }
        
        if bet_type == "moneyline":
            record["home_odds"] = self._convert_to_american(odds_data.get("home"))
            record["away_odds"] = self._convert_to_american(odds_data.get("away"))
        elif bet_type == "spread":
            hdp = odds_data.get("hdp", odds_data.get("handicap"))
            record["home_line"] = float(hdp) if hdp is not None else None
            record["away_line"] = -float(hdp) if hdp is not None else None
            record["home_odds"] = self._convert_to_american(odds_data.get("home"))
            record["away_odds"] = self._convert_to_american(odds_data.get("away"))
        elif bet_type == "total":
            points = odds_data.get("points", odds_data.get("hdp"))
            record["total"] = float(points) if points is not None else None
            record["over_odds"] = self._convert_to_american(odds_data.get("over"))
            record["under_odds"] = self._convert_to_american(odds_data.get("under"))
        
        return record
    
    def _convert_to_american(self, odds: Any) -> Optional[int]:
        """
        Convert decimal odds to American odds.
        
        Pinnacle returns decimal odds (e.g., 1.91, 2.50).
        American odds: -110, +150, etc.
        """
        if odds is None:
            return None
        
        try:
            odds = float(odds)
            
            # If already looks like American odds (> 50 or < -50)
            if odds > 50 or odds < -50:
                return int(odds)
            
            # Convert decimal to American
            if odds >= 2.0:
                # Positive American odds
                return int((odds - 1) * 100)
            elif odds > 1.0:
                # Negative American odds
                return int(-100 / (odds - 1))
            else:
                return None
                
        except (ValueError, TypeError, ZeroDivisionError):
            return None
    
    async def validate(self, data: Any) -> bool:
        """Validate Pinnacle odds data."""
        if not data:
            return False
        if not isinstance(data, list):
            return False
        return len(data) > 0
    
    async def get_sports_list(self) -> List[Dict[str, Any]]:
        """Get list of available sports from Pinnacle."""
        try:
            data = await self.get("/kit/v1/sports")
            return data if isinstance(data, list) else []
        except Exception as e:
            logger.error(f"[Pinnacle] Error fetching sports: {e}")
            return []
    
    async def get_leagues(self, sport_id: int) -> List[Dict[str, Any]]:
        """Get leagues for a sport."""
        try:
            data = await self.get("/kit/v1/leagues", params={"sport_id": sport_id})
            return data if isinstance(data, list) else []
        except Exception as e:
            logger.error(f"[Pinnacle] Error fetching leagues: {e}")
            return []
    
    async def get_event_details(self, event_id: int) -> Optional[Dict[str, Any]]:
        """Get detailed information for a specific event."""
        try:
            data = await self.get("/kit/v1/details", params={"event_id": event_id})
            return data
        except Exception as e:
            logger.error(f"[Pinnacle] Error fetching event details: {e}")
            return None
    
    async def get_archive_events(self, sport_id: int, page_num: int = 1) -> List[Dict[str, Any]]:
        """Get historical/archive events for a sport."""
        try:
            data = await self.get("/kit/v1/archive", params={
                "sport_id": sport_id,
                "page_num": page_num,
            })
            return data if isinstance(data, list) else []
        except Exception as e:
            logger.error(f"[Pinnacle] Error fetching archive: {e}")
            return []
    
    async def save_to_database(
        self,
        records: List[Dict[str, Any]],
        session: AsyncSession,
    ) -> int:
        """Save Pinnacle odds to database."""
        if not records:
            return 0
        
        saved_count = 0
        
        # Get or create Pinnacle sportsbook
        pinnacle_book = await self._get_or_create_pinnacle_sportsbook(session)
        
        # Group records by game
        games_map: Dict[str, List[Dict[str, Any]]] = {}
        for record in records:
            key = f"{record['sport_code']}_{record['external_id']}"
            if key not in games_map:
                games_map[key] = []
            games_map[key].append(record)
        
        for game_key, game_records in games_map.items():
            try:
                first_record = game_records[0]
                
                # Find or create the game
                game = await self._find_or_create_game(session, first_record)
                if not game:
                    continue
                
                # Save each bet type's odds
                for record in game_records:
                    bet_type = record["bet_type"]
                    
                    # Check for existing Pinnacle odds
                    existing = await session.execute(
                        select(Odds).where(
                            and_(
                                Odds.game_id == game.id,
                                Odds.sportsbook_id == pinnacle_book.id,
                                Odds.bet_type == bet_type,
                            )
                        )
                    )
                    existing_odds = existing.scalar_one_or_none()
                    
                    if existing_odds:
                        # Update existing
                        existing_odds.home_odds = record.get("home_odds")
                        existing_odds.away_odds = record.get("away_odds")
                        existing_odds.home_line = record.get("home_line")
                        existing_odds.away_line = record.get("away_line")
                        existing_odds.total = record.get("total")
                        existing_odds.over_odds = record.get("over_odds")
                        existing_odds.under_odds = record.get("under_odds")
                        existing_odds.recorded_at = datetime.utcnow()
                    else:
                        # Create new
                        new_odds = Odds(
                            game_id=game.id,
                            sportsbook_id=pinnacle_book.id,
                            sportsbook_key="pinnacle",
                            bet_type=bet_type,
                            home_odds=record.get("home_odds"),
                            away_odds=record.get("away_odds"),
                            home_line=record.get("home_line"),
                            away_line=record.get("away_line"),
                            total=record.get("total"),
                            over_odds=record.get("over_odds"),
                            under_odds=record.get("under_odds"),
                            is_opening=False,
                            recorded_at=datetime.utcnow(),
                        )
                        session.add(new_odds)
                        saved_count += 1
                
                await session.flush()
                
            except Exception as e:
                logger.error(f"[Pinnacle] Error saving game {game_key}: {e}")
                continue
        
        await session.commit()
        logger.info(f"[Pinnacle] Saved {saved_count} new odds records")
        return saved_count
    
    async def _get_or_create_pinnacle_sportsbook(
        self,
        session: AsyncSession,
    ) -> Sportsbook:
        """Get or create Pinnacle sportsbook record."""
        result = await session.execute(
            select(Sportsbook).where(Sportsbook.key == "pinnacle")
        )
        sportsbook = result.scalar_one_or_none()
        
        if not sportsbook:
            sportsbook = Sportsbook(
                name="Pinnacle",
                key="pinnacle",
                is_sharp=True,
                is_active=True,
                priority=1,
            )
            session.add(sportsbook)
            await session.flush()
            logger.info("[Pinnacle] Created Pinnacle sportsbook record")
        
        return sportsbook
    
    async def _find_or_create_game(
        self,
        session: AsyncSession,
        record: Dict[str, Any],
    ) -> Optional[Game]:
        """Find existing game or create new one."""
        external_id = record["external_id"]
        sport_code = record["sport_code"]
        
        # Try to find by external_id first
        result = await session.execute(
            select(Game).where(Game.external_id == external_id)
        )
        game = result.scalar_one_or_none()
        
        if game:
            return game
        
        home_team_name = record["home_team"]
        away_team_name = record["away_team"]
        
        # Get sport
        sport_result = await session.execute(
            select(Sport).where(Sport.code == sport_code)
        )
        sport = sport_result.scalar_one_or_none()
        
        if not sport:
            sport_names = {
                "NBA": "National Basketball Association",
                "NFL": "National Football League",
                "NCAAF": "NCAA Football",
                "NCAAB": "NCAA Basketball",
                "NHL": "National Hockey League",
                "MLB": "Major League Baseball",
                "WNBA": "Women's National Basketball Association",
                "CFL": "Canadian Football League",
                "ATP": "ATP Tennis",
                "WTA": "WTA Tennis",
            }
            sport = Sport(
                code=sport_code,
                name=sport_names.get(sport_code, sport_code),
                is_active=True,
            )
            session.add(sport)
            await session.flush()
        
        # Get or create teams
        home_team = await self._get_or_create_team(session, sport.id, home_team_name)
        away_team = await self._get_or_create_team(session, sport.id, away_team_name)
        
        # Parse game date
        commence_time = record.get("commence_time")
        game_date = self._parse_datetime(commence_time)
        
        # Try to find game by teams and approximate date
        date_start = game_date - timedelta(hours=12)
        date_end = game_date + timedelta(hours=12)
        
        result = await session.execute(
            select(Game).where(
                and_(
                    Game.sport_id == sport.id,
                    Game.home_team_id == home_team.id,
                    Game.away_team_id == away_team.id,
                    Game.scheduled_at >= date_start,
                    Game.scheduled_at <= date_end,
                )
            )
        )
        game = result.scalar_one_or_none()
        
        if game:
            if not game.external_id:
                game.external_id = external_id
            return game
        
        # Create new game
        game = Game(
            sport_id=sport.id,
            external_id=external_id,
            home_team_id=home_team.id,
            away_team_id=away_team.id,
            scheduled_at=game_date,
            status=GameStatus.SCHEDULED,
        )
        session.add(game)
        await session.flush()
        logger.info(f"[Pinnacle] Created game: {away_team_name} @ {home_team_name}")
        
        return game
    
    async def _get_or_create_team(
        self,
        session: AsyncSession,
        sport_id: UUID,
        team_name: str,
    ) -> Team:
        """Get or create team record."""
        result = await session.execute(
            select(Team).where(
                and_(
                    Team.sport_id == sport_id,
                    Team.name == team_name,
                )
            )
        )
        team = result.scalar_one_or_none()
        
        if not team:
            abbreviation = team_name[:3].upper() if len(team_name) >= 3 else team_name.upper()
            team = Team(
                sport_id=sport_id,
                external_id=f"pinnacle_{team_name.lower().replace(' ', '_')}",
                name=team_name,
                abbreviation=abbreviation,
                is_active=True,
            )
            session.add(team)
            await session.flush()
        
        return team
    
    def _parse_datetime(self, dt_str: Any) -> datetime:
        """Parse datetime string to naive UTC datetime."""
        if not dt_str:
            return datetime.utcnow()
        
        if isinstance(dt_str, datetime):
            if dt_str.tzinfo:
                utc_dt = dt_str.astimezone(timezone.utc)
                return datetime(
                    utc_dt.year, utc_dt.month, utc_dt.day,
                    utc_dt.hour, utc_dt.minute, utc_dt.second
                )
            return dt_str
        
        try:
            if "T" in str(dt_str):
                parsed = datetime.fromisoformat(str(dt_str).replace("Z", "+00:00"))
                if parsed.tzinfo:
                    utc_dt = parsed.astimezone(timezone.utc)
                    return datetime(
                        utc_dt.year, utc_dt.month, utc_dt.day,
                        utc_dt.hour, utc_dt.minute, utc_dt.second
                    )
                return parsed
        except:
            pass
        
        return datetime.utcnow()


# Create and register collector instance
pinnacle_collector = PinnacleCollector()
collector_manager.register(pinnacle_collector)


class PinnacleCollector(BaseCollector):
    """
    Collector for Pinnacle Odds via RapidAPI.
    
    Pinnacle is considered the sharpest sportsbook globally.
    Their closing lines are the industry benchmark for CLV (Closing Line Value).
    
    Features:
    - Pre-match odds (spreads, moneylines, totals)
    - Live odds updates
    - Historical odds snapshots
    - Automatic closing line capture
    - Line movement detection
    
    API Provider: DataMenu via RapidAPI
    Endpoint: pinnacle-odds-api.p.rapidapi.com
    """
    
    # Market types
    MARKET_MONEYLINE = 0
    MARKET_SPREAD = 1
    MARKET_TOTAL = 2
    
    MARKET_NAMES = {
        0: "moneyline",
        1: "spread",
        2: "total",
    }
    
    def __init__(self):
        # RapidAPI endpoint
        super().__init__(
            name="pinnacle",
            base_url="https://pinnacle-odds-api.p.rapidapi.com",
            rate_limit=100,     # Adjust based on your plan
            rate_window=60,     # Per minute
            timeout=30.0,
            max_retries=3,
        )
        self.api_key = settings.RAPIDAPI_KEY
        self._requests_used = 0
    
    def _get_headers(self) -> Dict[str, str]:
        """Get RapidAPI headers."""
        return {
            "Accept": "application/json",
            "X-RapidAPI-Key": self.api_key,
            "X-RapidAPI-Host": "pinnacle-odds-api.p.rapidapi.com",
        }
    
    async def collect(
        self,
        sport_code: str = None,
        include_live: bool = False,
        **kwargs,
    ) -> CollectorResult:
        """
        Collect Pinnacle odds for specified sports.
        
        Args:
            sport_code: Optional sport code (collects all if None)
            include_live: Include live/in-play odds
            
        Returns:
            CollectorResult with Pinnacle odds data
        """
        if not self.api_key:
            return CollectorResult(
                success=False,
                error="RapidAPI key not configured. Set RAPIDAPI_KEY in environment.",
            )
        
        sports_to_collect = (
            [sport_code] if sport_code 
            else list(PINNACLE_SPORT_IDS.keys())
        )
        
        all_odds = []
        errors = []
        successful_sports = []
        
        logger.info(f"[Pinnacle] 🎯 Starting collection for {len(sports_to_collect)} sport(s)")
        print(f"[Pinnacle] 🎯 Starting Pinnacle odds collection...")
        print(f"[Pinnacle] Sports: {', '.join(sports_to_collect)}")
        
        for sport in sports_to_collect:
            try:
                logger.info(f"[Pinnacle] Collecting {sport} odds...")
                print(f"[Pinnacle] 📊 Fetching {sport} odds from Pinnacle...")
                
                odds_data = await self._collect_sport_odds(sport, include_live)
                all_odds.extend(odds_data)
                successful_sports.append(sport)
                
                logger.info(f"[Pinnacle] ✅ Collected {len(odds_data)} odds records for {sport}")
                print(f"[Pinnacle] ✅ {sport}: {len(odds_data)} odds records")
                
            except Exception as e:
                logger.error(f"[Pinnacle] ❌ Error collecting {sport}: {e}")
                print(f"[Pinnacle] ❌ {sport} error: {e}")
                errors.append(f"{sport}: {str(e)}")
        
        logger.info(f"[Pinnacle] Collection complete: {len(all_odds)} total records")
        print(f"[Pinnacle] 🏁 Total: {len(all_odds)} odds records from {len(successful_sports)} sports")
        
        return CollectorResult(
            success=len(successful_sports) > 0,
            data=all_odds,
            records_count=len(all_odds),
            error="; ".join(errors) if errors else None,
            metadata={
                "source": "pinnacle",
                "sports_collected": sports_to_collect,
                "successful_sports": successful_sports,
                "failed_sports": [s for s in sports_to_collect if s not in successful_sports],
                "include_live": include_live,
            },
        )
    
    async def _collect_sport_odds(
        self,
        sport_code: str,
        include_live: bool = False,
    ) -> List[Dict[str, Any]]:
        """Collect odds for a single sport from Pinnacle."""
        sport_id = PINNACLE_SPORT_IDS.get(sport_code)
        if not sport_id:
            logger.warning(f"[Pinnacle] Unknown sport code: {sport_code}")
            return []
        
        # Try multiple endpoint patterns (RapidAPI providers vary)
        endpoints_to_try = [
            # DataMenu endpoints
            f"/v1/odds/{sport_id}",
            f"/odds/{sport_id}",
            f"/v1/sports/{sport_id}/odds",
            # Alternative patterns
            f"/kit/v1/markets?sport_id={sport_id}&is_have_odds=true",
        ]
        
        params = {
            "sport_id": sport_id,
            "is_have_odds": "true",
        }
        
        data = None
        last_error = None
        
        for endpoint in endpoints_to_try:
            try:
                logger.info(f"[Pinnacle] Trying endpoint: {endpoint}")
                
                # Some endpoints use query params, some use path params
                if "?" in endpoint:
                    data = await self.get(endpoint.split("?")[0], params=dict(x.split("=") for x in endpoint.split("?")[1].split("&")))
                else:
                    data = await self.get(endpoint, params=params)
                
                if data:
                    logger.info(f"[Pinnacle] ✅ Endpoint {endpoint} returned data")
                    break
                    
            except Exception as e:
                last_error = e
                logger.debug(f"[Pinnacle] Endpoint {endpoint} failed: {e}")
                continue
        
        if not data:
            # Try the events endpoint as fallback
            try:
                data = await self._fetch_events_with_odds(sport_id)
            except Exception as e:
                last_error = e
        
        if not data:
            raise Exception(f"All endpoints failed. Last error: {last_error}")
        
        return self._parse_pinnacle_response(data, sport_code)
    
    async def _fetch_events_with_odds(
        self,
        sport_id: int,
    ) -> List[Dict[str, Any]]:
        """Fetch events and odds separately then combine."""
        # Get fixtures/events first
        events_data = await self.get("/v1/fixtures", params={"sport_id": sport_id})
        
        if not events_data:
            return []
        
        # Get odds for the sport
        odds_data = await self.get("/v1/odds", params={"sport_id": sport_id})
        
        # Combine events with odds
        odds_by_event = {}
        if odds_data:
            for league in odds_data if isinstance(odds_data, list) else [odds_data]:
                for event_odds in league.get("events", []):
                    event_id = event_odds.get("id")
                    if event_id:
                        odds_by_event[event_id] = event_odds
        
        # Merge
        result = []
        for league in events_data if isinstance(events_data, list) else [events_data]:
            for event in league.get("events", []):
                event_id = event.get("id")
                if event_id in odds_by_event:
                    event["odds"] = odds_by_event[event_id]
                result.append(event)
        
        return result
    
    def _parse_pinnacle_response(
        self,
        data: Any,
        sport_code: str,
    ) -> List[Dict[str, Any]]:
        """
        Parse Pinnacle API response into normalized odds records.
        
        Handles multiple response formats from different RapidAPI providers.
        """
        parsed_odds = []
        
        # Handle different response structures
        events = []
        
        if isinstance(data, list):
            # List of leagues/events
            for item in data:
                if "events" in item:
                    events.extend(item["events"])
                elif "id" in item and ("home" in item or "home_team" in item):
                    events.append(item)
                elif "leagues" in item:
                    for league in item["leagues"]:
                        events.extend(league.get("events", []))
        elif isinstance(data, dict):
            if "events" in data:
                events = data["events"]
            elif "leagues" in data:
                for league in data["leagues"]:
                    events.extend(league.get("events", []))
            elif "data" in data:
                return self._parse_pinnacle_response(data["data"], sport_code)
        
        for event in events:
            try:
                event_odds = self._parse_event_odds(event, sport_code)
                parsed_odds.extend(event_odds)
            except Exception as e:
                logger.warning(f"[Pinnacle] Error parsing event: {e}")
                continue
        
        return parsed_odds
    
    def _parse_event_odds(
        self,
        event: Dict[str, Any],
        sport_code: str,
    ) -> List[Dict[str, Any]]:
        """Parse a single event into odds records."""
        records = []
        
        # Extract event info
        event_id = str(event.get("id", event.get("event_id", "")))
        home_team = event.get("home", event.get("home_team", event.get("homeTeam", "")))
        away_team = event.get("away", event.get("away_team", event.get("awayTeam", "")))
        
        # Parse start time
        start_time = event.get("starts", event.get("start_time", event.get("commence_time", event.get("date", ""))))
        
        if not event_id or not home_team or not away_team:
            return records
        
        # Get odds data
        odds_data = event.get("odds", event.get("markets", event.get("periods", [])))
        
        # Handle different odds structures
        if isinstance(odds_data, dict):
            # Direct odds object
            records.extend(self._parse_odds_object(
                odds_data, event_id, home_team, away_team, start_time, sport_code
            ))
        elif isinstance(odds_data, list):
            # List of markets/periods
            for market in odds_data:
                records.extend(self._parse_odds_object(
                    market, event_id, home_team, away_team, start_time, sport_code
                ))
        
        # Also check for direct market fields
        for market_type, market_name in [
            ("moneyline", "moneyline"), 
            ("spreads", "spread"), 
            ("totals", "total"),
            ("spread", "spread"),
            ("total", "total"),
        ]:
            if market_type in event:
                market_data = event[market_type]
                if isinstance(market_data, list):
                    for m in market_data:
                        record = self._create_odds_record(
                            m, market_name, event_id, home_team, away_team, start_time, sport_code
                        )
                        if record:
                            records.append(record)
                elif isinstance(market_data, dict):
                    record = self._create_odds_record(
                        market_data, market_name, event_id, home_team, away_team, start_time, sport_code
                    )
                    if record:
                        records.append(record)
        
        return records
    
    def _parse_odds_object(
        self,
        odds_data: Dict[str, Any],
        event_id: str,
        home_team: str,
        away_team: str,
        start_time: str,
        sport_code: str,
    ) -> List[Dict[str, Any]]:
        """Parse an odds object into records."""
        records = []
        
        # Check for period-based structure (Pinnacle uses periods)
        period = odds_data.get("period", odds_data.get("number", 0))
        if period != 0:
            # Skip non-full-game periods for now
            return records
        
        # Moneyline
        if "moneyline" in odds_data or "ml" in odds_data:
            ml = odds_data.get("moneyline", odds_data.get("ml", {}))
            record = {
                "sport_code": sport_code,
                "external_id": event_id,
                "home_team": home_team,
                "away_team": away_team,
                "commence_time": start_time,
                "sportsbook_key": "pinnacle",
                "sportsbook_name": "Pinnacle",
                "bet_type": "moneyline",
                "home_odds": self._convert_to_american(ml.get("home", ml.get("homePrice"))),
                "away_odds": self._convert_to_american(ml.get("away", ml.get("awayPrice"))),
                "home_line": None,
                "away_line": None,
                "total": None,
                "over_odds": None,
                "under_odds": None,
                "is_pinnacle": True,
                "recorded_at": datetime.utcnow().isoformat(),
            }
            records.append(record)
        
        # Spread
        if "spread" in odds_data or "spreads" in odds_data or "handicap" in odds_data:
            spread = odds_data.get("spread", odds_data.get("spreads", odds_data.get("handicap", {})))
            if isinstance(spread, list) and spread:
                spread = spread[0]  # Take first spread line
            
            home_spread = spread.get("hdp", spread.get("home_hdp", spread.get("homeHandicap")))
            record = {
                "sport_code": sport_code,
                "external_id": event_id,
                "home_team": home_team,
                "away_team": away_team,
                "commence_time": start_time,
                "sportsbook_key": "pinnacle",
                "sportsbook_name": "Pinnacle",
                "bet_type": "spread",
                "home_odds": self._convert_to_american(spread.get("home", spread.get("homePrice"))),
                "away_odds": self._convert_to_american(spread.get("away", spread.get("awayPrice"))),
                "home_line": float(home_spread) if home_spread is not None else None,
                "away_line": -float(home_spread) if home_spread is not None else None,
                "total": None,
                "over_odds": None,
                "under_odds": None,
                "is_pinnacle": True,
                "recorded_at": datetime.utcnow().isoformat(),
            }
            records.append(record)
        
        # Total
        if "total" in odds_data or "totals" in odds_data or "over_under" in odds_data:
            total = odds_data.get("total", odds_data.get("totals", odds_data.get("over_under", {})))
            if isinstance(total, list) and total:
                total = total[0]  # Take first total line
            
            points = total.get("points", total.get("hdp", total.get("line")))
            record = {
                "sport_code": sport_code,
                "external_id": event_id,
                "home_team": home_team,
                "away_team": away_team,
                "commence_time": start_time,
                "sportsbook_key": "pinnacle",
                "sportsbook_name": "Pinnacle",
                "bet_type": "total",
                "home_odds": None,
                "away_odds": None,
                "home_line": None,
                "away_line": None,
                "total": float(points) if points is not None else None,
                "over_odds": self._convert_to_american(total.get("over", total.get("overPrice"))),
                "under_odds": self._convert_to_american(total.get("under", total.get("underPrice"))),
                "is_pinnacle": True,
                "recorded_at": datetime.utcnow().isoformat(),
            }
            records.append(record)
        
        return records
    
    def _create_odds_record(
        self,
        market_data: Dict[str, Any],
        market_name: str,
        event_id: str,
        home_team: str,
        away_team: str,
        start_time: str,
        sport_code: str,
    ) -> Optional[Dict[str, Any]]:
        """Create a single odds record from market data."""
        record = {
            "sport_code": sport_code,
            "external_id": event_id,
            "home_team": home_team,
            "away_team": away_team,
            "commence_time": start_time,
            "sportsbook_key": "pinnacle",
            "sportsbook_name": "Pinnacle",
            "bet_type": market_name,
            "home_odds": None,
            "away_odds": None,
            "home_line": None,
            "away_line": None,
            "total": None,
            "over_odds": None,
            "under_odds": None,
            "is_pinnacle": True,
            "recorded_at": datetime.utcnow().isoformat(),
        }
        
        if market_name == "moneyline":
            record["home_odds"] = self._convert_to_american(market_data.get("home", market_data.get("homePrice")))
            record["away_odds"] = self._convert_to_american(market_data.get("away", market_data.get("awayPrice")))
        elif market_name == "spread":
            hdp = market_data.get("hdp", market_data.get("handicap", market_data.get("line")))
            record["home_line"] = float(hdp) if hdp is not None else None
            record["away_line"] = -float(hdp) if hdp is not None else None
            record["home_odds"] = self._convert_to_american(market_data.get("home", market_data.get("homePrice")))
            record["away_odds"] = self._convert_to_american(market_data.get("away", market_data.get("awayPrice")))
        elif market_name == "total":
            points = market_data.get("points", market_data.get("hdp", market_data.get("line")))
            record["total"] = float(points) if points is not None else None
            record["over_odds"] = self._convert_to_american(market_data.get("over", market_data.get("overPrice")))
            record["under_odds"] = self._convert_to_american(market_data.get("under", market_data.get("underPrice")))
        
        return record
    
    def _convert_to_american(self, odds: Any) -> Optional[int]:
        """
        Convert decimal odds to American odds.
        
        Pinnacle typically returns decimal odds.
        American odds: -110, +150, etc.
        Decimal odds: 1.91, 2.50, etc.
        """
        if odds is None:
            return None
        
        try:
            odds = float(odds)
            
            # If already looks like American odds
            if odds > 50 or odds < -50:
                return int(odds)
            
            # Convert decimal to American
            if odds >= 2.0:
                # Positive American odds
                return int((odds - 1) * 100)
            else:
                # Negative American odds
                return int(-100 / (odds - 1))
                
        except (ValueError, TypeError, ZeroDivisionError):
            return None
    
    async def validate(self, data: Any) -> bool:
        """Validate Pinnacle odds data."""
        if not data:
            return False
        if not isinstance(data, list):
            return False
        return len(data) > 0
    
    async def save_to_database(
        self,
        records: List[Dict[str, Any]],
        session: AsyncSession,
    ) -> int:
        """
        Save Pinnacle odds to database.
        
        Also captures closing lines and detects line movements.
        """
        if not records:
            return 0
        
        saved_count = 0
        
        # Get or create Pinnacle sportsbook
        pinnacle_book = await self._get_or_create_pinnacle_sportsbook(session)
        
        # Group records by game
        games_map: Dict[str, List[Dict[str, Any]]] = {}
        for record in records:
            key = f"{record['sport_code']}_{record['external_id']}"
            if key not in games_map:
                games_map[key] = []
            games_map[key].append(record)
        
        for game_key, game_records in games_map.items():
            try:
                # Get the first record for game info
                first_record = game_records[0]
                
                # Find or create the game
                game = await self._find_or_create_game(session, first_record)
                if not game:
                    continue
                
                # Save each bet type's odds
                for record in game_records:
                    bet_type = record["bet_type"]
                    
                    # Check for existing Pinnacle odds for this game/bet_type
                    existing = await session.execute(
                        select(Odds).where(
                            and_(
                                Odds.game_id == game.id,
                                Odds.sportsbook_id == pinnacle_book.id,
                                Odds.bet_type == bet_type,
                            )
                        )
                    )
                    existing_odds = existing.scalar_one_or_none()
                    
                    if existing_odds:
                        # Check for line movement
                        await self._check_line_movement(
                            session, game.id, bet_type, existing_odds, record
                        )
                        
                        # Update existing odds
                        existing_odds.home_odds = record.get("home_odds")
                        existing_odds.away_odds = record.get("away_odds")
                        existing_odds.home_line = record.get("home_line")
                        existing_odds.away_line = record.get("away_line")
                        existing_odds.total = record.get("total")
                        existing_odds.over_odds = record.get("over_odds")
                        existing_odds.under_odds = record.get("under_odds")
                        existing_odds.recorded_at = datetime.utcnow()
                    else:
                        # Create new odds record
                        new_odds = Odds(
                            game_id=game.id,
                            sportsbook_id=pinnacle_book.id,
                            sportsbook_key="pinnacle",
                            bet_type=bet_type,
                            home_odds=record.get("home_odds"),
                            away_odds=record.get("away_odds"),
                            home_line=record.get("home_line"),
                            away_line=record.get("away_line"),
                            total=record.get("total"),
                            over_odds=record.get("over_odds"),
                            under_odds=record.get("under_odds"),
                            is_opening=False,
                            recorded_at=datetime.utcnow(),
                        )
                        session.add(new_odds)
                        saved_count += 1
                
                await session.flush()
                
            except Exception as e:
                logger.error(f"[Pinnacle] Error saving game {game_key}: {e}")
                continue
        
        await session.commit()
        logger.info(f"[Pinnacle] Saved {saved_count} new odds records")
        return saved_count
    
    async def _get_or_create_pinnacle_sportsbook(
        self,
        session: AsyncSession,
    ) -> Sportsbook:
        """Get or create Pinnacle sportsbook record."""
        result = await session.execute(
            select(Sportsbook).where(Sportsbook.key == "pinnacle")
        )
        sportsbook = result.scalar_one_or_none()
        
        if not sportsbook:
            sportsbook = Sportsbook(
                name="Pinnacle",
                key="pinnacle",
                is_sharp=True,  # Pinnacle is THE sharp book
                is_active=True,
                priority=1,  # Highest priority for CLV
            )
            session.add(sportsbook)
            await session.flush()
            logger.info("[Pinnacle] Created Pinnacle sportsbook record")
        
        return sportsbook
    
    async def _find_or_create_game(
        self,
        session: AsyncSession,
        record: Dict[str, Any],
    ) -> Optional[Game]:
        """Find existing game or create new one."""
        external_id = record["external_id"]
        sport_code = record["sport_code"]
        
        # Try to find by external_id first
        result = await session.execute(
            select(Game).where(Game.external_id == external_id)
        )
        game = result.scalar_one_or_none()
        
        if game:
            return game
        
        # Try to find by teams and date
        home_team_name = record["home_team"]
        away_team_name = record["away_team"]
        
        # Get sport
        sport_result = await session.execute(
            select(Sport).where(Sport.code == sport_code)
        )
        sport = sport_result.scalar_one_or_none()
        
        if not sport:
            # Create sport
            sport_names = {
                "NBA": "National Basketball Association",
                "NFL": "National Football League",
                "NCAAF": "NCAA Football",
                "NCAAB": "NCAA Basketball",
                "NHL": "National Hockey League",
                "MLB": "Major League Baseball",
                "WNBA": "Women's National Basketball Association",
                "CFL": "Canadian Football League",
                "ATP": "ATP Tennis",
                "WTA": "WTA Tennis",
            }
            sport = Sport(
                code=sport_code,
                name=sport_names.get(sport_code, sport_code),
                is_active=True,
            )
            session.add(sport)
            await session.flush()
        
        # Get or create teams
        home_team = await self._get_or_create_team(session, sport.id, home_team_name)
        away_team = await self._get_or_create_team(session, sport.id, away_team_name)
        
        # Parse game date
        commence_time = record.get("commence_time")
        game_date = self._parse_datetime(commence_time)
        
        # Try to find game by teams and approximate date
        date_start = game_date - timedelta(hours=12)
        date_end = game_date + timedelta(hours=12)
        
        result = await session.execute(
            select(Game).where(
                and_(
                    Game.sport_id == sport.id,
                    Game.home_team_id == home_team.id,
                    Game.away_team_id == away_team.id,
                    Game.scheduled_at >= date_start,
                    Game.scheduled_at <= date_end,
                )
            )
        )
        game = result.scalar_one_or_none()
        
        if game:
            # Update external_id if not set
            if not game.external_id:
                game.external_id = external_id
            return game
        
        # Create new game
        game = Game(
            sport_id=sport.id,
            external_id=external_id,
            home_team_id=home_team.id,
            away_team_id=away_team.id,
            scheduled_at=game_date,
            status=GameStatus.SCHEDULED,
        )
        session.add(game)
        await session.flush()
        logger.info(f"[Pinnacle] Created game: {away_team_name} @ {home_team_name}")
        
        return game
    
    async def _get_or_create_team(
        self,
        session: AsyncSession,
        sport_id: UUID,
        team_name: str,
    ) -> Team:
        """Get or create team record."""
        result = await session.execute(
            select(Team).where(
                and_(
                    Team.sport_id == sport_id,
                    Team.name == team_name,
                )
            )
        )
        team = result.scalar_one_or_none()
        
        if not team:
            abbreviation = team_name[:3].upper() if len(team_name) >= 3 else team_name.upper()
            team = Team(
                sport_id=sport_id,
                external_id=f"pinnacle_{team_name.lower().replace(' ', '_')}",
                name=team_name,
                abbreviation=abbreviation,
                is_active=True,
            )
            session.add(team)
            await session.flush()
        
        return team
    
    def _parse_datetime(self, dt_str: Any) -> datetime:
        """Parse datetime string to naive UTC datetime."""
        if not dt_str:
            return datetime.utcnow()
        
        if isinstance(dt_str, datetime):
            if dt_str.tzinfo:
                utc_dt = dt_str.astimezone(timezone.utc)
                return datetime(
                    utc_dt.year, utc_dt.month, utc_dt.day,
                    utc_dt.hour, utc_dt.minute, utc_dt.second
                )
            return dt_str
        
        try:
            # Try ISO format
            if "T" in str(dt_str):
                parsed = datetime.fromisoformat(str(dt_str).replace("Z", "+00:00"))
                if parsed.tzinfo:
                    utc_dt = parsed.astimezone(timezone.utc)
                    return datetime(
                        utc_dt.year, utc_dt.month, utc_dt.day,
                        utc_dt.hour, utc_dt.minute, utc_dt.second
                    )
                return parsed
        except:
            pass
        
        return datetime.utcnow()
    
    async def _check_line_movement(
        self,
        session: AsyncSession,
        game_id: UUID,
        bet_type: str,
        existing_odds: Odds,
        new_record: Dict[str, Any],
    ) -> None:
        """Detect and record line movements."""
        old_line = None
        new_line = None
        
        if bet_type == "spread":
            old_line = existing_odds.home_line
            new_line = new_record.get("home_line")
        elif bet_type == "total":
            old_line = existing_odds.total
            new_line = new_record.get("total")
        
        if old_line is None or new_line is None:
            return
        
        movement = new_line - old_line
        if abs(movement) < 0.001:  # No significant movement
            return
        
        # Create movement record
        movement_record = OddsMovement(
            game_id=game_id,
            bet_type=bet_type,
            previous_line=old_line,
            current_line=new_line,
            movement=movement,
            is_steam=abs(movement) >= 1.0,  # Large sudden move
            is_reverse=False,  # Would need more context
            detected_at=datetime.utcnow(),
        )
        session.add(movement_record)
        logger.info(f"[Pinnacle] Line movement detected: {bet_type} {old_line} → {new_line} ({movement:+.1f})")
    
    async def capture_closing_lines(
        self,
        session: AsyncSession,
        sport_code: str = None,
    ) -> int:
        """
        Capture closing lines for games about to start.
        
        Run this before game starts to record Pinnacle closing lines for CLV calculation.
        """
        # Find games starting in the next 15 minutes
        now = datetime.utcnow()
        cutoff = now + timedelta(minutes=15)
        
        query = select(Game).where(
            and_(
                Game.scheduled_at >= now,
                Game.scheduled_at <= cutoff,
                Game.status == GameStatus.SCHEDULED,
            )
        )
        
        if sport_code:
            query = query.join(Sport).where(Sport.code == sport_code)
        
        result = await session.execute(query)
        games = result.scalars().all()
        
        captured = 0
        pinnacle_book = await self._get_or_create_pinnacle_sportsbook(session)
        
        for game in games:
            # Check if closing line already captured
            existing = await session.execute(
                select(ClosingLine).where(ClosingLine.game_id == game.id)
            )
            if existing.scalar_one_or_none():
                continue
            
            # Get current Pinnacle odds
            odds_result = await session.execute(
                select(Odds).where(
                    and_(
                        Odds.game_id == game.id,
                        Odds.sportsbook_id == pinnacle_book.id,
                    )
                )
            )
            odds_list = odds_result.scalars().all()
            
            if not odds_list:
                continue
            
            # Build closing line record
            closing_line = ClosingLine(
                game_id=game.id,
                source="pinnacle",
                recorded_at=datetime.utcnow(),
            )
            
            for odds in odds_list:
                if odds.bet_type == "spread":
                    closing_line.spread_home = odds.home_line
                    closing_line.spread_away = odds.away_line
                elif odds.bet_type == "moneyline":
                    closing_line.moneyline_home = odds.home_odds
                    closing_line.moneyline_away = odds.away_odds
                elif odds.bet_type == "total":
                    closing_line.total = odds.total
            
            session.add(closing_line)
            captured += 1
            logger.info(f"[Pinnacle] Captured closing line for game {game.id}")
        
        await session.commit()
        return captured
    
    async def get_sports_list(self) -> List[Dict[str, Any]]:
        """Get list of available sports from Pinnacle."""
        try:
            data = await self.get("/v1/sports")
            return data if data else []
        except Exception as e:
            logger.error(f"[Pinnacle] Error fetching sports: {e}")
            return []


# Create and register collector instance
pinnacle_collector = PinnacleCollector()
collector_manager.register(pinnacle_collector)
