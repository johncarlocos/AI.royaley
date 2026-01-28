"""
ROYALEY - College Football Data API Collector
Phase 1: Data Collection Services

Collects comprehensive NCAAF data from College Football Data API.
- Teams and conferences
- Game schedules and results
- Advanced team stats (SP+, SRS, talent composite)
- Player stats and recruiting
- Betting lines
- Play-by-play data

Data Source:
- https://api.collegefootballdata.com/
- Documentation: https://collegefootballdata.com/

FREE tier available with API key registration.

Tables Filled:
- sports - Sport definitions
- teams - Team info with conference/division
- players - Player info
- seasons - Season definitions
- games - Game records with scores
- team_stats - Team statistics by season
- player_stats - Player statistics
- odds - Betting lines (if Odds model available)
"""

import asyncio
import logging
from datetime import datetime, date, timedelta
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4
import random

import httpx

from sqlalchemy import select, and_, or_, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models import Sport, Team, Game, GameStatus, Player, PlayerStats, TeamStats, Season
from app.services.collectors.base_collector import (
    BaseCollector,
    CollectorResult,
    collector_manager,
)

logger = logging.getLogger(__name__)


# =============================================================================
# API CONFIGURATION
# =============================================================================

CFB_API_BASE = "https://api.collegefootballdata.com"

# Default API key (can be overridden by environment variable)
DEFAULT_API_KEY = "1GwqWpwDDQmlA33u4rED9E0Z7bRtRinhQQ4wRoeszLnd5EM5hTLrnus3f0vpdFLy"

# Power 5 + Group of 5 conferences
POWER_CONFERENCES = ["ACC", "Big Ten", "Big 12", "Pac-12", "SEC"]
GROUP_OF_5 = ["American Athletic", "Conference USA", "Mid-American", "Mountain West", "Sun Belt"]
FBS_CONFERENCES = POWER_CONFERENCES + GROUP_OF_5 + ["FBS Independents"]

# Team stat types to collect
TEAM_STAT_TYPES = [
    # Offensive stats
    "totalYards", "netPassingYards", "rushingYards", "passingTDs", "rushingTDs",
    "turnovers", "fumblesLost", "interceptions", "firstDowns",
    "thirdDownEff", "fourthDownEff", "totalPenaltiesYards",
    "passAttempts", "passCompletions", "rushingAttempts",
    # Defensive stats  
    "sacks", "tacklesForLoss", "interceptionYards", "interceptionTDs",
    # Advanced stats
    "possessionTime", "kickReturns", "kickReturnYards", "puntReturns", "puntReturnYards"
]

# Player stat categories
PLAYER_STAT_CATEGORIES = [
    "passing", "rushing", "receiving", "defensive", "kicking", "punting"
]


# =============================================================================
# COLLEGE FOOTBALL DATA COLLECTOR CLASS
# =============================================================================

class CollegeFootballDataCollector(BaseCollector):
    """Collector for NCAAF data via College Football Data API."""
    
    name = "cfbd"
    
    def __init__(self):
        super().__init__(
            name="cfbd",
            base_url=CFB_API_BASE,
            rate_limit=0.5,  # 2 requests per second
        )
        self._api_key = getattr(settings, 'CFBD_API_KEY', None) or DEFAULT_API_KEY
    
    def _get_headers(self) -> Dict[str, str]:
        """Get API headers with authentication."""
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Accept": "application/json",
        }
    
    async def _api_get(self, endpoint: str, params: Dict[str, Any] = None) -> Optional[Any]:
        """Make authenticated GET request to API."""
        url = f"{CFB_API_BASE}{endpoint}"
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    url,
                    headers=self._get_headers(),
                    params=params
                )
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 401:
                    logger.error("[CFBD] Authentication failed - check API key")
                    return None
                elif response.status_code == 429:
                    logger.warning("[CFBD] Rate limited - waiting...")
                    await asyncio.sleep(5)
                    return await self._api_get(endpoint, params)
                else:
                    logger.warning(f"[CFBD] API error {response.status_code}: {endpoint}")
                    return None
                    
        except Exception as e:
            logger.error(f"[CFBD] Request error: {e}")
            return None

    async def validate(self, data: Any) -> bool:
        """Validate collected data."""
        if data is None:
            return False
        if hasattr(data, 'success'):
            return data.success
        if isinstance(data, dict):
            return bool(data)
        return False

    # =========================================================================
    # MAIN COLLECT METHOD
    # =========================================================================
    
    async def collect(
        self,
        years_back: int = 10,
        collect_type: str = "all",
        fbs_only: bool = True,
    ) -> CollectorResult:
        """
        Collect NCAAF data from College Football Data API.
        
        Args:
            years_back: Number of years to collect (default: 10)
            collect_type: Type of data to collect:
                - "all": All data types
                - "teams": Teams only
                - "games": Games and schedules only
                - "stats": Team and player stats only
                - "recruiting": Recruiting data only
                - "ratings": SP+, SRS, talent composite
                - "lines": Betting lines only
            fbs_only: Only collect FBS teams (default: True)
        
        Returns:
            CollectorResult with collected data
        """
        current_year = datetime.now().year
        current_month = datetime.now().month
        
        data = {
            "teams": [],
            "players": [],
            "games": [],
            "team_stats": [],
            "player_stats": [],
            "seasons": [],
            "recruiting": [],
            "ratings": [],
            "lines": [],
        }
        total_records = 0
        errors = []
        
        # NCAAF season runs Aug-Jan, use calendar year
        # In Jan 2026, the 2025 season just ended
        if current_month <= 2:
            latest_year = current_year - 1
        else:
            latest_year = current_year
        
        start_year = latest_year - years_back + 1
        end_year = latest_year
        
        logger.info(f"[CFBD] Collecting NCAAF data for years {start_year} to {end_year}")
        
        # Collect teams (only need current)
        if collect_type in ["all", "teams"]:
            try:
                teams = await self._collect_teams(fbs_only)
                data["teams"].extend(teams)
                total_records += len(teams)
                logger.info(f"[CFBD] Collected {len(teams)} teams")
            except Exception as e:
                logger.warning(f"[CFBD] Error collecting teams: {e}")
                errors.append(f"teams: {str(e)[:50]}")
        
        for year in range(start_year, end_year + 1):
            logger.info(f"[CFBD] NCAAF {year}...")
            
            try:
                # Rate limiting
                await asyncio.sleep(random.uniform(0.5, 1.0))
                
                # Collect games
                if collect_type in ["all", "games"]:
                    games = await self._collect_games(year, fbs_only)
                    data["games"].extend(games)
                    total_records += len(games)
                    logger.info(f"[CFBD] NCAAF {year}: {len(games)} games")
                
                # Collect team stats
                if collect_type in ["all", "stats"]:
                    team_stats = await self._collect_team_stats(year)
                    data["team_stats"].extend(team_stats)
                    total_records += len(team_stats)
                    logger.info(f"[CFBD] NCAAF {year}: {len(team_stats)} team stats")
                
                # Collect player stats
                if collect_type in ["all", "stats"]:
                    player_stats = await self._collect_player_stats(year)
                    data["player_stats"].extend(player_stats)
                    total_records += len(player_stats)
                    logger.info(f"[CFBD] NCAAF {year}: {len(player_stats)} player stats")
                
                # Collect ratings (SP+, SRS, talent)
                if collect_type in ["all", "ratings"]:
                    ratings = await self._collect_ratings(year)
                    data["ratings"].extend(ratings)
                    total_records += len(ratings)
                    logger.info(f"[CFBD] NCAAF {year}: {len(ratings)} ratings")
                
                # Collect recruiting
                if collect_type in ["all", "recruiting"]:
                    recruiting = await self._collect_recruiting(year)
                    data["recruiting"].extend(recruiting)
                    total_records += len(recruiting)
                    logger.info(f"[CFBD] NCAAF {year}: {len(recruiting)} recruiting records")
                
                # Collect betting lines
                if collect_type in ["all", "lines"]:
                    lines = await self._collect_lines(year)
                    data["lines"].extend(lines)
                    total_records += len(lines)
                    logger.info(f"[CFBD] NCAAF {year}: {len(lines)} betting lines")
                
                # Add season record
                season_data = {
                    "sport_code": "NCAAF",
                    "year": year,
                    "name": str(year),
                }
                data["seasons"].append(season_data)
                
            except Exception as e:
                logger.warning(f"[CFBD] Error collecting NCAAF {year}: {e}")
                errors.append(f"NCAAF {year}: {str(e)[:50]}")
                continue
        
        logger.info(f"[CFBD] Total records collected: {total_records}")
        
        return CollectorResult(
            success=total_records > 0,
            data=data,
            records_count=total_records,
            error="; ".join(errors[:5]) if errors else None
        )

    # =========================================================================
    # TEAMS COLLECTION
    # =========================================================================
    
    async def _collect_teams(self, fbs_only: bool = True) -> List[Dict[str, Any]]:
        """Collect team information."""
        teams = []
        
        response = await self._api_get("/teams")
        if not response:
            return teams
        
        for team_data in response:
            try:
                # Filter by classification if FBS only
                classification = team_data.get("classification", "")
                if fbs_only and classification != "fbs":
                    continue
                
                team = {
                    "sport_code": "NCAAF",
                    "external_id": f"NCAAF_{team_data.get('id', team_data.get('school', ''))}",
                    "name": team_data.get("school", ""),
                    "abbreviation": team_data.get("abbreviation", team_data.get("school", "")[:4].upper()),
                    "city": team_data.get("location", {}).get("city", ""),
                    "conference": team_data.get("conference", ""),
                    "division": team_data.get("division", classification),
                    "logo_url": team_data.get("logos", [None])[0] if team_data.get("logos") else None,
                    "color": team_data.get("color", ""),
                    "alt_color": team_data.get("alt_color", ""),
                    "mascot": team_data.get("mascot", ""),
                }
                teams.append(team)
                
            except Exception as e:
                logger.debug(f"[CFBD] Error parsing team: {e}")
                continue
        
        return teams

    # =========================================================================
    # GAMES COLLECTION
    # =========================================================================
    
    async def _collect_games(self, year: int, fbs_only: bool = True) -> List[Dict[str, Any]]:
        """Collect game schedules and results."""
        games = []
        
        params = {
            "year": year,
            "seasonType": "regular",
        }
        if fbs_only:
            params["division"] = "fbs"
        
        # Regular season games
        response = await self._api_get("/games", params)
        if response:
            games.extend(await self._parse_games(response, year))
        
        # Postseason games
        params["seasonType"] = "postseason"
        response = await self._api_get("/games", params)
        if response:
            games.extend(await self._parse_games(response, year))
        
        return games
    
    async def _parse_games(self, response: List[Dict], year: int) -> List[Dict[str, Any]]:
        """Parse games from API response."""
        games = []
        
        for game_data in response:
            try:
                game_id = game_data.get("id")
                home_team = game_data.get("home_team", "")
                away_team = game_data.get("away_team", "")
                
                # Parse date
                start_date = game_data.get("start_date", "")
                if start_date:
                    try:
                        scheduled_at = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
                    except:
                        scheduled_at = datetime.now()
                else:
                    scheduled_at = datetime.now()
                
                # Get scores
                home_score = game_data.get("home_points")
                away_score = game_data.get("away_points")
                
                # Determine status
                completed = game_data.get("completed", False)
                status = "final" if completed else "scheduled"
                
                game = {
                    "sport_code": "NCAAF",
                    "external_id": f"NCAAF_CFB_{game_id}",
                    "year": year,
                    "week": game_data.get("week"),
                    "season_type": game_data.get("season_type", "regular"),
                    "home_team_name": home_team,
                    "away_team_name": away_team,
                    "scheduled_at": scheduled_at,
                    "home_score": int(home_score) if home_score is not None else None,
                    "away_score": int(away_score) if away_score is not None else None,
                    "status": status,
                    "venue": game_data.get("venue", ""),
                    "neutral_site": game_data.get("neutral_site", False),
                    "conference_game": game_data.get("conference_game", False),
                    "home_line_scores": game_data.get("home_line_scores", []),
                    "away_line_scores": game_data.get("away_line_scores", []),
                }
                games.append(game)
                
            except Exception as e:
                logger.debug(f"[CFBD] Error parsing game: {e}")
                continue
        
        return games

    # =========================================================================
    # TEAM STATS COLLECTION
    # =========================================================================
    
    async def _collect_team_stats(self, year: int) -> List[Dict[str, Any]]:
        """Collect team season statistics."""
        stats = []
        
        # Season team stats
        params = {"year": year}
        response = await self._api_get("/stats/season", params)
        
        if response:
            for stat_data in response:
                try:
                    team = stat_data.get("team", "")
                    stat_name = stat_data.get("statName", "")
                    stat_value = stat_data.get("statValue")
                    
                    if stat_value is not None:
                        try:
                            value = float(stat_value)
                        except (ValueError, TypeError):
                            continue
                        
                        stat = {
                            "sport_code": "NCAAF",
                            "team_name": team,
                            "year": year,
                            "stat_type": f"cfbd_{stat_name}",
                            "value": value,
                        }
                        stats.append(stat)
                        
                except Exception as e:
                    logger.debug(f"[CFBD] Error parsing team stat: {e}")
                    continue
        
        # Also get advanced team stats
        await asyncio.sleep(0.3)
        response = await self._api_get("/stats/season/advanced", params)
        
        if response:
            for stat_data in response:
                try:
                    team = stat_data.get("team", "")
                    
                    # Parse offense stats
                    offense = stat_data.get("offense", {})
                    for key, value in offense.items():
                        if value is not None:
                            try:
                                stat = {
                                    "sport_code": "NCAAF",
                                    "team_name": team,
                                    "year": year,
                                    "stat_type": f"cfbd_off_{key}",
                                    "value": float(value),
                                }
                                stats.append(stat)
                            except (ValueError, TypeError):
                                continue
                    
                    # Parse defense stats
                    defense = stat_data.get("defense", {})
                    for key, value in defense.items():
                        if value is not None:
                            try:
                                stat = {
                                    "sport_code": "NCAAF",
                                    "team_name": team,
                                    "year": year,
                                    "stat_type": f"cfbd_def_{key}",
                                    "value": float(value),
                                }
                                stats.append(stat)
                            except (ValueError, TypeError):
                                continue
                        
                except Exception as e:
                    logger.debug(f"[CFBD] Error parsing advanced stat: {e}")
                    continue
        
        return stats

    # =========================================================================
    # PLAYER STATS COLLECTION
    # =========================================================================
    
    async def _collect_player_stats(self, year: int) -> List[Dict[str, Any]]:
        """Collect player season statistics."""
        stats = []
        
        for category in PLAYER_STAT_CATEGORIES:
            await asyncio.sleep(0.3)
            
            params = {"year": year, "category": category}
            response = await self._api_get("/stats/player/season", params)
            
            if response:
                for player_data in response:
                    try:
                        player_name = player_data.get("player", "")
                        team = player_data.get("team", "")
                        stat_type = player_data.get("statType", "")
                        stat_value = player_data.get("stat")
                        
                        if stat_value is not None and player_name:
                            try:
                                value = float(stat_value)
                            except (ValueError, TypeError):
                                continue
                            
                            stat = {
                                "sport_code": "NCAAF",
                                "player_name": player_name,
                                "team_name": team,
                                "year": year,
                                "category": category,
                                "stat_type": f"cfbd_{category}_{stat_type}",
                                "value": value,
                            }
                            stats.append(stat)
                            
                    except Exception as e:
                        logger.debug(f"[CFBD] Error parsing player stat: {e}")
                        continue
        
        return stats

    # =========================================================================
    # RATINGS COLLECTION (SP+, SRS, Talent)
    # =========================================================================
    
    async def _collect_ratings(self, year: int) -> List[Dict[str, Any]]:
        """Collect team ratings (SP+, SRS, talent composite)."""
        ratings = []
        
        # SP+ ratings
        await asyncio.sleep(0.3)
        response = await self._api_get("/ratings/sp", {"year": year})
        
        if response:
            for rating_data in response:
                try:
                    team = rating_data.get("team", "")
                    
                    # Overall rating
                    if rating_data.get("rating") is not None:
                        ratings.append({
                            "sport_code": "NCAAF",
                            "team_name": team,
                            "year": year,
                            "rating_type": "cfbd_sp_rating",
                            "value": float(rating_data["rating"]),
                        })
                    
                    # Offense rating
                    offense = rating_data.get("offense", {})
                    if offense.get("rating") is not None:
                        ratings.append({
                            "sport_code": "NCAAF",
                            "team_name": team,
                            "year": year,
                            "rating_type": "cfbd_sp_offense",
                            "value": float(offense["rating"]),
                        })
                    
                    # Defense rating
                    defense = rating_data.get("defense", {})
                    if defense.get("rating") is not None:
                        ratings.append({
                            "sport_code": "NCAAF",
                            "team_name": team,
                            "year": year,
                            "rating_type": "cfbd_sp_defense",
                            "value": float(defense["rating"]),
                        })
                    
                    # Special teams rating
                    if rating_data.get("specialTeams", {}).get("rating") is not None:
                        ratings.append({
                            "sport_code": "NCAAF",
                            "team_name": team,
                            "year": year,
                            "rating_type": "cfbd_sp_special_teams",
                            "value": float(rating_data["specialTeams"]["rating"]),
                        })
                        
                except Exception as e:
                    logger.debug(f"[CFBD] Error parsing SP+ rating: {e}")
                    continue
        
        # SRS ratings
        await asyncio.sleep(0.3)
        response = await self._api_get("/ratings/srs", {"year": year})
        
        if response:
            for rating_data in response:
                try:
                    team = rating_data.get("team", "")
                    
                    if rating_data.get("rating") is not None:
                        ratings.append({
                            "sport_code": "NCAAF",
                            "team_name": team,
                            "year": year,
                            "rating_type": "cfbd_srs_rating",
                            "value": float(rating_data["rating"]),
                        })
                        
                except Exception as e:
                    logger.debug(f"[CFBD] Error parsing SRS rating: {e}")
                    continue
        
        # Team talent composite
        await asyncio.sleep(0.3)
        response = await self._api_get("/talent", {"year": year})
        
        if response:
            for talent_data in response:
                try:
                    team = talent_data.get("school", "")
                    
                    if talent_data.get("talent") is not None:
                        ratings.append({
                            "sport_code": "NCAAF",
                            "team_name": team,
                            "year": year,
                            "rating_type": "cfbd_talent_composite",
                            "value": float(talent_data["talent"]),
                        })
                        
                except Exception as e:
                    logger.debug(f"[CFBD] Error parsing talent: {e}")
                    continue
        
        return ratings

    # =========================================================================
    # RECRUITING COLLECTION
    # =========================================================================
    
    async def _collect_recruiting(self, year: int) -> List[Dict[str, Any]]:
        """Collect recruiting data."""
        recruiting = []
        
        # Team recruiting rankings
        await asyncio.sleep(0.3)
        response = await self._api_get("/recruiting/teams", {"year": year})
        
        if response:
            for recruit_data in response:
                try:
                    team = recruit_data.get("team", "")
                    
                    recruiting.append({
                        "sport_code": "NCAAF",
                        "team_name": team,
                        "year": year,
                        "recruit_type": "team_ranking",
                        "rank": recruit_data.get("rank"),
                        "points": recruit_data.get("points"),
                    })
                        
                except Exception as e:
                    logger.debug(f"[CFBD] Error parsing recruiting: {e}")
                    continue
        
        return recruiting

    # =========================================================================
    # BETTING LINES COLLECTION
    # =========================================================================
    
    async def _collect_lines(self, year: int) -> List[Dict[str, Any]]:
        """Collect betting lines."""
        lines = []
        
        params = {"year": year}
        response = await self._api_get("/lines", params)
        
        if response:
            for game_data in response:
                try:
                    game_id = game_data.get("id")
                    home_team = game_data.get("homeTeam", "")
                    away_team = game_data.get("awayTeam", "")
                    
                    for line_data in game_data.get("lines", []):
                        try:
                            line = {
                                "sport_code": "NCAAF",
                                "game_external_id": f"NCAAF_CFB_{game_id}",
                                "year": year,
                                "home_team": home_team,
                                "away_team": away_team,
                                "provider": line_data.get("provider", ""),
                                "spread": line_data.get("spread"),
                                "spread_open": line_data.get("spreadOpen"),
                                "home_moneyline": line_data.get("homeMoneyline"),
                                "away_moneyline": line_data.get("awayMoneyline"),
                                "over_under": line_data.get("overUnder"),
                                "over_under_open": line_data.get("overUnderOpen"),
                            }
                            lines.append(line)
                        except Exception as e:
                            logger.debug(f"[CFBD] Error parsing line: {e}")
                            continue
                        
                except Exception as e:
                    logger.debug(f"[CFBD] Error parsing game lines: {e}")
                    continue
        
        return lines

    # =========================================================================
    # DATABASE SAVE METHODS
    # =========================================================================
    
    async def save_to_database(self, data: Dict[str, Any], session: AsyncSession) -> int:
        """Save all collected data to database."""
        total_saved = 0
        
        try:
            # Save seasons first
            if data.get("seasons"):
                saved = await self._save_seasons(session, data["seasons"])
                total_saved += saved
                logger.info(f"[CFBD] Saved {saved} seasons")
            
            # Save teams
            if data.get("teams"):
                saved = await self._save_teams(session, data["teams"])
                total_saved += saved
                logger.info(f"[CFBD] Saved {saved} teams")
            
            # Save games
            if data.get("games"):
                saved = await self._save_games(session, data["games"])
                total_saved += saved
                logger.info(f"[CFBD] Saved {saved} games")
            
            # Save team stats (including ratings)
            if data.get("team_stats"):
                saved = await self._save_team_stats(session, data["team_stats"])
                total_saved += saved
                logger.info(f"[CFBD] Saved {saved} team stats")
            
            # Save ratings as team stats
            if data.get("ratings"):
                saved = await self._save_ratings(session, data["ratings"])
                total_saved += saved
                logger.info(f"[CFBD] Saved {saved} ratings")
            
            # Save player stats
            if data.get("player_stats"):
                saved = await self._save_player_stats(session, data["player_stats"])
                total_saved += saved
                logger.info(f"[CFBD] Saved {saved} player stats")
            
            # Save recruiting as team stats
            if data.get("recruiting"):
                saved = await self._save_recruiting(session, data["recruiting"])
                total_saved += saved
                logger.info(f"[CFBD] Saved {saved} recruiting records")
            
            await session.commit()
            
        except Exception as e:
            logger.error(f"[CFBD] Error saving to database: {e}")
            await session.rollback()
            raise
        
        return total_saved
    
    async def _get_or_create_sport(self, session: AsyncSession) -> Sport:
        """Get or create NCAAF sport record."""
        result = await session.execute(
            select(Sport).where(Sport.code == "NCAAF")
        )
        sport = result.scalar_one_or_none()
        
        if not sport:
            sport = Sport(
                code="NCAAF",
                name="NCAA Football",
                is_active=True
            )
            session.add(sport)
            await session.flush()
        
        return sport
    
    async def _get_or_create_season(
        self, 
        session: AsyncSession, 
        sport_id: UUID, 
        year: int
    ) -> Season:
        """Get or create season record."""
        result = await session.execute(
            select(Season).where(
                and_(
                    Season.sport_id == sport_id,
                    Season.year == year
                )
            )
        )
        season = result.scalar_one_or_none()
        
        if not season:
            # NCAAF season runs Aug-Jan
            start_date = date(year, 8, 1)
            end_date = date(year + 1, 1, 31)
            name = str(year)
            
            season = Season(
                sport_id=sport_id,
                year=year,
                name=name,
                start_date=start_date,
                end_date=end_date,
                is_current=(year == datetime.now().year)
            )
            session.add(season)
            await session.flush()
        
        return season
    
    async def _save_seasons(self, session: AsyncSession, seasons: List[Dict]) -> int:
        """Save season records."""
        saved = 0
        
        for season_data in seasons:
            try:
                sport = await self._get_or_create_sport(session)
                await self._get_or_create_season(session, sport.id, season_data["year"])
                saved += 1
            except Exception as e:
                logger.debug(f"[CFBD] Error saving season: {e}")
        
        await session.flush()
        return saved
    
    async def _save_teams(self, session: AsyncSession, teams: List[Dict]) -> int:
        """Save team records."""
        saved = 0
        
        for team_data in teams:
            try:
                sport = await self._get_or_create_sport(session)
                external_id = team_data["external_id"]
                team_name = team_data["name"]
                
                # Check by external_id first
                result = await session.execute(
                    select(Team).where(
                        and_(
                            Team.sport_id == sport.id,
                            Team.external_id == external_id
                        )
                    )
                )
                existing = result.scalar_one_or_none()
                
                # If not found by external_id, check by name (teams may exist from other collectors)
                if not existing:
                    result = await session.execute(
                        select(Team).where(
                            and_(
                                Team.sport_id == sport.id,
                                Team.name == team_name
                            )
                        )
                    )
                    existing = result.scalar_one_or_none()
                
                if existing:
                    # Update existing team
                    existing.conference = team_data.get("conference") or existing.conference
                    existing.division = team_data.get("division") or existing.division
                    existing.logo_url = team_data.get("logo_url") or existing.logo_url
                    # Update external_id if it was found by name
                    if existing.external_id != external_id:
                        existing.external_id = external_id
                else:
                    team = Team(
                        sport_id=sport.id,
                        external_id=external_id,
                        name=team_name,
                        abbreviation=team_data["abbreviation"],
                        city=team_data.get("city"),
                        conference=team_data.get("conference"),
                        division=team_data.get("division"),
                        logo_url=team_data.get("logo_url"),
                        is_active=True
                    )
                    session.add(team)
                
                saved += 1
                
            except Exception as e:
                logger.debug(f"[CFBD] Error saving team: {e}")
        
        await session.flush()
        return saved
    
    async def _find_team_by_name(self, session: AsyncSession, sport_id: UUID, team_name: str) -> Optional[Team]:
        """Find team by name (case-insensitive)."""
        result = await session.execute(
            select(Team).where(
                and_(
                    Team.sport_id == sport_id,
                    Team.name.ilike(team_name)
                )
            )
        )
        return result.scalar_one_or_none()
    
    async def _save_games(self, session: AsyncSession, games: List[Dict]) -> int:
        """Save game records."""
        saved = 0
        
        for game_data in games:
            try:
                sport = await self._get_or_create_sport(session)
                season = await self._get_or_create_season(session, sport.id, game_data["year"])
                
                external_id = game_data["external_id"]
                
                # Find teams by name
                home_team = await self._find_team_by_name(session, sport.id, game_data["home_team_name"])
                away_team = await self._find_team_by_name(session, sport.id, game_data["away_team_name"])
                
                if not home_team or not away_team:
                    continue
                
                result = await session.execute(
                    select(Game).where(Game.external_id == external_id)
                )
                existing = result.scalar_one_or_none()
                
                if existing:
                    if game_data.get("home_score") is not None:
                        existing.home_score = game_data["home_score"]
                    if game_data.get("away_score") is not None:
                        existing.away_score = game_data["away_score"]
                    if game_data.get("status") == "final":
                        existing.status = GameStatus.FINAL
                else:
                    status = GameStatus.FINAL if game_data.get("status") == "final" else GameStatus.SCHEDULED
                    
                    game = Game(
                        sport_id=sport.id,
                        season_id=season.id,
                        external_id=external_id,
                        home_team_id=home_team.id,
                        away_team_id=away_team.id,
                        scheduled_at=game_data["scheduled_at"],
                        home_score=game_data.get("home_score"),
                        away_score=game_data.get("away_score"),
                        status=status
                    )
                    session.add(game)
                
                saved += 1
                
            except Exception as e:
                logger.debug(f"[CFBD] Error saving game: {e}")
        
        await session.flush()
        return saved
    
    async def _save_team_stats(self, session: AsyncSession, stats: List[Dict]) -> int:
        """Save team statistics."""
        saved = 0
        
        for stat_data in stats:
            try:
                sport = await self._get_or_create_sport(session)
                season = await self._get_or_create_season(session, sport.id, stat_data["year"])
                
                team = await self._find_team_by_name(session, sport.id, stat_data["team_name"])
                if not team:
                    continue
                
                result = await session.execute(
                    select(TeamStats).where(
                        and_(
                            TeamStats.team_id == team.id,
                            TeamStats.season_id == season.id,
                            TeamStats.stat_type == stat_data["stat_type"]
                        )
                    )
                )
                existing = result.scalar_one_or_none()
                
                if existing:
                    existing.value = stat_data["value"]
                else:
                    stat = TeamStats(
                        team_id=team.id,
                        season_id=season.id,
                        stat_type=stat_data["stat_type"],
                        value=stat_data["value"]
                    )
                    session.add(stat)
                
                saved += 1
                
            except Exception as e:
                logger.debug(f"[CFBD] Error saving team stat: {e}")
        
        await session.flush()
        return saved
    
    async def _save_ratings(self, session: AsyncSession, ratings: List[Dict]) -> int:
        """Save ratings as team stats."""
        saved = 0
        
        for rating_data in ratings:
            try:
                sport = await self._get_or_create_sport(session)
                season = await self._get_or_create_season(session, sport.id, rating_data["year"])
                
                team = await self._find_team_by_name(session, sport.id, rating_data["team_name"])
                if not team:
                    continue
                
                stat_type = rating_data["rating_type"]
                
                result = await session.execute(
                    select(TeamStats).where(
                        and_(
                            TeamStats.team_id == team.id,
                            TeamStats.season_id == season.id,
                            TeamStats.stat_type == stat_type
                        )
                    )
                )
                existing = result.scalar_one_or_none()
                
                if existing:
                    existing.value = rating_data["value"]
                else:
                    stat = TeamStats(
                        team_id=team.id,
                        season_id=season.id,
                        stat_type=stat_type,
                        value=rating_data["value"]
                    )
                    session.add(stat)
                
                saved += 1
                
            except Exception as e:
                logger.debug(f"[CFBD] Error saving rating: {e}")
        
        await session.flush()
        return saved
    
    async def _save_recruiting(self, session: AsyncSession, recruiting: List[Dict]) -> int:
        """Save recruiting data as team stats."""
        saved = 0
        
        for recruit_data in recruiting:
            try:
                sport = await self._get_or_create_sport(session)
                season = await self._get_or_create_season(session, sport.id, recruit_data["year"])
                
                team = await self._find_team_by_name(session, sport.id, recruit_data["team_name"])
                if not team:
                    continue
                
                # Save rank
                if recruit_data.get("rank") is not None:
                    result = await session.execute(
                        select(TeamStats).where(
                            and_(
                                TeamStats.team_id == team.id,
                                TeamStats.season_id == season.id,
                                TeamStats.stat_type == "cfbd_recruiting_rank"
                            )
                        )
                    )
                    existing = result.scalar_one_or_none()
                    
                    if existing:
                        existing.value = float(recruit_data["rank"])
                    else:
                        stat = TeamStats(
                            team_id=team.id,
                            season_id=season.id,
                            stat_type="cfbd_recruiting_rank",
                            value=float(recruit_data["rank"])
                        )
                        session.add(stat)
                    saved += 1
                
                # Save points
                if recruit_data.get("points") is not None:
                    result = await session.execute(
                        select(TeamStats).where(
                            and_(
                                TeamStats.team_id == team.id,
                                TeamStats.season_id == season.id,
                                TeamStats.stat_type == "cfbd_recruiting_points"
                            )
                        )
                    )
                    existing = result.scalar_one_or_none()
                    
                    if existing:
                        existing.value = float(recruit_data["points"])
                    else:
                        stat = TeamStats(
                            team_id=team.id,
                            season_id=season.id,
                            stat_type="cfbd_recruiting_points",
                            value=float(recruit_data["points"])
                        )
                        session.add(stat)
                    saved += 1
                
            except Exception as e:
                logger.debug(f"[CFBD] Error saving recruiting: {e}")
        
        await session.flush()
        return saved
    
    async def _save_player_stats(self, session: AsyncSession, stats: List[Dict]) -> int:
        """Save player statistics."""
        saved = 0
        seen_players = {}
        
        for stat_data in stats:
            try:
                sport = await self._get_or_create_sport(session)
                season = await self._get_or_create_season(session, sport.id, stat_data["year"])
                
                player_name = stat_data["player_name"]
                player_key = f"{player_name}_{stat_data['team_name']}"
                
                # Find or create player
                if player_key in seen_players:
                    player = seen_players[player_key]
                else:
                    external_id = f"cfbd_NCAAF_{player_name.replace(' ', '_').lower()}"
                    
                    result = await session.execute(
                        select(Player).where(Player.external_id == external_id)
                    )
                    player = result.scalar_one_or_none()
                    
                    if not player:
                        # Find team
                        team = await self._find_team_by_name(session, sport.id, stat_data["team_name"])
                        team_id = team.id if team else None
                        
                        player = Player(
                            external_id=external_id,
                            team_id=team_id,
                            name=player_name,
                            is_active=True
                        )
                        session.add(player)
                        await session.flush()
                    
                    seen_players[player_key] = player
                
                # Save stat
                result = await session.execute(
                    select(PlayerStats).where(
                        and_(
                            PlayerStats.player_id == player.id,
                            PlayerStats.season_id == season.id,
                            PlayerStats.stat_type == stat_data["stat_type"]
                        )
                    )
                )
                existing = result.scalar_one_or_none()
                
                if existing:
                    existing.value = stat_data["value"]
                else:
                    stat = PlayerStats(
                        player_id=player.id,
                        season_id=season.id,
                        stat_type=stat_data["stat_type"],
                        value=stat_data["value"]
                    )
                    session.add(stat)
                
                saved += 1
                
            except Exception as e:
                logger.debug(f"[CFBD] Error saving player stat: {e}")
        
        await session.flush()
        return saved


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

cfbd_collector = CollegeFootballDataCollector()

# Register with collector manager
collector_manager.register(cfbd_collector)
logger.info("Registered collector: College Football Data")