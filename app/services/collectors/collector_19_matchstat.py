"""
ROYALEY - Matchstat Tennis API Collector
Phase 1: Data Collection Services

Collects comprehensive ATP/WTA tennis data from Matchstat.com via RapidAPI.
- Player rankings (ATP/WTA live rankings)
- Player profiles and career stats
- Head-to-head records
- Match results and schedules
- Surface performance stats
- Tournament data

Data Source:
- https://matchstat.com/
- RapidAPI: tennis-api-atp-wta-itf
- Endpoint: https://tennis-api-atp-wta-itf.p.rapidapi.com

$49/mo subscription via RapidAPI

Tables Filled:
- sports - Sport definitions (ATP, WTA)
- teams - Tournament/event entities
- players - Player info with rankings
- seasons - Season definitions
- games - Match records with scores
- team_stats - Tournament statistics
- player_stats - Detailed player stats (career, surface, serve/return)
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

RAPIDAPI_HOST = "tennis-api-atp-wta-itf.p.rapidapi.com"
RAPIDAPI_BASE = f"https://{RAPIDAPI_HOST}"

# Default API key (can be overridden by environment variable)
DEFAULT_API_KEY = "4d3f34e709msh3ef002f589a50a2p10d455jsnfd280f6cde41"

# Tennis tours
TENNIS_TOURS = {
    "ATP": {"name": "ATP Tour (Men's)", "code": "atp", "gender": "male"},
    "WTA": {"name": "WTA Tour (Women's)", "code": "wta", "gender": "female"},
}

# Surface types for tennis
SURFACES = ["Hard", "Clay", "Grass", "Indoor", "Carpet"]

# Grand Slam tournaments
GRAND_SLAMS = [
    "Australian Open",
    "Roland Garros", 
    "Wimbledon",
    "US Open"
]

# Player stat types to collect
PLAYER_STAT_TYPES = [
    # Career stats
    "career_titles", "career_wins", "career_losses", "career_win_pct",
    "career_prize_money", "current_ranking", "highest_ranking", "ranking_points",
    # Service stats
    "aces_per_match", "double_faults_per_match", "first_serve_pct",
    "first_serve_won_pct", "second_serve_won_pct", "break_points_saved_pct",
    "service_games_won_pct", "service_points_won_pct",
    # Return stats
    "first_return_won_pct", "second_return_won_pct", "break_points_converted_pct",
    "return_games_won_pct", "return_points_won_pct",
    # Surface stats
    "hard_wins", "hard_losses", "hard_win_pct",
    "clay_wins", "clay_losses", "clay_win_pct",
    "grass_wins", "grass_losses", "grass_win_pct",
    "indoor_wins", "indoor_losses", "indoor_win_pct",
    # Recent form
    "ytd_wins", "ytd_losses", "ytd_titles", "ytd_prize_money",
    # H2H stats
    "h2h_record",
]


# =============================================================================
# MATCHSTAT TENNIS COLLECTOR CLASS
# =============================================================================

class MatchstatCollector(BaseCollector):
    """Collector for Tennis data via Matchstat/RapidAPI."""
    
    name = "matchstat"
    
    def __init__(self):
        super().__init__(
            name="matchstat",
            base_url=RAPIDAPI_BASE,
            rate_limit=0.5,  # 2 requests per second max
        )
        self._api_key = getattr(settings, 'MATCHSTAT_API_KEY', None) or \
                        getattr(settings, 'RAPIDAPI_KEY', None) or \
                        DEFAULT_API_KEY
    
    def _get_headers(self) -> Dict[str, str]:
        """Get RapidAPI headers with authentication."""
        return {
            "X-RapidAPI-Key": self._api_key,
            "X-RapidAPI-Host": RAPIDAPI_HOST,
            "Accept": "application/json",
        }
    
    async def _api_get(self, endpoint: str, params: Dict[str, Any] = None) -> Optional[Any]:
        """Make authenticated GET request to RapidAPI."""
        url = f"{RAPIDAPI_BASE}{endpoint}"
        
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
                    logger.error("[Matchstat] Authentication failed - check RapidAPI key")
                    return None
                elif response.status_code == 403:
                    logger.error("[Matchstat] Access forbidden - check subscription")
                    return None
                elif response.status_code == 429:
                    logger.warning("[Matchstat] Rate limited - waiting...")
                    await asyncio.sleep(5)
                    return await self._api_get(endpoint, params)
                else:
                    logger.warning(f"[Matchstat] API error {response.status_code}: {endpoint}")
                    logger.debug(f"[Matchstat] Response: {response.text[:500]}")
                    return None
                    
        except Exception as e:
            logger.error(f"[Matchstat] Request error: {e}")
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
        tours: List[str] = None,
    ) -> CollectorResult:
        """
        Collect Tennis data from Matchstat API.
        
        Args:
            years_back: Number of years to collect (default: 10)
            collect_type: Type of data to collect:
                - "all": All data types
                - "rankings": Current rankings only
                - "players": Player profiles only
                - "matches": Match results only
                - "stats": Player stats only
                - "h2h": Head-to-head records only
            tours: List of tours to collect ["ATP", "WTA"] (default: both)
        
        Returns:
            CollectorResult with collected data
        """
        if tours is None:
            tours = ["ATP", "WTA"]
        
        current_year = datetime.now().year
        
        data = {
            "players": [],
            "rankings": [],
            "matches": [],
            "player_stats": [],
            "h2h": [],
            "tournaments": [],
            "seasons": [],
        }
        total_records = 0
        errors = []
        
        start_year = current_year - years_back + 1
        end_year = current_year
        
        logger.info(f"[Matchstat] Collecting Tennis data for {tours}, years {start_year} to {end_year}")
        
        for tour in tours:
            tour_code = TENNIS_TOURS[tour]["code"]
            logger.info(f"[Matchstat] Collecting {tour} data...")
            
            # Collect current rankings
            if collect_type in ["all", "rankings"]:
                try:
                    await asyncio.sleep(random.uniform(0.5, 1.0))
                    rankings = await self._collect_rankings(tour_code)
                    data["rankings"].extend(rankings)
                    total_records += len(rankings)
                    logger.info(f"[Matchstat] {tour}: {len(rankings)} rankings")
                except Exception as e:
                    logger.warning(f"[Matchstat] Error collecting {tour} rankings: {e}")
                    errors.append(f"{tour} rankings: {str(e)[:50]}")
            
            # Collect player profiles (from rankings)
            if collect_type in ["all", "players"]:
                try:
                    await asyncio.sleep(random.uniform(0.5, 1.0))
                    players = await self._collect_players(tour_code, data.get("rankings", []))
                    data["players"].extend(players)
                    total_records += len(players)
                    logger.info(f"[Matchstat] {tour}: {len(players)} players")
                except Exception as e:
                    logger.warning(f"[Matchstat] Error collecting {tour} players: {e}")
                    errors.append(f"{tour} players: {str(e)[:50]}")
            
            # Collect player stats
            if collect_type in ["all", "stats"]:
                try:
                    await asyncio.sleep(random.uniform(0.5, 1.0))
                    stats = await self._collect_player_stats(tour_code, data.get("players", []))
                    data["player_stats"].extend(stats)
                    total_records += len(stats)
                    logger.info(f"[Matchstat] {tour}: {len(stats)} player stats")
                except Exception as e:
                    logger.warning(f"[Matchstat] Error collecting {tour} stats: {e}")
                    errors.append(f"{tour} stats: {str(e)[:50]}")
            
            # Collect matches by year
            if collect_type in ["all", "matches"]:
                for year in range(start_year, end_year + 1):
                    try:
                        await asyncio.sleep(random.uniform(0.5, 1.0))
                        matches = await self._collect_matches(tour_code, year)
                        data["matches"].extend(matches)
                        total_records += len(matches)
                        logger.info(f"[Matchstat] {tour} {year}: {len(matches)} matches")
                        
                        # Add season record
                        season_data = {
                            "sport_code": tour,
                            "year": year,
                            "name": f"{tour} {year}",
                        }
                        data["seasons"].append(season_data)
                        
                    except Exception as e:
                        logger.warning(f"[Matchstat] Error collecting {tour} {year} matches: {e}")
                        errors.append(f"{tour} {year}: {str(e)[:50]}")
            
            # Collect tournaments
            if collect_type in ["all", "tournaments"]:
                try:
                    await asyncio.sleep(random.uniform(0.5, 1.0))
                    tournaments = await self._collect_tournaments(tour_code)
                    data["tournaments"].extend(tournaments)
                    total_records += len(tournaments)
                    logger.info(f"[Matchstat] {tour}: {len(tournaments)} tournaments")
                except Exception as e:
                    logger.warning(f"[Matchstat] Error collecting {tour} tournaments: {e}")
                    errors.append(f"{tour} tournaments: {str(e)[:50]}")
        
        logger.info(f"[Matchstat] Total records collected: {total_records}")
        
        return CollectorResult(
            success=total_records > 0,
            data=data,
            records_count=total_records,
            error="; ".join(errors[:5]) if errors else None
        )

    # =========================================================================
    # RANKINGS COLLECTION
    # =========================================================================
    
    async def _collect_rankings(self, tour_code: str) -> List[Dict[str, Any]]:
        """Collect current rankings for a tour."""
        rankings = []
        
        # Try different endpoint formats
        endpoints = [
            f"/tennis/v2/{tour_code}/live-ranking",
            f"/{tour_code}/live-ranking",
            f"/tennis/{tour_code}/rankings",
            f"/{tour_code}/rankings",
        ]
        
        response = None
        for endpoint in endpoints:
            response = await self._api_get(endpoint)
            if response:
                logger.info(f"[Matchstat] Rankings endpoint working: {endpoint}")
                break
        
        if not response:
            logger.warning(f"[Matchstat] No rankings response for {tour_code}")
            return rankings
        
        # Handle different response formats
        ranking_data = response if isinstance(response, list) else response.get("rankings", response.get("data", []))
        
        for rank_data in ranking_data:
            try:
                ranking = {
                    "tour": tour_code.upper(),
                    "rank": rank_data.get("rank", rank_data.get("position", rank_data.get("ranking"))),
                    "player_id": rank_data.get("player_id", rank_data.get("playerId", rank_data.get("id"))),
                    "player_name": rank_data.get("player_name", rank_data.get("playerName", rank_data.get("name", rank_data.get("full_name")))),
                    "country": rank_data.get("country", rank_data.get("nationality", rank_data.get("country_code"))),
                    "points": rank_data.get("points", rank_data.get("ranking_points", rank_data.get("rankingPoints"))),
                    "move": rank_data.get("move", rank_data.get("movement", 0)),
                    "tournaments_played": rank_data.get("tournaments_played", rank_data.get("tournamentsPlayed")),
                }
                rankings.append(ranking)
            except Exception as e:
                logger.debug(f"[Matchstat] Error parsing ranking: {e}")
                continue
        
        return rankings

    # =========================================================================
    # PLAYERS COLLECTION
    # =========================================================================
    
    async def _collect_players(self, tour_code: str, rankings: List[Dict]) -> List[Dict[str, Any]]:
        """Collect player profiles."""
        players = []
        seen_players = set()
        
        # First, get players from rankings
        for rank_data in rankings:
            if rank_data.get("tour", "").lower() != tour_code.lower():
                continue
                
            player_id = rank_data.get("player_id")
            player_name = rank_data.get("player_name", "")
            
            if not player_name or player_name in seen_players:
                continue
            seen_players.add(player_name)
            
            player = {
                "tour": tour_code.upper(),
                "external_id": f"matchstat_{tour_code}_{player_id}" if player_id else f"matchstat_{tour_code}_{player_name.replace(' ', '_').lower()}",
                "player_id": player_id,
                "name": player_name,
                "country": rank_data.get("country"),
                "current_ranking": rank_data.get("rank"),
                "ranking_points": rank_data.get("points"),
            }
            players.append(player)
        
        # Try to get additional player details
        for player in players[:100]:  # Limit to top 100 to avoid rate limits
            player_id = player.get("player_id")
            if not player_id:
                continue
            
            await asyncio.sleep(random.uniform(0.3, 0.5))
            
            # Try different player detail endpoints
            endpoints = [
                f"/tennis/v2/{tour_code}/player/{player_id}",
                f"/{tour_code}/player/{player_id}",
                f"/tennis/{tour_code}/players/{player_id}",
            ]
            
            for endpoint in endpoints:
                response = await self._api_get(endpoint)
                if response:
                    # Update player with additional details
                    player["birth_date"] = response.get("birth_date", response.get("birthDate", response.get("dob")))
                    player["height"] = response.get("height")
                    player["weight"] = response.get("weight")
                    player["turned_pro"] = response.get("turned_pro", response.get("turnedPro"))
                    player["plays"] = response.get("plays", response.get("hand"))  # Right/Left handed
                    player["backhand"] = response.get("backhand")  # One/Two handed
                    player["career_titles"] = response.get("career_titles", response.get("careerTitles", response.get("titles")))
                    player["career_wins"] = response.get("career_wins", response.get("careerWins", response.get("wins")))
                    player["career_losses"] = response.get("career_losses", response.get("careerLosses", response.get("losses")))
                    player["prize_money"] = response.get("prize_money", response.get("prizeMoney", response.get("career_prize_money")))
                    player["highest_ranking"] = response.get("highest_ranking", response.get("highestRanking", response.get("best_ranking")))
                    break
        
        return players

    # =========================================================================
    # PLAYER STATS COLLECTION
    # =========================================================================
    
    async def _collect_player_stats(self, tour_code: str, players: List[Dict]) -> List[Dict[str, Any]]:
        """Collect player statistics."""
        stats = []
        
        for player in players[:100]:  # Limit to top 100
            player_id = player.get("player_id")
            player_name = player.get("name", "")
            
            if not player_id:
                continue
            
            await asyncio.sleep(random.uniform(0.3, 0.5))
            
            # Try to get player stats
            endpoints = [
                f"/tennis/v2/{tour_code}/player/{player_id}/stats",
                f"/{tour_code}/player/{player_id}/stats",
                f"/tennis/{tour_code}/players/{player_id}/statistics",
            ]
            
            response = None
            for endpoint in endpoints:
                response = await self._api_get(endpoint)
                if response:
                    break
            
            if not response:
                # Use basic stats from player profile
                if player.get("career_wins") is not None:
                    wins = player.get("career_wins", 0)
                    losses = player.get("career_losses", 0)
                    total = wins + losses
                    win_pct = (wins / total * 100) if total > 0 else 0
                    
                    stats.append({
                        "tour": tour_code.upper(),
                        "player_name": player_name,
                        "player_id": player_id,
                        "stat_type": "matchstat_career_wins",
                        "value": wins,
                    })
                    stats.append({
                        "tour": tour_code.upper(),
                        "player_name": player_name,
                        "player_id": player_id,
                        "stat_type": "matchstat_career_losses",
                        "value": losses,
                    })
                    stats.append({
                        "tour": tour_code.upper(),
                        "player_name": player_name,
                        "player_id": player_id,
                        "stat_type": "matchstat_career_win_pct",
                        "value": win_pct,
                    })
                continue
            
            # Parse detailed stats
            stat_data = response if isinstance(response, dict) else {}
            
            # Service stats
            for stat_key in ["aces", "double_faults", "first_serve_pct", "first_serve_won", 
                           "second_serve_won", "break_points_saved", "service_games_won"]:
                value = stat_data.get(stat_key, stat_data.get(self._to_camel_case(stat_key)))
                if value is not None:
                    stats.append({
                        "tour": tour_code.upper(),
                        "player_name": player_name,
                        "player_id": player_id,
                        "stat_type": f"matchstat_{stat_key}",
                        "value": float(value) if isinstance(value, (int, float, str)) else 0,
                    })
            
            # Return stats
            for stat_key in ["first_return_won", "second_return_won", "break_points_converted", 
                           "return_games_won"]:
                value = stat_data.get(stat_key, stat_data.get(self._to_camel_case(stat_key)))
                if value is not None:
                    stats.append({
                        "tour": tour_code.upper(),
                        "player_name": player_name,
                        "player_id": player_id,
                        "stat_type": f"matchstat_{stat_key}",
                        "value": float(value) if isinstance(value, (int, float, str)) else 0,
                    })
            
            # Surface stats
            for surface in ["hard", "clay", "grass", "indoor"]:
                surface_data = stat_data.get(f"{surface}_stats", stat_data.get(surface, {}))
                if isinstance(surface_data, dict):
                    wins = surface_data.get("wins", 0)
                    losses = surface_data.get("losses", 0)
                    if wins or losses:
                        stats.append({
                            "tour": tour_code.upper(),
                            "player_name": player_name,
                            "player_id": player_id,
                            "stat_type": f"matchstat_{surface}_wins",
                            "value": wins,
                        })
                        stats.append({
                            "tour": tour_code.upper(),
                            "player_name": player_name,
                            "player_id": player_id,
                            "stat_type": f"matchstat_{surface}_losses",
                            "value": losses,
                        })
        
        return stats
    
    def _to_camel_case(self, snake_str: str) -> str:
        """Convert snake_case to camelCase."""
        components = snake_str.split('_')
        return components[0] + ''.join(x.title() for x in components[1:])

    # =========================================================================
    # MATCHES COLLECTION
    # =========================================================================
    
    async def _collect_matches(self, tour_code: str, year: int) -> List[Dict[str, Any]]:
        """Collect match results for a year."""
        matches = []
        
        # Try different match endpoints
        endpoints = [
            f"/tennis/v2/{tour_code}/results/{year}",
            f"/{tour_code}/results/{year}",
            f"/tennis/{tour_code}/matches/{year}",
            f"/{tour_code}/matches?year={year}",
        ]
        
        response = None
        for endpoint in endpoints:
            response = await self._api_get(endpoint)
            if response:
                logger.info(f"[Matchstat] Matches endpoint working: {endpoint}")
                break
        
        if not response:
            # Try getting matches by tournament
            tournaments = await self._collect_tournaments(tour_code)
            for tournament in tournaments[:20]:  # Limit tournaments
                tournament_id = tournament.get("tournament_id")
                if not tournament_id:
                    continue
                
                await asyncio.sleep(random.uniform(0.3, 0.5))
                
                tournament_matches = await self._api_get(
                    f"/tennis/v2/{tour_code}/tournament/{tournament_id}/matches",
                    {"year": year}
                )
                if tournament_matches:
                    match_list = tournament_matches if isinstance(tournament_matches, list) else tournament_matches.get("matches", [])
                    for match_data in match_list:
                        match = self._parse_match(match_data, tour_code, year, tournament.get("name"))
                        if match:
                            matches.append(match)
            
            return matches
        
        # Parse matches from response
        match_list = response if isinstance(response, list) else response.get("matches", response.get("results", response.get("data", [])))
        
        for match_data in match_list:
            match = self._parse_match(match_data, tour_code, year)
            if match:
                matches.append(match)
        
        return matches
    
    def _parse_match(self, match_data: Dict, tour_code: str, year: int, tournament_name: str = None) -> Optional[Dict[str, Any]]:
        """Parse a match from API response."""
        try:
            match_id = match_data.get("match_id", match_data.get("matchId", match_data.get("id")))
            
            # Get players
            player1 = match_data.get("player1", match_data.get("player1_name", match_data.get("home_player")))
            player2 = match_data.get("player2", match_data.get("player2_name", match_data.get("away_player")))
            
            if isinstance(player1, dict):
                player1 = player1.get("name", player1.get("full_name", ""))
            if isinstance(player2, dict):
                player2 = player2.get("name", player2.get("full_name", ""))
            
            if not player1 or not player2:
                return None
            
            # Get score
            score = match_data.get("score", match_data.get("result", ""))
            winner = match_data.get("winner", match_data.get("winner_name"))
            if isinstance(winner, dict):
                winner = winner.get("name", "")
            
            # Get match date
            match_date = match_data.get("date", match_data.get("match_date", match_data.get("startTime")))
            if isinstance(match_date, str):
                try:
                    scheduled_at = datetime.fromisoformat(match_date.replace("Z", "+00:00"))
                except:
                    scheduled_at = datetime(year, 1, 1)
            else:
                scheduled_at = datetime(year, 1, 1)
            
            # Get tournament
            tournament = tournament_name or match_data.get("tournament", match_data.get("tournament_name", match_data.get("event")))
            if isinstance(tournament, dict):
                tournament = tournament.get("name", "")
            
            # Determine status
            status = "final" if score and winner else "scheduled"
            
            match = {
                "tour": tour_code.upper(),
                "external_id": f"matchstat_{tour_code}_{match_id}" if match_id else f"matchstat_{tour_code}_{year}_{player1}_{player2}".replace(" ", "_"),
                "year": year,
                "player1_name": player1,
                "player2_name": player2,
                "winner_name": winner,
                "score": score,
                "tournament": tournament,
                "surface": match_data.get("surface"),
                "round": match_data.get("round", match_data.get("round_name")),
                "scheduled_at": scheduled_at,
                "status": status,
                "duration_minutes": match_data.get("duration", match_data.get("match_duration")),
                # Set scores (parse from score string if needed)
                "sets": self._parse_sets(score),
            }
            
            return match
            
        except Exception as e:
            logger.debug(f"[Matchstat] Error parsing match: {e}")
            return None
    
    def _parse_sets(self, score: str) -> List[Dict[str, int]]:
        """Parse set scores from score string like '6-4 7-5 6-3'."""
        sets = []
        if not score:
            return sets
        
        try:
            # Handle different score formats
            score = str(score).strip()
            set_scores = score.split()
            
            for set_score in set_scores:
                if '-' in set_score:
                    parts = set_score.replace('(', ' ').replace(')', '').split('-')
                    if len(parts) >= 2:
                        try:
                            p1_games = int(parts[0].split()[0])
                            p2_games = int(parts[1].split()[0])
                            sets.append({"p1": p1_games, "p2": p2_games})
                        except (ValueError, IndexError):
                            continue
        except:
            pass
        
        return sets

    # =========================================================================
    # TOURNAMENTS COLLECTION
    # =========================================================================
    
    async def _collect_tournaments(self, tour_code: str) -> List[Dict[str, Any]]:
        """Collect tournament information."""
        tournaments = []
        
        # Try different tournament endpoints
        endpoints = [
            f"/tennis/v2/{tour_code}/tournaments",
            f"/{tour_code}/tournaments",
            f"/tennis/{tour_code}/events",
            f"/{tour_code}/events",
        ]
        
        response = None
        for endpoint in endpoints:
            response = await self._api_get(endpoint)
            if response:
                break
        
        if not response:
            return tournaments
        
        tournament_list = response if isinstance(response, list) else response.get("tournaments", response.get("events", response.get("data", [])))
        
        for t_data in tournament_list:
            try:
                tournament = {
                    "tour": tour_code.upper(),
                    "tournament_id": t_data.get("tournament_id", t_data.get("tournamentId", t_data.get("id"))),
                    "name": t_data.get("name", t_data.get("tournament_name", t_data.get("title"))),
                    "location": t_data.get("location", t_data.get("city")),
                    "country": t_data.get("country"),
                    "surface": t_data.get("surface"),
                    "category": t_data.get("category", t_data.get("level")),  # Grand Slam, Masters, ATP 500, etc.
                    "prize_money": t_data.get("prize_money", t_data.get("prizeMoney")),
                    "start_date": t_data.get("start_date", t_data.get("startDate")),
                    "end_date": t_data.get("end_date", t_data.get("endDate")),
                }
                tournaments.append(tournament)
            except Exception as e:
                logger.debug(f"[Matchstat] Error parsing tournament: {e}")
                continue
        
        return tournaments

    # =========================================================================
    # HEAD-TO-HEAD COLLECTION
    # =========================================================================
    
    async def _collect_h2h(self, tour_code: str, player1_id: str, player2_id: str) -> Optional[Dict[str, Any]]:
        """Collect head-to-head record between two players."""
        
        # Try different H2H endpoints
        endpoints = [
            f"/tennis/v2/{tour_code}/h2h/{player1_id}/{player2_id}",
            f"/{tour_code}/h2h/{player1_id}/{player2_id}",
            f"/tennis/{tour_code}/head-to-head?player1={player1_id}&player2={player2_id}",
        ]
        
        response = None
        for endpoint in endpoints:
            response = await self._api_get(endpoint)
            if response:
                break
        
        if not response:
            return None
        
        try:
            h2h = {
                "tour": tour_code.upper(),
                "player1_id": player1_id,
                "player2_id": player2_id,
                "player1_wins": response.get("player1_wins", response.get("p1Wins", 0)),
                "player2_wins": response.get("player2_wins", response.get("p2Wins", 0)),
                "matches": response.get("matches", []),
            }
            return h2h
        except Exception as e:
            logger.debug(f"[Matchstat] Error parsing H2H: {e}")
            return None

    # =========================================================================
    # DATABASE SAVE METHODS
    # =========================================================================
    
    async def save_to_database(self, data: Dict[str, Any], session: AsyncSession) -> int:
        """Save all collected data to database with incremental commits."""
        total_saved = 0
        
        # Save seasons first
        if data.get("seasons"):
            try:
                saved = await self._save_seasons(session, data["seasons"])
                await session.commit()
                total_saved += saved
                logger.info(f"[Matchstat] Saved {saved} seasons ✓")
            except Exception as e:
                logger.error(f"[Matchstat] Error saving seasons: {e}")
                await session.rollback()
        
        # Save players
        if data.get("players") or data.get("rankings"):
            try:
                # Combine players from rankings if no separate players
                players = data.get("players", [])
                if not players and data.get("rankings"):
                    players = [{"tour": r["tour"], "name": r["player_name"], "external_id": f"matchstat_{r['tour']}_{r.get('player_id', r['player_name'])}",
                               "current_ranking": r["rank"], "ranking_points": r.get("points"), "country": r.get("country")} 
                              for r in data["rankings"]]
                
                saved = await self._save_players(session, players)
                await session.commit()
                total_saved += saved
                logger.info(f"[Matchstat] Saved {saved} players ✓")
            except Exception as e:
                logger.error(f"[Matchstat] Error saving players: {e}")
                await session.rollback()
        
        # Save matches
        if data.get("matches"):
            try:
                saved = await self._save_matches_batched(session, data["matches"])
                total_saved += saved
                logger.info(f"[Matchstat] Saved {saved} matches ✓")
            except Exception as e:
                logger.error(f"[Matchstat] Error saving matches: {e}")
                await session.rollback()
        
        # Save player stats
        if data.get("player_stats"):
            try:
                saved = await self._save_player_stats_batched(session, data["player_stats"])
                total_saved += saved
                logger.info(f"[Matchstat] Saved {saved} player stats ✓")
            except Exception as e:
                logger.error(f"[Matchstat] Error saving player stats: {e}")
                await session.rollback()
        
        # Save tournaments as teams
        if data.get("tournaments"):
            try:
                saved = await self._save_tournaments(session, data["tournaments"])
                await session.commit()
                total_saved += saved
                logger.info(f"[Matchstat] Saved {saved} tournaments ✓")
            except Exception as e:
                logger.error(f"[Matchstat] Error saving tournaments: {e}")
                await session.rollback()
        
        return total_saved
    
    async def _get_or_create_sport(self, session: AsyncSession, tour_code: str) -> Sport:
        """Get or create sport record for tennis tour."""
        result = await session.execute(
            select(Sport).where(Sport.code == tour_code)
        )
        sport = result.scalar_one_or_none()
        
        if not sport:
            tour_info = TENNIS_TOURS.get(tour_code, {"name": f"{tour_code} Tennis"})
            sport = Sport(
                code=tour_code,
                name=tour_info["name"],
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
            # Tennis season runs Jan-Nov
            start_date = date(year, 1, 1)
            end_date = date(year, 11, 30)
            
            season = Season(
                sport_id=sport_id,
                year=year,
                name=str(year),
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
        seen = set()
        
        for season_data in seasons:
            key = (season_data["sport_code"], season_data["year"])
            if key in seen:
                continue
            seen.add(key)
            
            try:
                sport = await self._get_or_create_sport(session, season_data["sport_code"])
                await self._get_or_create_season(session, sport.id, season_data["year"])
                saved += 1
            except Exception as e:
                logger.debug(f"[Matchstat] Error saving season: {e}")
        
        await session.flush()
        return saved
    
    async def _save_players(self, session: AsyncSession, players: List[Dict]) -> int:
        """Save player records."""
        saved = 0
        
        for player_data in players:
            try:
                tour = player_data.get("tour", "ATP")
                sport = await self._get_or_create_sport(session, tour)
                external_id = player_data.get("external_id", f"matchstat_{tour}_{player_data['name'].replace(' ', '_').lower()}")
                
                result = await session.execute(
                    select(Player).where(Player.external_id == external_id)
                )
                existing = result.scalar_one_or_none()
                
                if existing:
                    # Update existing player
                    if player_data.get("current_ranking"):
                        existing.jersey_number = str(player_data["current_ranking"])  # Store ranking in jersey_number
                    if player_data.get("country"):
                        existing.birth_place = player_data["country"]
                else:
                    player = Player(
                        external_id=external_id,
                        name=player_data["name"],
                        position=tour,  # Use position to store tour type
                        jersey_number=str(player_data.get("current_ranking", "")),
                        birth_place=player_data.get("country"),
                        height=player_data.get("height"),
                        weight=player_data.get("weight"),
                        is_active=True
                    )
                    session.add(player)
                
                saved += 1
                
            except Exception as e:
                logger.debug(f"[Matchstat] Error saving player: {e}")
        
        await session.flush()
        return saved
    
    async def _save_matches_batched(self, session: AsyncSession, matches: List[Dict], batch_size: int = 500) -> int:
        """Save match records in batches."""
        saved = 0
        total = len(matches)
        
        for i, match_data in enumerate(matches):
            try:
                tour = match_data.get("tour", "ATP")
                sport = await self._get_or_create_sport(session, tour)
                season = await self._get_or_create_season(session, sport.id, match_data["year"])
                
                external_id = match_data["external_id"]
                
                result = await session.execute(
                    select(Game).where(Game.external_id == external_id)
                )
                existing = result.scalar_one_or_none()
                
                if existing:
                    if match_data.get("status") == "final":
                        existing.status = GameStatus.FINAL
                else:
                    status = GameStatus.FINAL if match_data.get("status") == "final" else GameStatus.SCHEDULED
                    
                    # For tennis, we don't have team_ids, use None or create placeholder
                    game = Game(
                        sport_id=sport.id,
                        season_id=season.id,
                        external_id=external_id,
                        scheduled_at=match_data["scheduled_at"],
                        status=status
                    )
                    session.add(game)
                
                saved += 1
                
                # Commit in batches
                if (i + 1) % batch_size == 0:
                    await session.commit()
                    logger.info(f"[Matchstat] Matches progress: {i+1}/{total}")
                
            except Exception as e:
                logger.debug(f"[Matchstat] Error saving match: {e}")
        
        await session.commit()
        return saved
    
    async def _save_player_stats_batched(self, session: AsyncSession, stats: List[Dict], batch_size: int = 500) -> int:
        """Save player statistics in batches."""
        saved = 0
        total = len(stats)
        player_cache = {}
        
        for i, stat_data in enumerate(stats):
            try:
                tour = stat_data.get("tour", "ATP")
                sport = await self._get_or_create_sport(session, tour)
                
                # Get current year season
                current_year = datetime.now().year
                season = await self._get_or_create_season(session, sport.id, current_year)
                
                player_name = stat_data["player_name"]
                
                # Find player
                if player_name not in player_cache:
                    external_id = f"matchstat_{tour}_{player_name.replace(' ', '_').lower()}"
                    result = await session.execute(
                        select(Player).where(Player.external_id == external_id)
                    )
                    player_cache[player_name] = result.scalar_one_or_none()
                
                player = player_cache.get(player_name)
                if not player:
                    continue
                
                stat_type = stat_data["stat_type"]
                
                result = await session.execute(
                    select(PlayerStats).where(
                        and_(
                            PlayerStats.player_id == player.id,
                            PlayerStats.season_id == season.id,
                            PlayerStats.stat_type == stat_type
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
                        stat_type=stat_type,
                        value=stat_data["value"]
                    )
                    session.add(stat)
                
                saved += 1
                
                # Commit in batches
                if (i + 1) % batch_size == 0:
                    await session.commit()
                    logger.info(f"[Matchstat] Player stats progress: {i+1}/{total}")
                
            except Exception as e:
                logger.debug(f"[Matchstat] Error saving player stat: {e}")
        
        await session.commit()
        return saved
    
    async def _save_tournaments(self, session: AsyncSession, tournaments: List[Dict]) -> int:
        """Save tournament records as teams."""
        saved = 0
        
        for t_data in tournaments:
            try:
                tour = t_data.get("tour", "ATP")
                sport = await self._get_or_create_sport(session, tour)
                
                external_id = f"matchstat_tournament_{tour}_{t_data.get('tournament_id', t_data['name'].replace(' ', '_'))}"
                
                result = await session.execute(
                    select(Team).where(
                        and_(
                            Team.sport_id == sport.id,
                            Team.external_id == external_id
                        )
                    )
                )
                existing = result.scalar_one_or_none()
                
                if existing:
                    existing.conference = t_data.get("category")  # Store category (Grand Slam, Masters, etc.)
                else:
                    team = Team(
                        sport_id=sport.id,
                        external_id=external_id,
                        name=t_data["name"],
                        abbreviation=t_data["name"][:10].upper(),
                        city=t_data.get("location"),
                        conference=t_data.get("category"),
                        division=t_data.get("surface"),
                        is_active=True
                    )
                    session.add(team)
                
                saved += 1
                
            except Exception as e:
                logger.debug(f"[Matchstat] Error saving tournament: {e}")
        
        await session.flush()
        return saved


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

matchstat_collector = MatchstatCollector()

# Register with collector manager
collector_manager.register(matchstat_collector)
logger.info("Registered collector: Matchstat Tennis API")