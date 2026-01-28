"""
Collector 19: Matchstat Tennis API
RapidAPI Tennis API for ATP, WTA, and ITF data

API: https://rapidapi.com/jjrm365-kIFr3Nx_odV/api/tennis-api-atp-wta-itf
Rate Limit: 2 requests/second (0.5s delay)
Cost: $49/month

Endpoints used:
- /{tour}/ranking/singles/ - Rankings
- /{tour}/player/ - All players list
- /{tour}/player/profile/{id} - Player profile
- /{tour}/player/match-stats/{id} - Player match statistics
- /{tour}/player/surface-summary/{id} - Surface breakdown
- /{tour}/player/past-matches/{id} - Historical matches
- /{tour}/tournament/calendar/{year} - Tournament calendar
- /{tour}/tournament/results/{id} - Tournament match results
"""

import httpx
import asyncio
import logging
from datetime import datetime, date
from typing import Optional, Dict, List, Any
from .base_collector import BaseCollector

logger = logging.getLogger(__name__)


class MatchstatCollector(BaseCollector):
    """Collector for Matchstat Tennis API (ATP/WTA/ITF)"""
    
    name = "matchstat"
    
    # RapidAPI configuration
    BASE_URL = "https://tennis-api-atp-wta-itf.p.rapidapi.com/tennis/v2"
    API_KEY = "4d3f34e709msh3ef002f589a50a2p10d455jsnfd280f6cde41"
    RATE_LIMIT_DELAY = 0.5  # 2 requests per second
    
    TOURS = ["atp", "wta"]
    
    def __init__(self, db_session=None):
        super().__init__(
            name="matchstat",
            base_url=self.BASE_URL,
            rate_limit=2,  # 2 requests per second
            rate_window=1,
            timeout=30.0,
            max_retries=3
        )
        self.db_session = db_session
        self.headers = {
            "x-rapidapi-host": "tennis-api-atp-wta-itf.p.rapidapi.com",
            "x-rapidapi-key": self.API_KEY
        }
        logger.info("Registered collector: Matchstat Tennis API")
    
    def validate(self) -> bool:
        """Validate collector configuration"""
        return True
    
    async def _make_request(self, endpoint: str) -> Optional[Dict]:
        """Make API request with rate limiting"""
        url = f"{self.BASE_URL}{endpoint}"
        
        await asyncio.sleep(self.RATE_LIMIT_DELAY)
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, headers=self.headers)
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 404:
                    logger.warning(f"[Matchstat] API error 404: {endpoint}")
                    return None
                else:
                    logger.warning(f"[Matchstat] API error {response.status_code}: {endpoint}")
                    return None
                    
        except Exception as e:
            logger.error(f"[Matchstat] Request error for {endpoint}: {e}")
            return None
    
    # =====================================================
    # API FETCH METHODS
    # =====================================================
    
    async def fetch_rankings(self, tour: str) -> List[Dict]:
        """Fetch current rankings for a tour"""
        endpoint = f"/{tour}/ranking/singles/"
        data = await self._make_request(endpoint)
        
        if data and isinstance(data, dict) and "data" in data:
            return data["data"]
        elif data and isinstance(data, list):
            return data
        return []
    
    async def fetch_all_players(self, tour: str) -> List[Dict]:
        """Fetch all players for a tour"""
        endpoint = f"/{tour}/player/"
        data = await self._make_request(endpoint)
        
        if data and isinstance(data, dict) and "data" in data:
            return data["data"]
        elif data and isinstance(data, list):
            return data
        return []
    
    async def fetch_player_profile(self, tour: str, player_id: int) -> Optional[Dict]:
        """Fetch player profile"""
        endpoint = f"/{tour}/player/profile/{player_id}"
        data = await self._make_request(endpoint)
        
        if data and isinstance(data, dict):
            if "data" in data:
                return data["data"]
            return data
        return None
    
    async def fetch_player_match_stats(self, tour: str, player_id: int) -> Optional[Dict]:
        """Fetch player match statistics"""
        endpoint = f"/{tour}/player/match-stats/{player_id}"
        data = await self._make_request(endpoint)
        
        if data and isinstance(data, dict):
            if "data" in data:
                return data["data"]
            return data
        return None
    
    async def fetch_player_surface_summary(self, tour: str, player_id: int) -> Optional[Dict]:
        """Fetch player surface breakdown stats"""
        endpoint = f"/{tour}/player/surface-summary/{player_id}"
        data = await self._make_request(endpoint)
        
        if data and isinstance(data, dict):
            if "data" in data:
                return data["data"]
            return data
        return None
    
    async def fetch_player_past_matches(self, tour: str, player_id: int) -> List[Dict]:
        """Fetch player's past matches"""
        endpoint = f"/{tour}/player/past-matches/{player_id}"
        data = await self._make_request(endpoint)
        
        if data and isinstance(data, dict) and "data" in data:
            return data["data"]
        elif data and isinstance(data, list):
            return data
        return []
    
    async def fetch_tournament_calendar(self, tour: str, year: int) -> List[Dict]:
        """Fetch tournament calendar for a year"""
        endpoint = f"/{tour}/tournament/calendar/{year}"
        data = await self._make_request(endpoint)
        
        if data and isinstance(data, dict) and "data" in data:
            return data["data"]
        elif data and isinstance(data, list):
            return data
        return []
    
    async def fetch_tournament_results(self, tour: str, tournament_id: int) -> List[Dict]:
        """Fetch tournament match results"""
        endpoint = f"/{tour}/tournament/results/{tournament_id}"
        data = await self._make_request(endpoint)
        
        if data and isinstance(data, dict) and "data" in data:
            return data["data"]
        elif data and isinstance(data, list):
            return data
        return []
    
    async def fetch_fixtures_by_date(self, tour: str, date_str: str) -> List[Dict]:
        """Fetch fixtures/matches for a specific date (YYYY-MM-DD)"""
        endpoint = f"/{tour}/fixtures/{date_str}"
        data = await self._make_request(endpoint)
        
        if data and isinstance(data, dict) and "data" in data:
            return data["data"]
        elif data and isinstance(data, list):
            return data
        return []
    
    # =====================================================
    # MAIN COLLECTION METHOD
    # =====================================================
    
    async def collect(self, tours: List[str] = None, years_back: int = 1, collect_type: str = "all") -> 'CollectorResult':
        """
        Main collection method for Tennis data
        
        Args:
            tours: List of tours to collect ['atp', 'wta'] or None for both
            years_back: Number of years of historical data to collect
            collect_type: Type of data to collect ('all', 'rankings', 'players', 'matches', 'stats')
        
        Returns:
            CollectorResult with success status and data
        """
        from app.services.collectors.base_collector import CollectorResult
        
        if tours is None:
            tours = self.TOURS
        else:
            # Normalize tour names to lowercase
            tours = [t.lower() for t in tours]
        
        current_year = datetime.now().year
        start_year = current_year - years_back + 1
        
        logger.info(f"[Matchstat] Collecting Tennis data for {tours}, type={collect_type}, years {start_year} to {current_year}")
        
        total_records = 0
        all_data = {"rankings": [], "players": [], "matches": [], "tournaments": [], "stats": []}
        
        try:
            for tour in tours:
                tour_lower = tour.lower()
                logger.info(f"[Matchstat] Collecting {tour.upper()} data...")
                
                # 1. Collect rankings (always needed for player IDs)
                if collect_type in ["all", "rankings", "players", "stats"]:
                    rankings = await self.fetch_rankings(tour_lower)
                    logger.info(f"[Matchstat] {tour.upper()}: {len(rankings)} rankings")
                    
                    if rankings:
                        # Save rankings and get player IDs
                        player_ids = await self._save_rankings(rankings, tour.upper())
                        total_records += len(rankings)
                        all_data["rankings"].extend(rankings)
                        
                        # 2. Collect player profiles for ranked players (top 200)
                        if collect_type in ["all", "players", "stats"]:
                            top_players = player_ids[:200] if len(player_ids) > 200 else player_ids
                            profiles_collected = 0
                            stats_collected = 0
                            
                            for player_id in top_players:
                                # Get profile
                                if collect_type in ["all", "players"]:
                                    profile = await self.fetch_player_profile(tour_lower, player_id)
                                    if profile:
                                        await self._save_player_profile(profile, tour.upper())
                                        profiles_collected += 1
                                        all_data["players"].append(profile)
                                
                                # Get match stats
                                if collect_type in ["all", "stats"]:
                                    stats = await self.fetch_player_match_stats(tour_lower, player_id)
                                    if stats:
                                        await self._save_player_stats(stats, player_id, tour.upper())
                                        stats_collected += 1
                                        all_data["stats"].append(stats)
                                    
                                    # Get surface stats
                                    surface = await self.fetch_player_surface_summary(tour_lower, player_id)
                                    if surface:
                                        await self._save_surface_stats(surface, player_id, tour.upper())
                            
                            logger.info(f"[Matchstat] {tour.upper()}: {profiles_collected} profiles, {stats_collected} stats")
                            total_records += profiles_collected + stats_collected
                
                # 3. Collect tournaments and matches by year
                if collect_type in ["all", "matches"]:
                    for year in range(start_year, current_year + 1):
                        tournaments = await self.fetch_tournament_calendar(tour_lower, year)
                        logger.info(f"[Matchstat] {tour.upper()} {year}: {len(tournaments)} tournaments")
                        
                        if tournaments:
                            await self._save_tournaments(tournaments, tour.upper())
                            total_records += len(tournaments)
                            all_data["tournaments"].extend(tournaments)
                            
                            # Collect results for each tournament
                            matches_collected = 0
                            for tournament in tournaments[:50]:  # Limit to 50 tournaments per year
                                t_id = tournament.get("id")
                                if t_id:
                                    results = await self.fetch_tournament_results(tour_lower, t_id)
                                    if results:
                                        await self._save_matches_batched(results, tour.upper())
                                        matches_collected += len(results)
                                        all_data["matches"].extend(results)
                            
                            logger.info(f"[Matchstat] {tour.upper()} {year}: {matches_collected} matches")
                            total_records += matches_collected
            
            logger.info(f"[Matchstat] Total records collected: {total_records}")
            return CollectorResult(
                success=True,
                data=all_data,
                records_count=total_records,
                metadata={"tours": tours, "collect_type": collect_type, "years_back": years_back}
            )
        except Exception as e:
            logger.error(f"[Matchstat] Collection error: {e}")
            return CollectorResult(
                success=False,
                error=str(e),
                records_count=total_records,
                data=all_data if total_records > 0 else None
            )
    
    # =====================================================
    # DATABASE SAVE METHODS
    # =====================================================
    
    async def _save_rankings(self, rankings: List[Dict], tour: str) -> List[int]:
        """Save rankings and return player IDs"""
        if not self.db_session:
            return [r.get("id") or r.get("playerId") for r in rankings if r.get("id") or r.get("playerId")]
        
        from app.models import Player, Sport, Season
        
        # Get or create sport
        sport = self.db_session.query(Sport).filter_by(code=tour).first()
        if not sport:
            sport = Sport(code=tour, name=f"Tennis {tour}", active=True)
            self.db_session.add(sport)
            self.db_session.commit()
        
        # Get or create current season
        current_year = datetime.now().year
        season = self.db_session.query(Season).filter_by(
            sport_id=sport.id, year=current_year
        ).first()
        if not season:
            season = Season(
                sport_id=sport.id,
                year=current_year,
                name=f"{tour} {current_year}",
                start_date=date(current_year, 1, 1),
                end_date=date(current_year, 12, 31)
            )
            self.db_session.add(season)
            self.db_session.commit()
        
        player_ids = []
        
        for ranking in rankings:
            player_id = ranking.get("id") or ranking.get("playerId")
            if not player_id:
                continue
            
            player_ids.append(player_id)
            
            # Get or create player
            external_id = f"matchstat_{tour}_{player_id}"
            player = self.db_session.query(Player).filter_by(external_id=external_id).first()
            
            player_data = ranking.get("player", {}) if isinstance(ranking.get("player"), dict) else {}
            name = player_data.get("name") or ranking.get("name") or ranking.get("playerName", f"Player {player_id}")
            country = player_data.get("countryAcr") or ranking.get("countryAcr") or ranking.get("country", "")
            
            rank = ranking.get("ranking") or ranking.get("rank") or ranking.get("position")
            points = ranking.get("points") or ranking.get("rankingPoints", 0)
            
            if player:
                player.name = name
                player.jersey_number = str(rank) if rank else player.jersey_number
                player.position = tour
                player.birth_country = country
            else:
                player = Player(
                    external_id=external_id,
                    sport_id=sport.id,
                    name=name,
                    jersey_number=str(rank) if rank else None,
                    position=tour,
                    birth_country=country,
                    active=True
                )
                self.db_session.add(player)
        
        self.db_session.commit()
        return player_ids
    
    async def _save_player_profile(self, profile: Dict, tour: str):
        """Save player profile details"""
        if not self.db_session:
            return
        
        from app.models import Player, Sport
        
        sport = self.db_session.query(Sport).filter_by(code=tour).first()
        if not sport:
            return
        
        player_id = profile.get("id") or profile.get("playerId")
        if not player_id:
            return
        
        external_id = f"matchstat_{tour}_{player_id}"
        player = self.db_session.query(Player).filter_by(external_id=external_id).first()
        
        if not player:
            player = Player(
                external_id=external_id,
                sport_id=sport.id,
                name=profile.get("name", f"Player {player_id}"),
                position=tour,
                active=True
            )
            self.db_session.add(player)
        
        # Update profile fields
        if profile.get("name"):
            player.name = profile["name"]
        if profile.get("countryAcr"):
            player.birth_country = profile["countryAcr"]
        if profile.get("birthDate"):
            try:
                player.birth_date = datetime.fromisoformat(profile["birthDate"].replace("Z", "")).date()
            except:
                pass
        if profile.get("height"):
            player.height = str(profile["height"])
        if profile.get("weight"):
            player.weight = str(profile["weight"])
        
        self.db_session.commit()
    
    async def _save_player_stats(self, stats: Dict, player_id: int, tour: str):
        """Save player match statistics"""
        if not self.db_session or not stats:
            return
        
        from app.models import PlayerStat, Player, Sport, Season
        
        sport = self.db_session.query(Sport).filter_by(code=tour).first()
        if not sport:
            return
        
        external_id = f"matchstat_{tour}_{player_id}"
        player = self.db_session.query(Player).filter_by(external_id=external_id).first()
        if not player:
            return
        
        current_year = datetime.now().year
        season = self.db_session.query(Season).filter_by(
            sport_id=sport.id, year=current_year
        ).first()
        if not season:
            return
        
        # Flatten stats into key-value pairs
        stat_entries = []
        
        def flatten_dict(d, prefix=""):
            for key, value in d.items():
                if isinstance(value, dict):
                    flatten_dict(value, f"{prefix}{key}_")
                elif isinstance(value, (int, float, str)) and value is not None:
                    stat_entries.append((f"matchstat_{prefix}{key}", str(value)))
        
        flatten_dict(stats)
        
        # Save stats
        for stat_type, value in stat_entries:
            existing = self.db_session.query(PlayerStat).filter_by(
                player_id=player.id,
                season_id=season.id,
                stat_type=stat_type
            ).first()
            
            if existing:
                existing.value = value
            else:
                stat = PlayerStat(
                    player_id=player.id,
                    season_id=season.id,
                    stat_type=stat_type,
                    value=value
                )
                self.db_session.add(stat)
        
        self.db_session.commit()
    
    async def _save_surface_stats(self, surface: Dict, player_id: int, tour: str):
        """Save player surface statistics"""
        if not self.db_session or not surface:
            return
        
        from app.models import PlayerStat, Player, Sport, Season
        
        sport = self.db_session.query(Sport).filter_by(code=tour).first()
        if not sport:
            return
        
        external_id = f"matchstat_{tour}_{player_id}"
        player = self.db_session.query(Player).filter_by(external_id=external_id).first()
        if not player:
            return
        
        current_year = datetime.now().year
        season = self.db_session.query(Season).filter_by(
            sport_id=sport.id, year=current_year
        ).first()
        if not season:
            return
        
        # Handle both list and dict formats
        surfaces = surface if isinstance(surface, list) else [surface]
        
        for surf_data in surfaces:
            if not isinstance(surf_data, dict):
                continue
            
            surface_name = surf_data.get("court", {}).get("name", "") if isinstance(surf_data.get("court"), dict) else surf_data.get("surface", "unknown")
            surface_name = surface_name.lower().replace(" ", "_").replace(".", "")
            
            for key in ["wins", "losses", "winPct", "titles"]:
                if key in surf_data:
                    stat_type = f"matchstat_surface_{surface_name}_{key}"
                    value = str(surf_data[key])
                    
                    existing = self.db_session.query(PlayerStat).filter_by(
                        player_id=player.id,
                        season_id=season.id,
                        stat_type=stat_type
                    ).first()
                    
                    if existing:
                        existing.value = value
                    else:
                        stat = PlayerStat(
                            player_id=player.id,
                            season_id=season.id,
                            stat_type=stat_type,
                            value=value
                        )
                        self.db_session.add(stat)
        
        self.db_session.commit()
    
    async def _save_tournaments(self, tournaments: List[Dict], tour: str):
        """Save tournaments as teams"""
        if not self.db_session:
            return
        
        from app.models import Team, Sport
        
        sport = self.db_session.query(Sport).filter_by(code=tour).first()
        if not sport:
            return
        
        for tournament in tournaments:
            t_id = tournament.get("id")
            if not t_id:
                continue
            
            external_id = f"matchstat_tournament_{tour}_{t_id}"
            team = self.db_session.query(Team).filter_by(external_id=external_id).first()
            
            name = tournament.get("name", f"Tournament {t_id}")
            country = tournament.get("countryAcr") or tournament.get("country", {}).get("acronym", "")
            court = tournament.get("court", {}).get("name", "") if isinstance(tournament.get("court"), dict) else ""
            rank_type = tournament.get("rank", {}).get("name", "") if isinstance(tournament.get("rank"), dict) else ""
            
            if team:
                team.name = name
                team.country = country
                team.venue = f"{court} - {rank_type}".strip(" -")
            else:
                team = Team(
                    external_id=external_id,
                    sport_id=sport.id,
                    name=name,
                    abbreviation=tour,
                    country=country,
                    venue=f"{court} - {rank_type}".strip(" -"),
                    active=True
                )
                self.db_session.add(team)
        
        self.db_session.commit()
    
    async def _save_matches_batched(self, matches: List[Dict], tour: str, batch_size: int = 500):
        """Save matches in batches"""
        if not self.db_session or not matches:
            return
        
        from app.models import Game, Sport, Season, Team, Player
        
        sport = self.db_session.query(Sport).filter_by(code=tour).first()
        if not sport:
            return
        
        # Cache for seasons and players
        season_cache = {}
        player_cache = {}
        
        def get_season(year: int):
            if year not in season_cache:
                season = self.db_session.query(Season).filter_by(
                    sport_id=sport.id, year=year
                ).first()
                if not season:
                    season = Season(
                        sport_id=sport.id,
                        year=year,
                        name=f"{tour} {year}",
                        start_date=date(year, 1, 1),
                        end_date=date(year, 12, 31)
                    )
                    self.db_session.add(season)
                    self.db_session.commit()
                season_cache[year] = season
            return season_cache[year]
        
        def get_player_id(player_api_id: int):
            if player_api_id not in player_cache:
                external_id = f"matchstat_{tour}_{player_api_id}"
                player = self.db_session.query(Player).filter_by(external_id=external_id).first()
                player_cache[player_api_id] = player.id if player else None
            return player_cache[player_api_id]
        
        total = len(matches)
        saved = 0
        
        for i in range(0, total, batch_size):
            batch = matches[i:i + batch_size]
            
            for match in batch:
                match_id = match.get("id")
                if not match_id:
                    continue
                
                external_id = f"matchstat_match_{tour}_{match_id}"
                
                # Check if exists
                existing = self.db_session.query(Game).filter_by(external_id=external_id).first()
                if existing:
                    continue
                
                # Parse date
                match_date = None
                date_str = match.get("date")
                if date_str:
                    try:
                        match_date = datetime.fromisoformat(date_str.replace("Z", "")).date()
                    except:
                        pass
                
                if not match_date:
                    continue
                
                year = match_date.year
                season = get_season(year)
                
                # Get tournament info
                tournament = match.get("tournament", {})
                tournament_id = tournament.get("id") if isinstance(tournament, dict) else match.get("tournamentId")
                tournament_name = tournament.get("name", "") if isinstance(tournament, dict) else ""
                
                # Get players
                player1 = match.get("player1", {})
                player2 = match.get("player2", {})
                player1_id = player1.get("id") if isinstance(player1, dict) else match.get("player1Id")
                player2_id = player2.get("id") if isinstance(player2, dict) else match.get("player2Id")
                player1_name = player1.get("name", "") if isinstance(player1, dict) else ""
                player2_name = player2.get("name", "") if isinstance(player2, dict) else ""
                
                winner_id = match.get("match_winner") or match.get("winnerId")
                result = match.get("result", "")
                
                # Create game
                game = Game(
                    external_id=external_id,
                    sport_id=sport.id,
                    season_id=season.id,
                    game_date=match_date,
                    status="final",
                    venue=tournament_name,
                    notes=f"{player1_name} vs {player2_name}: {result}"
                )
                self.db_session.add(game)
                saved += 1
            
            self.db_session.commit()
            
            if i + batch_size < total:
                logger.info(f"[Matchstat] Saved {min(i + batch_size, total)}/{total} matches")
        
        logger.info(f"[Matchstat] Saved {saved} new matches")
    
    # =====================================================
    # CONVENIENCE METHODS FOR MASTER IMPORT
    # =====================================================
    
    async def collect_rankings_only(self, tours: List[str] = None) -> Dict[str, Any]:
        """Collect only rankings"""
        if tours is None:
            tours = self.TOURS
        
        total = 0
        for tour in tours:
            rankings = await self.fetch_rankings(tour.lower())
            if rankings:
                await self._save_rankings(rankings, tour.upper())
                total += len(rankings)
                logger.info(f"[Matchstat] {tour.upper()}: {len(rankings)} rankings")
        
        return {"records": total}
    
    async def collect_players_only(self, tours: List[str] = None, limit: int = 200) -> Dict[str, Any]:
        """Collect player profiles and stats for top players"""
        if tours is None:
            tours = self.TOURS
        
        total = 0
        for tour in tours:
            # Get player IDs from rankings
            rankings = await self.fetch_rankings(tour.lower())
            player_ids = [r.get("id") or r.get("playerId") for r in rankings if r.get("id") or r.get("playerId")]
            player_ids = player_ids[:limit]
            
            for player_id in player_ids:
                profile = await self.fetch_player_profile(tour.lower(), player_id)
                if profile:
                    await self._save_player_profile(profile, tour.upper())
                    total += 1
            
            logger.info(f"[Matchstat] {tour.upper()}: {total} profiles")
        
        return {"records": total}
    
    async def collect_player_stats_only(self, tours: List[str] = None, limit: int = 200) -> Dict[str, Any]:
        """Collect player statistics for top players"""
        if tours is None:
            tours = self.TOURS
        
        total = 0
        for tour in tours:
            rankings = await self.fetch_rankings(tour.lower())
            player_ids = [r.get("id") or r.get("playerId") for r in rankings if r.get("id") or r.get("playerId")]
            player_ids = player_ids[:limit]
            
            for player_id in player_ids:
                stats = await self.fetch_player_match_stats(tour.lower(), player_id)
                if stats:
                    await self._save_player_stats(stats, player_id, tour.upper())
                    total += 1
                
                surface = await self.fetch_player_surface_summary(tour.lower(), player_id)
                if surface:
                    await self._save_surface_stats(surface, player_id, tour.upper())
            
            logger.info(f"[Matchstat] {tour.upper()}: {total} player stats")
        
        return {"records": total}
    
    async def collect_matches_only(self, tours: List[str] = None, years_back: int = 1) -> Dict[str, Any]:
        """Collect match results from tournament calendar"""
        if tours is None:
            tours = self.TOURS
        
        current_year = datetime.now().year
        start_year = current_year - years_back + 1
        
        total = 0
        for tour in tours:
            for year in range(start_year, current_year + 1):
                tournaments = await self.fetch_tournament_calendar(tour.lower(), year)
                
                for tournament in tournaments[:50]:
                    t_id = tournament.get("id")
                    if t_id:
                        results = await self.fetch_tournament_results(tour.lower(), t_id)
                        if results:
                            await self._save_matches_batched(results, tour.upper())
                            total += len(results)
                
                logger.info(f"[Matchstat] {tour.upper()} {year}: collected matches")
        
        return {"records": total}
    
    async def collect_atp_only(self, years_back: int = 1) -> Dict[str, Any]:
        """Collect ATP data only"""
        return await self.collect(tours=["atp"], years_back=years_back)
    
    async def collect_wta_only(self, years_back: int = 1) -> Dict[str, Any]:
        """Collect WTA data only"""
        return await self.collect(tours=["wta"], years_back=years_back)


# Singleton instance for import
matchstat_collector = MatchstatCollector()