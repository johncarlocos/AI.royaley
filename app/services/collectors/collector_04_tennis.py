"""
Tennis Data Collector
Specialized collector for ATP and WTA tennis data
"""
import asyncio
from datetime import datetime, timedelta, date
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum

import httpx

from app.services.collectors.base_collector import BaseCollector, CollectorResult
from app.core.config import settings


class TennisTour(str, Enum):
    """Tennis tours"""
    ATP = "atp"
    WTA = "wta"


class TournamentLevel(str, Enum):
    """Tournament importance levels"""
    GRAND_SLAM = "grand_slam"
    MASTERS_1000 = "masters_1000"
    ATP_500 = "atp_500"
    ATP_250 = "atp_250"
    WTA_1000 = "wta_1000"
    WTA_500 = "wta_500"
    WTA_250 = "wta_250"
    CHALLENGER = "challenger"


class Surface(str, Enum):
    """Court surfaces"""
    HARD = "hard"
    CLAY = "clay"
    GRASS = "grass"
    CARPET = "carpet"


@dataclass
class TennisPlayer:
    """Tennis player data"""
    external_id: str
    name: str
    tour: TennisTour
    ranking: Optional[int] = None
    ranking_points: Optional[int] = None
    country: Optional[str] = None
    hand: Optional[str] = None  # R, L, or U (unknown)
    height_cm: Optional[int] = None
    age: Optional[int] = None
    
    # Performance stats
    ytd_wins: int = 0
    ytd_losses: int = 0
    career_wins: int = 0
    career_losses: int = 0
    career_titles: int = 0
    
    # Surface-specific records
    hard_wins: int = 0
    hard_losses: int = 0
    clay_wins: int = 0
    clay_losses: int = 0
    grass_wins: int = 0
    grass_losses: int = 0
    
    # Recent form (last 10 matches)
    recent_wins: int = 0
    recent_losses: int = 0


@dataclass
class TennisMatch:
    """Tennis match data"""
    external_id: str
    tour: TennisTour
    tournament_name: str
    tournament_level: TournamentLevel
    surface: Surface
    round_name: str
    scheduled_time: datetime
    
    player1_id: str
    player1_name: str
    player2_id: str
    player2_name: str
    
    # Rankings at time of match
    player1_ranking: Optional[int] = None
    player2_ranking: Optional[int] = None
    
    # Head-to-head
    h2h_player1_wins: int = 0
    h2h_player2_wins: int = 0
    
    # Result (if completed)
    winner_id: Optional[str] = None
    score: Optional[str] = None
    
    # Match stats
    duration_minutes: Optional[int] = None
    player1_aces: Optional[int] = None
    player2_aces: Optional[int] = None
    player1_double_faults: Optional[int] = None
    player2_double_faults: Optional[int] = None


@dataclass
class TennisTournament:
    """Tournament data"""
    external_id: str
    name: str
    tour: TennisTour
    level: TournamentLevel
    surface: Surface
    location: str
    country: str
    start_date: date
    end_date: date
    prize_money: Optional[int] = None
    draw_size: Optional[int] = None


class TennisCollector(BaseCollector):
    """
    Collector for ATP and WTA tennis data.
    
    Data sources:
    - TheOddsAPI for match odds
    - Tennis-specific APIs for rankings and stats
    """
    
    def __init__(self):
        super().__init__()
        self.odds_api_key = settings.ODDS_API_KEY
        self.odds_api_base = settings.ODDS_API_BASE_URL
        
        # Sport codes for TheOddsAPI
        self.sport_codes = {
            TennisTour.ATP: "tennis_atp_us_open",  # Changes by tournament
            TennisTour.WTA: "tennis_wta_us_open",
        }
        
        # Grand Slam tournament codes
        self.grand_slam_codes = {
            "australian_open": {
                TennisTour.ATP: "tennis_atp_australian_open",
                TennisTour.WTA: "tennis_wta_australian_open",
            },
            "french_open": {
                TennisTour.ATP: "tennis_atp_french_open",
                TennisTour.WTA: "tennis_wta_french_open",
            },
            "wimbledon": {
                TennisTour.ATP: "tennis_atp_wimbledon",
                TennisTour.WTA: "tennis_wta_wimbledon",
            },
            "us_open": {
                TennisTour.ATP: "tennis_atp_us_open",
                TennisTour.WTA: "tennis_wta_us_open",
            },
        }
    
    async def collect_rankings(
        self,
        tour: TennisTour,
        top_n: int = 100,
    ) -> CollectorResult:
        """
        Collect current rankings.
        
        Args:
            tour: ATP or WTA
            top_n: Number of top players to fetch
            
        Returns:
            CollectorResult with player rankings
        """
        self.logger.info(f"Collecting {tour.value.upper()} rankings (top {top_n})")
        
        try:
            # This would integrate with a tennis rankings API
            # Using placeholder data structure for now
            players = []
            
            # In production, fetch from:
            # - Official ATP/WTA API
            # - Tennis-data.co.uk
            # - SofaScore API
            # - Flashscore API
            
            # Simulated ranking data structure
            for i in range(1, min(top_n + 1, 101)):
                player = TennisPlayer(
                    external_id=f"{tour.value}_{i}",
                    name=f"Player {i}",
                    tour=tour,
                    ranking=i,
                    ranking_points=10000 - (i * 50),
                )
                players.append(player.__dict__)
            
            return CollectorResult(
                success=True,
                data=players,
                records_count=len(players),
                message=f"Collected {len(players)} {tour.value.upper()} rankings",
            )
            
        except Exception as e:
            self.logger.error(f"Error collecting rankings: {e}")
            return CollectorResult(
                success=False,
                data=[],
                records_count=0,
                message=str(e),
            )
    
    async def collect_matches(
        self,
        tour: TennisTour,
        tournament: Optional[str] = None,
        days_ahead: int = 7,
    ) -> CollectorResult:
        """
        Collect upcoming matches.
        
        Args:
            tour: ATP or WTA
            tournament: Specific tournament code (optional)
            days_ahead: Days of matches to fetch
            
        Returns:
            CollectorResult with match data
        """
        self.logger.info(f"Collecting {tour.value.upper()} matches")
        
        try:
            matches = []
            
            # Get matches from TheOddsAPI
            sport_code = self._get_sport_code(tour, tournament)
            
            if self.odds_api_key and sport_code:
                odds_matches = await self._fetch_odds_api_matches(sport_code)
                matches.extend(odds_matches)
            
            return CollectorResult(
                success=True,
                data=matches,
                records_count=len(matches),
                message=f"Collected {len(matches)} {tour.value.upper()} matches",
            )
            
        except Exception as e:
            self.logger.error(f"Error collecting matches: {e}")
            return CollectorResult(
                success=False,
                data=[],
                records_count=0,
                message=str(e),
            )
    
    async def collect_odds(
        self,
        tour: TennisTour,
        tournament: Optional[str] = None,
    ) -> CollectorResult:
        """
        Collect current odds for matches.
        
        Args:
            tour: ATP or WTA
            tournament: Specific tournament code (optional)
            
        Returns:
            CollectorResult with odds data
        """
        self.logger.info(f"Collecting {tour.value.upper()} odds")
        
        if not self.odds_api_key:
            return CollectorResult(
                success=False,
                data=[],
                records_count=0,
                message="TheOddsAPI key not configured",
            )
        
        try:
            sport_code = self._get_sport_code(tour, tournament)
            
            if not sport_code:
                return CollectorResult(
                    success=False,
                    data=[],
                    records_count=0,
                    message=f"No sport code available for {tour.value}",
                )
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.odds_api_base}/sports/{sport_code}/odds",
                    params={
                        "apiKey": self.odds_api_key,
                        "regions": "us,eu",
                        "markets": "h2h",  # Moneyline for tennis
                        "oddsFormat": "american",
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    odds_data = self._parse_tennis_odds(data, tour)
                    
                    return CollectorResult(
                        success=True,
                        data=odds_data,
                        records_count=len(odds_data),
                        message=f"Collected odds for {len(odds_data)} matches",
                    )
                else:
                    return CollectorResult(
                        success=False,
                        data=[],
                        records_count=0,
                        message=f"API error: {response.status_code}",
                    )
                    
        except Exception as e:
            self.logger.error(f"Error collecting odds: {e}")
            return CollectorResult(
                success=False,
                data=[],
                records_count=0,
                message=str(e),
            )
    
    async def collect_head_to_head(
        self,
        player1_id: str,
        player2_id: str,
    ) -> CollectorResult:
        """
        Collect head-to-head record between two players.
        
        Args:
            player1_id: First player's external ID
            player2_id: Second player's external ID
            
        Returns:
            CollectorResult with H2H data
        """
        self.logger.info(f"Collecting H2H: {player1_id} vs {player2_id}")
        
        try:
            # Fetch H2H data from tennis API
            # This would integrate with a tennis stats API
            
            h2h_data = {
                "player1_id": player1_id,
                "player2_id": player2_id,
                "player1_wins": 0,
                "player2_wins": 0,
                "matches": [],
                "surface_breakdown": {
                    "hard": {"player1": 0, "player2": 0},
                    "clay": {"player1": 0, "player2": 0},
                    "grass": {"player1": 0, "player2": 0},
                },
            }
            
            return CollectorResult(
                success=True,
                data=h2h_data,
                records_count=1,
                message="Collected H2H record",
            )
            
        except Exception as e:
            self.logger.error(f"Error collecting H2H: {e}")
            return CollectorResult(
                success=False,
                data={},
                records_count=0,
                message=str(e),
            )
    
    async def collect_player_stats(
        self,
        player_id: str,
        year: Optional[int] = None,
    ) -> CollectorResult:
        """
        Collect detailed player statistics.
        
        Args:
            player_id: Player's external ID
            year: Year for stats (defaults to current year)
            
        Returns:
            CollectorResult with player stats
        """
        if year is None:
            year = datetime.now().year
        
        self.logger.info(f"Collecting stats for player {player_id} ({year})")
        
        try:
            # Fetch detailed player stats
            # This would integrate with a tennis stats API
            
            stats = {
                "player_id": player_id,
                "year": year,
                "matches_played": 0,
                "matches_won": 0,
                "titles": 0,
                
                # Serve stats
                "first_serve_percentage": 0.0,
                "first_serve_won_percentage": 0.0,
                "second_serve_won_percentage": 0.0,
                "aces_per_match": 0.0,
                "double_faults_per_match": 0.0,
                
                # Return stats
                "first_return_won_percentage": 0.0,
                "second_return_won_percentage": 0.0,
                "break_points_won_percentage": 0.0,
                
                # Surface performance
                "hard_court_win_pct": 0.0,
                "clay_court_win_pct": 0.0,
                "grass_court_win_pct": 0.0,
                
                # Set/tiebreak performance
                "tiebreaks_won_percentage": 0.0,
                "deciding_sets_won_percentage": 0.0,
            }
            
            return CollectorResult(
                success=True,
                data=stats,
                records_count=1,
                message=f"Collected stats for player {player_id}",
            )
            
        except Exception as e:
            self.logger.error(f"Error collecting player stats: {e}")
            return CollectorResult(
                success=False,
                data={},
                records_count=0,
                message=str(e),
            )
    
    async def collect_tournament_info(
        self,
        tournament_id: str,
    ) -> CollectorResult:
        """
        Collect tournament information.
        
        Args:
            tournament_id: Tournament's external ID
            
        Returns:
            CollectorResult with tournament data
        """
        self.logger.info(f"Collecting tournament info: {tournament_id}")
        
        try:
            # Fetch tournament data
            tournament = {
                "external_id": tournament_id,
                "name": tournament_id.replace("_", " ").title(),
                "tour": "ATP",
                "level": "grand_slam",
                "surface": "hard",
                "location": "Unknown",
                "country": "Unknown",
                "start_date": datetime.now().date().isoformat(),
                "end_date": (datetime.now() + timedelta(days=14)).date().isoformat(),
            }
            
            return CollectorResult(
                success=True,
                data=tournament,
                records_count=1,
                message=f"Collected tournament info for {tournament_id}",
            )
            
        except Exception as e:
            self.logger.error(f"Error collecting tournament info: {e}")
            return CollectorResult(
                success=False,
                data={},
                records_count=0,
                message=str(e),
            )
    
    async def collect_all(
        self,
        tour: TennisTour,
    ) -> Dict[str, CollectorResult]:
        """
        Collect all tennis data for a tour.
        
        Args:
            tour: ATP or WTA
            
        Returns:
            Dictionary of CollectorResults by data type
        """
        self.logger.info(f"Collecting all {tour.value.upper()} data")
        
        results = {}
        
        # Collect rankings
        results["rankings"] = await self.collect_rankings(tour)
        
        # Collect matches
        results["matches"] = await self.collect_matches(tour)
        
        # Collect odds
        results["odds"] = await self.collect_odds(tour)
        
        return results
    
    def _get_sport_code(
        self,
        tour: TennisTour,
        tournament: Optional[str] = None,
    ) -> Optional[str]:
        """Get TheOddsAPI sport code for tour/tournament"""
        if tournament and tournament in self.grand_slam_codes:
            return self.grand_slam_codes[tournament].get(tour)
        
        return self.sport_codes.get(tour)
    
    async def _fetch_odds_api_matches(
        self,
        sport_code: str,
    ) -> List[Dict[str, Any]]:
        """Fetch matches from TheOddsAPI"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.odds_api_base}/sports/{sport_code}/events",
                    params={
                        "apiKey": self.odds_api_key,
                    }
                )
                
                if response.status_code == 200:
                    events = response.json()
                    return self._parse_tennis_events(events)
                
                return []
                
        except Exception as e:
            self.logger.error(f"Error fetching from OddsAPI: {e}")
            return []
    
    def _parse_tennis_events(
        self,
        events: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Parse tennis events from TheOddsAPI"""
        matches = []
        
        for event in events:
            match = {
                "external_id": event.get("id"),
                "scheduled_time": event.get("commence_time"),
                "player1_name": event.get("home_team"),
                "player2_name": event.get("away_team"),
                "sport_key": event.get("sport_key"),
                "sport_title": event.get("sport_title"),
            }
            matches.append(match)
        
        return matches
    
    def _parse_tennis_odds(
        self,
        data: List[Dict[str, Any]],
        tour: TennisTour,
    ) -> List[Dict[str, Any]]:
        """Parse tennis odds from TheOddsAPI"""
        odds_data = []
        
        for event in data:
            event_odds = {
                "external_id": event.get("id"),
                "tour": tour.value,
                "player1_name": event.get("home_team"),
                "player2_name": event.get("away_team"),
                "scheduled_time": event.get("commence_time"),
                "bookmakers": [],
            }
            
            for bookmaker in event.get("bookmakers", []):
                book_odds = {
                    "sportsbook": bookmaker.get("key"),
                    "last_update": bookmaker.get("last_update"),
                    "markets": [],
                }
                
                for market in bookmaker.get("markets", []):
                    if market.get("key") == "h2h":
                        market_data = {
                            "market_type": "moneyline",
                            "outcomes": [],
                        }
                        
                        for outcome in market.get("outcomes", []):
                            market_data["outcomes"].append({
                                "name": outcome.get("name"),
                                "price": outcome.get("price"),
                            })
                        
                        book_odds["markets"].append(market_data)
                
                event_odds["bookmakers"].append(book_odds)
            
            odds_data.append(event_odds)
        
        return odds_data


# Feature calculations for tennis predictions

def calculate_tennis_features(
    player1: TennisPlayer,
    player2: TennisPlayer,
    surface: Surface,
    tournament_level: TournamentLevel,
    h2h: Dict[str, Any],
) -> Dict[str, float]:
    """
    Calculate prediction features for a tennis match.
    
    Args:
        player1: First player data
        player2: Second player data
        surface: Court surface
        tournament_level: Tournament importance
        h2h: Head-to-head record
        
    Returns:
        Dictionary of feature values
    """
    features = {}
    
    # Ranking features
    features["ranking_diff"] = (player2.ranking or 500) - (player1.ranking or 500)
    features["ranking_points_diff"] = (player1.ranking_points or 0) - (player2.ranking_points or 0)
    
    # Overall win percentage
    p1_total = player1.ytd_wins + player1.ytd_losses
    p2_total = player2.ytd_wins + player2.ytd_losses
    
    features["p1_win_pct"] = player1.ytd_wins / p1_total if p1_total > 0 else 0.5
    features["p2_win_pct"] = player2.ytd_wins / p2_total if p2_total > 0 else 0.5
    features["win_pct_diff"] = features["p1_win_pct"] - features["p2_win_pct"]
    
    # Surface-specific performance
    if surface == Surface.HARD:
        p1_surf = player1.hard_wins / (player1.hard_wins + player1.hard_losses) if (player1.hard_wins + player1.hard_losses) > 0 else 0.5
        p2_surf = player2.hard_wins / (player2.hard_wins + player2.hard_losses) if (player2.hard_wins + player2.hard_losses) > 0 else 0.5
    elif surface == Surface.CLAY:
        p1_surf = player1.clay_wins / (player1.clay_wins + player1.clay_losses) if (player1.clay_wins + player1.clay_losses) > 0 else 0.5
        p2_surf = player2.clay_wins / (player2.clay_wins + player2.clay_losses) if (player2.clay_wins + player2.clay_losses) > 0 else 0.5
    elif surface == Surface.GRASS:
        p1_surf = player1.grass_wins / (player1.grass_wins + player1.grass_losses) if (player1.grass_wins + player1.grass_losses) > 0 else 0.5
        p2_surf = player2.grass_wins / (player2.grass_wins + player2.grass_losses) if (player2.grass_wins + player2.grass_losses) > 0 else 0.5
    else:
        p1_surf = 0.5
        p2_surf = 0.5
    
    features["p1_surface_win_pct"] = p1_surf
    features["p2_surface_win_pct"] = p2_surf
    features["surface_win_pct_diff"] = p1_surf - p2_surf
    
    # Recent form
    p1_recent_total = player1.recent_wins + player1.recent_losses
    p2_recent_total = player2.recent_wins + player2.recent_losses
    
    features["p1_recent_form"] = player1.recent_wins / p1_recent_total if p1_recent_total > 0 else 0.5
    features["p2_recent_form"] = player2.recent_wins / p2_recent_total if p2_recent_total > 0 else 0.5
    features["recent_form_diff"] = features["p1_recent_form"] - features["p2_recent_form"]
    
    # Head-to-head
    h2h_total = h2h.get("player1_wins", 0) + h2h.get("player2_wins", 0)
    if h2h_total > 0:
        features["h2h_advantage"] = h2h.get("player1_wins", 0) / h2h_total - 0.5
    else:
        features["h2h_advantage"] = 0.0
    
    features["h2h_matches"] = h2h_total
    
    # Tournament level encoding
    level_weights = {
        TournamentLevel.GRAND_SLAM: 1.0,
        TournamentLevel.MASTERS_1000: 0.8,
        TournamentLevel.WTA_1000: 0.8,
        TournamentLevel.ATP_500: 0.6,
        TournamentLevel.WTA_500: 0.6,
        TournamentLevel.ATP_250: 0.4,
        TournamentLevel.WTA_250: 0.4,
        TournamentLevel.CHALLENGER: 0.2,
    }
    features["tournament_level"] = level_weights.get(tournament_level, 0.5)
    
    # Experience features
    features["p1_career_matches"] = player1.career_wins + player1.career_losses
    features["p2_career_matches"] = player2.career_wins + player2.career_losses
    features["experience_diff"] = features["p1_career_matches"] - features["p2_career_matches"]
    
    # Title count
    features["p1_titles"] = player1.career_titles
    features["p2_titles"] = player2.career_titles
    features["title_diff"] = player1.career_titles - player2.career_titles
    
    # Age factor (if available)
    if player1.age and player2.age:
        features["age_diff"] = player2.age - player1.age
        # Peak age range for tennis is roughly 24-30
        features["p1_peak_factor"] = 1.0 - abs(player1.age - 27) / 15
        features["p2_peak_factor"] = 1.0 - abs(player2.age - 27) / 15
    else:
        features["age_diff"] = 0.0
        features["p1_peak_factor"] = 0.5
        features["p2_peak_factor"] = 0.5
    
    return features
