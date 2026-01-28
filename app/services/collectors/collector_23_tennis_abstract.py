"""
Collector 23: Tennis Abstract (Jeff Sackmann GitHub Data)
Comprehensive ATP and WTA tennis data from Tennis Abstract GitHub repos

Data Source: https://github.com/JeffSackmann/tennis_atp and tennis_wta
License: Creative Commons Attribution-NonCommercial-ShareAlike 4.0

Data Available:
- Match results: 1968-present (ATP), 1968-present (WTA)
- Player information: IDs, names, DOB, country, height, hand
- Rankings: 1973-present (ATP), 1975-present (WTA)
- Match statistics: 1991-present (ATP tour level)

CSV Files Structure:
- atp_matches_YYYY.csv / wta_matches_YYYY.csv - Match results by year
- atp_players.csv / wta_players.csv - Player master file
- atp_rankings_XXs.csv / wta_rankings_XXs.csv - Rankings by decade
- atp_rankings_current.csv / wta_rankings_current.csv - Current rankings

Match Columns:
tourney_id, tourney_name, surface, draw_size, tourney_level, tourney_date,
match_num, winner_id, winner_seed, winner_entry, winner_name, winner_hand,
winner_ht, winner_ioc, winner_age, loser_id, loser_seed, loser_entry,
loser_name, loser_hand, loser_ht, loser_ioc, loser_age, score, best_of, round, 
minutes, w_ace, w_df, w_svpt, w_1stIn, w_1stWon, w_2ndWon, w_SvGms, w_bpSaved, 
w_bpFaced, l_ace, l_df, l_svpt, l_1stIn, l_1stWon, l_2ndWon, l_SvGms, l_bpSaved, l_bpFaced,
winner_rank, winner_rank_points, loser_rank, loser_rank_points

Tournament Levels:
G = Grand Slam, M = Masters 1000, A = ATP 500/250, D = Davis Cup, 
F = Tour Finals, O = Olympics, C = Challenger

FREE - Web Scraping/GitHub CSV Download
"""

import httpx
import asyncio
import logging
import csv
import io
from datetime import datetime, date, timedelta
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field
from .base_collector import BaseCollector, CollectorResult

logger = logging.getLogger(__name__)


@dataclass
class TennisPlayer:
    """Tennis player data structure"""
    player_id: str
    first_name: str
    last_name: str
    name: str
    hand: Optional[str] = None  # R, L, U (unknown)
    birth_date: Optional[date] = None
    country_code: Optional[str] = None
    height_cm: Optional[int] = None
    tour: str = "ATP"  # ATP or WTA


@dataclass
class TennisMatch:
    """Tennis match data structure"""
    # Required fields (no defaults) must come first
    match_id: str
    tourney_id: str
    tourney_name: str
    tourney_date: date
    tourney_level: str
    surface: str
    round_name: str
    best_of: int
    winner_id: str
    winner_name: str
    loser_id: str
    loser_name: str
    
    # Optional fields (with defaults) come after
    winner_seed: Optional[int] = None
    winner_rank: Optional[int] = None
    winner_rank_points: Optional[int] = None
    loser_seed: Optional[int] = None
    loser_rank: Optional[int] = None
    loser_rank_points: Optional[int] = None
    
    score: Optional[str] = None
    minutes: Optional[int] = None
    
    # Winner stats
    w_ace: Optional[int] = None
    w_df: Optional[int] = None
    w_svpt: Optional[int] = None
    w_1stIn: Optional[int] = None
    w_1stWon: Optional[int] = None
    w_2ndWon: Optional[int] = None
    w_SvGms: Optional[int] = None
    w_bpSaved: Optional[int] = None
    w_bpFaced: Optional[int] = None
    
    # Loser stats
    l_ace: Optional[int] = None
    l_df: Optional[int] = None
    l_svpt: Optional[int] = None
    l_1stIn: Optional[int] = None
    l_1stWon: Optional[int] = None
    l_2ndWon: Optional[int] = None
    l_SvGms: Optional[int] = None
    l_bpSaved: Optional[int] = None
    l_bpFaced: Optional[int] = None
    
    tour: str = "ATP"


@dataclass
class TennisRanking:
    """Tennis ranking data structure"""
    ranking_date: date
    ranking: int
    player_id: str
    ranking_points: Optional[int] = None
    tour: str = "ATP"


class TennisAbstractCollector(BaseCollector):
    """
    Collector for Tennis Abstract data from Jeff Sackmann's GitHub
    
    Features:
    - ATP and WTA match results (1968-present)
    - Player information with DOB, country, height, hand
    - Historical rankings (1973-present)
    - Match statistics (aces, double faults, serve points, etc.)
    - Surface-specific data (Hard, Clay, Grass, Carpet)
    - Tournament levels (Grand Slam, Masters, 500, 250, Challenger)
    """
    
    name = "tennis_abstract"
    
    # GitHub raw URLs
    ATP_BASE_URL = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master"
    WTA_BASE_URL = "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master"
    
    # Tournament level mapping
    TOURNEY_LEVELS = {
        'G': 'Grand Slam',
        'M': 'Masters 1000',
        'A': 'ATP 500/250',
        'F': 'Tour Finals',
        'D': 'Davis Cup',
        'O': 'Olympics',
        'C': 'Challenger'
    }
    
    # Surface mapping
    SURFACES = {
        'Hard': 'hard',
        'Clay': 'clay',
        'Grass': 'grass',
        'Carpet': 'carpet',
        'H': 'hard',
        'C': 'clay',
        'G': 'grass'
    }
    
    # Round name mapping
    ROUNDS = {
        'F': 'Final',
        'SF': 'Semifinal',
        'QF': 'Quarterfinal',
        'R16': 'Round of 16',
        'R32': 'Round of 32',
        'R64': 'Round of 64',
        'R128': 'Round of 128',
        'RR': 'Round Robin',
        'BR': 'Bronze Medal Match',
        'ER': 'Early Rounds',
        'Q1': 'Qualifying Round 1',
        'Q2': 'Qualifying Round 2',
        'Q3': 'Qualifying Round 3'
    }
    
    def __init__(self, db_session=None):
        super().__init__(
            name="tennis_abstract",
            base_url=self.ATP_BASE_URL,
            rate_limit=10,
            rate_window=1,
            timeout=60.0,
            max_retries=3
        )
        self.db_session = db_session
        logger.info("Registered collector: Tennis Abstract (Jeff Sackmann GitHub)")
    
    def validate(self) -> bool:
        """Validate collector configuration"""
        return True
    
    async def _download_csv(self, url: str) -> Optional[str]:
        """Download CSV file from GitHub"""
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.get(url)
                
                if response.status_code == 200:
                    return response.text
                elif response.status_code == 404:
                    logger.debug(f"[TennisAbstract] File not found: {url}")
                    return None
                else:
                    logger.warning(f"[TennisAbstract] Error {response.status_code}: {url}")
                    return None
                    
        except Exception as e:
            logger.error(f"[TennisAbstract] Download error for {url}: {e}")
            return None
    
    def _parse_csv(self, csv_text: str) -> List[Dict[str, Any]]:
        """Parse CSV text into list of dictionaries"""
        rows = []
        try:
            reader = csv.DictReader(io.StringIO(csv_text))
            for row in reader:
                rows.append(row)
        except Exception as e:
            logger.error(f"[TennisAbstract] CSV parse error: {e}")
        return rows
    
    def _safe_int(self, value: Any) -> Optional[int]:
        """Safely convert value to int"""
        if value is None or value == '' or value == 'NA':
            return None
        try:
            return int(float(value))
        except (ValueError, TypeError):
            return None
    
    def _safe_float(self, value: Any) -> Optional[float]:
        """Safely convert value to float"""
        if value is None or value == '' or value == 'NA':
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    
    def _parse_date(self, date_str: str) -> Optional[date]:
        """Parse date from YYYYMMDD format"""
        if not date_str or len(date_str) < 8:
            return None
        try:
            return datetime.strptime(date_str[:8], '%Y%m%d').date()
        except (ValueError, TypeError):
            return None
    
    def _parse_birth_date(self, date_str: str) -> Optional[date]:
        """Parse birth date (YYYYMMDD or YYYY format)"""
        if not date_str:
            return None
        try:
            if len(date_str) >= 8:
                return datetime.strptime(date_str[:8], '%Y%m%d').date()
            elif len(date_str) == 4:
                # Just year - use Jan 1
                return date(int(date_str), 1, 1)
        except (ValueError, TypeError):
            pass
        return None
    
    # =========================================================================
    # PLAYER DATA METHODS
    # =========================================================================
    
    async def fetch_players(self, tour: str = "ATP") -> List[Dict[str, Any]]:
        """Fetch all players for a tour"""
        base_url = self.ATP_BASE_URL if tour.upper() == "ATP" else self.WTA_BASE_URL
        filename = "atp_players.csv" if tour.upper() == "ATP" else "wta_players.csv"
        url = f"{base_url}/{filename}"
        
        csv_text = await self._download_csv(url)
        if not csv_text:
            return []
        
        players = []
        rows = self._parse_csv(csv_text)
        
        for row in rows:
            player = {
                'player_id': row.get('player_id', ''),
                'first_name': row.get('first_name', ''),
                'last_name': row.get('last_name', ''),
                'name': f"{row.get('first_name', '')} {row.get('last_name', '')}".strip(),
                'hand': row.get('hand'),
                'birth_date': self._parse_birth_date(row.get('birth_date', '')),
                'country_code': row.get('country_code', ''),
                'height_cm': self._safe_int(row.get('height')),
                'tour': tour.upper()
            }
            if player['player_id']:
                players.append(player)
        
        logger.info(f"[TennisAbstract] Fetched {len(players)} {tour} players")
        return players
    
    # =========================================================================
    # RANKING DATA METHODS
    # =========================================================================
    
    async def fetch_rankings_current(self, tour: str = "ATP") -> List[Dict[str, Any]]:
        """Fetch current rankings"""
        base_url = self.ATP_BASE_URL if tour.upper() == "ATP" else self.WTA_BASE_URL
        filename = "atp_rankings_current.csv" if tour.upper() == "ATP" else "wta_rankings_current.csv"
        url = f"{base_url}/{filename}"
        
        csv_text = await self._download_csv(url)
        if not csv_text:
            return []
        
        rankings = []
        rows = self._parse_csv(csv_text)
        
        for row in rows:
            ranking = {
                'ranking_date': self._parse_date(row.get('ranking_date', '')),
                'ranking': self._safe_int(row.get('ranking')) or self._safe_int(row.get('rank')),
                'player_id': row.get('player_id') or row.get('player'),
                'ranking_points': self._safe_int(row.get('ranking_points')) or self._safe_int(row.get('points')),
                'tour': tour.upper()
            }
            if ranking['player_id'] and ranking['ranking']:
                rankings.append(ranking)
        
        logger.info(f"[TennisAbstract] Fetched {len(rankings)} current {tour} rankings")
        return rankings
    
    async def fetch_rankings_historical(self, tour: str = "ATP", decades: List[str] = None) -> List[Dict[str, Any]]:
        """Fetch historical rankings by decade"""
        if decades is None:
            decades = ['70', '80', '90', '00', '10', '20']
        
        base_url = self.ATP_BASE_URL if tour.upper() == "ATP" else self.WTA_BASE_URL
        prefix = "atp" if tour.upper() == "ATP" else "wta"
        
        all_rankings = []
        
        for decade in decades:
            filename = f"{prefix}_rankings_{decade}s.csv"
            url = f"{base_url}/{filename}"
            
            csv_text = await self._download_csv(url)
            if not csv_text:
                continue
            
            rows = self._parse_csv(csv_text)
            
            for row in rows:
                ranking = {
                    'ranking_date': self._parse_date(row.get('ranking_date', '')),
                    'ranking': self._safe_int(row.get('ranking')) or self._safe_int(row.get('rank')),
                    'player_id': row.get('player_id') or row.get('player'),
                    'ranking_points': self._safe_int(row.get('ranking_points')) or self._safe_int(row.get('points')),
                    'tour': tour.upper()
                }
                if ranking['player_id'] and ranking['ranking']:
                    all_rankings.append(ranking)
            
            logger.info(f"[TennisAbstract] Fetched {len(rows)} {tour} rankings from {decade}s")
            await asyncio.sleep(0.2)  # Rate limiting
        
        return all_rankings
    
    # =========================================================================
    # MATCH DATA METHODS
    # =========================================================================
    
    async def fetch_matches_year(self, year: int, tour: str = "ATP") -> List[Dict[str, Any]]:
        """Fetch matches for a specific year"""
        base_url = self.ATP_BASE_URL if tour.upper() == "ATP" else self.WTA_BASE_URL
        prefix = "atp" if tour.upper() == "ATP" else "wta"
        filename = f"{prefix}_matches_{year}.csv"
        url = f"{base_url}/{filename}"
        
        csv_text = await self._download_csv(url)
        if not csv_text:
            return []
        
        matches = []
        rows = self._parse_csv(csv_text)
        
        for row in rows:
            match = self._parse_match_row(row, tour.upper())
            if match:
                matches.append(match)
        
        logger.info(f"[TennisAbstract] Fetched {len(matches)} {tour} matches from {year}")
        return matches
    
    def _parse_match_row(self, row: Dict[str, Any], tour: str) -> Optional[Dict[str, Any]]:
        """Parse a single match row from CSV"""
        try:
            tourney_date = self._parse_date(row.get('tourney_date', ''))
            if not tourney_date:
                return None
            
            match = {
                # Tournament info
                'tourney_id': row.get('tourney_id', ''),
                'tourney_name': row.get('tourney_name', ''),
                'tourney_date': tourney_date,
                'tourney_level': row.get('tourney_level', ''),
                'tourney_level_name': self.TOURNEY_LEVELS.get(row.get('tourney_level', ''), 'Other'),
                'surface': self.SURFACES.get(row.get('surface', ''), row.get('surface', '').lower()),
                'draw_size': self._safe_int(row.get('draw_size')),
                
                # Match info
                'match_num': self._safe_int(row.get('match_num')),
                'round': row.get('round', ''),
                'round_name': self.ROUNDS.get(row.get('round', ''), row.get('round', '')),
                'best_of': self._safe_int(row.get('best_of')) or 3,
                'score': row.get('score', ''),
                'minutes': self._safe_int(row.get('minutes')),
                
                # Winner info
                'winner_id': row.get('winner_id', ''),
                'winner_name': row.get('winner_name', ''),
                'winner_seed': self._safe_int(row.get('winner_seed')),
                'winner_entry': row.get('winner_entry', ''),
                'winner_hand': row.get('winner_hand', ''),
                'winner_ht': self._safe_int(row.get('winner_ht')),
                'winner_ioc': row.get('winner_ioc', ''),
                'winner_age': self._safe_float(row.get('winner_age')),
                'winner_rank': self._safe_int(row.get('winner_rank')),
                'winner_rank_points': self._safe_int(row.get('winner_rank_points')),
                
                # Loser info
                'loser_id': row.get('loser_id', ''),
                'loser_name': row.get('loser_name', ''),
                'loser_seed': self._safe_int(row.get('loser_seed')),
                'loser_entry': row.get('loser_entry', ''),
                'loser_hand': row.get('loser_hand', ''),
                'loser_ht': self._safe_int(row.get('loser_ht')),
                'loser_ioc': row.get('loser_ioc', ''),
                'loser_age': self._safe_float(row.get('loser_age')),
                'loser_rank': self._safe_int(row.get('loser_rank')),
                'loser_rank_points': self._safe_int(row.get('loser_rank_points')),
                
                # Winner match stats
                'w_ace': self._safe_int(row.get('w_ace')),
                'w_df': self._safe_int(row.get('w_df')),
                'w_svpt': self._safe_int(row.get('w_svpt')),
                'w_1stIn': self._safe_int(row.get('w_1stIn')),
                'w_1stWon': self._safe_int(row.get('w_1stWon')),
                'w_2ndWon': self._safe_int(row.get('w_2ndWon')),
                'w_SvGms': self._safe_int(row.get('w_SvGms')),
                'w_bpSaved': self._safe_int(row.get('w_bpSaved')),
                'w_bpFaced': self._safe_int(row.get('w_bpFaced')),
                
                # Loser match stats
                'l_ace': self._safe_int(row.get('l_ace')),
                'l_df': self._safe_int(row.get('l_df')),
                'l_svpt': self._safe_int(row.get('l_svpt')),
                'l_1stIn': self._safe_int(row.get('l_1stIn')),
                'l_1stWon': self._safe_int(row.get('l_1stWon')),
                'l_2ndWon': self._safe_int(row.get('l_2ndWon')),
                'l_SvGms': self._safe_int(row.get('l_SvGms')),
                'l_bpSaved': self._safe_int(row.get('l_bpSaved')),
                'l_bpFaced': self._safe_int(row.get('l_bpFaced')),
                
                'tour': tour
            }
            
            # Create unique match ID
            match['match_id'] = f"{tour}_{match['tourney_id']}_{match['match_num']}"
            
            return match
            
        except Exception as e:
            logger.debug(f"[TennisAbstract] Error parsing match row: {e}")
            return None
    
    async def fetch_matches_historical(self, tour: str = "ATP", years_back: int = 10) -> List[Dict[str, Any]]:
        """Fetch historical matches for multiple years"""
        current_year = datetime.now().year
        start_year = current_year - years_back + 1
        
        all_matches = []
        
        for year in range(start_year, current_year + 1):
            matches = await self.fetch_matches_year(year, tour)
            all_matches.extend(matches)
            await asyncio.sleep(0.3)  # Rate limiting
        
        logger.info(f"[TennisAbstract] Fetched {len(all_matches)} total {tour} matches ({start_year}-{current_year})")
        return all_matches
    
    # =========================================================================
    # HEAD-TO-HEAD CALCULATION
    # =========================================================================
    
    def calculate_h2h(self, matches: List[Dict], player1_id: str, player2_id: str) -> Dict[str, Any]:
        """Calculate head-to-head record between two players"""
        h2h_matches = []
        p1_wins = 0
        p2_wins = 0
        p1_surface_wins = {'hard': 0, 'clay': 0, 'grass': 0}
        p2_surface_wins = {'hard': 0, 'clay': 0, 'grass': 0}
        
        for match in matches:
            winner_id = str(match.get('winner_id', ''))
            loser_id = str(match.get('loser_id', ''))
            
            if (winner_id == player1_id and loser_id == player2_id) or \
               (winner_id == player2_id and loser_id == player1_id):
                h2h_matches.append(match)
                surface = match.get('surface', 'hard')
                
                if winner_id == player1_id:
                    p1_wins += 1
                    if surface in p1_surface_wins:
                        p1_surface_wins[surface] += 1
                else:
                    p2_wins += 1
                    if surface in p2_surface_wins:
                        p2_surface_wins[surface] += 1
        
        return {
            'player1_id': player1_id,
            'player2_id': player2_id,
            'total_matches': len(h2h_matches),
            'player1_wins': p1_wins,
            'player2_wins': p2_wins,
            'player1_surface_wins': p1_surface_wins,
            'player2_surface_wins': p2_surface_wins,
            'matches': h2h_matches
        }
    
    # =========================================================================
    # MAIN COLLECTION METHOD
    # =========================================================================
    
    async def collect(self, tours: List[str] = None, years_back: int = 10, 
                     collect_type: str = "all") -> CollectorResult:
        """
        Main collection method for Tennis Abstract data
        
        Args:
            tours: List of tours ['ATP', 'WTA'] or None for both
            years_back: Number of years of historical data
            collect_type: 'all', 'players', 'rankings', 'matches'
        
        Returns:
            CollectorResult with all collected data
        """
        if tours is None:
            tours = ["ATP", "WTA"]
        
        data = {
            'players': [],
            'rankings': [],
            'matches': [],
            'tournaments': set()
        }
        
        try:
            for tour in tours:
                tour = tour.upper()
                
                # Collect players
                if collect_type in ['all', 'players']:
                    players = await self.fetch_players(tour)
                    data['players'].extend(players)
                    await asyncio.sleep(0.2)
                
                # Collect current rankings
                if collect_type in ['all', 'rankings']:
                    rankings = await self.fetch_rankings_current(tour)
                    data['rankings'].extend(rankings)
                    await asyncio.sleep(0.2)
                
                # Collect matches
                if collect_type in ['all', 'matches']:
                    matches = await self.fetch_matches_historical(tour, years_back)
                    data['matches'].extend(matches)
                    
                    # Extract unique tournaments
                    for match in matches:
                        tourney_key = f"{match.get('tourney_id')}_{match.get('tourney_name')}"
                        data['tournaments'].add(tourney_key)
            
            data['tournaments'] = list(data['tournaments'])
            
            total_records = len(data['players']) + len(data['rankings']) + len(data['matches'])
            
            logger.info(f"[TennisAbstract] Collection complete: {total_records} total records")
            logger.info(f"  Players: {len(data['players'])}")
            logger.info(f"  Rankings: {len(data['rankings'])}")
            logger.info(f"  Matches: {len(data['matches'])}")
            logger.info(f"  Tournaments: {len(data['tournaments'])}")
            
            return CollectorResult(
                success=True,
                data=data,
                records_count=total_records,
                metadata={'tours': tours, 'years_back': years_back}
            )
            
        except Exception as e:
            logger.error(f"[TennisAbstract] Collection error: {e}")
            return CollectorResult(
                success=False,
                error=str(e),
                data=data,
                records_count=0
            )
    
    async def collect_historical(self, years_back: int = 10) -> CollectorResult:
        """Collect historical data for both tours"""
        return await self.collect(tours=["ATP", "WTA"], years_back=years_back, collect_type="all")
    
    async def collect_sport(self, tour: str, years_back: int = 10) -> CollectorResult:
        """Collect data for a specific tour"""
        return await self.collect(tours=[tour], years_back=years_back, collect_type="all")
    
    # =========================================================================
    # DATABASE SAVE METHODS
    # =========================================================================
    
    async def save_to_database(self, data: Dict[str, Any], session=None) -> Dict[str, int]:
        """
        Save collected data to database
        
        Args:
            data: Dictionary with players, rankings, matches
            session: Database session (optional, uses self.db_session if not provided)
        
        Returns:
            Dictionary with counts of saved records
        """
        session = session or self.db_session
        if session is None:
            logger.error("[TennisAbstract] No database session available")
            return {'players': 0, 'rankings': 0, 'matches': 0, 'stats': 0}
        
        counts = {
            'players': 0,
            'rankings': 0,
            'matches': 0,
            'stats': 0,
            'teams': 0,
            'seasons': 0
        }
        
        try:
            from app.models.models import Sport, Season, Team, Player, Game, PlayerStats, TeamStats
            from sqlalchemy import select
            
            # Get or create ATP sport
            result = await session.execute(
                select(Sport).where(Sport.code == "ATP")
            )
            atp_sport = result.scalar_one_or_none()
            
            if not atp_sport:
                atp_sport = Sport(code="ATP", name="ATP Tennis", is_active=True)
                session.add(atp_sport)
                await session.flush()
            
            # Get or create WTA sport
            result = await session.execute(
                select(Sport).where(Sport.code == "WTA")
            )
            wta_sport = result.scalar_one_or_none()
            
            if not wta_sport:
                wta_sport = Sport(code="WTA", name="WTA Tennis", is_active=True)
                session.add(wta_sport)
                await session.flush()
            
            # Save players
            if data.get('players'):
                counts['players'] = await self._save_players(
                    data['players'], atp_sport, wta_sport, session
                )
            
            # Save matches (as games with player stats)
            if data.get('matches'):
                match_counts = await self._save_matches(
                    data['matches'], atp_sport, wta_sport, session
                )
                counts['matches'] = match_counts['games']
                counts['stats'] = match_counts['stats']
                counts['teams'] = match_counts['teams']
                counts['seasons'] = match_counts['seasons']
            
            await session.commit()
            
            logger.info(f"[TennisAbstract] Saved to database: {counts}")
            return counts
            
        except Exception as e:
            logger.error(f"[TennisAbstract] Database save error: {e}")
            await session.rollback()
            return counts
    
    async def _save_players(self, players: List[Dict], atp_sport: Any, 
                           wta_sport: Any, session: Any) -> int:
        """Save players to database"""
        from app.models.models import Player
        from sqlalchemy import select
        
        saved = 0
        
        for player_data in players:
            tour = player_data.get('tour', 'ATP')
            external_id = f"tennis_abstract_{tour}_{player_data.get('player_id')}"
            
            # Check if exists
            result = await session.execute(
                select(Player).where(Player.external_id == external_id)
            )
            existing = result.scalar_one_or_none()
            
            if existing:
                continue
            
            # Parse birth date
            birth_date = player_data.get('birth_date')
            if isinstance(birth_date, str):
                birth_date = self._parse_birth_date(birth_date)
            
            # Parse height
            height = player_data.get('height_cm')
            height_str = f"{height} cm" if height else None
            
            player = Player(
                external_id=external_id,
                name=player_data.get('name', ''),
                position=player_data.get('hand', ''),  # Store hand as position
                birth_date=birth_date,
                height=height_str,
                is_active=True
            )
            session.add(player)
            saved += 1
            
            if saved % 500 == 0:
                await session.flush()
                logger.info(f"[TennisAbstract] Saved {saved} players...")
        
        await session.flush()
        logger.info(f"[TennisAbstract] Saved {saved} new players")
        return saved
    
    async def _save_matches(self, matches: List[Dict], atp_sport: Any,
                           wta_sport: Any, session: Any) -> Dict[str, int]:
        """Save matches as games with player stats"""
        from app.models.models import Game, Team, Season, PlayerStats, Player, GameStatus
        from sqlalchemy import select
        from datetime import datetime
        
        counts = {'games': 0, 'stats': 0, 'teams': 0, 'seasons': 0}
        team_cache = {}
        season_cache = {}
        player_cache = {}
        
        for match in matches:
            try:
                tour = match.get('tour', 'ATP')
                sport = atp_sport if tour == 'ATP' else wta_sport
                
                match_id = match.get('match_id', '')
                external_id = f"tennis_abstract_{match_id}"
                
                # Check if game exists
                result = await session.execute(
                    select(Game).where(Game.external_id == external_id)
                )
                existing = result.scalar_one_or_none()
                
                if existing:
                    continue
                
                tourney_date = match.get('tourney_date')
                if isinstance(tourney_date, str):
                    tourney_date = self._parse_date(tourney_date)
                
                if not tourney_date:
                    continue
                
                year = tourney_date.year
                
                # Get or create season
                season_key = f"{tour}_{year}"
                if season_key not in season_cache:
                    result = await session.execute(
                        select(Season).where(
                            Season.sport_id == sport.id,
                            Season.year == year
                        )
                    )
                    season = result.scalar_one_or_none()
                    
                    if not season:
                        season = Season(
                            sport_id=sport.id,
                            year=year,
                            name=f"{tour} {year}",
                            start_date=date(year, 1, 1),
                            end_date=date(year, 12, 31),
                            is_current=(year == datetime.now().year)
                        )
                        session.add(season)
                        await session.flush()
                        counts['seasons'] += 1
                    
                    season_cache[season_key] = season
                
                season = season_cache[season_key]
                
                # Get or create "teams" for winner and loser (tennis uses players as teams)
                winner_id = match.get('winner_id', '')
                loser_id = match.get('loser_id', '')
                winner_name = match.get('winner_name', 'Unknown')
                loser_name = match.get('loser_name', 'Unknown')
                
                # Winner team
                winner_team_key = f"{tour}_{winner_id}"
                if winner_team_key not in team_cache:
                    result = await session.execute(
                        select(Team).where(
                            Team.sport_id == sport.id,
                            Team.external_id == f"tennis_{tour}_{winner_id}"
                        )
                    )
                    winner_team = result.scalar_one_or_none()
                    
                    if not winner_team:
                        abbrev = self._get_player_abbrev(winner_name)
                        winner_team = Team(
                            sport_id=sport.id,
                            external_id=f"tennis_{tour}_{winner_id}",
                            name=winner_name,
                            abbreviation=abbrev,
                            city=match.get('winner_ioc', ''),
                            is_active=True
                        )
                        session.add(winner_team)
                        await session.flush()
                        counts['teams'] += 1
                    
                    team_cache[winner_team_key] = winner_team
                
                # Loser team
                loser_team_key = f"{tour}_{loser_id}"
                if loser_team_key not in team_cache:
                    result = await session.execute(
                        select(Team).where(
                            Team.sport_id == sport.id,
                            Team.external_id == f"tennis_{tour}_{loser_id}"
                        )
                    )
                    loser_team = result.scalar_one_or_none()
                    
                    if not loser_team:
                        abbrev = self._get_player_abbrev(loser_name)
                        loser_team = Team(
                            sport_id=sport.id,
                            external_id=f"tennis_{tour}_{loser_id}",
                            name=loser_name,
                            abbreviation=abbrev,
                            city=match.get('loser_ioc', ''),
                            is_active=True
                        )
                        session.add(loser_team)
                        await session.flush()
                        counts['teams'] += 1
                    
                    team_cache[loser_team_key] = loser_team
                
                winner_team = team_cache[winner_team_key]
                loser_team = team_cache[loser_team_key]
                
                # Create game (winner is always home team for tennis)
                game = Game(
                    sport_id=sport.id,
                    season_id=season.id,
                    external_id=external_id,
                    home_team_id=winner_team.id,
                    away_team_id=loser_team.id,
                    scheduled_at=datetime.combine(tourney_date, datetime.min.time()),
                    status=GameStatus.FINAL,
                    home_score=1,  # Winner always gets 1
                    away_score=0,  # Loser gets 0
                    weather={
                        'tournament': match.get('tourney_name'),
                        'surface': match.get('surface'),
                        'round': match.get('round_name'),
                        'score': match.get('score'),
                        'minutes': match.get('minutes'),
                        'tourney_level': match.get('tourney_level_name')
                    }
                )
                session.add(game)
                await session.flush()
                counts['games'] += 1
                
                # Save player stats for winner
                winner_stats = await self._create_player_stats(match, 'winner', season.id, game.id, session)
                if winner_stats:
                    counts['stats'] += len(winner_stats)
                
                # Save player stats for loser
                loser_stats = await self._create_player_stats(match, 'loser', season.id, game.id, session)
                if loser_stats:
                    counts['stats'] += len(loser_stats)
                
                if counts['games'] % 1000 == 0:
                    await session.flush()
                    logger.info(f"[TennisAbstract] Saved {counts['games']} matches...")
                    
            except Exception as e:
                logger.debug(f"[TennisAbstract] Error saving match: {e}")
                continue
        
        await session.flush()
        logger.info(f"[TennisAbstract] Saved {counts['games']} matches, {counts['stats']} stats")
        return counts
    
    async def _create_player_stats(self, match: Dict, role: str, season_id: Any, 
                            game_id: Any, session: Any) -> List[Any]:
        """Create player stats records for a match"""
        from app.models.models import PlayerStats, Player
        from sqlalchemy import select
        
        stats_list = []
        prefix = 'w_' if role == 'winner' else 'l_'
        player_id = match.get(f'{role}_id')
        tour = match.get('tour', 'ATP')
        
        # Get or create player
        external_id = f"tennis_abstract_{tour}_{player_id}"
        result = await session.execute(
            select(Player).where(Player.external_id == external_id)
        )
        player = result.scalar_one_or_none()
        
        if not player:
            # Create player if not exists
            player = Player(
                external_id=external_id,
                name=match.get(f'{role}_name', 'Unknown'),
                position=match.get(f'{role}_hand', ''),
                is_active=True
            )
            session.add(player)
            await session.flush()
        
        # Define stats to save (truncate stat_type to 50 chars)
        stat_mapping = {
            'aces': f'{prefix}ace',
            'double_faults': f'{prefix}df',
            'serve_points': f'{prefix}svpt',
            'first_serve_in': f'{prefix}1stIn',
            'first_serve_won': f'{prefix}1stWon',
            'second_serve_won': f'{prefix}2ndWon',
            'serve_games': f'{prefix}SvGms',
            'break_points_saved': f'{prefix}bpSaved',
            'break_points_faced': f'{prefix}bpFaced'
        }
        
        # Add ranking stats
        if role == 'winner':
            stat_mapping['ranking'] = 'winner_rank'
            stat_mapping['ranking_points'] = 'winner_rank_points'
        else:
            stat_mapping['ranking'] = 'loser_rank'
            stat_mapping['ranking_points'] = 'loser_rank_points'
        
        for stat_type, match_key in stat_mapping.items():
            value = match.get(match_key)
            if value is not None:
                stat = PlayerStats(
                    player_id=player.id,
                    game_id=game_id,
                    season_id=season_id,
                    stat_type=stat_type[:50],  # Truncate to 50 chars
                    value=float(value)
                )
                session.add(stat)
                stats_list.append(stat)
        
        return stats_list
    
    def _get_player_abbrev(self, name: str) -> str:
        """Generate abbreviation from player name"""
        if not name:
            return "UNK"
        
        parts = name.split()
        if len(parts) >= 2:
            # First initial + last name (max 3 chars)
            return (parts[0][0] + parts[-1][:2]).upper()
        elif len(parts) == 1:
            return parts[0][:3].upper()
        return "UNK"
    
    # =========================================================================
    # CONVENIENCE METHODS FOR MASTER IMPORT
    # =========================================================================
    
    async def collect_atp_only(self, years_back: int = 10) -> CollectorResult:
        """Collect ATP data only"""
        return await self.collect(tours=["ATP"], years_back=years_back)
    
    async def collect_wta_only(self, years_back: int = 10) -> CollectorResult:
        """Collect WTA data only"""
        return await self.collect(tours=["WTA"], years_back=years_back)
    
    async def collect_players_only(self, tours: List[str] = None) -> CollectorResult:
        """Collect only player data"""
        return await self.collect(tours=tours, years_back=1, collect_type="players")
    
    async def collect_rankings_only(self, tours: List[str] = None) -> CollectorResult:
        """Collect only ranking data"""
        return await self.collect(tours=tours, years_back=1, collect_type="rankings")
    
    async def collect_matches_only(self, tours: List[str] = None, years_back: int = 10) -> CollectorResult:
        """Collect only match data"""
        return await self.collect(tours=tours, years_back=years_back, collect_type="matches")


# Singleton instance
tennis_abstract_collector = TennisAbstractCollector()