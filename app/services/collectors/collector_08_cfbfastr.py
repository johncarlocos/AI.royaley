"""
ROYALEY - cfbfastR Data Collector
Phase 1: Data Collection Services

Collects comprehensive College Football (NCAAF) play-by-play data from sportsdataverse/cfbfastR.
Features: Play-by-play (2004-present), EPA, SP+ ratings, recruiting data, 70+ features.

Data Source: https://github.com/sportsdataverse/cfbfastR-data
Documentation: https://cfbfastr.sportsdataverse.org/

FREE data - no API key required!

Key Data Types:
- Play-by-play: Every play with EPA, WPA, success rate (2004-present)
- Team info: 130+ FBS teams with conferences, colors, logos
- SP+ Ratings: Bill Connelly's predictive ratings
- Recruiting: Team recruiting rankings and player stars
- Betting lines: Spreads, totals, moneylines
- Game schedules: Full schedules with results
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

import httpx
import pandas as pd
import numpy as np

from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models import Sport, Team, Game, GameStatus
from app.services.collectors.base_collector import (
    BaseCollector,
    CollectorResult,
    collector_manager,
)

logger = logging.getLogger(__name__)


# =============================================================================
# SPORTSDATAVERSE CFB DATA URLS (Direct GitHub Raw Files)
# =============================================================================

# Primary data source - raw GitHub files from cfbfastR-data repo (MASTER branch)
CFBVERSE_RAW = "https://raw.githubusercontent.com/sportsdataverse/cfbfastR-data/master"

# sportsdataverse-data releases (newer consolidated data)
SPORTSDATA_RELEASES = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download"

CFBFASTR_URLS = {
    # Play-by-play (2002-2020 in cfbfastR-data, newer in sportsdataverse-data)
    # Parquet - older data (2002-2020)
    "pbp_parquet_old": f"{CFBVERSE_RAW}/data/parquet/play_by_play_{{year}}.parquet",
    # CSV - older data (2002-2020)  
    "pbp_csv": f"{CFBVERSE_RAW}/pbp/csv/play_by_play_{{year}}.csv.gz",
    # RDS - older data
    "pbp_rds": f"{CFBVERSE_RAW}/pbp/rds/play_by_play_{{year}}.rds",
    
    # Newer PBP data from sportsdataverse-data releases (2021+)
    "pbp_parquet_new": f"{SPORTSDATA_RELEASES}/cfbfastR_cfb_pbp/play_by_play_{{year}}.parquet",
    
    # Schedules from cfbfastR-data
    "schedules_parquet": f"{CFBVERSE_RAW}/schedules/parquet/schedules_{{year}}.parquet",
    "schedules_csv": f"{CFBVERSE_RAW}/schedules/csv/schedules_{{year}}.csv",
    
    # Team information
    "teams_parquet": f"{CFBVERSE_RAW}/teams/parquet/teams.parquet",
    "teams_csv": f"{CFBVERSE_RAW}/teams/csv/teams.csv",
    "team_info_parquet": f"{CFBVERSE_RAW}/team_info/parquet/cfb_team_info.parquet",
    "team_info_csv": f"{CFBVERSE_RAW}/team_info/csv/cfb_team_info.csv",
    
    # Betting lines  
    "betting_parquet": f"{CFBVERSE_RAW}/betting/parquet/betting_{{year}}.parquet",
    "betting_csv": f"{CFBVERSE_RAW}/betting/csv/betting_{{year}}.csv",
    
    # Player stats
    "player_stats_parquet": f"{CFBVERSE_RAW}/player_stats/parquet/player_stats_{{year}}.parquet",
    "player_stats_csv": f"{CFBVERSE_RAW}/player_stats/csv/player_stats_{{year}}.csv",
    
    # Rosters
    "rosters_parquet": f"{CFBVERSE_RAW}/rosters/parquet/rosters_{{year}}.parquet",
    "rosters_csv": f"{CFBVERSE_RAW}/rosters/csv/rosters_{{year}}.csv",
}


# =============================================================================
# KEY PLAY-BY-PLAY FEATURES (70+ columns)
# =============================================================================

CFB_PBP_KEY_COLUMNS = [
    # Game/Play Identifiers
    "id_play", "game_id", "drive_id", "game_play_number",
    "home", "away", "pos_team", "def_pos_team",
    "season", "season_type", "week", "game_date",
    "period", "clock.minutes", "clock.seconds",
    
    # Scores
    "home_score", "away_score", "pos_team_score", "def_pos_team_score",
    "score_diff", "pos_score_diff",
    
    # Play Details
    "play_type", "play_text", "down", "distance", "yards_to_goal",
    "yards_gained", "drive_number", "drive_play_number",
    "start.yardsToEndzone", "end.yardsToEndzone",
    
    # Expected Points Added (EPA) - THE KEY METRIC
    "EPA", "ep_before", "ep_after",
    "home_EPA", "away_EPA",
    "def_EPA", 
    "EPA.explosive", "EPA.success", "EPA.success_standard",
    "EPA.passing", "EPA.rushing",
    
    # Win Probability
    "home_wp", "away_wp", "home_wp_before", "away_wp_before",
    "home_wp_post", "away_wp_post",
    "wpa", 
    
    # Success Metrics
    "success", "first_down", "firstD_by_pass", "firstD_by_rush",
    "third_down_converted", "third_down_failed",
    "fourth_down_converted", "fourth_down_failed",
    "explosiveness",
    
    # Passing
    "pass", "completion", "incompletion",
    "pass_attempt", "sack", "interception",
    "pass_td", "passing_yards", "air_yards",
    "passer_player_name", "passer_id",
    
    # Rushing
    "rush", "rush_td",
    "rushing_yards",
    "rusher_player_name", "rusher_id",
    
    # Receiving
    "receiver_player_name", "receiver_id",
    "receiving_yards", "target",
    
    # Drive Information
    "drive_start_yard_line", "drive_end_yard_line",
    "drive_yards", "drive_result", "drive_pts",
    "drive.start_period", "drive.end_period",
    
    # Special Teams
    "kickoff_play", "punt", "fg_attempt", "fg_made",
    "kickoff_return_yards", "punt_return_yards",
    
    # Penalties
    "penalty_flag", "penalty_yards", "penalty_text",
    
    # Turnovers
    "turnover", "fumble", "fumble_lost", "int", "int_td",
    
    # Garbage time filter
    "garbage_time", "scoring_play", "red_zone",
]


# =============================================================================
# FBS TEAMS AND CONFERENCES (2024 Realignment)
# =============================================================================

# Power 4 Conferences (2024+)
CFB_CONFERENCES = {
    "ACC": [
        "Boston College", "California", "Clemson", "Duke", "Florida State",
        "Georgia Tech", "Louisville", "Miami", "NC State", "North Carolina",
        "Pittsburgh", "SMU", "Stanford", "Syracuse", "Virginia", "Virginia Tech",
        "Wake Forest"
    ],
    "Big 12": [
        "Arizona", "Arizona State", "Baylor", "BYU", "Central Florida",
        "Cincinnati", "Colorado", "Houston", "Iowa State", "Kansas", 
        "Kansas State", "Oklahoma State", "TCU", "Texas Tech", "UCF",
        "Utah", "West Virginia"
    ],
    "Big Ten": [
        "Illinois", "Indiana", "Iowa", "Maryland", "Michigan",
        "Michigan State", "Minnesota", "Nebraska", "Northwestern",
        "Ohio State", "Oregon", "Penn State", "Purdue", "Rutgers",
        "UCLA", "USC", "Washington", "Wisconsin"
    ],
    "SEC": [
        "Alabama", "Arkansas", "Auburn", "Florida", "Georgia",
        "Kentucky", "LSU", "Mississippi State", "Missouri", "Oklahoma",
        "Ole Miss", "South Carolina", "Tennessee", "Texas", "Texas A&M",
        "Vanderbilt"
    ],
    # Group of 5
    "American": [
        "Army", "Charlotte", "East Carolina", "FAU", "Memphis",
        "Navy", "North Texas", "Rice", "South Florida", "Temple",
        "Tulane", "Tulsa", "UAB", "UTSA"
    ],
    "Conference USA": [
        "FIU", "Jacksonville State", "Kennesaw State", "Liberty",
        "Louisiana Tech", "Middle Tennessee", "New Mexico State",
        "Sam Houston", "UTEP", "Western Kentucky"
    ],
    "MAC": [
        "Akron", "Ball State", "Bowling Green", "Buffalo", "Central Michigan",
        "Eastern Michigan", "Kent State", "Miami (OH)", "Northern Illinois",
        "Ohio", "Toledo", "Western Michigan"
    ],
    "Mountain West": [
        "Air Force", "Boise State", "Colorado State", "Fresno State",
        "Hawaii", "Nevada", "New Mexico", "San Diego State",
        "San Jose State", "UNLV", "Utah State", "Wyoming"
    ],
    "Sun Belt": [
        "Appalachian State", "Arkansas State", "Coastal Carolina",
        "Georgia Southern", "Georgia State", "James Madison",
        "Louisiana", "Marshall", "Old Dominion", "South Alabama",
        "Southern Miss", "Texas State", "Troy", "ULM"
    ],
    # Independents
    "FBS Independents": [
        "Notre Dame", "UConn", "UMass"
    ],
}

# Team name variations and abbreviations
CFB_TEAM_MAP = {
    # Team Name -> (Full Name, Abbreviation, Mascot)
    "Alabama": ("Alabama Crimson Tide", "ALA", "Crimson Tide"),
    "Arizona": ("Arizona Wildcats", "ARIZ", "Wildcats"),
    "Arizona State": ("Arizona State Sun Devils", "ASU", "Sun Devils"),
    "Arkansas": ("Arkansas Razorbacks", "ARK", "Razorbacks"),
    "Auburn": ("Auburn Tigers", "AUB", "Tigers"),
    "Baylor": ("Baylor Bears", "BAY", "Bears"),
    "Boise State": ("Boise State Broncos", "BSU", "Broncos"),
    "Boston College": ("Boston College Eagles", "BC", "Eagles"),
    "BYU": ("BYU Cougars", "BYU", "Cougars"),
    "California": ("California Golden Bears", "CAL", "Golden Bears"),
    "Central Florida": ("UCF Knights", "UCF", "Knights"),
    "UCF": ("UCF Knights", "UCF", "Knights"),
    "Cincinnati": ("Cincinnati Bearcats", "CIN", "Bearcats"),
    "Clemson": ("Clemson Tigers", "CLEM", "Tigers"),
    "Colorado": ("Colorado Buffaloes", "COLO", "Buffaloes"),
    "Duke": ("Duke Blue Devils", "DUKE", "Blue Devils"),
    "Florida": ("Florida Gators", "FLA", "Gators"),
    "Florida State": ("Florida State Seminoles", "FSU", "Seminoles"),
    "Georgia": ("Georgia Bulldogs", "UGA", "Bulldogs"),
    "Georgia Tech": ("Georgia Tech Yellow Jackets", "GT", "Yellow Jackets"),
    "Houston": ("Houston Cougars", "HOU", "Cougars"),
    "Illinois": ("Illinois Fighting Illini", "ILL", "Fighting Illini"),
    "Indiana": ("Indiana Hoosiers", "IND", "Hoosiers"),
    "Iowa": ("Iowa Hawkeyes", "IOWA", "Hawkeyes"),
    "Iowa State": ("Iowa State Cyclones", "ISU", "Cyclones"),
    "Kansas": ("Kansas Jayhawks", "KU", "Jayhawks"),
    "Kansas State": ("Kansas State Wildcats", "KSU", "Wildcats"),
    "Kentucky": ("Kentucky Wildcats", "UK", "Wildcats"),
    "LSU": ("LSU Tigers", "LSU", "Tigers"),
    "Louisville": ("Louisville Cardinals", "LOU", "Cardinals"),
    "Maryland": ("Maryland Terrapins", "MD", "Terrapins"),
    "Miami": ("Miami Hurricanes", "MIA", "Hurricanes"),
    "Michigan": ("Michigan Wolverines", "MICH", "Wolverines"),
    "Michigan State": ("Michigan State Spartans", "MSU", "Spartans"),
    "Minnesota": ("Minnesota Golden Gophers", "MINN", "Golden Gophers"),
    "Mississippi State": ("Mississippi State Bulldogs", "MSST", "Bulldogs"),
    "Missouri": ("Missouri Tigers", "MIZ", "Tigers"),
    "Nebraska": ("Nebraska Cornhuskers", "NEB", "Cornhuskers"),
    "North Carolina": ("North Carolina Tar Heels", "UNC", "Tar Heels"),
    "NC State": ("NC State Wolfpack", "NCST", "Wolfpack"),
    "Northwestern": ("Northwestern Wildcats", "NW", "Wildcats"),
    "Notre Dame": ("Notre Dame Fighting Irish", "ND", "Fighting Irish"),
    "Ohio State": ("Ohio State Buckeyes", "OSU", "Buckeyes"),
    "Oklahoma": ("Oklahoma Sooners", "OU", "Sooners"),
    "Oklahoma State": ("Oklahoma State Cowboys", "OKST", "Cowboys"),
    "Ole Miss": ("Ole Miss Rebels", "MISS", "Rebels"),
    "Oregon": ("Oregon Ducks", "ORE", "Ducks"),
    "Oregon State": ("Oregon State Beavers", "ORST", "Beavers"),
    "Penn State": ("Penn State Nittany Lions", "PSU", "Nittany Lions"),
    "Pittsburgh": ("Pittsburgh Panthers", "PITT", "Panthers"),
    "Purdue": ("Purdue Boilermakers", "PUR", "Boilermakers"),
    "Rutgers": ("Rutgers Scarlet Knights", "RUT", "Scarlet Knights"),
    "SMU": ("SMU Mustangs", "SMU", "Mustangs"),
    "South Carolina": ("South Carolina Gamecocks", "SC", "Gamecocks"),
    "Stanford": ("Stanford Cardinal", "STAN", "Cardinal"),
    "Syracuse": ("Syracuse Orange", "SYR", "Orange"),
    "TCU": ("TCU Horned Frogs", "TCU", "Horned Frogs"),
    "Tennessee": ("Tennessee Volunteers", "TENN", "Volunteers"),
    "Texas": ("Texas Longhorns", "TEX", "Longhorns"),
    "Texas A&M": ("Texas A&M Aggies", "TAMU", "Aggies"),
    "Texas Tech": ("Texas Tech Red Raiders", "TTU", "Red Raiders"),
    "UCLA": ("UCLA Bruins", "UCLA", "Bruins"),
    "USC": ("USC Trojans", "USC", "Trojans"),
    "Utah": ("Utah Utes", "UTAH", "Utes"),
    "Vanderbilt": ("Vanderbilt Commodores", "VAN", "Commodores"),
    "Virginia": ("Virginia Cavaliers", "UVA", "Cavaliers"),
    "Virginia Tech": ("Virginia Tech Hokies", "VT", "Hokies"),
    "Wake Forest": ("Wake Forest Demon Deacons", "WAKE", "Demon Deacons"),
    "Washington": ("Washington Huskies", "UW", "Huskies"),
    "Washington State": ("Washington State Cougars", "WSU", "Cougars"),
    "West Virginia": ("West Virginia Mountaineers", "WVU", "Mountaineers"),
    "Wisconsin": ("Wisconsin Badgers", "WIS", "Badgers"),
    # Additional G5 teams
    "Air Force": ("Air Force Falcons", "AFA", "Falcons"),
    "Appalachian State": ("Appalachian State Mountaineers", "APP", "Mountaineers"),
    "Army": ("Army Black Knights", "ARMY", "Black Knights"),
    "Boise State": ("Boise State Broncos", "BSU", "Broncos"),
    "Coastal Carolina": ("Coastal Carolina Chanticleers", "CCU", "Chanticleers"),
    "Colorado State": ("Colorado State Rams", "CSU", "Rams"),
    "Fresno State": ("Fresno State Bulldogs", "FRES", "Bulldogs"),
    "Hawaii": ("Hawaii Rainbow Warriors", "HAW", "Rainbow Warriors"),
    "James Madison": ("James Madison Dukes", "JMU", "Dukes"),
    "Liberty": ("Liberty Flames", "LIB", "Flames"),
    "Louisiana": ("Louisiana Ragin' Cajuns", "ULL", "Ragin' Cajuns"),
    "Marshall": ("Marshall Thundering Herd", "MRSH", "Thundering Herd"),
    "Memphis": ("Memphis Tigers", "MEM", "Tigers"),
    "Navy": ("Navy Midshipmen", "NAVY", "Midshipmen"),
    "Nevada": ("Nevada Wolf Pack", "NEV", "Wolf Pack"),
    "Northern Illinois": ("Northern Illinois Huskies", "NIU", "Huskies"),
    "San Diego State": ("San Diego State Aztecs", "SDSU", "Aztecs"),
    "San Jose State": ("San Jose State Spartans", "SJSU", "Spartans"),
    "Toledo": ("Toledo Rockets", "TOL", "Rockets"),
    "Troy": ("Troy Trojans", "TROY", "Trojans"),
    "Tulane": ("Tulane Green Wave", "TUL", "Green Wave"),
    "UNLV": ("UNLV Rebels", "UNLV", "Rebels"),
    "UTEP": ("UTEP Miners", "UTEP", "Miners"),
    "UTSA": ("UTSA Roadrunners", "UTSA", "Roadrunners"),
    "Western Kentucky": ("Western Kentucky Hilltoppers", "WKU", "Hilltoppers"),
    "Wyoming": ("Wyoming Cowboys", "WYO", "Cowboys"),
}


class CFBFastRCollector(BaseCollector):
    """
    Collector for cfbfastR/sportsdataverse College Football data.
    
    Features:
    - Play-by-play data from 2004-present
    - Expected Points Added (EPA) - gold standard metric for CFB
    - SP+ ratings (Bill Connelly's predictive system)
    - Win Probability (WP) models
    - Team recruiting rankings and player ratings
    - Betting lines with spreads, totals, moneylines
    - Full game schedules and results
    - 70+ analytical features per play
    
    FREE - No API key required!
    
    Covers:
    - 130+ FBS teams
    - 120+ FCS teams
    - Bowl games, CFP, conference championships
    - Regular season (2004-present)
    """
    
    def __init__(self):
        super().__init__(
            name="cfbfastr",
            base_url="https://github.com/sportsdataverse/cfbfastR-data/releases/download",
            rate_limit=30,  # Conservative for GitHub
            rate_window=60,
            timeout=300.0,  # Long timeout for large parquet files
            max_retries=3,
        )
        self.data_dir = Path(settings.MODEL_STORAGE_PATH) / "cfb_data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._teams_cache: Optional[pd.DataFrame] = None
        
    # =========================================================================
    # MAIN COLLECT METHOD
    # =========================================================================
    
    async def collect(
        self,
        sport_code: str = "NCAAF",
        collect_type: str = "schedules",
        years: List[int] = None,
        **kwargs,
    ) -> CollectorResult:
        """
        Collect NCAAF data from cfbfastR/sportsdataverse.
        
        Args:
            sport_code: Must be "NCAAF" or "CFB"
            collect_type: "schedules", "pbp", "sp_ratings", "recruiting", 
                          "betting", "team_stats", "all"
            years: List of years (default: last 5 years, available 2004-present)
            
        Returns:
            CollectorResult with collected data
        """
        if sport_code not in ["NCAAF", "CFB", "ncaaf", "cfb"]:
            return CollectorResult(
                success=False,
                error="cfbfastR only supports NCAAF/CFB data",
                records_count=0,
            )
        
        current_year = datetime.now().year
        if years is None:
            years = list(range(current_year - 4, current_year + 1))
        
        # Ensure years are within cfbfastR-data range (2002+)
        years = [y for y in years if y >= 2002]
        
        all_data = {
            "games": [],
            "sp_ratings": [],
            "recruiting": [],
            "betting_lines": [],
            "team_stats": [],
            "player_stats": [],
        }
        errors = []
        
        try:
            if collect_type in ["schedules", "all"]:
                games = await self._collect_schedules(years)
                all_data["games"] = games
                logger.info(f"[cfbfastR] Collected {len(games)} games")
                
            if collect_type in ["sp_ratings", "all"]:
                sp_ratings = await self._collect_sp_ratings(years)
                all_data["sp_ratings"] = sp_ratings
                logger.info(f"[cfbfastR] Collected {len(sp_ratings)} SP+ ratings")
                
            if collect_type in ["recruiting", "all"]:
                recruiting = await self._collect_recruiting(years)
                all_data["recruiting"] = recruiting
                logger.info(f"[cfbfastR] Collected {len(recruiting)} recruiting records")
                
            if collect_type in ["betting", "all"]:
                betting = await self._collect_betting_lines(years)
                all_data["betting_lines"] = betting
                logger.info(f"[cfbfastR] Collected {len(betting)} betting lines")
                
            if collect_type in ["team_stats", "all"]:
                team_stats = await self._collect_team_stats(years)
                all_data["team_stats"] = team_stats
                
            if collect_type in ["player_stats", "all"]:
                player_stats = await self._collect_player_stats(years)
                all_data["player_stats"] = player_stats
                
        except Exception as e:
            logger.error(f"[cfbfastR] Collection error: {e}")
            errors.append(str(e))
        
        total_records = sum(len(v) for v in all_data.values() if isinstance(v, list))
        
        return CollectorResult(
            success=len(errors) == 0 or total_records > 0,
            data=all_data,
            records_count=total_records,
            error="; ".join(errors) if errors else None,
            metadata={"years": years, "collect_type": collect_type},
        )
    
    # =========================================================================
    # SCHEDULES / GAMES
    # =========================================================================
    
    async def _collect_schedules(self, years: List[int]) -> List[Dict[str, Any]]:
        """
        Collect game schedules with results.
        
        CFB schedule data includes:
        - Regular season games
        - Conference championship games
        - Bowl games
        - CFP games
        
        Note: Primary method is extracting from PBP data since 
        dedicated schedule files may not be available.
        """
        games = []
        
        for year in years:
            try:
                df = None
                
                # Try dedicated schedules first (less likely to work)
                try:
                    url = CFBFASTR_URLS["schedules_parquet"].format(year=year)
                    df = await self._download_parquet(url)
                except:
                    pass
                
                if df is None or len(df) == 0:
                    try:
                        url = CFBFASTR_URLS["schedules_csv"].format(year=year)
                        df = await self._download_csv(url)
                    except:
                        pass
                
                # Extract from PBP as primary fallback (most reliable)
                if df is None or len(df) == 0:
                    df = await self._extract_games_from_pbp(year)
                
                if df is not None and len(df) > 0:
                    for _, row in df.iterrows():
                        game = self._parse_schedule_row(row, year)
                        if game:
                            games.append(game)
                    
                    logger.info(f"[cfbfastR] {year}: {len(df)} games loaded")
                else:
                    logger.warning(f"[cfbfastR] No schedule data for {year}")
                    
            except Exception as e:
                logger.error(f"[cfbfastR] Schedule {year} error: {e}")
        
        logger.info(f"[cfbfastR] Total {len(games)} games collected")
        return games
    
    def _parse_schedule_row(self, row, year: int) -> Optional[Dict[str, Any]]:
        """Parse a schedule/game dataframe row to game dict."""
        try:
            # Handle various column naming conventions
            game_id = row.get("game_id") or row.get("id") or row.get("gameId")
            if pd.isna(game_id):
                return None
            
            # Get teams - handle various column names
            home_team = (row.get("home_team") or row.get("home") or 
                         row.get("homeTeam") or "")
            away_team = (row.get("away_team") or row.get("away") or 
                         row.get("awayTeam") or "")
            
            if not home_team or not away_team:
                return None
            
            home_team = str(home_team).strip()
            away_team = str(away_team).strip()
            
            if home_team == "nan" or away_team == "nan":
                return None
            
            # Get scores
            home_score = row.get("home_points") or row.get("home_score") or row.get("homeScore")
            away_score = row.get("away_points") or row.get("away_score") or row.get("awayScore")
            
            home_score = int(home_score) if pd.notna(home_score) else None
            away_score = int(away_score) if pd.notna(away_score) else None
            
            # Parse game date
            game_date = row.get("start_date") or row.get("game_date") or row.get("date")
            if pd.isna(game_date):
                return None
            
            if isinstance(game_date, str):
                # Handle various date formats
                try:
                    if "T" in game_date:
                        game_date = datetime.fromisoformat(game_date.replace("Z", "+00:00"))
                    else:
                        game_date = datetime.strptime(game_date[:10], "%Y-%m-%d")
                except:
                    return None
            elif hasattr(game_date, 'to_pydatetime'):
                game_date = game_date.to_pydatetime()
            
            # Determine status
            completed = row.get("completed") if pd.notna(row.get("completed")) else None
            if completed == True or (home_score is not None and away_score is not None):
                status = "final"
            elif game_date < datetime.now():
                status = "final"
            else:
                status = "scheduled"
            
            # Get week
            week = row.get("week") or row.get("calendar_week") or 0
            week = int(week) if pd.notna(week) else 0
            
            # Get team info
            home_info = self._get_team_info(home_team)
            away_info = self._get_team_info(away_team)
            
            # Get additional info
            conference_game = bool(row.get("conference_game")) if pd.notna(row.get("conference_game")) else None
            neutral_site = bool(row.get("neutral_site")) if pd.notna(row.get("neutral_site")) else False
            
            return {
                "sport_code": "NCAAF",
                "external_id": f"cfb_{game_id}",
                "game_id": str(game_id),
                "home_team": {
                    "name": home_info[0],
                    "abbreviation": home_info[1],
                },
                "away_team": {
                    "name": away_info[0],
                    "abbreviation": away_info[1],
                },
                "game_date": game_date.isoformat(),
                "status": status,
                "home_score": home_score,
                "away_score": away_score,
                "season": int(row.get("season", year)),
                "week": week,
                "season_type": row.get("season_type", "regular"),
                "home_conference": row.get("home_conference") or row.get("homeConference"),
                "away_conference": row.get("away_conference") or row.get("awayConference"),
                "conference_game": conference_game,
                "neutral_site": neutral_site,
                "venue": row.get("venue") or row.get("venue_id"),
                # Betting info if available
                "spread": float(row.get("spread")) if pd.notna(row.get("spread")) else None,
                "over_under": float(row.get("over_under")) if pd.notna(row.get("over_under")) else None,
            }
            
        except Exception as e:
            logger.debug(f"[cfbfastR] Failed to parse schedule row: {e}")
            return None
    
    def _get_team_info(self, team_name: str) -> Tuple[str, str]:
        """Get (full_name, abbreviation) for a team."""
        # Clean the name
        team_name = team_name.strip()
        
        # Check direct mapping
        if team_name in CFB_TEAM_MAP:
            info = CFB_TEAM_MAP[team_name]
            return (info[0], info[1])
        
        # Check variations
        for key, info in CFB_TEAM_MAP.items():
            if team_name.lower() == key.lower():
                return (info[0], info[1])
            if team_name.lower() in info[0].lower():
                return (info[0], info[1])
        
        # Default: use name as-is with 4-char abbreviation
        abbrev = team_name.replace(" ", "")[:4].upper()
        return (team_name, abbrev)
    
    async def _extract_games_from_pbp(self, year: int) -> Optional[pd.DataFrame]:
        """Extract unique games from PBP data as fallback."""
        try:
            # Try newer sportsdataverse-data first (2021+)
            if year >= 2021:
                url = CFBFASTR_URLS["pbp_parquet_new"].format(year=year)
            else:
                # Older data in cfbfastR-data repo
                url = CFBFASTR_URLS["pbp_parquet_old"].format(year=year)
            
            cols = ["game_id", "home", "away", "game_date", "season", "week",
                    "home_score", "away_score", "season_type"]
            
            df = await self._download_parquet(url, columns=cols)
            
            if df is None or len(df) == 0:
                # Try CSV as fallback
                url = CFBFASTR_URLS["pbp_csv"].format(year=year)
                df = await self._download_csv(url)
            
            if df is None or len(df) == 0:
                return None
            
            # Get unique games with final scores
            # Handle potential column variations
            home_col = "home" if "home" in df.columns else "home_team"
            away_col = "away" if "away" in df.columns else "away_team"
            
            if home_col not in df.columns or away_col not in df.columns:
                logger.debug(f"[cfbfastR] PBP columns: {df.columns.tolist()}")
                return None
            
            agg_dict = {
                home_col: "first",
                away_col: "first",
            }
            
            # Add optional columns if they exist
            for col in ["game_date", "season", "week", "season_type", "home_score", "away_score"]:
                if col in df.columns:
                    agg_dict[col] = "first" if col not in ["home_score", "away_score"] else "max"
            
            games = df.groupby("game_id").agg(agg_dict).reset_index()
            
            # Rename columns to match schedule format
            games = games.rename(columns={
                home_col: "home_team",
                away_col: "away_team",
            })
            
            logger.info(f"[cfbfastR] Extracted {len(games)} games from {year} PBP")
            return games
            
        except Exception as e:
            logger.debug(f"[cfbfastR] PBP extraction failed for {year}: {e}")
            return None
    
    # =========================================================================
    # PLAY-BY-PLAY DATA (THE GOLD)
    # =========================================================================
    
    async def collect_pbp(
        self,
        years: List[int] = None,
        columns: List[str] = None,
        save_to_disk: bool = True,
    ) -> CollectorResult:
        """
        Collect play-by-play data with EPA, WPA, and 70+ features.
        
        WARNING: Large files (~300MB per season compressed)
        
        Args:
            years: List of years (2002-present)
            columns: Specific columns to load (None = key columns only)
            save_to_disk: Save parquet files locally for faster reuse
            
        Returns:
            CollectorResult with PBP summary info
        """
        current_year = datetime.now().year
        if years is None:
            years = [current_year]
        
        # Ensure valid years (cfbfastR starts at 2002)
        years = [y for y in years if 2002 <= y <= current_year]
        
        if columns is None:
            columns = CFB_PBP_KEY_COLUMNS
        
        results = []
        errors = []
        
        for year in years:
            try:
                # Check if cached first
                cache_path = self.data_dir / f"pbp_{year}.parquet"
                
                if cache_path.exists() and not save_to_disk:
                    logger.info(f"[cfbfastR] Loading cached PBP for {year}")
                    available_cols = pd.read_parquet(cache_path, columns=None).columns.tolist()
                    load_cols = [c for c in columns if c in available_cols]
                    df = pd.read_parquet(cache_path, columns=load_cols if load_cols else None)
                else:
                    # Try newer sportsdataverse-data first (2021+)
                    df = None
                    if year >= 2021:
                        url = CFBFASTR_URLS["pbp_parquet_new"].format(year=year)
                        logger.info(f"[cfbfastR] Trying sportsdataverse-data for {year}...")
                        df = await self._download_parquet(url, columns=columns)
                    
                    # Fall back to cfbfastR-data repo (2002-2020)
                    if df is None:
                        url = CFBFASTR_URLS["pbp_parquet_old"].format(year=year)
                        logger.info(f"[cfbfastR] Trying cfbfastR-data for {year}...")
                        df = await self._download_parquet(url, columns=columns)
                    
                    # Try CSV as last resort
                    if df is None:
                        url = CFBFASTR_URLS["pbp_csv"].format(year=year)
                        logger.info(f"[cfbfastR] Trying CSV for {year}...")
                        df = await self._download_csv(url)
                    
                    if df is not None and save_to_disk:
                        df.to_parquet(cache_path)
                        logger.info(f"[cfbfastR] Cached PBP {year} to {cache_path}")
                
                if df is not None and len(df) > 0:
                    results.append({
                        "year": year,
                        "plays": len(df),
                        "columns": len(df.columns),
                        "size_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
                    })
                    logger.info(f"[cfbfastR] {year}: {len(df):,} plays loaded")
                else:
                    logger.warning(f"[cfbfastR] No PBP data for {year}")
                    
            except Exception as e:
                logger.error(f"[cfbfastR] PBP {year} error: {e}")
                errors.append(f"{year}: {str(e)[:50]}")
        
        total_plays = sum(r["plays"] for r in results)
        
        return CollectorResult(
            success=len(results) > 0,
            data=results,
            records_count=total_plays,
            error="; ".join(errors) if errors else None,
            metadata={"years": years, "type": "play_by_play"},
        )
    
    async def get_team_epa(
        self,
        season: int = None,
        weeks: List[int] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate team EPA statistics from play-by-play data.
        
        Returns dict like:
        {
            "Alabama": {
                "pass_epa_per_play": 0.15,
                "rush_epa_per_play": 0.08,
                "total_epa_per_play": 0.12,
                "success_rate": 0.52,
                "explosiveness": 1.25,
                "plays": 1024
            },
            ...
        }
        """
        if season is None:
            season = datetime.now().year
        
        team_epa = {}
        
        try:
            # Check cache first
            cache_path = self.data_dir / f"pbp_{season}.parquet"
            
            if cache_path.exists():
                logger.info(f"[cfbfastR] Loading cached PBP for EPA calculation")
                cols = ["pos_team", "play_type", "EPA", "success", "week", "explosiveness"]
                available = pd.read_parquet(cache_path, columns=None).columns.tolist()
                load_cols = [c for c in cols if c in available]
                pbp = pd.read_parquet(cache_path, columns=load_cols)
            else:
                # Download - select URL based on year
                if season >= 2021:
                    url = CFBFASTR_URLS["pbp_parquet_new"].format(year=season)
                else:
                    url = CFBFASTR_URLS["pbp_parquet_old"].format(year=season)
                cols = ["pos_team", "play_type", "EPA", "success", "week", "explosiveness"]
                pbp = await self._download_parquet(url, columns=cols)
            
            if pbp is None:
                return team_epa
            
            # Normalize column names
            pbp.columns = pbp.columns.str.lower()
            
            # Filter weeks if specified
            if weeks and "week" in pbp.columns:
                pbp = pbp[pbp["week"].isin(weeks)]
            
            # Filter to pass/run plays only
            if "play_type" in pbp.columns:
                pbp = pbp[pbp["play_type"].isin(["pass", "rush", "Pass", "Rush"])]
            
            # Get EPA column (might be named differently)
            epa_col = "epa" if "epa" in pbp.columns else "EPA" if "EPA" in pbp.columns else None
            team_col = "pos_team" if "pos_team" in pbp.columns else None
            
            if epa_col is None or team_col is None:
                logger.warning(f"[cfbfastR] Required columns not found for EPA calculation")
                return team_epa
            
            pbp = pbp[pbp[team_col].notna()]
            
            # Aggregate by team
            for team in pbp[team_col].unique():
                team_plays = pbp[pbp[team_col] == team]
                
                if "play_type" in pbp.columns:
                    pass_plays = team_plays[team_plays["play_type"].isin(["pass", "Pass"])]
                    rush_plays = team_plays[team_plays["play_type"].isin(["rush", "Rush"])]
                else:
                    pass_plays = pd.DataFrame()
                    rush_plays = pd.DataFrame()
                
                team_epa[team] = {
                    "pass_epa_per_play": float(pass_plays[epa_col].mean()) if len(pass_plays) > 0 else 0.0,
                    "rush_epa_per_play": float(rush_plays[epa_col].mean()) if len(rush_plays) > 0 else 0.0,
                    "total_epa_per_play": float(team_plays[epa_col].mean()) if len(team_plays) > 0 else 0.0,
                    "success_rate": float((team_plays["success"] == 1).mean()) if "success" in team_plays.columns and len(team_plays) > 0 else 0.0,
                    "explosiveness": float(team_plays["explosiveness"].mean()) if "explosiveness" in team_plays.columns and len(team_plays) > 0 else 0.0,
                    "plays": len(team_plays),
                }
            
            logger.info(f"[cfbfastR] Calculated EPA for {len(team_epa)} teams")
            
        except Exception as e:
            logger.error(f"[cfbfastR] EPA calculation error: {e}")
        
        return team_epa
    
    # =========================================================================
    # SP+ RATINGS (Bill Connelly's Advanced Metrics)
    # =========================================================================
    
    async def _collect_sp_ratings(self, years: List[int]) -> List[Dict[str, Any]]:
        """
        Collect SP+ ratings - the gold standard for CFB team evaluation.
        
        NOTE: SP+ ratings are NOT available in the cfbfastR-data repository.
        They require the CollegeFootballData API (CFBD_API_KEY).
        This method is a placeholder for future API integration.
        
        SP+ Components:
        - Overall rating
        - Offensive rating
        - Defensive rating  
        - Special teams rating
        - Second-order wins
        - Strength of schedule
        """
        logger.warning("[cfbfastR] SP+ ratings not available in cfbfastR-data repo. "
                      "These require the CollegeFootballData API.")
        return []
    
    def _parse_sp_rating(self, row, year: int) -> Optional[Dict[str, Any]]:
        """Parse SP+ rating row."""
        try:
            team = row.get("team") or row.get("school")
            if pd.isna(team):
                return None
            
            return {
                "team": str(team),
                "season": year,
                "conference": row.get("conference"),
                # Overall SP+ rating
                "sp_overall": float(row.get("rating") or row.get("overall") or 0) if pd.notna(row.get("rating") or row.get("overall")) else None,
                # Offensive SP+
                "sp_offense": float(row.get("offense.rating") or row.get("offense") or 0) if pd.notna(row.get("offense.rating") or row.get("offense")) else None,
                # Defensive SP+
                "sp_defense": float(row.get("defense.rating") or row.get("defense") or 0) if pd.notna(row.get("defense.rating") or row.get("defense")) else None,
                # Special teams
                "sp_special_teams": float(row.get("specialTeams.rating") or row.get("specialTeams") or 0) if pd.notna(row.get("specialTeams.rating") or row.get("specialTeams")) else None,
                # Rankings
                "ranking": int(row.get("ranking") or row.get("rank") or 0) if pd.notna(row.get("ranking") or row.get("rank")) else None,
                "offense_ranking": int(row.get("offense.ranking") or 0) if pd.notna(row.get("offense.ranking")) else None,
                "defense_ranking": int(row.get("defense.ranking") or 0) if pd.notna(row.get("defense.ranking")) else None,
                # Second order wins (expected wins based on performance)
                "second_order_wins": float(row.get("secondOrderWins") or 0) if pd.notna(row.get("secondOrderWins")) else None,
                # Strength of schedule
                "sos": float(row.get("sos") or 0) if pd.notna(row.get("sos")) else None,
            }
            
        except Exception as e:
            logger.debug(f"[cfbfastR] SP+ parse error: {e}")
            return None
    
    # =========================================================================
    # RECRUITING DATA
    # =========================================================================
    
    async def _collect_recruiting(self, years: List[int]) -> List[Dict[str, Any]]:
        """
        Collect team recruiting rankings.
        
        NOTE: Recruiting data is NOT available in the cfbfastR-data repository.
        It requires the CollegeFootballData API (CFBD_API_KEY).
        This method is a placeholder for future API integration.
        
        Includes:
        - Team ranking (247Sports Composite)
        - Total points
        - 5-star, 4-star, 3-star counts
        - Average player rating
        """
        logger.warning("[cfbfastR] Recruiting data not available in cfbfastR-data repo. "
                      "This requires the CollegeFootballData API.")
        return []
    
    def _parse_recruiting(self, row, year: int) -> Optional[Dict[str, Any]]:
        """Parse recruiting row."""
        try:
            team = row.get("team") or row.get("school")
            if pd.isna(team):
                return None
            
            return {
                "team": str(team),
                "year": year,
                "rank": int(row.get("rank") or 0) if pd.notna(row.get("rank")) else None,
                "points": float(row.get("points") or 0) if pd.notna(row.get("points")) else None,
                "total_commits": int(row.get("total") or row.get("commits") or 0) if pd.notna(row.get("total") or row.get("commits")) else None,
                "five_star": int(row.get("5star") or row.get("fiveStars") or 0) if pd.notna(row.get("5star") or row.get("fiveStars")) else 0,
                "four_star": int(row.get("4star") or row.get("fourStars") or 0) if pd.notna(row.get("4star") or row.get("fourStars")) else 0,
                "three_star": int(row.get("3star") or row.get("threeStars") or 0) if pd.notna(row.get("3star") or row.get("threeStars")) else 0,
                "avg_rating": float(row.get("averageRating") or row.get("avg_rating") or 0) if pd.notna(row.get("averageRating") or row.get("avg_rating")) else None,
            }
            
        except Exception as e:
            logger.debug(f"[cfbfastR] Recruiting parse error: {e}")
            return None
    
    # =========================================================================
    # BETTING LINES
    # =========================================================================
    
    async def _collect_betting_lines(self, years: List[int]) -> List[Dict[str, Any]]:
        """Collect betting lines for games."""
        betting_lines = []
        
        for year in years:
            try:
                url = CFBFASTR_URLS["betting_parquet"].format(year=year)
                df = await self._download_parquet(url)
                
                if df is None or len(df) == 0:
                    url = CFBFASTR_URLS["betting_csv"].format(year=year)
                    df = await self._download_csv(url)
                
                if df is not None and len(df) > 0:
                    for _, row in df.iterrows():
                        line = self._parse_betting_line(row, year)
                        if line:
                            betting_lines.append(line)
                    
                    logger.info(f"[cfbfastR] {year}: {len(df)} betting lines")
                    
            except Exception as e:
                logger.debug(f"[cfbfastR] Betting {year} error: {e}")
        
        return betting_lines
    
    def _parse_betting_line(self, row, year: int) -> Optional[Dict[str, Any]]:
        """Parse betting line row."""
        try:
            game_id = row.get("game_id") or row.get("id")
            if pd.isna(game_id):
                return None
            
            return {
                "game_id": str(game_id),
                "season": year,
                "provider": row.get("provider") or "consensus",
                "spread": float(row.get("spread") or 0) if pd.notna(row.get("spread")) else None,
                "spread_open": float(row.get("spread_open") or 0) if pd.notna(row.get("spread_open")) else None,
                "over_under": float(row.get("over_under") or row.get("overUnder") or 0) if pd.notna(row.get("over_under") or row.get("overUnder")) else None,
                "over_under_open": float(row.get("over_under_open") or 0) if pd.notna(row.get("over_under_open")) else None,
                "home_moneyline": int(row.get("home_moneyline") or row.get("homeMoneyline") or 0) if pd.notna(row.get("home_moneyline") or row.get("homeMoneyline")) else None,
                "away_moneyline": int(row.get("away_moneyline") or row.get("awayMoneyline") or 0) if pd.notna(row.get("away_moneyline") or row.get("awayMoneyline")) else None,
            }
            
        except Exception as e:
            logger.debug(f"[cfbfastR] Betting parse error: {e}")
            return None
    
    # =========================================================================
    # TEAM STATS
    # =========================================================================
    
    async def _collect_team_stats(self, years: List[int]) -> List[Dict[str, Any]]:
        """
        Collect team statistics.
        
        NOTE: Team stats are NOT available in the cfbfastR-data repository as separate files.
        Team statistics can be derived from PBP data using get_team_epa().
        """
        logger.warning("[cfbfastR] Team stats not available as separate files. "
                      "Use get_team_epa() to calculate from PBP data.")
        return []
    
    # =========================================================================
    # PLAYER STATS
    # =========================================================================
    
    async def _collect_player_stats(self, years: List[int]) -> List[Dict[str, Any]]:
        """Collect player statistics."""
        player_stats = []
        
        for year in years:
            try:
                url = CFBFASTR_URLS["player_stats_parquet"].format(year=year)
                df = await self._download_parquet(url)
                
                if df is None or len(df) == 0:
                    url = CFBFASTR_URLS["player_stats_csv"].format(year=year)
                    df = await self._download_csv(url)
                
                if df is not None and len(df) > 0:
                    for _, row in df.iterrows():
                        stats = row.to_dict()
                        stats["season"] = year
                        player_stats.append(stats)
                    
                    logger.info(f"[cfbfastR] {year}: {len(df)} player stats")
                    
            except Exception as e:
                logger.debug(f"[cfbfastR] Player stats {year} error: {e}")
        
        return player_stats
    
    # =========================================================================
    # HISTORICAL DATA COLLECTION
    # =========================================================================
    
    async def collect_historical(
        self,
        start_year: int = 2014,
        end_year: int = None,
        include_pbp: bool = False,
        **kwargs,
    ) -> CollectorResult:
        """
        Collect historical NCAAF data.
        
        Args:
            start_year: First year (2002 minimum for cfbfastR-data)
            end_year: Last year (default: current year)
            include_pbp: Download full PBP data (WARNING: large files)
            
        Returns:
            CollectorResult with historical data
        """
        if end_year is None:
            end_year = datetime.now().year
        
        # Ensure valid range for cfbfastR-data (2002+)
        start_year = max(start_year, 2002)
        
        years = list(range(start_year, end_year + 1))
        
        logger.info(f"[cfbfastR] Collecting {len(years)} years of data ({start_year}-{end_year})")
        
        result = await self.collect(
            sport_code="NCAAF",
            collect_type="all",
            years=years,
        )
        
        if include_pbp:
            pbp_result = await self.collect_pbp(years=years, save_to_disk=True)
            result.metadata["pbp_summary"] = pbp_result.data
        
        return result
    
    # =========================================================================
    # FILE DOWNLOAD HELPERS
    # =========================================================================
    
    async def _download_parquet(
        self,
        url: str,
        columns: List[str] = None,
    ) -> Optional[pd.DataFrame]:
        """Download and parse a parquet file from URL."""
        try:
            client = await self.get_client()
            
            logger.debug(f"[cfbfastR] Downloading: {url}")
            response = await client.get(url, follow_redirects=True, timeout=180.0)
            response.raise_for_status()
            
            # Read parquet from bytes
            import pyarrow.parquet as pq
            
            buffer = BytesIO(response.content)
            table = pq.read_table(buffer)
            
            # Get available columns
            available_cols = table.column_names
            
            if columns:
                # Filter to available columns
                load_cols = [c for c in columns if c in available_cols]
                if load_cols:
                    table = table.select(load_cols)
            
            df = table.to_pandas()
            
            logger.debug(f"[cfbfastR] Downloaded {len(df):,} rows")
            return df
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.debug(f"[cfbfastR] File not found: {url}")
            else:
                logger.error(f"[cfbfastR] HTTP error for {url}: {e}")
            return None
        except Exception as e:
            logger.debug(f"[cfbfastR] Parquet download error for {url}: {e}")
            return None
    
    async def _download_csv(self, url: str) -> Optional[pd.DataFrame]:
        """Download and parse a CSV file from URL."""
        try:
            client = await self.get_client()
            response = await client.get(url, follow_redirects=True, timeout=120.0)
            response.raise_for_status()
            
            df = pd.read_csv(BytesIO(response.content))
            
            return df
            
        except Exception as e:
            logger.debug(f"[cfbfastR] CSV download error for {url}: {e}")
            return None
    
    # =========================================================================
    # DATABASE OPERATIONS
    # =========================================================================
    
    async def save_to_database(
        self,
        data: Dict[str, List[Dict]],
        session: AsyncSession,
    ) -> int:
        """Save collected data to database."""
        total_saved = 0
        
        if data.get("games"):
            saved = await self._save_games(data["games"], session)
            total_saved += saved
        
        return total_saved
    
    async def _save_games(
        self,
        games_data: List[Dict[str, Any]],
        session: AsyncSession,
    ) -> int:
        """Save game records to database."""
        saved_count = 0
        
        # Get or create NCAAF sport
        sport_result = await session.execute(
            select(Sport).where(Sport.code == "NCAAF")
        )
        sport = sport_result.scalar_one_or_none()
        
        if not sport:
            logger.error("[cfbfastR] NCAAF sport not found in database")
            return 0
        
        for game_data in games_data:
            try:
                # Get or create teams
                home_team = await self._get_or_create_team(
                    session, sport.id, game_data["home_team"]
                )
                away_team = await self._get_or_create_team(
                    session, sport.id, game_data["away_team"]
                )
                
                if not home_team or not away_team:
                    continue
                
                external_id = game_data.get("external_id")
                
                # Check if game exists by external_id
                existing = await session.execute(
                    select(Game).where(Game.external_id == external_id)
                )
                game = existing.scalars().first()
                
                if game:
                    # Update scores
                    if game_data.get("home_score") is not None:
                        game.home_score = game_data["home_score"]
                    if game_data.get("away_score") is not None:
                        game.away_score = game_data["away_score"]
                    if game_data.get("status"):
                        game.status = GameStatus(game_data["status"])
                else:
                    # Parse date
                    game_date_str = game_data.get("game_date")
                    if not game_date_str:
                        continue
                    
                    if isinstance(game_date_str, datetime):
                        scheduled_dt = game_date_str
                    else:
                        scheduled_dt = datetime.fromisoformat(game_date_str.replace("Z", "+00:00"))
                    
                    if scheduled_dt.tzinfo:
                        scheduled_dt = scheduled_dt.replace(tzinfo=None)
                    
                    # Check for duplicates by teams and date
                    date_start = scheduled_dt - timedelta(hours=12)
                    date_end = scheduled_dt + timedelta(hours=12)
                    
                    dup_check = await session.execute(
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
                    existing_game = dup_check.scalars().first()
                    
                    if existing_game:
                        # Update
                        if game_data.get("home_score") is not None:
                            existing_game.home_score = game_data["home_score"]
                        if game_data.get("away_score") is not None:
                            existing_game.away_score = game_data["away_score"]
                        if external_id and not existing_game.external_id:
                            existing_game.external_id = external_id
                    else:
                        # Create new
                        game = Game(
                            sport_id=sport.id,
                            external_id=external_id,
                            home_team_id=home_team.id,
                            away_team_id=away_team.id,
                            scheduled_at=scheduled_dt,
                            status=GameStatus(game_data.get("status", "scheduled")),
                            home_score=game_data.get("home_score"),
                            away_score=game_data.get("away_score"),
                        )
                        session.add(game)
                        saved_count += 1
                        
            except Exception as e:
                logger.error(f"[cfbfastR] Error saving game: {e}")
                continue
        
        await session.commit()
        logger.info(f"[cfbfastR] Saved {saved_count} games to database")
        
        return saved_count
    
    async def _get_or_create_team(
        self,
        session: AsyncSession,
        sport_id: UUID,
        team_data: Dict[str, Any],
    ) -> Optional[Team]:
        """Get or create team record."""
        team_name = team_data.get("name")
        abbreviation = team_data.get("abbreviation")
        
        if not team_name and not abbreviation:
            return None
        
        # Try by name
        if team_name:
            result = await session.execute(
                select(Team).where(
                    and_(
                        Team.sport_id == sport_id,
                        Team.name == team_name,
                    )
                )
            )
            team = result.scalar_one_or_none()
            if team:
                return team
        
        # Try by abbreviation
        if abbreviation:
            result = await session.execute(
                select(Team).where(
                    and_(
                        Team.sport_id == sport_id,
                        Team.abbreviation == abbreviation,
                    )
                )
            )
            team = result.scalar_one_or_none()
            if team:
                return team
        
        # Determine conference
        conference = None
        for conf, teams in CFB_CONFERENCES.items():
            for t in teams:
                if team_name and (t.lower() in team_name.lower() or team_name.lower() in t.lower()):
                    conference = conf
                    break
            if conference:
                break
        
        # Create new
        team = Team(
            sport_id=sport_id,
            external_id=f"cfb_{abbreviation}" if abbreviation else None,
            name=team_name,
            abbreviation=abbreviation,
            conference=conference,
            is_active=True,
        )
        session.add(team)
        await session.flush()
        
        return team
    
    async def save_historical_to_database(
        self,
        games_data: List[Dict[str, Any]],
        session: AsyncSession,
    ) -> Tuple[int, int]:
        """Save historical games to database."""
        saved = await self._save_games(games_data, session)
        return saved, 0
    
    async def validate(self, data: Any) -> bool:
        """Validate collected data."""
        if not isinstance(data, dict):
            return False
        return True


# =============================================================================
# SINGLETON & REGISTRATION
# =============================================================================

cfbfastr_collector = CFBFastRCollector()

try:
    collector_manager.register("cfbfastr", cfbfastr_collector)
except:
    pass  # Already registered
