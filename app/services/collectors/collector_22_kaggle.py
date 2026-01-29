"""
ROYALEY - Kaggle Datasets Collector
Phase 1: Data Collection Services

Collector 22: Kaggle sports datasets for backtesting and model training.

Downloads comprehensive historical sports data from Kaggle datasets:
- NFL: Games, scores, betting data, player stats, injuries
- NBA: Games, odds, player stats, betting data, injuries
- MLB: Game data, player stats
- NHL: Game data, player stats
- Soccer: Match results, betting odds
- College Football: Game stats, betting data
- College Basketball: Game stats
- Tennis: Match results
- UFC/MMA: Fight data, odds

INJURY DATASETS (10+ years historical):
- NBA: loganlauton/nba-injury-stats-1951-2023 (70+ years, ~50,000 records)
- NBA: ghopkins/nba-injuries-2010-2018 (detailed, ~5,000 records)
- NFL: thedevastator/nfl-injury-analysis-2012-2017 (~8,000 records)
- NFL: rishidamarla/concussions-in-the-nfl-20122014 (~500 records)

Data Source: Kaggle Datasets (https://www.kaggle.com/datasets)
API Documentation: https://github.com/Kaggle/kaggle-api

REQUIRES Kaggle API credentials:
- Create account at kaggle.com
- Generate API token at https://www.kaggle.com/account
- Save kaggle.json to ~/.kaggle/ or set KAGGLE_USERNAME and KAGGLE_KEY env vars

Tables Filled:
- sports - Sport definitions
- teams - Team info
- players - Player info
- games - Game records with scores
- odds - Historical betting odds
- player_stats - Player statistics
- team_stats - Team statistics
- seasons - Season definitions
- injuries - Historical injury data (NEW)

Usage:
    from app.services.collectors import kaggle_collector
    
    # Collect all sports datasets
    result = await kaggle_collector.collect()
    
    # Collect specific sport
    result = await kaggle_collector.collect(sports=["nfl"])
    
    # Collect historical injuries (NEW)
    result = await kaggle_collector.collect_injuries()
    result = await kaggle_collector.collect_injuries(sports=["NBA"])
    
    # Save to database
    await kaggle_collector.save_to_database(result.data)
"""

import asyncio
import logging
import os
import zipfile
import tempfile
import shutil
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4

import pandas as pd

from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import (
    Sport, Team, Season, Player, PlayerStats, TeamStats, 
    Game, GameStatus, Odds, Venue, Sportsbook
)
from app.services.collectors.base_collector import (
    BaseCollector,
    CollectorResult,
    collector_manager,
)

logger = logging.getLogger(__name__)


# =============================================================================
# KAGGLE DATASET CONFIGURATION
# =============================================================================

# Priority Kaggle datasets for sports betting/prediction
KAGGLE_DATASETS = {
    # NFL Datasets
    "nfl_betting": {
        "dataset": "tobycrabtree/nfl-scores-and-betting-data",
        "sport": "NFL",
        "description": "NFL scores and betting data 1966-present",
        "files": ["spreadspoke_scores.csv", "nfl_teams.csv", "nfl_stadiums.csv"],
        "data_types": ["games", "odds", "teams", "venues"],
    },
    "nfl_play_by_play": {
        "dataset": "maxhorowitz/nflplaybyplay2009to2016",
        "sport": "NFL",
        "description": "Detailed NFL play-by-play data 2009-2018",
        "files": [],  # Will use all CSV files
        "data_types": ["games", "team_stats"],
    },
    "nfl_game_data": {
        "dataset": "keonim/nfl-game-scores-dataset-2017-2023",
        "sport": "NFL", 
        "description": "NFL game scores and plays 2017-2025",
        "files": [],
        "data_types": ["games", "team_stats"],
    },
    
    # NBA Datasets
    "nba_betting": {
        "dataset": "ehallmar/nba-historical-stats-and-betting-data",
        "sport": "NBA",
        "description": "NBA historical stats and betting data",
        "files": [],
        "data_types": ["games", "odds", "player_stats"],
    },
    "nba_odds": {
        "dataset": "erichqiu/nba-odds-and-scores",
        "sport": "NBA",
        "description": "NBA odds and scores",
        "files": [],
        "data_types": ["games", "odds"],
    },
    "nba_database": {
        "dataset": "wyattowalsh/basketball",
        "sport": "NBA",
        "description": "Comprehensive NBA database",
        "files": [],
        "data_types": ["games", "players", "teams", "player_stats"],
    },
    
    # MLB Datasets  
    "mlb_games": {
        "dataset": "cristobalmitchell/mlb-game-data",
        "sport": "MLB",
        "description": "MLB game data 2010-2020",
        "files": [],
        "data_types": ["games", "team_stats"],
    },
    "mlb_players": {
        "dataset": "seanlahman/the-history-of-baseball",
        "sport": "MLB",
        "description": "Historical MLB player/team data",
        "files": [],
        "data_types": ["players", "teams", "player_stats"],
    },
    
    # NHL Datasets
    "nhl_game_data": {
        "dataset": "martinellis/nhl-game-data",
        "sport": "NHL",
        "description": "NHL game data",
        "files": [],
        "data_types": ["games", "teams", "players"],
    },
    
    # Soccer Datasets
    "soccer_betting": {
        "dataset": "mexwell/historical-football-resultsbetting-odds-data",
        "sport": "SOCCER",
        "description": "Historical football/soccer results and betting odds",
        "files": [],
        "data_types": ["games", "odds", "teams"],
    },
    
    # College Football
    "cfb_game_stats": {
        "dataset": "jeffgallini/college-football-team-stats-2019",
        "sport": "NCAAF",
        "description": "College football team stats",
        "files": [],
        "data_types": ["team_stats", "games"],
    },
    
    # College Basketball
    "ncaab_data": {
        "dataset": "andrewsundberg/college-basketball-dataset",
        "sport": "NCAAB",
        "description": "College basketball data",
        "files": [],
        "data_types": ["games", "teams", "team_stats"],
    },
    
    # Tennis
    "atp_tennis": {
        "dataset": "guillemservera/atp-tennis-grand-slam-data",
        "sport": "TENNIS",
        "description": "ATP Tennis Grand Slam data",
        "files": [],
        "data_types": ["games", "players", "player_stats"],
    },
    
    # UFC/MMA
    "ufc_data": {
        "dataset": "rajeevw/ufcdata",
        "sport": "MMA",
        "description": "UFC/MMA fight data with odds",
        "files": [],
        "data_types": ["games", "players", "odds"],
    },
}

# =============================================================================
# KAGGLE INJURY DATASETS - Historical injury data (10+ years)
# =============================================================================

KAGGLE_INJURY_DATASETS = {
    # NBA Injury Datasets
    "nba_injuries_historical": {
        "dataset": "loganlauton/nba-injury-stats-1951-2023",
        "sport": "NBA",
        "description": "NBA Injury Stats 1951-2023 (70+ years)",
        "data_types": ["injuries"],
        "expected_records": 50000,
    },
    "nba_injuries_detailed": {
        "dataset": "ghopkins/nba-injuries-2010-2018",
        "sport": "NBA",
        "description": "NBA Injuries 2010-2020 with game context",
        "data_types": ["injuries"],
        "expected_records": 5000,
    },
    
    # NFL Injury Datasets
    "nfl_injuries_analysis": {
        "dataset": "thedevastator/nfl-injury-analysis-2012-2017",
        "sport": "NFL",
        "description": "NFL Injury Analysis 2012-2017",
        "data_types": ["injuries"],
        "expected_records": 8000,
    },
    "nfl_concussions": {
        "dataset": "rishidamarla/concussions-in-the-nfl-20122014",
        "sport": "NFL",
        "description": "NFL Concussions 2012-2014",
        "data_types": ["injuries"],
        "expected_records": 500,
    },
}

# Sport configurations
SPORT_CONFIG = {
    "NFL": {"code": "NFL", "name": "National Football League", "season_start_month": 9, "season_end_month": 2},
    "NBA": {"code": "NBA", "name": "National Basketball Association", "season_start_month": 10, "season_end_month": 6},
    "MLB": {"code": "MLB", "name": "Major League Baseball", "season_start_month": 3, "season_end_month": 10},
    "NHL": {"code": "NHL", "name": "National Hockey League", "season_start_month": 10, "season_end_month": 6},
    "SOCCER": {"code": "SOCCER", "name": "International Soccer", "season_start_month": 8, "season_end_month": 5},
    "NCAAF": {"code": "NCAAF", "name": "NCAA Football", "season_start_month": 8, "season_end_month": 1},
    "NCAAB": {"code": "NCAAB", "name": "NCAA Basketball", "season_start_month": 11, "season_end_month": 4},
    "TENNIS": {"code": "TENNIS", "name": "Professional Tennis", "season_start_month": 1, "season_end_month": 12},
    "MMA": {"code": "MMA", "name": "Mixed Martial Arts", "season_start_month": 1, "season_end_month": 12},
}


# =============================================================================
# KAGGLE COLLECTOR CLASS
# =============================================================================

class KaggleCollector(BaseCollector):
    """
    Collector for Kaggle sports datasets.
    
    Downloads and processes multiple Kaggle datasets containing:
    - Historical game results
    - Betting odds/lines
    - Player statistics
    - Team statistics
    """
    
    def __init__(self):
        super().__init__(
            name="kaggle",
            base_url="https://www.kaggle.com/api/v1",
            rate_limit=10,  # Kaggle has stricter rate limits
            rate_window=60,
            timeout=300.0,  # Longer timeout for large downloads
        )
        self.kaggle_api = None
        self.download_path = Path(tempfile.gettempdir()) / "royaley_kaggle"
        self._ensure_download_dir()
        logger.info("Registered collector: Kaggle Datasets")
    
    def _ensure_download_dir(self):
        """Create download directory if it doesn't exist."""
        self.download_path.mkdir(parents=True, exist_ok=True)
    
    def _load_kaggle_api(self) -> bool:
        """
        Load and authenticate Kaggle API.
        
        Returns:
            True if API loaded successfully, False otherwise
        """
        if self.kaggle_api is not None:
            return True
        
        try:
            # Set credentials from environment if available
            kaggle_username = os.environ.get("KAGGLE_USERNAME", "royaley")
            kaggle_key = os.environ.get("KAGGLE_KEY")
            
            if kaggle_key:
                os.environ["KAGGLE_USERNAME"] = kaggle_username
                os.environ["KAGGLE_KEY"] = kaggle_key
            
            # Import and authenticate
            from kaggle.api.kaggle_api_extended import KaggleApi
            self.kaggle_api = KaggleApi()
            self.kaggle_api.authenticate()
            
            logger.info("[Kaggle] API authenticated successfully")
            return True
            
        except ImportError:
            logger.error("[Kaggle] kaggle package not installed. Install with: pip install kaggle")
            return False
        except Exception as e:
            logger.error(f"[Kaggle] Authentication failed: {e}")
            logger.info("[Kaggle] Please ensure kaggle.json is in ~/.kaggle/ or set KAGGLE_USERNAME and KAGGLE_KEY")
            return False
    
    async def validate(self, data: Any) -> bool:
        """Validate collected data."""
        if not data:
            return False
        return isinstance(data, dict) and len(data) > 0
    
    def _download_dataset(self, dataset_ref: str, force: bool = False) -> Optional[Path]:
        """
        Download a Kaggle dataset.
        
        Args:
            dataset_ref: Dataset reference (owner/dataset-name)
            force: Force re-download if exists
            
        Returns:
            Path to downloaded dataset directory or None if failed
        """
        if not self._load_kaggle_api():
            return None
        
        dataset_dir = self.download_path / dataset_ref.replace("/", "_")
        
        # Check if already downloaded
        if dataset_dir.exists() and not force:
            csv_files = list(dataset_dir.glob("*.csv"))
            if csv_files:
                logger.info(f"[Kaggle] Using cached dataset: {dataset_ref}")
                return dataset_dir
        
        try:
            # Download dataset
            logger.info(f"[Kaggle] Downloading dataset: {dataset_ref}")
            self.kaggle_api.dataset_download_files(
                dataset_ref,
                path=str(dataset_dir),
                unzip=True,
                force=force,
                quiet=False
            )
            
            logger.info(f"[Kaggle] Downloaded to: {dataset_dir}")
            return dataset_dir
            
        except Exception as e:
            logger.error(f"[Kaggle] Failed to download {dataset_ref}: {e}")
            return None
    
    def _read_csv_safely(self, filepath: Path) -> Optional[pd.DataFrame]:
        """
        Read CSV file with error handling.
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            DataFrame or None if failed
        """
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(filepath, encoding=encoding, low_memory=False)
                    logger.debug(f"[Kaggle] Read {filepath.name}: {len(df)} rows")
                    return df
                except UnicodeDecodeError:
                    continue
            
            logger.warning(f"[Kaggle] Could not decode {filepath.name}")
            return None
            
        except Exception as e:
            logger.error(f"[Kaggle] Error reading {filepath.name}: {e}")
            return None
    
    async def collect(
        self,
        sports: Optional[List[str]] = None,
        datasets: Optional[List[str]] = None,
        years_back: int = 10,
        force_download: bool = False,
        **kwargs
    ) -> CollectorResult:
        """
        Collect data from Kaggle datasets.
        
        Args:
            sports: List of sports to collect (None = all)
            datasets: Specific dataset keys to collect (None = all)
            years_back: Number of years of historical data
            force_download: Force re-download of datasets
            
        Returns:
            CollectorResult with collected data
        """
        if not self._load_kaggle_api():
            return CollectorResult(
                success=False,
                error="Kaggle API not available. Install kaggle and configure credentials.",
                records_count=0
            )
        
        # Filter datasets by sport if specified
        target_datasets = KAGGLE_DATASETS
        if sports:
            sports_upper = [s.upper() for s in sports]
            target_datasets = {
                k: v for k, v in KAGGLE_DATASETS.items()
                if v["sport"] in sports_upper
            }
        
        if datasets:
            target_datasets = {
                k: v for k, v in target_datasets.items()
                if k in datasets
            }
        
        logger.info(f"[Kaggle] Collecting {len(target_datasets)} datasets...")
        
        all_data = {
            "games": [],
            "odds": [],
            "teams": [],
            "players": [],
            "player_stats": [],
            "team_stats": [],
            "venues": [],
            "injuries": [],
        }
        
        total_records = 0
        errors = []
        
        for dataset_key, config in target_datasets.items():
            try:
                logger.info(f"[Kaggle] Processing: {dataset_key} ({config['sport']})")
                
                # Download dataset
                dataset_dir = self._download_dataset(config["dataset"], force=force_download)
                if not dataset_dir:
                    errors.append(f"Failed to download {dataset_key}")
                    continue
                
                # Find CSV files
                csv_files = list(dataset_dir.glob("**/*.csv"))
                if not csv_files:
                    logger.warning(f"[Kaggle] No CSV files found in {dataset_key}")
                    continue
                
                # Process each CSV file
                for csv_file in csv_files:
                    df = self._read_csv_safely(csv_file)
                    if df is None or df.empty:
                        continue
                    
                    # Parse based on dataset type
                    parsed = await self._parse_dataset(
                        df=df,
                        dataset_key=dataset_key,
                        config=config,
                        filename=csv_file.name,
                        years_back=years_back
                    )
                    
                    # Merge parsed data
                    for key in all_data:
                        if key in parsed:
                            all_data[key].extend(parsed[key])
                            total_records += len(parsed[key])
                
                logger.info(f"[Kaggle] Completed {dataset_key}: {total_records} total records")
                
                # Small delay between datasets
                await asyncio.sleep(1.0)
                
            except Exception as e:
                logger.error(f"[Kaggle] Error processing {dataset_key}: {e}")
                errors.append(str(e))
        
        logger.info(f"[Kaggle] Collection complete: {total_records} total records")
        
        return CollectorResult(
            success=total_records > 0,
            data=all_data,
            records_count=total_records,
            error="; ".join(errors) if errors else None,
            metadata={
                "datasets_processed": len(target_datasets),
                "sports": list(set(d["sport"] for d in target_datasets.values())),
            }
        )
    
    async def _parse_dataset(
        self,
        df: pd.DataFrame,
        dataset_key: str,
        config: Dict,
        filename: str,
        years_back: int
    ) -> Dict[str, List[Dict]]:
        """
        Parse a dataset DataFrame based on its type.
        
        Args:
            df: DataFrame to parse
            dataset_key: Key identifying the dataset
            config: Dataset configuration
            filename: Original filename
            years_back: Years of historical data to include
            
        Returns:
            Dict with parsed data by category
        """
        result = {
            "games": [],
            "odds": [],
            "teams": [],
            "players": [],
            "player_stats": [],
            "team_stats": [],
            "venues": [],
            "injuries": [],
        }
        
        sport = config["sport"]
        min_year = datetime.now().year - years_back
        
        # Route to appropriate parser
        if dataset_key == "nfl_betting":
            if "score" in filename.lower():
                result = self._parse_nfl_betting_scores(df, sport, min_year)
            elif "team" in filename.lower():
                result["teams"] = self._parse_nfl_teams(df, sport)
            elif "stadium" in filename.lower():
                result["venues"] = self._parse_nfl_stadiums(df)
                
        elif dataset_key in ["nba_betting", "nba_odds"]:
            result = self._parse_nba_betting(df, sport, min_year)
            
        elif dataset_key == "nba_database":
            result = self._parse_nba_database(df, filename, sport, min_year)
            
        elif dataset_key in ["mlb_games", "mlb_players"]:
            result = self._parse_mlb_data(df, filename, sport, min_year)
            
        elif dataset_key == "nhl_game_data":
            result = self._parse_nhl_data(df, filename, sport, min_year)
            
        elif dataset_key == "soccer_betting":
            result = self._parse_soccer_betting(df, sport, min_year)
            
        elif dataset_key in ["cfb_game_stats", "ncaab_data"]:
            result = self._parse_college_data(df, sport, min_year)
            
        elif dataset_key == "atp_tennis":
            result = self._parse_tennis_data(df, sport, min_year)
            
        elif dataset_key == "ufc_data":
            result = self._parse_ufc_data(df, sport, min_year)
        
        # Injury datasets
        elif dataset_key in ["nba_injuries_historical", "nba_injuries_detailed"]:
            result["injuries"] = self._parse_nba_injuries(df, sport)
            
        elif dataset_key in ["nfl_injuries_analysis", "nfl_concussions"]:
            result["injuries"] = self._parse_nfl_injuries(df, sport)
            
        else:
            # Generic parser
            result = self._parse_generic(df, sport, min_year)
        
        return result
    
    # =========================================================================
    # NFL PARSERS
    # =========================================================================
    
    def _parse_nfl_betting_scores(self, df: pd.DataFrame, sport: str, min_year: int) -> Dict[str, List[Dict]]:
        """Parse NFL scores and betting data."""
        result = {"games": [], "odds": [], "teams": [], "team_stats": []}
        
        # Expected columns: schedule_date, schedule_season, team_home, team_away, 
        # score_home, score_away, spread_favorite, over_under_line
        
        # Standardize column names
        df.columns = df.columns.str.lower().str.strip()
        
        # Date column detection
        date_col = None
        for col in ['schedule_date', 'date', 'game_date', 'gamedate']:
            if col in df.columns:
                date_col = col
                break
        
        # Season/year column detection
        year_col = None
        for col in ['schedule_season', 'season', 'year', 'schedule_year']:
            if col in df.columns:
                year_col = col
                break
        
        if not date_col and not year_col:
            logger.warning("[Kaggle] No date/season column found in NFL betting data")
            return result
        
        teams_seen = set()
        
        for _, row in df.iterrows():
            try:
                # Get year
                year = None
                if year_col and pd.notna(row.get(year_col)):
                    year = int(row[year_col])
                elif date_col and pd.notna(row.get(date_col)):
                    try:
                        game_date = pd.to_datetime(row[date_col])
                        year = game_date.year
                    except:
                        continue
                
                if not year or year < min_year:
                    continue
                
                # Get teams
                home_team = row.get('team_home', row.get('home_team', row.get('home')))
                away_team = row.get('team_away', row.get('away_team', row.get('away')))
                
                if pd.isna(home_team) or pd.isna(away_team):
                    continue
                
                home_team = str(home_team).strip()
                away_team = str(away_team).strip()
                
                # Track teams
                teams_seen.add(home_team)
                teams_seen.add(away_team)
                
                # Get scores
                home_score = row.get('score_home', row.get('home_score', row.get('pts_home')))
                away_score = row.get('score_away', row.get('away_score', row.get('pts_away')))
                
                # Parse game date
                game_date = None
                if date_col and pd.notna(row.get(date_col)):
                    try:
                        game_date = pd.to_datetime(row[date_col])
                    except:
                        game_date = datetime(year, 9, 1)  # Default to season start
                else:
                    game_date = datetime(year, 9, 1)
                
                # Create game record
                game = {
                    "sport": sport,
                    "home_team": home_team,
                    "away_team": away_team,
                    "scheduled_at": game_date.isoformat() if game_date else None,
                    "home_score": int(home_score) if pd.notna(home_score) else None,
                    "away_score": int(away_score) if pd.notna(away_score) else None,
                    "status": "final" if pd.notna(home_score) else "scheduled",
                    "season_year": year,
                    "external_id": f"kaggle_nfl_{year}_{home_team}_{away_team}_{game_date.strftime('%Y%m%d') if game_date else ''}",
                }
                result["games"].append(game)
                
                # Parse odds if available
                spread = row.get('spread_favorite', row.get('spread', row.get('spread_line')))
                total = row.get('over_under_line', row.get('total', row.get('over_under')))
                favorite = row.get('team_favorite_id', row.get('favorite', row.get('team_favorite')))
                
                if pd.notna(spread) or pd.notna(total):
                    odds_record = {
                        "game_external_id": game["external_id"],
                        "sport": sport,
                        "sportsbook": "market_consensus",
                        "bet_type": "spread",
                        "home_line": float(spread) if pd.notna(spread) and str(favorite) == str(home_team) else (float(spread) * -1 if pd.notna(spread) else None),
                        "away_line": float(spread) if pd.notna(spread) and str(favorite) == str(away_team) else (float(spread) * -1 if pd.notna(spread) else None),
                        "total": float(total) if pd.notna(total) else None,
                        "recorded_at": game_date.isoformat() if game_date else None,
                    }
                    result["odds"].append(odds_record)
                    
            except Exception as e:
                logger.debug(f"[Kaggle] Error parsing NFL row: {e}")
                continue
        
        # Create team records
        for team_name in teams_seen:
            result["teams"].append({
                "sport": sport,
                "name": team_name,
                "abbreviation": self._get_team_abbrev(team_name, sport),
                "external_id": f"kaggle_{sport.lower()}_{team_name.lower().replace(' ', '_')}",
            })
        
        logger.info(f"[Kaggle] Parsed NFL betting: {len(result['games'])} games, {len(result['odds'])} odds, {len(result['teams'])} teams")
        return result
    
    def _parse_nfl_teams(self, df: pd.DataFrame, sport: str) -> List[Dict]:
        """Parse NFL teams data."""
        teams = []
        df.columns = df.columns.str.lower().str.strip()
        
        for _, row in df.iterrows():
            try:
                name = row.get('team_name', row.get('name', row.get('team')))
                if pd.isna(name):
                    continue
                
                teams.append({
                    "sport": sport,
                    "name": str(name).strip(),
                    "abbreviation": str(row.get('team_id', row.get('abbreviation', ''))).strip()[:10] or self._get_team_abbrev(str(name), sport),
                    "city": str(row.get('team_city', row.get('city', ''))).strip() if pd.notna(row.get('team_city', row.get('city'))) else None,
                    "conference": str(row.get('team_conference', row.get('conference', ''))).strip() if pd.notna(row.get('team_conference', row.get('conference'))) else None,
                    "division": str(row.get('team_division', row.get('division', ''))).strip() if pd.notna(row.get('team_division', row.get('division'))) else None,
                    "external_id": f"kaggle_{sport.lower()}_{str(name).lower().replace(' ', '_')}",
                })
            except Exception as e:
                logger.debug(f"[Kaggle] Error parsing NFL team: {e}")
                continue
        
        logger.info(f"[Kaggle] Parsed {len(teams)} NFL teams")
        return teams
    
    def _parse_nfl_stadiums(self, df: pd.DataFrame) -> List[Dict]:
        """Parse NFL stadiums/venues data."""
        venues = []
        df.columns = df.columns.str.lower().str.strip()
        
        for _, row in df.iterrows():
            try:
                name = row.get('stadium_name', row.get('name', row.get('venue')))
                if pd.isna(name):
                    continue
                
                venues.append({
                    "name": str(name).strip(),
                    "city": str(row.get('stadium_location', row.get('city', ''))).strip() if pd.notna(row.get('stadium_location', row.get('city'))) else None,
                    "state": str(row.get('state', '')).strip() if pd.notna(row.get('state')) else None,
                    "is_dome": bool(row.get('stadium_type', '').lower() in ['dome', 'indoor', 'retractable']) if pd.notna(row.get('stadium_type')) else False,
                    "surface": str(row.get('stadium_surface', row.get('surface', ''))).strip() if pd.notna(row.get('stadium_surface', row.get('surface'))) else None,
                    "capacity": int(row.get('stadium_capacity', row.get('capacity', 0))) if pd.notna(row.get('stadium_capacity', row.get('capacity'))) else None,
                })
            except Exception as e:
                logger.debug(f"[Kaggle] Error parsing stadium: {e}")
                continue
        
        logger.info(f"[Kaggle] Parsed {len(venues)} stadiums")
        return venues
    
    # =========================================================================
    # NBA PARSERS  
    # =========================================================================
    
    def _parse_nba_betting(self, df: pd.DataFrame, sport: str, min_year: int) -> Dict[str, List[Dict]]:
        """Parse NBA betting and odds data."""
        result = {"games": [], "odds": [], "teams": [], "player_stats": []}
        df.columns = df.columns.str.lower().str.strip()
        
        # Date column detection
        date_col = None
        for col in ['date', 'game_date', 'gamedate', 'datetime']:
            if col in df.columns:
                date_col = col
                break
        
        teams_seen = set()
        
        for _, row in df.iterrows():
            try:
                # Parse date
                game_date = None
                year = None
                if date_col and pd.notna(row.get(date_col)):
                    try:
                        game_date = pd.to_datetime(row[date_col])
                        year = game_date.year
                    except:
                        continue
                
                if not year or year < min_year:
                    continue
                
                # Get teams
                home_team = row.get('home_team', row.get('home', row.get('team_home')))
                away_team = row.get('away_team', row.get('visitor', row.get('team_away', row.get('away'))))
                
                if pd.isna(home_team) or pd.isna(away_team):
                    continue
                
                home_team = str(home_team).strip()
                away_team = str(away_team).strip()
                teams_seen.add(home_team)
                teams_seen.add(away_team)
                
                # Scores
                home_score = row.get('home_score', row.get('pts_home', row.get('home_pts')))
                away_score = row.get('away_score', row.get('pts_away', row.get('away_pts', row.get('visitor_pts'))))
                
                # Game record
                game = {
                    "sport": sport,
                    "home_team": home_team,
                    "away_team": away_team,
                    "scheduled_at": game_date.isoformat() if game_date else None,
                    "home_score": int(home_score) if pd.notna(home_score) else None,
                    "away_score": int(away_score) if pd.notna(away_score) else None,
                    "status": "final" if pd.notna(home_score) else "scheduled",
                    "season_year": year,
                    "external_id": f"kaggle_nba_{year}_{home_team}_{away_team}_{game_date.strftime('%Y%m%d') if game_date else ''}",
                }
                result["games"].append(game)
                
                # Odds
                spread = row.get('spread', row.get('line', row.get('point_spread')))
                total = row.get('total', row.get('over_under', row.get('ou')))
                ml_home = row.get('home_ml', row.get('ml_home', row.get('home_moneyline')))
                ml_away = row.get('away_ml', row.get('ml_away', row.get('away_moneyline')))
                
                if pd.notna(spread) or pd.notna(total) or pd.notna(ml_home):
                    odds_record = {
                        "game_external_id": game["external_id"],
                        "sport": sport,
                        "sportsbook": "market_consensus",
                        "bet_type": "spread",
                        "home_line": float(spread) if pd.notna(spread) else None,
                        "away_line": -float(spread) if pd.notna(spread) else None,
                        "home_odds": int(ml_home) if pd.notna(ml_home) else None,
                        "away_odds": int(ml_away) if pd.notna(ml_away) else None,
                        "total": float(total) if pd.notna(total) else None,
                        "recorded_at": game_date.isoformat() if game_date else None,
                    }
                    result["odds"].append(odds_record)
                    
            except Exception as e:
                logger.debug(f"[Kaggle] Error parsing NBA row: {e}")
                continue
        
        # Create team records
        for team_name in teams_seen:
            result["teams"].append({
                "sport": sport,
                "name": team_name,
                "abbreviation": self._get_team_abbrev(team_name, sport),
                "external_id": f"kaggle_{sport.lower()}_{team_name.lower().replace(' ', '_')}",
            })
        
        logger.info(f"[Kaggle] Parsed NBA betting: {len(result['games'])} games, {len(result['odds'])} odds")
        return result
    
    def _parse_nba_database(self, df: pd.DataFrame, filename: str, sport: str, min_year: int) -> Dict[str, List[Dict]]:
        """Parse comprehensive NBA database."""
        result = {"games": [], "odds": [], "teams": [], "players": [], "player_stats": []}
        df.columns = df.columns.str.lower().str.strip()
        
        filename_lower = filename.lower()
        
        if "player" in filename_lower and "stat" in filename_lower:
            # Player stats file
            for _, row in df.iterrows():
                try:
                    player_name = row.get('player_name', row.get('player', row.get('name')))
                    if pd.isna(player_name):
                        continue
                    
                    season = row.get('season', row.get('year'))
                    if pd.notna(season):
                        try:
                            year = int(str(season)[:4])
                            if year < min_year:
                                continue
                        except:
                            pass
                    
                    # Add player
                    result["players"].append({
                        "sport": sport,
                        "name": str(player_name).strip(),
                        "external_id": f"kaggle_nba_{str(player_name).lower().replace(' ', '_')}",
                        "position": str(row.get('position', row.get('pos', ''))).strip() if pd.notna(row.get('position', row.get('pos'))) else None,
                    })
                    
                    # Common NBA stats
                    stat_columns = ['pts', 'reb', 'ast', 'stl', 'blk', 'fg_pct', '3p_pct', 'ft_pct', 'min', 'gp']
                    for stat_col in stat_columns:
                        if stat_col in df.columns and pd.notna(row.get(stat_col)):
                            result["player_stats"].append({
                                "sport": sport,
                                "player_name": str(player_name).strip(),
                                "stat_type": f"kaggle_{stat_col}",
                                "value": float(row[stat_col]),
                                "season_year": year if 'year' in dir() else None,
                            })
                            
                except Exception as e:
                    logger.debug(f"[Kaggle] Error parsing NBA player stat: {e}")
                    continue
                    
        elif "team" in filename_lower:
            # Teams file
            for _, row in df.iterrows():
                try:
                    name = row.get('team_name', row.get('name', row.get('team')))
                    if pd.isna(name):
                        continue
                    
                    result["teams"].append({
                        "sport": sport,
                        "name": str(name).strip(),
                        "abbreviation": str(row.get('abbreviation', row.get('team_id', '')))[:10] or self._get_team_abbrev(str(name), sport),
                        "city": str(row.get('city', '')).strip() if pd.notna(row.get('city')) else None,
                        "conference": str(row.get('conference', '')).strip() if pd.notna(row.get('conference')) else None,
                        "division": str(row.get('division', '')).strip() if pd.notna(row.get('division')) else None,
                        "external_id": f"kaggle_{sport.lower()}_{str(name).lower().replace(' ', '_')}",
                    })
                except Exception as e:
                    continue
                    
        elif "game" in filename_lower:
            # Games file
            return self._parse_nba_betting(df, sport, min_year)
        
        return result
    
    # =========================================================================
    # MLB PARSERS
    # =========================================================================
    
    def _parse_mlb_data(self, df: pd.DataFrame, filename: str, sport: str, min_year: int) -> Dict[str, List[Dict]]:
        """Parse MLB game and player data."""
        result = {"games": [], "odds": [], "teams": [], "players": [], "player_stats": [], "team_stats": []}
        df.columns = df.columns.str.lower().str.strip()
        
        filename_lower = filename.lower()
        
        if "game" in filename_lower or "score" in filename_lower:
            # Games data
            date_col = None
            for col in ['date', 'game_date', 'datetime']:
                if col in df.columns:
                    date_col = col
                    break
            
            teams_seen = set()
            
            for _, row in df.iterrows():
                try:
                    game_date = None
                    year = None
                    if date_col and pd.notna(row.get(date_col)):
                        try:
                            game_date = pd.to_datetime(row[date_col])
                            year = game_date.year
                        except:
                            continue
                    
                    if not year or year < min_year:
                        continue
                    
                    home_team = row.get('home_team', row.get('home', row.get('home_team_name')))
                    away_team = row.get('away_team', row.get('away', row.get('away_team_name')))
                    
                    if pd.isna(home_team) or pd.isna(away_team):
                        continue
                    
                    home_team = str(home_team).strip()
                    away_team = str(away_team).strip()
                    teams_seen.add(home_team)
                    teams_seen.add(away_team)
                    
                    home_score = row.get('home_score', row.get('home_runs', row.get('home_final_score')))
                    away_score = row.get('away_score', row.get('away_runs', row.get('away_final_score')))
                    
                    game = {
                        "sport": sport,
                        "home_team": home_team,
                        "away_team": away_team,
                        "scheduled_at": game_date.isoformat() if game_date else None,
                        "home_score": int(home_score) if pd.notna(home_score) else None,
                        "away_score": int(away_score) if pd.notna(away_score) else None,
                        "status": "final" if pd.notna(home_score) else "scheduled",
                        "season_year": year,
                        "external_id": f"kaggle_mlb_{year}_{home_team}_{away_team}_{game_date.strftime('%Y%m%d') if game_date else ''}",
                    }
                    result["games"].append(game)
                    
                except Exception as e:
                    continue
            
            for team_name in teams_seen:
                result["teams"].append({
                    "sport": sport,
                    "name": team_name,
                    "abbreviation": self._get_team_abbrev(team_name, sport),
                    "external_id": f"kaggle_{sport.lower()}_{team_name.lower().replace(' ', '_')}",
                })
                
        elif "player" in filename_lower or "batting" in filename_lower or "pitching" in filename_lower:
            # Player data
            for _, row in df.iterrows():
                try:
                    player_name = row.get('player_name', row.get('name', row.get('player')))
                    if pd.isna(player_name):
                        # Try to construct from first/last name
                        first = row.get('first_name', row.get('namefirst', ''))
                        last = row.get('last_name', row.get('namelast', ''))
                        if pd.notna(first) and pd.notna(last):
                            player_name = f"{first} {last}"
                        else:
                            continue
                    
                    year = row.get('year', row.get('yearid', row.get('season')))
                    if pd.notna(year):
                        year = int(year)
                        if year < min_year:
                            continue
                    
                    result["players"].append({
                        "sport": sport,
                        "name": str(player_name).strip(),
                        "external_id": f"kaggle_mlb_{str(player_name).lower().replace(' ', '_')}",
                        "position": str(row.get('position', row.get('pos', ''))).strip() if pd.notna(row.get('position', row.get('pos'))) else None,
                    })
                    
                    # MLB batting stats
                    batting_stats = ['ab', 'h', 'hr', 'rbi', 'r', 'sb', 'bb', 'so', 'avg', 'obp', 'slg', 'ops']
                    for stat in batting_stats:
                        if stat in df.columns and pd.notna(row.get(stat)):
                            result["player_stats"].append({
                                "sport": sport,
                                "player_name": str(player_name).strip(),
                                "stat_type": f"kaggle_batting_{stat}",
                                "value": float(row[stat]),
                                "season_year": year,
                            })
                    
                    # MLB pitching stats
                    pitching_stats = ['w', 'l', 'era', 'ip', 'so', 'bb', 'sv', 'whip', 'h', 'er']
                    for stat in pitching_stats:
                        col_name = stat if stat in df.columns else f"p_{stat}"
                        if col_name in df.columns and pd.notna(row.get(col_name)):
                            result["player_stats"].append({
                                "sport": sport,
                                "player_name": str(player_name).strip(),
                                "stat_type": f"kaggle_pitching_{stat}",
                                "value": float(row[col_name]),
                                "season_year": year,
                            })
                            
                except Exception as e:
                    continue
        
        return result
    
    # =========================================================================
    # NHL PARSERS
    # =========================================================================
    
    def _parse_nhl_data(self, df: pd.DataFrame, filename: str, sport: str, min_year: int) -> Dict[str, List[Dict]]:
        """Parse NHL game data."""
        result = {"games": [], "odds": [], "teams": [], "players": [], "player_stats": []}
        df.columns = df.columns.str.lower().str.strip()
        
        filename_lower = filename.lower()
        
        if "game" in filename_lower:
            date_col = None
            for col in ['date', 'game_date', 'datetime', 'date_time']:
                if col in df.columns:
                    date_col = col
                    break
            
            teams_seen = set()
            
            for _, row in df.iterrows():
                try:
                    game_date = None
                    year = None
                    if date_col and pd.notna(row.get(date_col)):
                        try:
                            game_date = pd.to_datetime(row[date_col])
                            year = game_date.year
                        except:
                            continue
                    
                    season = row.get('season')
                    if pd.notna(season) and not year:
                        year = int(str(season)[:4])
                    
                    if not year or year < min_year:
                        continue
                    
                    home_team = row.get('home_team', row.get('home', row.get('home_team_name')))
                    away_team = row.get('away_team', row.get('away', row.get('away_team_name')))
                    
                    if pd.isna(home_team) or pd.isna(away_team):
                        continue
                    
                    home_team = str(home_team).strip()
                    away_team = str(away_team).strip()
                    teams_seen.add(home_team)
                    teams_seen.add(away_team)
                    
                    home_score = row.get('home_goals', row.get('home_score'))
                    away_score = row.get('away_goals', row.get('away_score'))
                    
                    game = {
                        "sport": sport,
                        "home_team": home_team,
                        "away_team": away_team,
                        "scheduled_at": game_date.isoformat() if game_date else None,
                        "home_score": int(home_score) if pd.notna(home_score) else None,
                        "away_score": int(away_score) if pd.notna(away_score) else None,
                        "status": "final" if pd.notna(home_score) else "scheduled",
                        "season_year": year,
                        "external_id": f"kaggle_nhl_{year}_{home_team}_{away_team}_{game_date.strftime('%Y%m%d') if game_date else ''}",
                    }
                    result["games"].append(game)
                    
                except Exception as e:
                    continue
            
            for team_name in teams_seen:
                result["teams"].append({
                    "sport": sport,
                    "name": team_name,
                    "abbreviation": self._get_team_abbrev(team_name, sport),
                    "external_id": f"kaggle_{sport.lower()}_{team_name.lower().replace(' ', '_')}",
                })
                
        elif "player" in filename_lower:
            for _, row in df.iterrows():
                try:
                    player_name = row.get('player_name', row.get('name', row.get('player')))
                    if pd.isna(player_name):
                        continue
                    
                    result["players"].append({
                        "sport": sport,
                        "name": str(player_name).strip(),
                        "external_id": f"kaggle_nhl_{str(player_name).lower().replace(' ', '_')}",
                        "position": str(row.get('position', row.get('pos', ''))).strip() if pd.notna(row.get('position', row.get('pos'))) else None,
                    })
                    
                    # NHL stats
                    stats = ['goals', 'assists', 'points', 'pim', 'plus_minus', 'shots', 'toi', 'gp']
                    for stat in stats:
                        if stat in df.columns and pd.notna(row.get(stat)):
                            result["player_stats"].append({
                                "sport": sport,
                                "player_name": str(player_name).strip(),
                                "stat_type": f"kaggle_{stat}",
                                "value": float(row[stat]),
                            })
                except Exception as e:
                    continue
        
        return result
    
    # =========================================================================
    # SOCCER PARSERS
    # =========================================================================
    
    def _parse_soccer_betting(self, df: pd.DataFrame, sport: str, min_year: int) -> Dict[str, List[Dict]]:
        """Parse soccer/football betting data."""
        result = {"games": [], "odds": [], "teams": []}
        df.columns = df.columns.str.lower().str.strip()
        
        date_col = None
        for col in ['date', 'match_date', 'datetime']:
            if col in df.columns:
                date_col = col
                break
        
        teams_seen = set()
        
        for _, row in df.iterrows():
            try:
                game_date = None
                year = None
                if date_col and pd.notna(row.get(date_col)):
                    try:
                        game_date = pd.to_datetime(row[date_col])
                        year = game_date.year
                    except:
                        continue
                
                if not year or year < min_year:
                    continue
                
                home_team = row.get('hometeam', row.get('home_team', row.get('home')))
                away_team = row.get('awayteam', row.get('away_team', row.get('away')))
                
                if pd.isna(home_team) or pd.isna(away_team):
                    continue
                
                home_team = str(home_team).strip()
                away_team = str(away_team).strip()
                teams_seen.add(home_team)
                teams_seen.add(away_team)
                
                home_score = row.get('fthg', row.get('home_score', row.get('hg')))
                away_score = row.get('ftag', row.get('away_score', row.get('ag')))
                
                game = {
                    "sport": sport,
                    "home_team": home_team,
                    "away_team": away_team,
                    "scheduled_at": game_date.isoformat() if game_date else None,
                    "home_score": int(home_score) if pd.notna(home_score) else None,
                    "away_score": int(away_score) if pd.notna(away_score) else None,
                    "status": "final" if pd.notna(home_score) else "scheduled",
                    "season_year": year,
                    "external_id": f"kaggle_soccer_{year}_{home_team}_{away_team}_{game_date.strftime('%Y%m%d') if game_date else ''}",
                }
                result["games"].append(game)
                
                # Soccer betting odds (B365, BW, IW, etc. are common bookmaker columns)
                odds_home = row.get('b365h', row.get('psh', row.get('odd_h')))
                odds_draw = row.get('b365d', row.get('psd', row.get('odd_d')))
                odds_away = row.get('b365a', row.get('psa', row.get('odd_a')))
                
                if pd.notna(odds_home) or pd.notna(odds_away):
                    odds_record = {
                        "game_external_id": game["external_id"],
                        "sport": sport,
                        "sportsbook": "bet365",
                        "bet_type": "moneyline",
                        "home_odds": int(self._decimal_to_american(float(odds_home))) if pd.notna(odds_home) else None,
                        "away_odds": int(self._decimal_to_american(float(odds_away))) if pd.notna(odds_away) else None,
                        "recorded_at": game_date.isoformat() if game_date else None,
                    }
                    result["odds"].append(odds_record)
                    
            except Exception as e:
                continue
        
        for team_name in teams_seen:
            result["teams"].append({
                "sport": sport,
                "name": team_name,
                "abbreviation": self._get_team_abbrev(team_name, sport),
                "external_id": f"kaggle_{sport.lower()}_{team_name.lower().replace(' ', '_')}",
            })
        
        return result
    
    # =========================================================================
    # COLLEGE SPORTS PARSERS
    # =========================================================================
    
    def _parse_college_data(self, df: pd.DataFrame, sport: str, min_year: int) -> Dict[str, List[Dict]]:
        """Parse college football/basketball data."""
        result = {"games": [], "teams": [], "team_stats": []}
        df.columns = df.columns.str.lower().str.strip()
        
        teams_seen = set()
        
        for _, row in df.iterrows():
            try:
                # Try to get teams
                team1 = row.get('team', row.get('school', row.get('team_name')))
                team2 = row.get('opponent', row.get('opp', row.get('opponent_name')))
                
                year = row.get('year', row.get('season'))
                if pd.notna(year):
                    year = int(year)
                    if year < min_year:
                        continue
                
                if pd.notna(team1):
                    team1 = str(team1).strip()
                    teams_seen.add(team1)
                    
                    # Team stats
                    stat_columns = ['wins', 'losses', 'points', 'points_allowed', 'ppg', 'oppg']
                    for stat in stat_columns:
                        if stat in df.columns and pd.notna(row.get(stat)):
                            result["team_stats"].append({
                                "sport": sport,
                                "team_name": team1,
                                "stat_type": f"kaggle_{stat}",
                                "value": float(row[stat]),
                                "season_year": year,
                            })
                
                if pd.notna(team2):
                    team2 = str(team2).strip()
                    teams_seen.add(team2)
                    
            except Exception as e:
                continue
        
        for team_name in teams_seen:
            result["teams"].append({
                "sport": sport,
                "name": team_name,
                "abbreviation": self._get_team_abbrev(team_name, sport),
                "external_id": f"kaggle_{sport.lower()}_{team_name.lower().replace(' ', '_')}",
            })
        
        return result
    
    # =========================================================================
    # TENNIS PARSERS
    # =========================================================================
    
    def _parse_tennis_data(self, df: pd.DataFrame, sport: str, min_year: int) -> Dict[str, List[Dict]]:
        """Parse tennis match data."""
        result = {"games": [], "players": [], "player_stats": []}
        df.columns = df.columns.str.lower().str.strip()
        
        players_seen = set()
        
        for _, row in df.iterrows():
            try:
                date_col = None
                for col in ['date', 'tourney_date', 'match_date']:
                    if col in df.columns:
                        date_col = col
                        break
                
                game_date = None
                year = None
                if date_col and pd.notna(row.get(date_col)):
                    try:
                        game_date = pd.to_datetime(row[date_col])
                        year = game_date.year
                    except:
                        pass
                
                if not year or year < min_year:
                    continue
                
                player1 = row.get('winner_name', row.get('player1', row.get('winner')))
                player2 = row.get('loser_name', row.get('player2', row.get('loser')))
                
                if pd.isna(player1) or pd.isna(player2):
                    continue
                
                player1 = str(player1).strip()
                player2 = str(player2).strip()
                players_seen.add(player1)
                players_seen.add(player2)
                
                score = row.get('score', '')
                
                # Create match record as a "game"
                game = {
                    "sport": sport,
                    "home_team": player1,  # Winner
                    "away_team": player2,  # Loser
                    "scheduled_at": game_date.isoformat() if game_date else None,
                    "status": "final",
                    "season_year": year,
                    "external_id": f"kaggle_tennis_{year}_{player1}_{player2}_{game_date.strftime('%Y%m%d') if game_date else ''}",
                }
                result["games"].append(game)
                
            except Exception as e:
                continue
        
        for player_name in players_seen:
            result["players"].append({
                "sport": sport,
                "name": player_name,
                "external_id": f"kaggle_tennis_{player_name.lower().replace(' ', '_')}",
            })
        
        return result
    
    # =========================================================================
    # UFC/MMA PARSERS
    # =========================================================================
    
    def _parse_ufc_data(self, df: pd.DataFrame, sport: str, min_year: int) -> Dict[str, List[Dict]]:
        """Parse UFC/MMA fight data."""
        result = {"games": [], "players": [], "odds": []}
        df.columns = df.columns.str.lower().str.strip()
        
        players_seen = set()
        
        for _, row in df.iterrows():
            try:
                date_col = None
                for col in ['date', 'event_date', 'fight_date']:
                    if col in df.columns:
                        date_col = col
                        break
                
                game_date = None
                year = None
                if date_col and pd.notna(row.get(date_col)):
                    try:
                        game_date = pd.to_datetime(row[date_col])
                        year = game_date.year
                    except:
                        pass
                
                if not year or year < min_year:
                    continue
                
                fighter1 = row.get('r_fighter', row.get('fighter1', row.get('red_fighter')))
                fighter2 = row.get('b_fighter', row.get('fighter2', row.get('blue_fighter')))
                
                if pd.isna(fighter1) or pd.isna(fighter2):
                    continue
                
                fighter1 = str(fighter1).strip()
                fighter2 = str(fighter2).strip()
                players_seen.add(fighter1)
                players_seen.add(fighter2)
                
                winner = row.get('winner', row.get('w_fighter'))
                
                # Create fight record
                game = {
                    "sport": sport,
                    "home_team": fighter1,
                    "away_team": fighter2,
                    "scheduled_at": game_date.isoformat() if game_date else None,
                    "home_score": 1 if pd.notna(winner) and str(winner).strip() == fighter1 else 0,
                    "away_score": 1 if pd.notna(winner) and str(winner).strip() == fighter2 else 0,
                    "status": "final",
                    "season_year": year,
                    "external_id": f"kaggle_mma_{year}_{fighter1}_{fighter2}_{game_date.strftime('%Y%m%d') if game_date else ''}",
                }
                result["games"].append(game)
                
                # Odds if available
                r_odds = row.get('r_odds', row.get('fighter1_odds'))
                b_odds = row.get('b_odds', row.get('fighter2_odds'))
                
                if pd.notna(r_odds) or pd.notna(b_odds):
                    result["odds"].append({
                        "game_external_id": game["external_id"],
                        "sport": sport,
                        "sportsbook": "market_consensus",
                        "bet_type": "moneyline",
                        "home_odds": int(r_odds) if pd.notna(r_odds) else None,
                        "away_odds": int(b_odds) if pd.notna(b_odds) else None,
                        "recorded_at": game_date.isoformat() if game_date else None,
                    })
                    
            except Exception as e:
                continue
        
        for player_name in players_seen:
            result["players"].append({
                "sport": sport,
                "name": player_name,
                "external_id": f"kaggle_mma_{player_name.lower().replace(' ', '_')}",
            })
        
        return result
    
    # =========================================================================
    # GENERIC PARSER
    # =========================================================================
    
    def _parse_generic(self, df: pd.DataFrame, sport: str, min_year: int) -> Dict[str, List[Dict]]:
        """Generic parser for unknown dataset formats."""
        result = {"games": [], "teams": [], "players": [], "player_stats": [], "team_stats": []}
        df.columns = df.columns.str.lower().str.strip()
        
        # Try to identify what kind of data this is
        col_set = set(df.columns)
        
        # Check for game-like data
        if any(x in col_set for x in ['home_team', 'away_team', 'home', 'away', 'team_home', 'team_away']):
            return self._parse_nba_betting(df, sport, min_year)
        
        # Check for player stats
        if any(x in col_set for x in ['player', 'player_name', 'name']) and any(x in col_set for x in ['pts', 'points', 'goals', 'score']):
            for _, row in df.iterrows():
                try:
                    player = row.get('player', row.get('player_name', row.get('name')))
                    if pd.isna(player):
                        continue
                    
                    result["players"].append({
                        "sport": sport,
                        "name": str(player).strip(),
                        "external_id": f"kaggle_{sport.lower()}_{str(player).lower().replace(' ', '_')}",
                    })
                except:
                    continue
        
        return result
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def _get_team_abbrev(self, team_name: str, sport: str) -> str:
        """Generate team abbreviation from name."""
        if not team_name:
            return "UNK"
        
        # Remove common words
        name = team_name.upper()
        for word in ["THE", "FC", "SC", "AFC", "BC"]:
            name = name.replace(word, "").strip()
        
        words = name.split()
        if len(words) >= 2:
            return (words[0][:1] + words[-1][:2]).upper()[:3]
        elif len(words) == 1:
            return words[0][:3].upper()
        return "UNK"
    
    def _decimal_to_american(self, decimal_odds: float) -> int:
        """Convert decimal odds to American format."""
        if decimal_odds >= 2.0:
            return int((decimal_odds - 1) * 100)
        else:
            return int(-100 / (decimal_odds - 1))
    
    # =========================================================================
    # INJURY PARSERS
    # =========================================================================
    
    def _find_column(self, df: pd.DataFrame, possible_names: List[str]) -> Optional[str]:
        """Find a column from a list of possible names (case-insensitive)."""
        df_columns_lower = {col.lower(): col for col in df.columns}
        for name in possible_names:
            if name.lower() in df_columns_lower:
                return df_columns_lower[name.lower()]
        return None
    
    def _parse_nba_injuries(self, df: pd.DataFrame, sport: str) -> List[Dict[str, Any]]:
        """
        Parse NBA injury data from Kaggle datasets.
        
        Handles:
        - loganlauton/nba-injury-stats-1951-2023
        - ghopkins/nba-injuries-2010-2018
        """
        injuries = []
        
        logger.info(f"[Kaggle] Parsing NBA injuries from {len(df)} rows")
        logger.debug(f"[Kaggle] Columns: {list(df.columns)}")
        
        # Column mappings for NBA injury datasets
        player_cols = ['Player', 'player', 'PLAYER', 'player_name', 'Name', 'Relinquished', 'Acquired']
        team_cols = ['Team', 'team', 'TEAM', 'team_name', 'Tm', 'Team_Name']
        injury_cols = ['Injury', 'injury', 'INJURY', 'injury_type', 'Notes', 'Description', 'type']
        status_cols = ['Status', 'status', 'STATUS', 'Designation', 'Category']
        position_cols = ['Position', 'position', 'POS', 'Pos']
        date_cols = ['Date', 'date', 'DATE', 'injury_date', 'Game_Date', 'date_injured']
        season_cols = ['Season', 'season', 'SEASON', 'Year', 'year']
        games_missed_cols = ['Games_Missed', 'games_missed', 'GM', 'G', 'Games']
        
        # Find actual columns
        player_col = self._find_column(df, player_cols)
        team_col = self._find_column(df, team_cols)
        injury_col = self._find_column(df, injury_cols)
        status_col = self._find_column(df, status_cols)
        position_col = self._find_column(df, position_cols)
        date_col = self._find_column(df, date_cols)
        season_col = self._find_column(df, season_cols)
        games_missed_col = self._find_column(df, games_missed_cols)
        
        if not player_col:
            logger.warning(f"[Kaggle] No player column found for NBA injuries")
            return []
        
        logger.info(f"[Kaggle] Found columns - Player: {player_col}, Team: {team_col}, Injury: {injury_col}")
        
        for idx, row in df.iterrows():
            try:
                player_name = str(row.get(player_col, '')).strip()
                if not player_name or player_name.lower() in ['nan', 'none', '']:
                    continue
                
                injury = {
                    'player_name': player_name[:200],
                    'team_name': None,
                    'injury_type': None,
                    'status': 'Historical',
                    'position': None,
                    'sport_code': sport,
                    'source': 'kaggle_nba_injuries',
                    'season': None,
                    'games_missed': None,
                }
                
                # Team
                if team_col and pd.notna(row.get(team_col)):
                    team_val = str(row[team_col]).strip()
                    if team_val.lower() not in ['nan', 'none', '']:
                        injury['team_name'] = team_val[:100]
                
                # Injury type
                if injury_col and pd.notna(row.get(injury_col)):
                    injury_val = str(row[injury_col]).strip()
                    if injury_val.lower() not in ['nan', 'none', '']:
                        injury['injury_type'] = injury_val[:200]
                
                # Status
                if status_col and pd.notna(row.get(status_col)):
                    status_val = str(row[status_col]).strip()
                    if status_val.lower() not in ['nan', 'none', '']:
                        injury['status'] = status_val[:50]
                
                # Position
                if position_col and pd.notna(row.get(position_col)):
                    pos_val = str(row[position_col]).strip()
                    if pos_val.lower() not in ['nan', 'none', '']:
                        injury['position'] = pos_val[:50]
                
                # Season
                if season_col and pd.notna(row.get(season_col)):
                    try:
                        season_val = row[season_col]
                        if isinstance(season_val, (int, float)):
                            injury['season'] = str(int(season_val))
                        else:
                            injury['season'] = str(season_val)[:20]
                    except:
                        pass
                
                # Games missed
                if games_missed_col and pd.notna(row.get(games_missed_col)):
                    try:
                        injury['games_missed'] = int(row[games_missed_col])
                    except:
                        pass
                
                # Injury date
                if date_col and pd.notna(row.get(date_col)):
                    try:
                        injury['injury_date'] = pd.to_datetime(row[date_col]).date()
                    except:
                        pass
                
                injuries.append(injury)
                
            except Exception as e:
                logger.debug(f"[Kaggle] Error parsing NBA injury row {idx}: {e}")
                continue
        
        logger.info(f"[Kaggle] Parsed {len(injuries)} NBA injuries")
        return injuries
    
    def _parse_nfl_injuries(self, df: pd.DataFrame, sport: str) -> List[Dict[str, Any]]:
        """
        Parse NFL injury data from Kaggle datasets.
        
        Handles:
        - thedevastator/nfl-injury-analysis-2012-2017
        - rishidamarla/concussions-in-the-nfl-20122014
        """
        injuries = []
        
        logger.info(f"[Kaggle] Parsing NFL injuries from {len(df)} rows")
        logger.debug(f"[Kaggle] Columns: {list(df.columns)}")
        
        # Column mappings for NFL injury datasets
        player_cols = ['Player', 'player', 'PLAYER', 'player_name', 'Name', 'Player Name']
        team_cols = ['Team', 'team', 'TEAM', 'team_name', 'Team Name']
        position_cols = ['Position', 'position', 'POS', 'Pos', 'Player Position']
        injury_cols = ['Injury', 'injury', 'INJURY', 'injury_type', 'Body_Part', 'Injury Type', 'type', 'Injury Location']
        status_cols = ['Status', 'status', 'STATUS', 'Game_Status', 'Designation', 'Injury Status']
        season_cols = ['Season', 'season', 'Year', 'year', 'Season Year']
        week_cols = ['Week', 'week', 'WEEK', 'Game Week']
        games_missed_cols = ['Games_Missed', 'games_missed', 'Weeks Missed']
        
        # Find actual columns
        player_col = self._find_column(df, player_cols)
        team_col = self._find_column(df, team_cols)
        position_col = self._find_column(df, position_cols)
        injury_col = self._find_column(df, injury_cols)
        status_col = self._find_column(df, status_cols)
        season_col = self._find_column(df, season_cols)
        week_col = self._find_column(df, week_cols)
        games_missed_col = self._find_column(df, games_missed_cols)
        
        if not player_col:
            logger.warning(f"[Kaggle] No player column found for NFL injuries")
            return []
        
        logger.info(f"[Kaggle] Found columns - Player: {player_col}, Team: {team_col}, Injury: {injury_col}")
        
        for idx, row in df.iterrows():
            try:
                player_name = str(row.get(player_col, '')).strip()
                if not player_name or player_name.lower() in ['nan', 'none', '']:
                    continue
                
                injury = {
                    'player_name': player_name[:200],
                    'team_name': None,
                    'injury_type': None,
                    'status': 'Historical',
                    'position': None,
                    'sport_code': sport,
                    'source': 'kaggle_nfl_injuries',
                    'season': None,
                    'week': None,
                    'games_missed': None,
                }
                
                # Team
                if team_col and pd.notna(row.get(team_col)):
                    team_val = str(row[team_col]).strip()
                    if team_val.lower() not in ['nan', 'none', '']:
                        injury['team_name'] = team_val[:100]
                
                # Position
                if position_col and pd.notna(row.get(position_col)):
                    pos_val = str(row[position_col]).strip()
                    if pos_val.lower() not in ['nan', 'none', '']:
                        injury['position'] = pos_val[:50]
                
                # Injury type
                if injury_col and pd.notna(row.get(injury_col)):
                    injury_val = str(row[injury_col]).strip()
                    if injury_val.lower() not in ['nan', 'none', '']:
                        injury['injury_type'] = injury_val[:200]
                
                # Status
                if status_col and pd.notna(row.get(status_col)):
                    status_val = str(row[status_col]).strip()
                    if status_val.lower() not in ['nan', 'none', '']:
                        injury['status'] = status_val[:50]
                
                # Season
                if season_col and pd.notna(row.get(season_col)):
                    try:
                        season_val = row[season_col]
                        if isinstance(season_val, (int, float)):
                            injury['season'] = str(int(season_val))
                        else:
                            injury['season'] = str(season_val)[:20]
                    except:
                        pass
                
                # Week
                if week_col and pd.notna(row.get(week_col)):
                    try:
                        injury['week'] = int(row[week_col])
                    except:
                        pass
                
                # Games missed
                if games_missed_col and pd.notna(row.get(games_missed_col)):
                    try:
                        injury['games_missed'] = int(row[games_missed_col])
                    except:
                        pass
                
                injuries.append(injury)
                
            except Exception as e:
                logger.debug(f"[Kaggle] Error parsing NFL injury row {idx}: {e}")
                continue
        
        logger.info(f"[Kaggle] Parsed {len(injuries)} NFL injuries")
        return injuries
    
    # =========================================================================
    # DATABASE SAVE METHODS
    # =========================================================================
    
    async def save_to_database(self, data: Dict[str, List[Dict]], session: Optional[AsyncSession] = None) -> Dict[str, int]:
        """
        Save collected data to database.
        
        Args:
            data: Collected data dictionary
            session: Optional database session
            
        Returns:
            Dict with counts of saved records by type
        """
        from app.core.database import get_async_session
        
        counts = {
            "sports": 0,
            "seasons": 0,
            "teams": 0,
            "players": 0,
            "games": 0,
            "odds": 0,
            "player_stats": 0,
            "team_stats": 0,
            "venues": 0,
            "injuries": 0,
        }
        
        try:
            async with get_async_session() as session:
                # 1. Create/update sports
                sport_cache = {}
                sports_in_data = set()
                
                for key in ["games", "teams", "players", "odds"]:
                    for item in data.get(key, []):
                        if "sport" in item:
                            sports_in_data.add(item["sport"])
                
                for sport_code in sports_in_data:
                    config = SPORT_CONFIG.get(sport_code, {"code": sport_code, "name": sport_code})
                    
                    result = await session.execute(
                        select(Sport).where(Sport.code == config["code"])
                    )
                    sport = result.scalar_one_or_none()
                    
                    if not sport:
                        sport = Sport(
                            code=config["code"],
                            name=config["name"],
                            is_active=True
                        )
                        session.add(sport)
                        await session.flush()
                        counts["sports"] += 1
                    
                    sport_cache[sport_code] = sport.id
                
                # 2. Create seasons
                season_cache = {}
                years_in_data = set()
                
                for game in data.get("games", []):
                    if "season_year" in game and game["season_year"]:
                        years_in_data.add((game.get("sport"), game["season_year"]))
                
                for sport_code, year in years_in_data:
                    if sport_code not in sport_cache:
                        continue
                    
                    sport_id = sport_cache[sport_code]
                    config = SPORT_CONFIG.get(sport_code, {"season_start_month": 1, "season_end_month": 12})
                    
                    result = await session.execute(
                        select(Season).where(
                            and_(Season.sport_id == sport_id, Season.year == year)
                        )
                    )
                    season = result.scalar_one_or_none()
                    
                    if not season:
                        start_month = config.get("season_start_month", 1)
                        end_month = config.get("season_end_month", 12)
                        
                        start_date = date(year, start_month, 1)
                        end_year = year + 1 if end_month < start_month else year
                        end_date = date(end_year, end_month, 28)
                        
                        season = Season(
                            sport_id=sport_id,
                            year=year,
                            name=f"{year} Season",
                            start_date=start_date,
                            end_date=end_date,
                            is_current=(year == datetime.now().year)
                        )
                        session.add(season)
                        await session.flush()
                        counts["seasons"] += 1
                    
                    season_cache[(sport_code, year)] = season.id
                
                # 3. Create teams
                team_cache = {}
                for team_data in data.get("teams", []):
                    try:
                        sport_code = team_data.get("sport")
                        if sport_code not in sport_cache:
                            continue
                        
                        sport_id = sport_cache[sport_code]
                        team_name = team_data.get("name", "")
                        external_id = team_data.get("external_id", f"kaggle_{team_name.lower().replace(' ', '_')}")
                        
                        # Check for existing team
                        result = await session.execute(
                            select(Team).where(
                                and_(
                                    Team.sport_id == sport_id,
                                    Team.name == team_name
                                )
                            )
                        )
                        team = result.scalar_one_or_none()
                        
                        if not team:
                            # Try by external_id
                            result = await session.execute(
                                select(Team).where(
                                    and_(
                                        Team.sport_id == sport_id,
                                        Team.external_id == external_id
                                    )
                                )
                            )
                            team = result.scalar_one_or_none()
                        
                        if not team:
                            team = Team(
                                sport_id=sport_id,
                                external_id=external_id,
                                name=team_name,
                                abbreviation=team_data.get("abbreviation", self._get_team_abbrev(team_name, sport_code))[:10],
                                city=team_data.get("city"),
                                conference=team_data.get("conference"),
                                division=team_data.get("division"),
                                is_active=True
                            )
                            session.add(team)
                            await session.flush()
                            counts["teams"] += 1
                        
                        team_cache[(sport_code, team_name)] = team.id
                        
                    except Exception as e:
                        logger.debug(f"[Kaggle] Error saving team: {e}")
                        continue
                
                # 4. Create players
                player_cache = {}
                for player_data in data.get("players", []):
                    try:
                        sport_code = player_data.get("sport")
                        player_name = player_data.get("name", "")
                        external_id = player_data.get("external_id", f"kaggle_{player_name.lower().replace(' ', '_')}")
                        
                        # Check existing
                        result = await session.execute(
                            select(Player).where(Player.external_id == external_id)
                        )
                        player = result.scalar_one_or_none()
                        
                        if not player:
                            player = Player(
                                external_id=external_id,
                                name=player_name,
                                position=player_data.get("position"),
                                is_active=True
                            )
                            session.add(player)
                            await session.flush()
                            counts["players"] += 1
                        
                        player_cache[(sport_code, player_name)] = player.id
                        
                    except Exception as e:
                        logger.debug(f"[Kaggle] Error saving player: {e}")
                        continue
                
                # 5. Create games
                game_cache = {}
                for game_data in data.get("games", []):
                    try:
                        sport_code = game_data.get("sport")
                        if sport_code not in sport_cache:
                            continue
                        
                        sport_id = sport_cache[sport_code]
                        external_id = game_data.get("external_id")
                        
                        # Get team IDs
                        home_team_key = (sport_code, game_data.get("home_team"))
                        away_team_key = (sport_code, game_data.get("away_team"))
                        
                        home_team_id = team_cache.get(home_team_key)
                        away_team_id = team_cache.get(away_team_key)
                        
                        if not home_team_id or not away_team_id:
                            continue
                        
                        season_key = (sport_code, game_data.get("season_year"))
                        season_id = season_cache.get(season_key)
                        
                        # Check existing
                        result = await session.execute(
                            select(Game).where(Game.external_id == external_id)
                        )
                        game = result.scalar_one_or_none()
                        
                        if not game:
                            scheduled_at = game_data.get("scheduled_at")
                            if scheduled_at:
                                scheduled_at = pd.to_datetime(scheduled_at)
                            else:
                                scheduled_at = datetime.now()
                            
                            status_str = game_data.get("status", "scheduled")
                            status = GameStatus.FINAL if status_str == "final" else GameStatus.SCHEDULED
                            
                            game = Game(
                                sport_id=sport_id,
                                season_id=season_id,
                                external_id=external_id,
                                home_team_id=home_team_id,
                                away_team_id=away_team_id,
                                scheduled_at=scheduled_at,
                                status=status,
                                home_score=game_data.get("home_score"),
                                away_score=game_data.get("away_score"),
                            )
                            session.add(game)
                            await session.flush()
                            counts["games"] += 1
                        
                        game_cache[external_id] = game.id
                        
                    except Exception as e:
                        logger.debug(f"[Kaggle] Error saving game: {e}")
                        continue
                
                # 6. Create odds
                for odds_data in data.get("odds", []):
                    try:
                        game_ext_id = odds_data.get("game_external_id")
                        game_id = game_cache.get(game_ext_id)
                        
                        if not game_id:
                            continue
                        
                        # Check for duplicate odds
                        result = await session.execute(
                            select(Odds).where(
                                and_(
                                    Odds.game_id == game_id,
                                    Odds.bet_type == odds_data.get("bet_type", "spread"),
                                    Odds.sportsbook_key == odds_data.get("sportsbook", "market_consensus")
                                )
                            )
                        )
                        existing = result.scalar_one_or_none()
                        
                        if not existing:
                            odds = Odds(
                                game_id=game_id,
                                sportsbook_key=odds_data.get("sportsbook", "market_consensus"),
                                bet_type=odds_data.get("bet_type", "spread"),
                                home_line=odds_data.get("home_line"),
                                away_line=odds_data.get("away_line"),
                                home_odds=odds_data.get("home_odds"),
                                away_odds=odds_data.get("away_odds"),
                                total=odds_data.get("total"),
                                is_opening=False,
                            )
                            session.add(odds)
                            counts["odds"] += 1
                            
                    except Exception as e:
                        logger.debug(f"[Kaggle] Error saving odds: {e}")
                        continue
                
                # 7. Create player stats
                for stat_data in data.get("player_stats", []):
                    try:
                        sport_code = stat_data.get("sport")
                        player_name = stat_data.get("player_name")
                        player_key = (sport_code, player_name)
                        player_id = player_cache.get(player_key)
                        
                        if not player_id:
                            continue
                        
                        season_key = (sport_code, stat_data.get("season_year"))
                        season_id = season_cache.get(season_key)
                        
                        stat_type = stat_data.get("stat_type", "kaggle_stat")
                        # Truncate stat_type to 50 chars
                        if len(stat_type) > 50:
                            stat_type = stat_type[:50]
                        
                        stat = PlayerStats(
                            player_id=player_id,
                            season_id=season_id,
                            stat_type=stat_type,
                            value=float(stat_data.get("value", 0)),
                        )
                        session.add(stat)
                        counts["player_stats"] += 1
                        
                    except Exception as e:
                        logger.debug(f"[Kaggle] Error saving player stat: {e}")
                        continue
                
                # 8. Create team stats
                for stat_data in data.get("team_stats", []):
                    try:
                        sport_code = stat_data.get("sport")
                        team_name = stat_data.get("team_name")
                        team_key = (sport_code, team_name)
                        team_id = team_cache.get(team_key)
                        
                        if not team_id:
                            continue
                        
                        season_key = (sport_code, stat_data.get("season_year"))
                        season_id = season_cache.get(season_key)
                        
                        stat_type = stat_data.get("stat_type", "kaggle_stat")
                        if len(stat_type) > 50:
                            stat_type = stat_type[:50]
                        
                        stat = TeamStats(
                            team_id=team_id,
                            season_id=season_id,
                            stat_type=stat_type,
                            value=float(stat_data.get("value", 0)),
                        )
                        session.add(stat)
                        counts["team_stats"] += 1
                        
                    except Exception as e:
                        logger.debug(f"[Kaggle] Error saving team stat: {e}")
                        continue
                
                # 9. Create injuries
                for injury_data in data.get("injuries", []):
                    try:
                        from app.models.injury_models import Injury
                        
                        player_name = injury_data.get("player_name", "")
                        if not player_name:
                            continue
                        
                        sport_code = injury_data.get("sport_code", "")
                        
                        # Find team if available
                        team_id = None
                        team_name = injury_data.get("team_name")
                        if team_name and sport_code:
                            team_key = (sport_code, team_name)
                            team_id = team_cache.get(team_key)
                            
                            # Try partial match
                            if not team_id:
                                for (s, t), tid in team_cache.items():
                                    if s == sport_code and (team_name.lower() in t.lower() or t.lower() in team_name.lower()):
                                        team_id = tid
                                        break
                        
                        # Create unique external_id for deduplication
                        season = injury_data.get("season", "")
                        injury_type = (injury_data.get("injury_type") or "")[:30]
                        external_id = f"kg_{sport_code}_{player_name}_{season}_{injury_type}".replace(" ", "_")[:100]
                        
                        # Check if already exists
                        existing = await session.execute(
                            select(Injury).where(Injury.external_id == external_id)
                        )
                        if existing.scalar_one_or_none():
                            continue
                        
                        injury = Injury(
                            team_id=team_id,
                            sport_code=sport_code,
                            player_name=player_name[:200],
                            position=injury_data.get("position", "")[:50] if injury_data.get("position") else None,
                            injury_type=injury_data.get("injury_type", "")[:200] if injury_data.get("injury_type") else None,
                            status=(injury_data.get("status") or "Historical")[:50],
                            source=(injury_data.get("source") or "kaggle")[:50],
                            external_id=external_id,
                        )
                        session.add(injury)
                        counts["injuries"] += 1
                        
                        # Commit in batches to avoid memory issues
                        if counts["injuries"] % 1000 == 0:
                            await session.commit()
                            logger.info(f"[Kaggle] Saved {counts['injuries']} injuries...")
                        
                    except ImportError:
                        logger.warning("[Kaggle] Injury model not available, skipping injuries")
                        break
                    except Exception as e:
                        logger.debug(f"[Kaggle] Error saving injury: {e}")
                        continue
                
                await session.commit()
                logger.info(f"[Kaggle] Database save complete: {counts}")
                
        except Exception as e:
            logger.error(f"[Kaggle] Database save error: {e}")
            raise
        
        return counts
    
    async def collect_historical(self, years_back: int = 10, **kwargs) -> CollectorResult:
        """
        Collect historical data for all sports.
        
        Args:
            years_back: Number of years of historical data
            
        Returns:
            CollectorResult with all historical data
        """
        logger.info(f"[Kaggle] Collecting {years_back} years of historical data for all sports...")
        return await self.collect(years_back=years_back, **kwargs)
    
    async def collect_sport(self, sport: str, years_back: int = 10, **kwargs) -> CollectorResult:
        """
        Collect data for a specific sport.
        
        Args:
            sport: Sport code (NFL, NBA, MLB, NHL, etc.)
            years_back: Number of years of historical data
            
        Returns:
            CollectorResult with sport-specific data
        """
        logger.info(f"[Kaggle] Collecting {years_back} years of {sport} data...")
        return await self.collect(sports=[sport], years_back=years_back, **kwargs)
    
    async def collect_injuries(
        self,
        sports: Optional[List[str]] = None,
        force_download: bool = False,
        **kwargs
    ) -> CollectorResult:
        """
        Collect historical injury data from Kaggle datasets.
        
        This method specifically targets injury datasets:
        - NBA: loganlauton/nba-injury-stats-1951-2023 (70+ years)
        - NBA: ghopkins/nba-injuries-2010-2018 (detailed)
        - NFL: thedevastator/nfl-injury-analysis-2012-2017
        - NFL: rishidamarla/concussions-in-the-nfl-20122014
        
        Args:
            sports: List of sports to collect (None = all: NBA, NFL)
            force_download: Force re-download of datasets
            
        Returns:
            CollectorResult with injury data
            
        Usage:
            # Collect all injury data
            result = await kaggle_collector.collect_injuries()
            
            # Collect NBA injuries only
            result = await kaggle_collector.collect_injuries(sports=["NBA"])
            
            # Save to database
            counts = await kaggle_collector.save_to_database(result.data)
        """
        if not self._load_kaggle_api():
            return CollectorResult(
                success=False,
                error="Kaggle API not available. Install kaggle and configure credentials.",
                records_count=0
            )
        
        # Filter injury datasets by sport if specified
        target_datasets = KAGGLE_INJURY_DATASETS
        if sports:
            sports_upper = [s.upper() for s in sports]
            target_datasets = {
                k: v for k, v in KAGGLE_INJURY_DATASETS.items()
                if v["sport"] in sports_upper
            }
        
        logger.info(f"[Kaggle] Collecting injury data from {len(target_datasets)} datasets...")
        
        all_data = {
            "games": [],
            "odds": [],
            "teams": [],
            "players": [],
            "player_stats": [],
            "team_stats": [],
            "venues": [],
            "injuries": [],
        }
        
        total_records = 0
        errors = []
        
        for dataset_key, config in target_datasets.items():
            try:
                logger.info(f"[Kaggle] Processing injury dataset: {dataset_key} ({config['sport']})")
                
                # Download dataset
                dataset_dir = self._download_dataset(config["dataset"], force=force_download)
                if not dataset_dir:
                    errors.append(f"Failed to download {dataset_key}")
                    continue
                
                # Find CSV files
                csv_files = list(dataset_dir.glob("**/*.csv"))
                if not csv_files:
                    logger.warning(f"[Kaggle] No CSV files found in {dataset_key}")
                    continue
                
                # Process each CSV file
                for csv_file in csv_files:
                    df = self._read_csv_safely(csv_file)
                    if df is None or df.empty:
                        continue
                    
                    # Parse injuries based on sport
                    sport = config["sport"]
                    if sport == "NBA":
                        injuries = self._parse_nba_injuries(df, sport)
                    elif sport == "NFL":
                        injuries = self._parse_nfl_injuries(df, sport)
                    else:
                        continue
                    
                    all_data["injuries"].extend(injuries)
                    total_records += len(injuries)
                
                logger.info(f"[Kaggle] Completed {dataset_key}: {len(all_data['injuries'])} total injuries")
                
                # Small delay between datasets
                await asyncio.sleep(1.0)
                
            except Exception as e:
                logger.error(f"[Kaggle] Error processing {dataset_key}: {e}")
                errors.append(str(e))
        
        logger.info(f"[Kaggle] Injury collection complete: {total_records} total records")
        
        return CollectorResult(
            success=total_records > 0,
            data=all_data,
            records_count=total_records,
            error="; ".join(errors) if errors else None,
            metadata={
                "datasets_processed": len(target_datasets),
                "sports": list(set(d["sport"] for d in target_datasets.values())),
                "data_type": "injuries",
            }
        )


# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

# Create singleton instance
kaggle_collector = KaggleCollector()

# Register with collector manager
collector_manager.register(kaggle_collector)

logger.info("Registered collector: Kaggle Datasets")