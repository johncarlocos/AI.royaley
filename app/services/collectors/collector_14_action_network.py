"""
ROYALEY - Action Network Public Betting Scraper Collector
Phase 1: Data Collection Services

Collects public betting data from actionnetwork.com using Selenium.
Based on: https://github.com/timsonater/action-network-scraper

Data Sources:
- Action Network Public Betting Pages
- URL: https://www.actionnetwork.com/{sport}/public-betting

Key Data Types:
- Public betting percentages (% of bets)
- Money percentages (% of money)
- Line movement tracking
- Sharp vs public indicators
- Reverse line movement detection

Sports Supported:
- NFL, NCAAF, NBA, NCAAB, NHL, MLB

Tables Filled:
- public_betting
- public_betting_history
- sharp_money_indicators
- fade_public_records
- consensus_lines (updated)

Requirements:
- Selenium WebDriver
- Chrome/Chromium browser
- chromedriver

Docker Requirements:
- chromium, chromium-chromedriver packages
- selenium python package
"""

import asyncio
import logging
import os
import re
import json
import hashlib
from datetime import datetime, date, timedelta
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from sqlalchemy import select, and_, or_, func as sql_func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.core.config import settings
from app.core.database import Base
from app.services.collectors.base_collector import (
    BaseCollector,
    CollectorResult,
    collector_manager,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Action Network base URLs
ACTION_NETWORK_BASE = "https://www.actionnetwork.com"

# Sport configurations
SPORT_CONFIGS = {
    "NFL": {
        "code": "NFL",
        "url_path": "nfl",
        "season_type": "week",  # week-based schedule
        "weeks_per_season": 18,
        "season_start_month": 9,  # September
        "public_betting_url": f"{ACTION_NETWORK_BASE}/nfl/public-betting",
        "scores_url": f"{ACTION_NETWORK_BASE}/nfl/scores",
    },
    "NCAAF": {
        "code": "NCAAF",
        "url_path": "ncaaf",
        "season_type": "week",
        "weeks_per_season": 15,
        "season_start_month": 8,  # August
        "public_betting_url": f"{ACTION_NETWORK_BASE}/ncaaf/public-betting",
        "scores_url": f"{ACTION_NETWORK_BASE}/ncaaf/scores",
    },
    "NBA": {
        "code": "NBA",
        "url_path": "nba",
        "season_type": "date",  # date-based schedule
        "season_start_month": 10,  # October
        "public_betting_url": f"{ACTION_NETWORK_BASE}/nba/public-betting",
        "scores_url": f"{ACTION_NETWORK_BASE}/nba/scores",
    },
    "NCAAB": {
        "code": "NCAAB",
        "url_path": "ncaab",
        "season_type": "date",
        "season_start_month": 11,  # November
        "public_betting_url": f"{ACTION_NETWORK_BASE}/ncaab/public-betting",
        "scores_url": f"{ACTION_NETWORK_BASE}/ncaab/scores",
    },
    "NHL": {
        "code": "NHL",
        "url_path": "nhl",
        "season_type": "date",
        "season_start_month": 10,  # October
        "public_betting_url": f"{ACTION_NETWORK_BASE}/nhl/public-betting",
        "scores_url": f"{ACTION_NETWORK_BASE}/nhl/scores",
    },
    "MLB": {
        "code": "MLB",
        "url_path": "mlb",
        "season_type": "date",
        "season_start_month": 3,  # March
        "public_betting_url": f"{ACTION_NETWORK_BASE}/mlb/public-betting",
        "scores_url": f"{ACTION_NETWORK_BASE}/mlb/scores",
    },
}

# Sharp money thresholds
SHARP_MONEY_THRESHOLD = 15  # % difference between bet% and money% indicates sharp action
RLM_THRESHOLD = 0.5  # Half point line movement against public
FADE_PUBLIC_THRESHOLD = 70  # Fade when public > 70%


# =============================================================================
# SELENIUM SETUP HELPERS
# =============================================================================

def get_webdriver():
    """
    Create and configure Selenium WebDriver.
    Uses Chrome in headless mode for Docker compatibility.
    """
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.service import Service
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
    except ImportError:
        logger.error("[ActionNetwork] Selenium not installed. Install with: pip install selenium")
        return None
    
    # Chrome options for headless operation
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    
    # Additional options to avoid detection
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option("useAutomationExtension", False)
    
    try:
        # Try to find chromedriver
        chromedriver_paths = [
            "/usr/bin/chromedriver",
            "/usr/local/bin/chromedriver",
            "/opt/chrome/chromedriver",
            "chromedriver",
        ]
        
        service = None
        for path in chromedriver_paths:
            if os.path.exists(path):
                service = Service(executable_path=path)
                break
        
        if service:
            driver = webdriver.Chrome(service=service, options=chrome_options)
        else:
            # Try default location
            driver = webdriver.Chrome(options=chrome_options)
        
        # Set implicit wait
        driver.implicitly_wait(10)
        
        # Execute CDP command to avoid detection
        driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
            "source": """
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                })
            """
        })
        
        return driver
        
    except Exception as e:
        logger.error(f"[ActionNetwork] Failed to create WebDriver: {e}")
        return None


def close_webdriver(driver):
    """Safely close WebDriver."""
    if driver:
        try:
            driver.quit()
        except Exception as e:
            logger.warning(f"[ActionNetwork] Error closing WebDriver: {e}")


# =============================================================================
# ACTION NETWORK COLLECTOR CLASS
# =============================================================================

class ActionNetworkCollector(BaseCollector):
    """
    Selenium-based scraper for Action Network public betting data.
    
    Features:
    - Public betting percentages (% of bets and % of money)
    - Line movement tracking
    - Sharp vs public indicators
    - Reverse line movement detection
    - Fade-the-public opportunities
    
    Supports: NFL, NCAAF, NBA, NCAAB, NHL, MLB
    
    Note: Historical data availability depends on Action Network's archive.
    Typically 2-3 seasons are available.
    """
    
    def __init__(self):
        super().__init__(
            name="action_network",
            base_url=ACTION_NETWORK_BASE,
            rate_limit=30,  # Be respectful - 30 requests per minute
            rate_window=60,
        )
        self._driver = None
    
    async def get_driver(self):
        """Get or create Selenium WebDriver."""
        if self._driver is None:
            self._driver = get_webdriver()
        return self._driver
    
    async def close(self):
        """Close WebDriver."""
        close_webdriver(self._driver)
        self._driver = None

    # =========================================================================
    # MAIN COLLECT METHOD
    # =========================================================================
    
    async def collect(
        self,
        sports: List[str] = None,
        days_back: int = 7,
        collect_type: str = "all"
    ) -> CollectorResult:
        """
        Collect public betting data from Action Network.
        
        Args:
            sports: List of sports to collect (default: all 6)
            days_back: Number of days to look back for historical data
            collect_type: Type of data to collect:
                - "all": All data types
                - "current": Today's games only
                - "historical": Historical games
        
        Returns:
            CollectorResult with collected data
        """
        if sports is None:
            sports = ["NFL", "NCAAF", "NBA", "NCAAB", "NHL", "MLB"]
        
        # Validate sports
        valid_sports = [s.upper() for s in sports if s.upper() in SPORT_CONFIGS]
        if not valid_sports:
            logger.warning("[ActionNetwork] No valid sports specified")
            return CollectorResult(success=False, data={}, records_count=0, error="No valid sports")
        
        logger.info(f"[ActionNetwork] Collecting public betting data for: {valid_sports}")
        logger.info(f"[ActionNetwork] Days back: {days_back}, Type: {collect_type}")
        
        data = {
            "public_betting": [],
            "sharp_indicators": [],
            "line_movements": [],
        }
        total_records = 0
        
        try:
            # Initialize WebDriver
            driver = await self.get_driver()
            if driver is None:
                return CollectorResult(
                    success=False,
                    data=data,
                    records_count=0,
                    error="Failed to initialize WebDriver. Ensure Selenium and ChromeDriver are installed."
                )
            
            for sport in valid_sports:
                sport_config = SPORT_CONFIGS[sport]
                
                try:
                    # Collect current public betting data
                    if collect_type in ["all", "current"]:
                        current_data = await self._scrape_public_betting(driver, sport_config)
                        data["public_betting"].extend(current_data)
                        total_records += len(current_data)
                        logger.info(f"[ActionNetwork] {sport}: {len(current_data)} current games")
                    
                    # Collect historical data
                    if collect_type in ["all", "historical"] and days_back > 0:
                        historical_data = await self._scrape_historical(driver, sport_config, days_back)
                        data["public_betting"].extend(historical_data)
                        total_records += len(historical_data)
                        logger.info(f"[ActionNetwork] {sport}: {len(historical_data)} historical games")
                    
                    # Rate limiting between sports
                    await asyncio.sleep(2)
                    
                except Exception as e:
                    logger.warning(f"[ActionNetwork] Error collecting {sport}: {e}")
                    continue
            
            # Analyze for sharp indicators
            sharp_indicators = self._detect_sharp_indicators(data["public_betting"])
            data["sharp_indicators"] = sharp_indicators
            total_records += len(sharp_indicators)
            
            logger.info(f"[ActionNetwork] Total records collected: {total_records}")
            logger.info(f"[ActionNetwork] Sharp indicators detected: {len(sharp_indicators)}")
            
            return CollectorResult(
                success=True,
                data=data,
                records_count=total_records,
            )
            
        except Exception as e:
            logger.error(f"[ActionNetwork] Collection error: {e}")
            import traceback
            traceback.print_exc()
            return CollectorResult(
                success=False,
                data=data,
                records_count=total_records,
                error=str(e)
            )
        finally:
            await self.close()

    # =========================================================================
    # SCRAPING METHODS
    # =========================================================================
    
    async def _scrape_public_betting(
        self,
        driver,
        sport_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Scrape current public betting data for a sport."""
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.common.exceptions import TimeoutException, NoSuchElementException
        
        games = []
        sport_code = sport_config["code"]
        url = sport_config["public_betting_url"]
        
        try:
            logger.info(f"[ActionNetwork] Scraping {sport_code} from {url}")
            driver.get(url)
            
            # Wait for page to load
            await asyncio.sleep(3)
            
            # Wait for game containers to load
            try:
                WebDriverWait(driver, 15).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "[class*='game'], [class*='Game'], [class*='matchup'], [class*='Matchup']"))
                )
            except TimeoutException:
                logger.warning(f"[ActionNetwork] Timeout waiting for games on {sport_code}")
                return games
            
            # Get page source and parse
            page_source = driver.page_source
            games = self._parse_public_betting_page(page_source, sport_code)
            
        except Exception as e:
            logger.warning(f"[ActionNetwork] Error scraping {sport_code}: {e}")
        
        return games
    
    async def _scrape_historical(
        self,
        driver,
        sport_config: Dict[str, Any],
        days_back: int
    ) -> List[Dict[str, Any]]:
        """Scrape historical public betting data."""
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        
        all_games = []
        sport_code = sport_config["code"]
        base_url = sport_config["scores_url"]
        
        # Generate dates to scrape
        today = date.today()
        dates_to_scrape = [today - timedelta(days=i) for i in range(1, days_back + 1)]
        
        for scrape_date in dates_to_scrape:
            try:
                # Format URL with date
                date_str = scrape_date.strftime("%Y%m%d")
                url = f"{base_url}?date={date_str}"
                
                logger.debug(f"[ActionNetwork] Scraping {sport_code} for {scrape_date}")
                driver.get(url)
                
                # Wait for page to load
                await asyncio.sleep(2)
                
                # Get page source and parse
                page_source = driver.page_source
                games = self._parse_scores_page(page_source, sport_code, scrape_date)
                all_games.extend(games)
                
                # Rate limiting
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.debug(f"[ActionNetwork] Error scraping {sport_code} for {scrape_date}: {e}")
                continue
        
        return all_games
    
    def _parse_public_betting_page(
        self,
        html: str,
        sport_code: str
    ) -> List[Dict[str, Any]]:
        """Parse public betting page HTML."""
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            logger.error("[ActionNetwork] BeautifulSoup not installed. Install with: pip install beautifulsoup4")
            return []
        
        games = []
        soup = BeautifulSoup(html, 'html.parser')
        
        # Action Network specific selectors (2024/2025 structure)
        game_containers = (
            soup.select('[class*="public-betting__game"]') or
            soup.select('[class*="mobile-public-betting__game-info"]') or
            soup.select('[class*="game-info__teams"]') or
            soup.select('[class*="game-card"]') or
            soup.select('[class*="GameCard"]') or
            soup.select('[class*="matchup"]') or
            soup.select('[data-testid*="game"]')
        )
        
        logger.debug(f"[ActionNetwork] Found {len(game_containers)} game containers")
        
        for container in game_containers:
            try:
                game_data = self._extract_game_data(container, sport_code)
                if game_data:
                    games.append(game_data)
            except Exception as e:
                logger.debug(f"[ActionNetwork] Error parsing game: {e}")
                continue
        
        # If no games found with containers, try row-based parsing
        if not games:
            games = self._parse_action_network_rows(soup, sport_code)
        
        # If still no games, try fallback
        if not games:
            games = self._parse_page_fallback(soup, sport_code)
        
        return games
    
    def _parse_action_network_rows(
        self,
        soup,
        sport_code: str
    ) -> List[Dict[str, Any]]:
        """Parse Action Network using row-based structure."""
        games = []
        
        # Find all game-info sections which contain team pairs
        game_sections = soup.select('[class*="game-info"]')
        
        # Group by finding team pairs
        team_elements = soup.select('[class*="game-info__team-info"], [class*="game-info__team--desktop"]')
        
        logger.debug(f"[ActionNetwork] Found {len(team_elements)} team elements")
        
        # Extract all percentage values from page
        import re
        page_text = soup.get_text()
        all_percentages = re.findall(r'(\d{1,2})%', page_text)
        
        # Find team names
        team_name_elements = soup.select('[class*="game-info__team-info"] span, [class*="team-name"], [class*="TeamName"]')
        team_names = []
        for el in team_name_elements:
            name = el.get_text(strip=True)
            if name and len(name) > 2 and not name.isdigit():
                team_names.append(name)
        
        # Also try to find team names from links
        team_links = soup.select('a[href*="/teams/"]')
        for link in team_links:
            name = link.get_text(strip=True)
            if name and len(name) > 2 and name not in team_names:
                team_names.append(name)
        
        logger.debug(f"[ActionNetwork] Found team names: {team_names[:20]}")
        logger.debug(f"[ActionNetwork] Found percentages: {all_percentages[:20]}")
        
        # Pair teams (away, home) and create games
        # Typically teams are listed in pairs with away first
        i = 0
        pct_idx = 0
        while i < len(team_names) - 1:
            away_team = self._clean_team_name(team_names[i])
            home_team = self._clean_team_name(team_names[i + 1])
            
            if away_team and home_team and away_team != home_team:
                game_data = {
                    "sport_code": sport_code,
                    "away_team": away_team,
                    "home_team": home_team,
                    "game_date": date.today(),
                    "source": "action_network",
                    "scraped_at": datetime.now(),
                }
                
                # Try to assign percentages (typically 2 per bet type: away%, home%)
                # Structure: spread_away, spread_home, ml_away, ml_home, over, under
                if pct_idx + 1 < len(all_percentages):
                    game_data["spread_away_bet_pct"] = float(all_percentages[pct_idx])
                    game_data["spread_home_bet_pct"] = float(all_percentages[pct_idx + 1])
                    pct_idx += 2
                
                if pct_idx + 1 < len(all_percentages):
                    game_data["ml_away_bet_pct"] = float(all_percentages[pct_idx])
                    game_data["ml_home_bet_pct"] = float(all_percentages[pct_idx + 1])
                    pct_idx += 2
                
                if pct_idx + 1 < len(all_percentages):
                    game_data["total_over_bet_pct"] = float(all_percentages[pct_idx])
                    game_data["total_under_bet_pct"] = float(all_percentages[pct_idx + 1])
                    pct_idx += 2
                
                games.append(game_data)
            
            i += 2  # Move to next pair
        
        return games
    
    def _parse_scores_page(
        self,
        html: str,
        sport_code: str,
        game_date: date
    ) -> List[Dict[str, Any]]:
        """Parse scores/results page HTML."""
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            return []
        
        games = []
        soup = BeautifulSoup(html, 'html.parser')
        
        # Find game containers
        game_containers = (
            soup.select('[class*="scoreboard"]') or
            soup.select('[class*="game-card"]') or
            soup.select('[class*="GameCard"]')
        )
        
        for container in game_containers:
            try:
                game_data = self._extract_game_data(container, sport_code)
                if game_data:
                    game_data["game_date"] = game_date
                    games.append(game_data)
            except Exception as e:
                continue
        
        return games
    
    def _extract_game_data(
        self,
        container,
        sport_code: str
    ) -> Optional[Dict[str, Any]]:
        """Extract game data from a container element."""
        try:
            # Action Network specific selectors
            team_elements = (
                container.select('[class*="game-info__team-info"]') or
                container.select('[class*="game-info__team--desktop"]') or
                container.select('[class*="team-name"]') or
                container.select('[class*="team"], [class*="Team"]')
            )
            
            if len(team_elements) < 2:
                # Try finding team names from text
                team_elements = container.find_all(['span', 'div', 'a'], string=re.compile(r'^[A-Z][a-z]{2,}'))
            
            if len(team_elements) < 2:
                return None
            
            # Extract team names (typically away team first, then home)
            away_team = self._clean_team_name(team_elements[0].get_text(strip=True))
            home_team = self._clean_team_name(team_elements[1].get_text(strip=True))
            
            if not away_team or not home_team or away_team == home_team:
                return None
            
            # Skip if looks like navigation text
            skip_words = ['home', 'odds', 'picks', 'teams', 'betting', 'news', 'futures', 'props']
            if away_team.lower() in skip_words or home_team.lower() in skip_words:
                return None
            
            game_data = {
                "sport_code": sport_code,
                "home_team": home_team,
                "away_team": away_team,
                "game_date": date.today(),
                "game_time": self._extract_game_time(container),
                "source": "action_network",
                "scraped_at": datetime.now(),
            }
            
            # Extract percentages from container
            container_text = container.get_text()
            pct_matches = re.findall(r'(\d{1,2})%', container_text)
            
            if len(pct_matches) >= 2:
                game_data["spread_away_bet_pct"] = float(pct_matches[0])
                game_data["spread_home_bet_pct"] = float(pct_matches[1])
            
            if len(pct_matches) >= 4:
                game_data["ml_away_bet_pct"] = float(pct_matches[2])
                game_data["ml_home_bet_pct"] = float(pct_matches[3])
            
            if len(pct_matches) >= 6:
                game_data["total_over_bet_pct"] = float(pct_matches[4])
                game_data["total_under_bet_pct"] = float(pct_matches[5])
            
            # Extract spread betting data (from specific elements if available)
            spread_data = self._extract_spread_data(container)
            for key, val in spread_data.items():
                if val is not None and key not in game_data:
                    game_data[key] = val
            
            # Extract moneyline data
            ml_data = self._extract_moneyline_data(container)
            for key, val in ml_data.items():
                if val is not None and key not in game_data:
                    game_data[key] = val
            
            # Extract total (over/under) data
            total_data = self._extract_total_data(container)
            for key, val in total_data.items():
                if val is not None and key not in game_data:
                    game_data[key] = val
            
            # Extract scores if available
            scores = self._extract_scores(container)
            game_data.update(scores)
            
            return game_data
            
        except Exception as e:
            logger.debug(f"[ActionNetwork] Error extracting game data: {e}")
            return None
    
    def _extract_spread_data(self, container) -> Dict[str, Any]:
        """Extract spread betting percentages."""
        data = {
            "spread_home_bet_pct": None,
            "spread_away_bet_pct": None,
            "spread_home_money_pct": None,
            "spread_away_money_pct": None,
            "spread_bet_count": None,
            "spread_line": None,
            "spread_opening_line": None,
        }
        
        try:
            # Look for spread section
            spread_section = container.select_one('[class*="spread"], [data-market="spread"]')
            if spread_section:
                percentages = self._extract_percentages(spread_section)
                if percentages:
                    data["spread_away_bet_pct"] = percentages.get("bet_pct_1")
                    data["spread_home_bet_pct"] = percentages.get("bet_pct_2")
                    data["spread_away_money_pct"] = percentages.get("money_pct_1")
                    data["spread_home_money_pct"] = percentages.get("money_pct_2")
                
                # Extract spread line
                line_match = re.search(r'([+-]?\d+\.?\d*)', spread_section.get_text())
                if line_match:
                    data["spread_line"] = float(line_match.group(1))
        except:
            pass
        
        return data
    
    def _extract_moneyline_data(self, container) -> Dict[str, Any]:
        """Extract moneyline betting percentages."""
        data = {
            "ml_home_bet_pct": None,
            "ml_away_bet_pct": None,
            "ml_home_money_pct": None,
            "ml_away_money_pct": None,
            "ml_bet_count": None,
            "ml_home_odds": None,
            "ml_away_odds": None,
        }
        
        try:
            ml_section = container.select_one('[class*="moneyline"], [class*="money-line"], [data-market="moneyline"]')
            if ml_section:
                percentages = self._extract_percentages(ml_section)
                if percentages:
                    data["ml_away_bet_pct"] = percentages.get("bet_pct_1")
                    data["ml_home_bet_pct"] = percentages.get("bet_pct_2")
                    data["ml_away_money_pct"] = percentages.get("money_pct_1")
                    data["ml_home_money_pct"] = percentages.get("money_pct_2")
                
                # Extract odds
                odds_matches = re.findall(r'([+-]\d+)', ml_section.get_text())
                if len(odds_matches) >= 2:
                    data["ml_away_odds"] = int(odds_matches[0])
                    data["ml_home_odds"] = int(odds_matches[1])
        except:
            pass
        
        return data
    
    def _extract_total_data(self, container) -> Dict[str, Any]:
        """Extract total (over/under) betting percentages."""
        data = {
            "total_over_bet_pct": None,
            "total_under_bet_pct": None,
            "total_over_money_pct": None,
            "total_under_money_pct": None,
            "total_bet_count": None,
            "total_line": None,
            "total_opening_line": None,
        }
        
        try:
            total_section = container.select_one('[class*="total"], [class*="over-under"], [data-market="total"]')
            if total_section:
                percentages = self._extract_percentages(total_section)
                if percentages:
                    data["total_over_bet_pct"] = percentages.get("bet_pct_1")
                    data["total_under_bet_pct"] = percentages.get("bet_pct_2")
                    data["total_over_money_pct"] = percentages.get("money_pct_1")
                    data["total_under_money_pct"] = percentages.get("money_pct_2")
                
                # Extract total line
                line_match = re.search(r'(\d+\.?\d*)', total_section.get_text())
                if line_match:
                    data["total_line"] = float(line_match.group(1))
        except:
            pass
        
        return data
    
    def _extract_percentages(self, section) -> Dict[str, Optional[float]]:
        """Extract percentage values from a section."""
        data = {
            "bet_pct_1": None,
            "bet_pct_2": None,
            "money_pct_1": None,
            "money_pct_2": None,
        }
        
        try:
            text = section.get_text()
            # Find all percentages (format: XX%)
            pct_matches = re.findall(r'(\d+)%', text)
            
            if len(pct_matches) >= 2:
                data["bet_pct_1"] = float(pct_matches[0])
                data["bet_pct_2"] = float(pct_matches[1])
            
            if len(pct_matches) >= 4:
                data["money_pct_1"] = float(pct_matches[2])
                data["money_pct_2"] = float(pct_matches[3])
            elif len(pct_matches) == 2:
                # If only 2 percentages, they might be bet percentages
                # Money percentages would need to be calculated or are same as bet
                data["money_pct_1"] = data["bet_pct_1"]
                data["money_pct_2"] = data["bet_pct_2"]
        except:
            pass
        
        return data
    
    def _extract_game_time(self, container) -> Optional[str]:
        """Extract game time from container."""
        try:
            time_element = container.select_one('[class*="time"], [class*="Time"], time')
            if time_element:
                return time_element.get_text(strip=True)
        except:
            pass
        return None
    
    def _extract_scores(self, container) -> Dict[str, Optional[int]]:
        """Extract scores if available."""
        scores = {
            "home_score": None,
            "away_score": None,
            "game_status": "scheduled",
        }
        
        try:
            score_elements = container.select('[class*="score"], [class*="Score"]')
            if len(score_elements) >= 2:
                away_score = re.search(r'\d+', score_elements[0].get_text())
                home_score = re.search(r'\d+', score_elements[1].get_text())
                
                if away_score:
                    scores["away_score"] = int(away_score.group())
                if home_score:
                    scores["home_score"] = int(home_score.group())
                
                if scores["home_score"] is not None:
                    scores["game_status"] = "final"
        except:
            pass
        
        return scores
    
    def _clean_team_name(self, name: str) -> str:
        """Clean team name."""
        if not name:
            return ""
        # Remove common prefixes/suffixes
        name = re.sub(r'^(at|vs\.?|@)\s*', '', name, flags=re.IGNORECASE)
        name = re.sub(r'\s*\(.*\)$', '', name)
        # Remove rotation numbers (e.g., "NE110", "SEA109", "BOS501")
        name = re.sub(r'[A-Z]{2,4}\d{3,4}$', '', name)
        # Remove trailing state abbreviations and numbers (e.g., "PatriotsNE110" -> "Patriots")
        name = re.sub(r'([a-z])([A-Z]{2}\d+)$', r'\1', name)
        # Remove any remaining trailing numbers
        name = re.sub(r'\d+$', '', name)
        # Remove trailing abbreviations that got stuck (e.g., "RockiesCOL" -> "Rockies")
        name = re.sub(r'([a-z])([A-Z]{2,3})$', r'\1', name)
        return name.strip()
    
    def _parse_page_fallback(self, soup, sport_code: str) -> List[Dict[str, Any]]:
        """Fallback parsing method using JSON data."""
        games = []
        
        try:
            # Look for JSON data in script tags
            scripts = soup.find_all('script', type='application/json')
            for script in scripts:
                try:
                    data = json.loads(script.string)
                    if isinstance(data, dict):
                        games_data = self._extract_games_from_json(data, sport_code)
                        games.extend(games_data)
                except:
                    continue
            
            # Also try __NEXT_DATA__ for Next.js apps
            next_data = soup.find('script', id='__NEXT_DATA__')
            if next_data:
                try:
                    data = json.loads(next_data.string)
                    games_data = self._extract_games_from_json(data, sport_code)
                    games.extend(games_data)
                except:
                    pass
        except:
            pass
        
        return games
    
    def _extract_games_from_json(
        self,
        data: Dict,
        sport_code: str
    ) -> List[Dict[str, Any]]:
        """Extract games from JSON data."""
        games = []
        
        def search_for_games(obj, depth=0):
            if depth > 10:
                return []
            
            found = []
            if isinstance(obj, dict):
                # Check if this looks like a game object
                if all(key in obj for key in ['homeTeam', 'awayTeam']) or \
                   all(key in obj for key in ['home_team', 'away_team']):
                    found.append(self._convert_json_game(obj, sport_code))
                
                # Check for games array
                for key in ['games', 'events', 'matchups', 'contests']:
                    if key in obj and isinstance(obj[key], list):
                        for item in obj[key]:
                            found.extend(search_for_games(item, depth + 1))
                
                # Recurse into other objects
                for value in obj.values():
                    found.extend(search_for_games(value, depth + 1))
            
            elif isinstance(obj, list):
                for item in obj:
                    found.extend(search_for_games(item, depth + 1))
            
            return found
        
        games = search_for_games(data)
        return [g for g in games if g is not None]
    
    def _convert_json_game(self, obj: Dict, sport_code: str) -> Optional[Dict[str, Any]]:
        """Convert JSON game object to standard format."""
        try:
            # Try different key formats
            home_team = obj.get('homeTeam', obj.get('home_team', {}))
            away_team = obj.get('awayTeam', obj.get('away_team', {}))
            
            if isinstance(home_team, dict):
                home_name = home_team.get('name', home_team.get('displayName', ''))
            else:
                home_name = str(home_team)
            
            if isinstance(away_team, dict):
                away_name = away_team.get('name', away_team.get('displayName', ''))
            else:
                away_name = str(away_team)
            
            if not home_name or not away_name:
                return None
            
            game_data = {
                "sport_code": sport_code,
                "home_team": home_name,
                "away_team": away_name,
                "game_date": date.today(),
                "source": "action_network",
                "scraped_at": datetime.now(),
            }
            
            # Extract betting percentages if available
            betting = obj.get('betting', obj.get('publicBetting', {}))
            if betting:
                spread = betting.get('spread', {})
                ml = betting.get('moneyline', {})
                total = betting.get('total', {})
                
                if spread:
                    game_data["spread_home_bet_pct"] = spread.get('homeBetPct', spread.get('home_bet_pct'))
                    game_data["spread_away_bet_pct"] = spread.get('awayBetPct', spread.get('away_bet_pct'))
                    game_data["spread_home_money_pct"] = spread.get('homeMoneyPct', spread.get('home_money_pct'))
                    game_data["spread_away_money_pct"] = spread.get('awayMoneyPct', spread.get('away_money_pct'))
                    game_data["spread_line"] = spread.get('line')
                
                if ml:
                    game_data["ml_home_bet_pct"] = ml.get('homeBetPct', ml.get('home_bet_pct'))
                    game_data["ml_away_bet_pct"] = ml.get('awayBetPct', ml.get('away_bet_pct'))
                    game_data["ml_home_odds"] = ml.get('homeOdds', ml.get('home_odds'))
                    game_data["ml_away_odds"] = ml.get('awayOdds', ml.get('away_odds'))
                
                if total:
                    game_data["total_over_bet_pct"] = total.get('overBetPct', total.get('over_bet_pct'))
                    game_data["total_under_bet_pct"] = total.get('underBetPct', total.get('under_bet_pct'))
                    game_data["total_line"] = total.get('line')
            
            return game_data
            
        except:
            return None

    # =========================================================================
    # SHARP INDICATOR DETECTION
    # =========================================================================
    
    def _detect_sharp_indicators(
        self,
        games: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Detect sharp money indicators from public betting data.
        
        Indicators:
        1. Reverse Line Movement (RLM)
        2. Money % significantly higher than Bet %
        3. Steam moves
        """
        indicators = []
        
        for game in games:
            # Detect sharp action on spread
            spread_indicator = self._check_sharp_spread(game)
            if spread_indicator:
                indicators.append(spread_indicator)
                game["is_sharp_spread"] = True
                game["sharp_side_spread"] = spread_indicator["sharp_side"]
            
            # Detect sharp action on moneyline
            ml_indicator = self._check_sharp_moneyline(game)
            if ml_indicator:
                indicators.append(ml_indicator)
                game["is_sharp_ml"] = True
                game["sharp_side_ml"] = ml_indicator["sharp_side"]
            
            # Detect sharp action on total
            total_indicator = self._check_sharp_total(game)
            if total_indicator:
                indicators.append(total_indicator)
                game["is_sharp_total"] = True
                game["sharp_side_total"] = total_indicator["sharp_side"]
            
            # Detect reverse line movement
            rlm_spread = self._check_rlm_spread(game)
            if rlm_spread:
                game["is_rlm_spread"] = True
                indicators.append(rlm_spread)
            
            rlm_total = self._check_rlm_total(game)
            if rlm_total:
                game["is_rlm_total"] = True
                indicators.append(rlm_total)
        
        return indicators
    
    def _check_sharp_spread(self, game: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check for sharp action on spread."""
        home_bet_pct = game.get("spread_home_bet_pct")
        home_money_pct = game.get("spread_home_money_pct")
        away_bet_pct = game.get("spread_away_bet_pct")
        away_money_pct = game.get("spread_away_money_pct")
        
        if not all([home_bet_pct, home_money_pct, away_bet_pct, away_money_pct]):
            return None
        
        # Check home side divergence
        home_divergence = home_money_pct - home_bet_pct
        away_divergence = away_money_pct - away_bet_pct
        
        if home_divergence >= SHARP_MONEY_THRESHOLD:
            return {
                "sport_code": game.get("sport_code"),
                "home_team": game.get("home_team"),
                "away_team": game.get("away_team"),
                "game_date": game.get("game_date"),
                "bet_type": "spread",
                "indicator_type": "money_pct_divergence",
                "sharp_side": "home",
                "public_bet_pct": away_bet_pct,
                "money_pct": home_money_pct,
                "divergence": home_divergence,
                "detected_at": datetime.now(),
            }
        
        if away_divergence >= SHARP_MONEY_THRESHOLD:
            return {
                "sport_code": game.get("sport_code"),
                "home_team": game.get("home_team"),
                "away_team": game.get("away_team"),
                "game_date": game.get("game_date"),
                "bet_type": "spread",
                "indicator_type": "money_pct_divergence",
                "sharp_side": "away",
                "public_bet_pct": home_bet_pct,
                "money_pct": away_money_pct,
                "divergence": away_divergence,
                "detected_at": datetime.now(),
            }
        
        return None
    
    def _check_sharp_moneyline(self, game: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check for sharp action on moneyline."""
        home_bet_pct = game.get("ml_home_bet_pct")
        home_money_pct = game.get("ml_home_money_pct")
        away_bet_pct = game.get("ml_away_bet_pct")
        away_money_pct = game.get("ml_away_money_pct")
        
        if not all([home_bet_pct, home_money_pct]):
            return None
        
        home_divergence = (home_money_pct or 0) - (home_bet_pct or 0)
        
        if home_divergence >= SHARP_MONEY_THRESHOLD:
            return {
                "sport_code": game.get("sport_code"),
                "home_team": game.get("home_team"),
                "away_team": game.get("away_team"),
                "game_date": game.get("game_date"),
                "bet_type": "moneyline",
                "indicator_type": "money_pct_divergence",
                "sharp_side": "home",
                "public_bet_pct": away_bet_pct,
                "money_pct": home_money_pct,
                "divergence": home_divergence,
                "detected_at": datetime.now(),
            }
        
        if home_divergence <= -SHARP_MONEY_THRESHOLD:
            return {
                "sport_code": game.get("sport_code"),
                "home_team": game.get("home_team"),
                "away_team": game.get("away_team"),
                "game_date": game.get("game_date"),
                "bet_type": "moneyline",
                "indicator_type": "money_pct_divergence",
                "sharp_side": "away",
                "public_bet_pct": home_bet_pct,
                "money_pct": away_money_pct,
                "divergence": abs(home_divergence),
                "detected_at": datetime.now(),
            }
        
        return None
    
    def _check_sharp_total(self, game: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check for sharp action on total."""
        over_bet_pct = game.get("total_over_bet_pct")
        over_money_pct = game.get("total_over_money_pct")
        under_bet_pct = game.get("total_under_bet_pct")
        under_money_pct = game.get("total_under_money_pct")
        
        if not all([over_bet_pct, over_money_pct]):
            return None
        
        over_divergence = (over_money_pct or 0) - (over_bet_pct or 0)
        
        if over_divergence >= SHARP_MONEY_THRESHOLD:
            return {
                "sport_code": game.get("sport_code"),
                "home_team": game.get("home_team"),
                "away_team": game.get("away_team"),
                "game_date": game.get("game_date"),
                "bet_type": "total",
                "indicator_type": "money_pct_divergence",
                "sharp_side": "over",
                "public_bet_pct": under_bet_pct,
                "money_pct": over_money_pct,
                "divergence": over_divergence,
                "detected_at": datetime.now(),
            }
        
        if over_divergence <= -SHARP_MONEY_THRESHOLD:
            return {
                "sport_code": game.get("sport_code"),
                "home_team": game.get("home_team"),
                "away_team": game.get("away_team"),
                "game_date": game.get("game_date"),
                "bet_type": "total",
                "indicator_type": "money_pct_divergence",
                "sharp_side": "under",
                "public_bet_pct": over_bet_pct,
                "money_pct": under_money_pct,
                "divergence": abs(over_divergence),
                "detected_at": datetime.now(),
            }
        
        return None
    
    def _check_rlm_spread(self, game: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check for reverse line movement on spread."""
        opening_line = game.get("spread_opening_line")
        current_line = game.get("spread_line")
        home_bet_pct = game.get("spread_home_bet_pct")
        
        if not all([opening_line, current_line, home_bet_pct]):
            return None
        
        line_movement = current_line - opening_line
        
        # RLM: Line moves against public
        # If public is on home (>50%) but line moves toward home (more positive)
        if home_bet_pct > 50 and line_movement >= RLM_THRESHOLD:
            return {
                "sport_code": game.get("sport_code"),
                "home_team": game.get("home_team"),
                "away_team": game.get("away_team"),
                "game_date": game.get("game_date"),
                "bet_type": "spread",
                "indicator_type": "reverse_line_movement",
                "sharp_side": "away",
                "public_bet_pct": home_bet_pct,
                "line_before": opening_line,
                "line_after": current_line,
                "line_movement": line_movement,
                "detected_at": datetime.now(),
            }
        
        # If public is on away (home<50%) but line moves toward away (more negative)
        if home_bet_pct < 50 and line_movement <= -RLM_THRESHOLD:
            return {
                "sport_code": game.get("sport_code"),
                "home_team": game.get("home_team"),
                "away_team": game.get("away_team"),
                "game_date": game.get("game_date"),
                "bet_type": "spread",
                "indicator_type": "reverse_line_movement",
                "sharp_side": "home",
                "public_bet_pct": 100 - home_bet_pct,
                "line_before": opening_line,
                "line_after": current_line,
                "line_movement": line_movement,
                "detected_at": datetime.now(),
            }
        
        return None
    
    def _check_rlm_total(self, game: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check for reverse line movement on total."""
        opening_line = game.get("total_opening_line")
        current_line = game.get("total_line")
        over_bet_pct = game.get("total_over_bet_pct")
        
        if not all([opening_line, current_line, over_bet_pct]):
            return None
        
        line_movement = current_line - opening_line
        
        # RLM: Line moves against public
        if over_bet_pct > 50 and line_movement >= RLM_THRESHOLD:
            return {
                "sport_code": game.get("sport_code"),
                "home_team": game.get("home_team"),
                "away_team": game.get("away_team"),
                "game_date": game.get("game_date"),
                "bet_type": "total",
                "indicator_type": "reverse_line_movement",
                "sharp_side": "under",
                "public_bet_pct": over_bet_pct,
                "line_before": opening_line,
                "line_after": current_line,
                "line_movement": line_movement,
                "detected_at": datetime.now(),
            }
        
        if over_bet_pct < 50 and line_movement <= -RLM_THRESHOLD:
            return {
                "sport_code": game.get("sport_code"),
                "home_team": game.get("home_team"),
                "away_team": game.get("away_team"),
                "game_date": game.get("game_date"),
                "bet_type": "total",
                "indicator_type": "reverse_line_movement",
                "sharp_side": "over",
                "public_bet_pct": 100 - over_bet_pct,
                "line_before": opening_line,
                "line_after": current_line,
                "line_movement": line_movement,
                "detected_at": datetime.now(),
            }
        
        return None

    # =========================================================================
    # DATABASE SAVE METHODS
    # =========================================================================
    
    async def save_to_database(self, data: Dict[str, Any], session: AsyncSession) -> int:
        """Save all collected data to database."""
        total_saved = 0
        
        try:
            # Import models
            from app.models.public_betting_models import (
                PublicBetting, PublicBettingHistory, SharpMoneyIndicator
            )
            
            # Save public betting data
            if data.get("public_betting"):
                saved = await self._save_public_betting(session, data["public_betting"])
                total_saved += saved
                logger.info(f"[ActionNetwork] Saved {saved} public betting records")
            
            # Save sharp indicators
            if data.get("sharp_indicators"):
                saved = await self._save_sharp_indicators(session, data["sharp_indicators"])
                total_saved += saved
                logger.info(f"[ActionNetwork] Saved {saved} sharp indicators")
            
            await session.commit()
            
        except ImportError as e:
            logger.warning(f"[ActionNetwork] Models not found, skipping database save: {e}")
            logger.warning("[ActionNetwork] Run database migration to create public_betting tables")
        except Exception as e:
            logger.error(f"[ActionNetwork] Error saving to database: {e}")
            await session.rollback()
            raise
        
        return total_saved
    
    async def _save_public_betting(
        self,
        session: AsyncSession,
        records: List[Dict[str, Any]]
    ) -> int:
        """Save public betting records to database."""
        from app.models.public_betting_models import PublicBetting
        
        saved = 0
        updated = 0
        
        for record in records:
            try:
                sport_code = record.get("sport_code", "")
                home_team = record.get("home_team", "")
                away_team = record.get("away_team", "")
                game_date = record.get("game_date", date.today())
                
                if not all([sport_code, home_team, away_team]):
                    continue
                
                # Check for existing record
                result = await session.execute(
                    select(PublicBetting).where(
                        and_(
                            PublicBetting.sport_code == sport_code,
                            PublicBetting.home_team == home_team,
                            PublicBetting.away_team == away_team,
                            PublicBetting.game_date == game_date,
                            PublicBetting.source == "action_network"
                        )
                    )
                )
                existing = result.scalar_one_or_none()
                
                if existing:
                    # Update existing record
                    for key, value in record.items():
                        if hasattr(existing, key) and value is not None:
                            setattr(existing, key, value)
                    existing.updated_at = datetime.now()
                    updated += 1
                else:
                    # Create new record
                    pb_record = PublicBetting(
                        sport_code=sport_code,
                        home_team=home_team,
                        away_team=away_team,
                        game_date=game_date,
                        game_time=record.get("game_time"),
                        spread_home_bet_pct=record.get("spread_home_bet_pct"),
                        spread_away_bet_pct=record.get("spread_away_bet_pct"),
                        spread_home_money_pct=record.get("spread_home_money_pct"),
                        spread_away_money_pct=record.get("spread_away_money_pct"),
                        spread_bet_count=record.get("spread_bet_count"),
                        spread_line=record.get("spread_line"),
                        spread_opening_line=record.get("spread_opening_line"),
                        ml_home_bet_pct=record.get("ml_home_bet_pct"),
                        ml_away_bet_pct=record.get("ml_away_bet_pct"),
                        ml_home_money_pct=record.get("ml_home_money_pct"),
                        ml_away_money_pct=record.get("ml_away_money_pct"),
                        ml_bet_count=record.get("ml_bet_count"),
                        ml_home_odds=record.get("ml_home_odds"),
                        ml_away_odds=record.get("ml_away_odds"),
                        total_over_bet_pct=record.get("total_over_bet_pct"),
                        total_under_bet_pct=record.get("total_under_bet_pct"),
                        total_over_money_pct=record.get("total_over_money_pct"),
                        total_under_money_pct=record.get("total_under_money_pct"),
                        total_bet_count=record.get("total_bet_count"),
                        total_line=record.get("total_line"),
                        total_opening_line=record.get("total_opening_line"),
                        is_sharp_spread=record.get("is_sharp_spread"),
                        is_sharp_ml=record.get("is_sharp_ml"),
                        is_sharp_total=record.get("is_sharp_total"),
                        sharp_side_spread=record.get("sharp_side_spread"),
                        sharp_side_ml=record.get("sharp_side_ml"),
                        sharp_side_total=record.get("sharp_side_total"),
                        is_rlm_spread=record.get("is_rlm_spread"),
                        is_rlm_total=record.get("is_rlm_total"),
                        home_score=record.get("home_score"),
                        away_score=record.get("away_score"),
                        game_status=record.get("game_status", "scheduled"),
                        source="action_network",
                    )
                    session.add(pb_record)
                    saved += 1
                
            except Exception as e:
                logger.debug(f"[ActionNetwork] Error saving public betting record: {e}")
                continue
        
        await session.flush()
        logger.info(f"[ActionNetwork] Public betting: {saved} new, {updated} updated")
        return saved + updated
    
    async def _save_sharp_indicators(
        self,
        session: AsyncSession,
        indicators: List[Dict[str, Any]]
    ) -> int:
        """Save sharp money indicators to database."""
        from app.models.public_betting_models import SharpMoneyIndicator
        
        saved = 0
        
        for indicator in indicators:
            try:
                sharp_indicator = SharpMoneyIndicator(
                    sport_code=indicator.get("sport_code", ""),
                    home_team=indicator.get("home_team", ""),
                    away_team=indicator.get("away_team", ""),
                    game_date=indicator.get("game_date", date.today()),
                    bet_type=indicator.get("bet_type", ""),
                    indicator_type=indicator.get("indicator_type", ""),
                    sharp_side=indicator.get("sharp_side", ""),
                    public_bet_pct=indicator.get("public_bet_pct"),
                    money_pct=indicator.get("money_pct"),
                    divergence=indicator.get("divergence"),
                    line_before=indicator.get("line_before"),
                    line_after=indicator.get("line_after"),
                    line_movement=indicator.get("line_movement"),
                    detected_at=indicator.get("detected_at", datetime.now()),
                    source="action_network",
                )
                session.add(sharp_indicator)
                saved += 1
                
            except Exception as e:
                logger.debug(f"[ActionNetwork] Error saving sharp indicator: {e}")
                continue
        
        await session.flush()
        return saved

    # =========================================================================
    # VALIDATION METHOD
    # =========================================================================
    
    async def validate(self, data: Any) -> bool:
        """Validate collected data."""
        if data is None:
            return False
        if not isinstance(data, dict):
            return False
        
        has_public_betting = len(data.get("public_betting", [])) > 0
        
        return has_public_betting


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

action_network_collector = ActionNetworkCollector()

# Register with collector manager
collector_manager.register(action_network_collector)
logger.info("Registered collector: Action Network")