"""
ROYALEY - Populate Master Teams & Source Registry
Phase 2: Insert canonical team records + register all data sources.

Run: python -m scripts.populate_master_data

This is idempotent — safe to run multiple times (uses ON CONFLICT DO NOTHING).
"""

import asyncio
import logging
import sys
import os
from uuid import uuid4

# Allow running as script from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text
from app.core.database import db_manager

logger = logging.getLogger(__name__)

# =============================================================================
# OFFICIAL TEAM ROSTERS (sport_code, canonical_name, abbreviation, city, conference, division)
# =============================================================================

NFL_TEAMS = [
    ("Arizona Cardinals", "ARI", "Glendale", "NFC", "NFC West"),
    ("Atlanta Falcons", "ATL", "Atlanta", "NFC", "NFC South"),
    ("Baltimore Ravens", "BAL", "Baltimore", "AFC", "AFC North"),
    ("Buffalo Bills", "BUF", "Buffalo", "AFC", "AFC East"),
    ("Carolina Panthers", "CAR", "Charlotte", "NFC", "NFC South"),
    ("Chicago Bears", "CHI", "Chicago", "NFC", "NFC North"),
    ("Cincinnati Bengals", "CIN", "Cincinnati", "AFC", "AFC North"),
    ("Cleveland Browns", "CLE", "Cleveland", "AFC", "AFC North"),
    ("Dallas Cowboys", "DAL", "Dallas", "NFC", "NFC East"),
    ("Denver Broncos", "DEN", "Denver", "AFC", "AFC West"),
    ("Detroit Lions", "DET", "Detroit", "NFC", "NFC North"),
    ("Green Bay Packers", "GB", "Green Bay", "NFC", "NFC North"),
    ("Houston Texans", "HOU", "Houston", "AFC", "AFC South"),
    ("Indianapolis Colts", "IND", "Indianapolis", "AFC", "AFC South"),
    ("Jacksonville Jaguars", "JAX", "Jacksonville", "AFC", "AFC South"),
    ("Kansas City Chiefs", "KC", "Kansas City", "AFC", "AFC West"),
    ("Las Vegas Raiders", "LV", "Las Vegas", "AFC", "AFC West"),
    ("Los Angeles Chargers", "LAC", "Los Angeles", "AFC", "AFC West"),
    ("Los Angeles Rams", "LAR", "Los Angeles", "NFC", "NFC West"),
    ("Miami Dolphins", "MIA", "Miami", "AFC", "AFC East"),
    ("Minnesota Vikings", "MIN", "Minneapolis", "NFC", "NFC North"),
    ("New England Patriots", "NE", "Foxborough", "AFC", "AFC East"),
    ("New Orleans Saints", "NO", "New Orleans", "NFC", "NFC South"),
    ("New York Giants", "NYG", "East Rutherford", "NFC", "NFC East"),
    ("New York Jets", "NYJ", "East Rutherford", "AFC", "AFC East"),
    ("Philadelphia Eagles", "PHI", "Philadelphia", "NFC", "NFC East"),
    ("Pittsburgh Steelers", "PIT", "Pittsburgh", "AFC", "AFC North"),
    ("San Francisco 49ers", "SF", "Santa Clara", "NFC", "NFC West"),
    ("Seattle Seahawks", "SEA", "Seattle", "NFC", "NFC West"),
    ("Tampa Bay Buccaneers", "TB", "Tampa", "NFC", "NFC South"),
    ("Tennessee Titans", "TEN", "Nashville", "AFC", "AFC South"),
    ("Washington Commanders", "WAS", "Landover", "NFC", "NFC East"),
]

NBA_TEAMS = [
    ("Atlanta Hawks", "ATL", "Atlanta", "Eastern", "Southeast"),
    ("Boston Celtics", "BOS", "Boston", "Eastern", "Atlantic"),
    ("Brooklyn Nets", "BKN", "Brooklyn", "Eastern", "Atlantic"),
    ("Charlotte Hornets", "CHA", "Charlotte", "Eastern", "Southeast"),
    ("Chicago Bulls", "CHI", "Chicago", "Eastern", "Central"),
    ("Cleveland Cavaliers", "CLE", "Cleveland", "Eastern", "Central"),
    ("Dallas Mavericks", "DAL", "Dallas", "Western", "Southwest"),
    ("Denver Nuggets", "DEN", "Denver", "Western", "Northwest"),
    ("Detroit Pistons", "DET", "Detroit", "Eastern", "Central"),
    ("Golden State Warriors", "GSW", "San Francisco", "Western", "Pacific"),
    ("Houston Rockets", "HOU", "Houston", "Western", "Southwest"),
    ("Indiana Pacers", "IND", "Indianapolis", "Eastern", "Central"),
    ("Los Angeles Clippers", "LAC", "Los Angeles", "Western", "Pacific"),
    ("Los Angeles Lakers", "LAL", "Los Angeles", "Western", "Pacific"),
    ("Memphis Grizzlies", "MEM", "Memphis", "Western", "Southwest"),
    ("Miami Heat", "MIA", "Miami", "Eastern", "Southeast"),
    ("Milwaukee Bucks", "MIL", "Milwaukee", "Eastern", "Central"),
    ("Minnesota Timberwolves", "MIN", "Minneapolis", "Western", "Northwest"),
    ("New Orleans Pelicans", "NOP", "New Orleans", "Western", "Southwest"),
    ("New York Knicks", "NYK", "New York", "Eastern", "Atlantic"),
    ("Oklahoma City Thunder", "OKC", "Oklahoma City", "Western", "Northwest"),
    ("Orlando Magic", "ORL", "Orlando", "Eastern", "Southeast"),
    ("Philadelphia 76ers", "PHI", "Philadelphia", "Eastern", "Atlantic"),
    ("Phoenix Suns", "PHX", "Phoenix", "Western", "Pacific"),
    ("Portland Trail Blazers", "POR", "Portland", "Western", "Northwest"),
    ("Sacramento Kings", "SAC", "Sacramento", "Western", "Pacific"),
    ("San Antonio Spurs", "SAS", "San Antonio", "Western", "Southwest"),
    ("Toronto Raptors", "TOR", "Toronto", "Eastern", "Atlantic"),
    ("Utah Jazz", "UTA", "Salt Lake City", "Western", "Northwest"),
    ("Washington Wizards", "WAS", "Washington", "Eastern", "Southeast"),
]

MLB_TEAMS = [
    ("Arizona Diamondbacks", "ARI", "Phoenix", "NL", "NL West"),
    ("Atlanta Braves", "ATL", "Atlanta", "NL", "NL East"),
    ("Baltimore Orioles", "BAL", "Baltimore", "AL", "AL East"),
    ("Boston Red Sox", "BOS", "Boston", "AL", "AL East"),
    ("Chicago Cubs", "CHC", "Chicago", "NL", "NL Central"),
    ("Chicago White Sox", "CWS", "Chicago", "AL", "AL Central"),
    ("Cincinnati Reds", "CIN", "Cincinnati", "NL", "NL Central"),
    ("Cleveland Guardians", "CLE", "Cleveland", "AL", "AL Central"),
    ("Colorado Rockies", "COL", "Denver", "NL", "NL West"),
    ("Detroit Tigers", "DET", "Detroit", "AL", "AL Central"),
    ("Houston Astros", "HOU", "Houston", "AL", "AL West"),
    ("Kansas City Royals", "KC", "Kansas City", "AL", "AL Central"),
    ("Los Angeles Angels", "LAA", "Anaheim", "AL", "AL West"),
    ("Los Angeles Dodgers", "LAD", "Los Angeles", "NL", "NL West"),
    ("Miami Marlins", "MIA", "Miami", "NL", "NL East"),
    ("Milwaukee Brewers", "MIL", "Milwaukee", "NL", "NL Central"),
    ("Minnesota Twins", "MIN", "Minneapolis", "AL", "AL Central"),
    ("New York Mets", "NYM", "New York", "NL", "NL East"),
    ("New York Yankees", "NYY", "New York", "AL", "AL East"),
    ("Oakland Athletics", "OAK", "Oakland", "AL", "AL West"),
    ("Philadelphia Phillies", "PHI", "Philadelphia", "NL", "NL East"),
    ("Pittsburgh Pirates", "PIT", "Pittsburgh", "NL", "NL Central"),
    ("San Diego Padres", "SD", "San Diego", "NL", "NL West"),
    ("San Francisco Giants", "SF", "San Francisco", "NL", "NL West"),
    ("Seattle Mariners", "SEA", "Seattle", "AL", "AL West"),
    ("St. Louis Cardinals", "STL", "St. Louis", "NL", "NL Central"),
    ("Tampa Bay Rays", "TB", "St. Petersburg", "AL", "AL East"),
    ("Texas Rangers", "TEX", "Arlington", "AL", "AL West"),
    ("Toronto Blue Jays", "TOR", "Toronto", "AL", "AL East"),
    ("Washington Nationals", "WSH", "Washington", "NL", "NL East"),
]

NHL_TEAMS = [
    ("Anaheim Ducks", "ANA", "Anaheim", "Western", "Pacific"),
    ("Arizona Coyotes", "ARI", "Tempe", "Western", "Central"),
    ("Boston Bruins", "BOS", "Boston", "Eastern", "Atlantic"),
    ("Buffalo Sabres", "BUF", "Buffalo", "Eastern", "Atlantic"),
    ("Calgary Flames", "CGY", "Calgary", "Western", "Pacific"),
    ("Carolina Hurricanes", "CAR", "Raleigh", "Eastern", "Metropolitan"),
    ("Chicago Blackhawks", "CHI", "Chicago", "Western", "Central"),
    ("Colorado Avalanche", "COL", "Denver", "Western", "Central"),
    ("Columbus Blue Jackets", "CBJ", "Columbus", "Eastern", "Metropolitan"),
    ("Dallas Stars", "DAL", "Dallas", "Western", "Central"),
    ("Detroit Red Wings", "DET", "Detroit", "Eastern", "Atlantic"),
    ("Edmonton Oilers", "EDM", "Edmonton", "Western", "Pacific"),
    ("Florida Panthers", "FLA", "Sunrise", "Eastern", "Atlantic"),
    ("Los Angeles Kings", "LAK", "Los Angeles", "Western", "Pacific"),
    ("Minnesota Wild", "MIN", "Saint Paul", "Western", "Central"),
    ("Montreal Canadiens", "MTL", "Montreal", "Eastern", "Atlantic"),
    ("Nashville Predators", "NSH", "Nashville", "Western", "Central"),
    ("New Jersey Devils", "NJD", "Newark", "Eastern", "Metropolitan"),
    ("New York Islanders", "NYI", "Elmont", "Eastern", "Metropolitan"),
    ("New York Rangers", "NYR", "New York", "Eastern", "Metropolitan"),
    ("Ottawa Senators", "OTT", "Ottawa", "Eastern", "Atlantic"),
    ("Philadelphia Flyers", "PHI", "Philadelphia", "Eastern", "Metropolitan"),
    ("Pittsburgh Penguins", "PIT", "Pittsburgh", "Eastern", "Metropolitan"),
    ("San Jose Sharks", "SJS", "San Jose", "Western", "Pacific"),
    ("Seattle Kraken", "SEA", "Seattle", "Western", "Pacific"),
    ("St. Louis Blues", "STL", "St. Louis", "Western", "Central"),
    ("Tampa Bay Lightning", "TBL", "Tampa", "Eastern", "Atlantic"),
    ("Toronto Maple Leafs", "TOR", "Toronto", "Eastern", "Atlantic"),
    ("Utah Hockey Club", "UTA", "Salt Lake City", "Western", "Central"),
    ("Vancouver Canucks", "VAN", "Vancouver", "Western", "Pacific"),
    ("Vegas Golden Knights", "VGK", "Las Vegas", "Western", "Pacific"),
    ("Washington Capitals", "WSH", "Washington", "Eastern", "Metropolitan"),
    ("Winnipeg Jets", "WPG", "Winnipeg", "Western", "Central"),
]

WNBA_TEAMS = [
    ("Atlanta Dream", "ATL", "Atlanta", "Eastern", None),
    ("Chicago Sky", "CHI", "Chicago", "Eastern", None),
    ("Connecticut Sun", "CON", "Uncasville", "Eastern", None),
    ("Dallas Wings", "DAL", "Arlington", "Western", None),
    ("Golden State Valkyries", "GSV", "San Francisco", "Western", None),
    ("Indiana Fever", "IND", "Indianapolis", "Eastern", None),
    ("Las Vegas Aces", "LVA", "Las Vegas", "Western", None),
    ("Los Angeles Sparks", "LAS", "Los Angeles", "Western", None),
    ("Minnesota Lynx", "MIN", "Minneapolis", "Western", None),
    ("New York Liberty", "NYL", "Brooklyn", "Eastern", None),
    ("Phoenix Mercury", "PHO", "Phoenix", "Western", None),
    ("Seattle Storm", "SEA", "Seattle", "Western", None),
    ("Washington Mystics", "WAS", "Washington", "Eastern", None),
]

CFL_TEAMS = [
    ("BC Lions", "BC", "Vancouver", "West", None),
    ("Calgary Stampeders", "CGY", "Calgary", "West", None),
    ("Edmonton Elks", "EDM", "Edmonton", "West", None),
    ("Hamilton Tiger-Cats", "HAM", "Hamilton", "East", None),
    ("Montreal Alouettes", "MTL", "Montreal", "East", None),
    ("Ottawa Redblacks", "OTT", "Ottawa", "East", None),
    ("Saskatchewan Roughriders", "SSK", "Regina", "West", None),
    ("Toronto Argonauts", "TOR", "Toronto", "East", None),
    ("Winnipeg Blue Bombers", "WPG", "Winnipeg", "West", None),
]

# =============================================================================
# SOURCE REGISTRY
# =============================================================================

SOURCES = [
    # key, name, source_type, priority, teams, players, games, odds, stats, sports
    ("espn", "ESPN API", "api", 5, True, True, True, False, True, ["NFL","NBA","MLB","NHL","WNBA","NCAAF","NCAAB"]),
    ("odds_api", "The Odds API", "api", 3, False, False, True, True, False, ["NFL","NBA","MLB","NHL","NCAAF","NCAAB","WNBA","ATP","WTA","CFL"]),
    ("pinnacle", "Pinnacle", "api", 1, False, False, True, True, False, ["NFL","NBA","MLB","NHL","NCAAF","NCAAB","ATP","WTA"]),
    ("tennis_abstract", "Tennis Abstract", "scraper", 10, False, True, True, False, True, ["ATP","WTA"]),
    ("weather_openweather", "OpenWeatherMap", "api", 5, False, False, False, False, False, ["NFL","MLB","NCAAF","CFL"]),
    ("sportsdb", "TheSportsDB", "api", 15, True, True, True, False, False, ["NFL","NBA","MLB","NHL","WNBA","CFL"]),
    ("nflfastr", "nflfastR", "file", 2, True, True, True, False, True, ["NFL"]),
    ("cfbfastr", "cfbfastR", "file", 2, True, True, True, False, True, ["NCAAF"]),
    ("baseballr", "baseballR", "file", 2, True, True, True, False, True, ["MLB"]),
    ("hockeyr", "hockeyR", "file", 2, True, True, True, False, True, ["NHL"]),
    ("wehoop", "wehoop", "file", 2, True, True, True, False, True, ["WNBA"]),
    ("hoopr", "hoopR", "file", 2, True, True, True, False, True, ["NBA"]),
    ("cfl_api", "CFL API", "api", 5, True, True, True, False, True, ["CFL"]),
    ("action_network", "Action Network", "scraper", 8, False, False, False, False, False, ["NFL","NBA","MLB","NHL","NCAAF","NCAAB"]),
    ("nhl_api", "NHL Official API", "api", 3, True, True, True, False, True, ["NHL"]),
    ("sportsipy", "Sportsipy", "scraper", 20, True, True, True, False, True, ["NFL","NBA","MLB","NHL","NCAAF","NCAAB"]),
    ("basketball_ref", "Basketball Reference", "scraper", 8, True, True, True, False, True, ["NBA","WNBA"]),
    ("cfbd", "CollegeFootballData", "api", 4, True, True, True, False, True, ["NCAAF"]),
    ("matchstat", "MatchStat", "scraper", 12, False, True, True, False, True, ["ATP","WTA"]),
    ("realgm", "RealGM", "scraper", 15, True, True, False, False, True, ["NBA","WNBA"]),
    ("nextgenstats", "Next Gen Stats", "api", 6, False, True, False, False, True, ["NFL"]),
    ("kaggle", "Kaggle Datasets", "file", 25, True, True, True, False, True, ["NFL","NBA","MLB","NHL","NCAAF","NCAAB"]),
    ("balldontlie", "BallDontLie", "api", 8, True, True, True, False, True, ["NFL","NBA","MLB","NCAAF","NCAAB","ATP","WTA"]),
    ("polymarket", "Polymarket", "api", 30, False, False, False, True, False, ["NFL","NBA","MLB"]),
    ("kalshi", "Kalshi", "api", 30, False, False, False, True, False, ["NFL","NBA","MLB"]),
    ("weatherstack", "Weatherstack", "api", 10, False, False, False, False, False, ["NFL","MLB","NCAAF"]),
]


async def populate():
    """Main population function."""
    await db_manager.initialize()

    async with db_manager.session() as session:
        # =====================================================================
        # 1. Populate source_registry
        # =====================================================================
        print("\n[Phase 2] Populating source_registry...")
        for src in SOURCES:
            key, name, stype, priority, teams, players, games, odds, stats, sports = src
            await session.execute(text("""
                INSERT INTO source_registry (id, key, name, source_type, priority, is_active,
                    provides_teams, provides_players, provides_games, provides_odds, provides_stats,
                    sports_covered)
                VALUES (gen_random_uuid(), :key, :name, :stype, :priority, true,
                    :teams, :players, :games, :odds, :stats, :sports)
                ON CONFLICT (key) DO NOTHING
            """), {
                "key": key, "name": name, "stype": stype, "priority": priority,
                "teams": teams, "players": players, "games": games,
                "odds": odds, "stats": stats, "sports": sports,
            })
        print(f"  ✅ Registered {len(SOURCES)} data sources")

        # =====================================================================
        # 2. Populate master_teams for all team sports
        # =====================================================================
        sport_teams = {
            "NFL": NFL_TEAMS, "NBA": NBA_TEAMS, "MLB": MLB_TEAMS,
            "NHL": NHL_TEAMS, "WNBA": WNBA_TEAMS, "CFL": CFL_TEAMS,
        }

        total = 0
        for sport_code, teams_list in sport_teams.items():
            for name, abbr, city, conf, div in teams_list:
                await session.execute(text("""
                    INSERT INTO master_teams (id, sport_code, canonical_name, abbreviation,
                        city, conference, division, is_active)
                    VALUES (gen_random_uuid(), :sport, :name, :abbr, :city, :conf, :div, true)
                    ON CONFLICT ON CONSTRAINT uq_master_teams_sport_name DO NOTHING
                """), {
                    "sport": sport_code, "name": name, "abbr": abbr,
                    "city": city, "conf": conf, "div": div,
                })
            total += len(teams_list)
            print(f"  ✅ {sport_code}: {len(teams_list)} teams")

        # Note: ATP/WTA use master_players, not master_teams.
        # NCAAF/NCAAB have too many teams for static list — they'll be auto-created
        # from source data during the mapping phase.

        print(f"\n  Total master teams inserted: {total}")

        # =====================================================================
        # 3. Fix sportsbook priority (Issue #8 from audit)
        # =====================================================================
        print("\n[Phase 2b] Fixing sportsbook priorities...")
        sharp_books = {
            "pinnacle": (1, True),
            "bookmaker": (5, True),
            "betcris": (8, True),
            "circa": (10, True),
        }
        for book_key, (priority, is_sharp) in sharp_books.items():
            result = await session.execute(text("""
                UPDATE sportsbooks SET is_sharp = :sharp, priority = :pri
                WHERE key = :key
            """), {"sharp": is_sharp, "pri": priority, "key": book_key})
            if result.rowcount > 0:
                print(f"  ✅ {book_key}: priority={priority}, is_sharp={is_sharp}")

        # Set consumer books to lower priority
        await session.execute(text("""
            UPDATE sportsbooks SET priority = 50
            WHERE is_sharp = false AND priority = 100
        """))

        await session.commit()
        print("\n✅ Phase 2 complete!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(populate())
