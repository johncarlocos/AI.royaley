"""
ROYALEY - Canonical Team Data
Contains official team rosters for all team-based sports.

Format: (canonical_name, abbreviation, city, conference, division)
"""

# =============================================================================
# NFL TEAMS (32)
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

# =============================================================================
# NBA TEAMS (30)
# =============================================================================

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

# =============================================================================
# MLB TEAMS (30)
# =============================================================================

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

# =============================================================================
# NHL TEAMS (32)
# =============================================================================

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
    ("Vancouver Canucks", "VAN", "Vancouver", "Western", "Pacific"),
    ("Vegas Golden Knights", "VGK", "Las Vegas", "Western", "Pacific"),
    ("Washington Capitals", "WSH", "Washington", "Eastern", "Metropolitan"),
    ("Winnipeg Jets", "WPG", "Winnipeg", "Western", "Central"),
]

# =============================================================================
# WNBA TEAMS (13)
# =============================================================================

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

# =============================================================================
# CFL TEAMS (9)
# =============================================================================

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
# TEAM ALIASES (nickname → canonical name)
# =============================================================================

TEAM_ALIASES = {
    # NFL nicknames
    "sooners": "Oklahoma Sooners",
    "crimson tide": "Alabama Crimson Tide",
    "golden eagles": "Southern Miss Golden Eagles",
    # Common abbreviation differences
    "la lakers": "Los Angeles Lakers",
    "la clippers": "Los Angeles Clippers",
    "la rams": "Los Angeles Rams",
    "la chargers": "Los Angeles Chargers",
    "la dodgers": "Los Angeles Dodgers",
    "la angels": "Los Angeles Angels",
    "ny giants": "New York Giants",
    "ny jets": "New York Jets",
    "ny knicks": "New York Knicks",
    "ny mets": "New York Mets",
    "ny yankees": "New York Yankees",
    "ny rangers": "New York Rangers",
    "ny islanders": "New York Islanders",
    "sf 49ers": "San Francisco 49ers",
    "sf giants": "San Francisco Giants",
    "gb packers": "Green Bay Packers",
    "tb buccaneers": "Tampa Bay Buccaneers",
    "tb lightning": "Tampa Bay Lightning",
    "tb rays": "Tampa Bay Rays",
    "kc chiefs": "Kansas City Chiefs",
    "kc royals": "Kansas City Royals",
    "lv raiders": "Las Vegas Raiders",
    "ne patriots": "New England Patriots",
    "no saints": "New Orleans Saints",
    "stl cardinals": "St. Louis Cardinals",
    "stl blues": "St. Louis Blues",
    "washington football team": "Washington Commanders",
}

# =============================================================================
# ALL TEAMS BY SPORT
# =============================================================================

ALL_SPORT_TEAMS = {
    "NFL": NFL_TEAMS,
    "NBA": NBA_TEAMS,
    "MLB": MLB_TEAMS,
    "NHL": NHL_TEAMS,
    "WNBA": WNBA_TEAMS,
    "CFL": CFL_TEAMS,
}

# Note: ATP/WTA use master_players, not master_teams.
# NCAAF/NCAAB have too many teams for static list — they'll be auto-created
# from source data during the mapping phase.
