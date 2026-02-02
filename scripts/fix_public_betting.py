"""
ROYALEY - Fix public_betting: Match to master_games by sport + date + team nickname

The public_betting table has 118 rows with:
  - game_id = NULL (no FK link)
  - Team names as nicknames only ("Longhorns", "Wolverines")
  - Some corrupted names ("AggiesTA&M", "AR-Pine BluffARPB")
  - Multiple sports (NCAAF, NCAAB, NBA, MLB, NHL)

Strategy: For each row, find master_games on that date for that sport,
then match by checking if the nickname appears in the team's canonical name.

Run: python -m scripts.fix_public_betting
"""

import asyncio
import logging
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text
from app.core.database import db_manager

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Manual nickname → search terms mapping for tricky cases
NICKNAME_OVERRIDES = {
    # Concatenated junk
    "AggiesTA&M": "Aggies",
    "AR-Pine BluffARPB": "Pine Bluff",
    "American UAMER": "American",
    "Austin PeayPEAY": "Austin Peay",
    "BellarmineBELL": "Bellarmine",
    "Boston UBU": "Boston University",
    "BroncosB": "Broncos",
    "BucknellBUCK": "Bucknell",
    "C. ArkansasCARK": "Central Arkansas",
    "ColgateCOLG": "Colgate",
    "E. Carolina": "East Carolina",
    "E. Kentucky": "Eastern Kentucky",
    "Grambling StGRAM": "Grambling",
    "HuskiesU": "Huskies",
    "IUIN": "Indiana",
    "LA Tech": "Louisiana Tech",
    "Loyola (MD)L-MD": "Loyola Maryland",
    "MS Valley StMVSU": "Mississippi Valley",
    "Miami (FL)": "Miami",
    "N. Iowa": "Northern Iowa",
    "N. Mexico St": "New Mexico State",
    "NavyNAVY": "Navy",
    "RedHawksM-": "RedHawks",
    "S. Carolina": "South Carolina",
    "S. Florida": "South Florida",
    "S. Illinois": "Southern Illinois",
    "SD State": "San Diego State",
    "Southern USOU": "Southern",
    "ArmyARMY": "Army",
    "W. Kentucky": "Western Kentucky",
    "FI": "FIU",
    "IPF": "IUPUI",
    "LS": "LSU",
    "TC": "TCU",
    "UA": "UAB",
    "US": "USC",
}


def clean_nickname(name: str) -> str:
    """Clean a public_betting team name for matching."""
    if not name:
        return ""
    # Apply overrides first
    if name in NICKNAME_OVERRIDES:
        return NICKNAME_OVERRIDES[name]
    return name.strip()


def team_matches(nickname: str, canonical_name: str) -> bool:
    """Check if a nickname matches a canonical team name."""
    if not nickname or not canonical_name:
        return False
    
    nick_lower = nickname.lower().strip()
    canon_lower = canonical_name.lower().strip()
    
    # Exact match
    if nick_lower == canon_lower:
        return True
    
    # Nickname is contained in canonical name
    if nick_lower in canon_lower:
        return True
    
    # Canonical name ends with nickname (e.g. "Texas Longhorns" ends with "Longhorns")
    if canon_lower.endswith(nick_lower):
        return True
    
    # Canonical name starts with nickname (e.g. "Oregon" starts with "Oregon")
    if canon_lower.startswith(nick_lower):
        return True
    
    # Last word of canonical matches nickname
    canon_parts = canon_lower.split()
    if canon_parts and canon_parts[-1] == nick_lower:
        return True
    
    # First word of canonical matches nickname
    if canon_parts and canon_parts[0] == nick_lower:
        return True
    
    return False


async def fix_public_betting():
    logger.info("Initializing database connection...")
    await db_manager.initialize()
    logger.info("Database connection initialized successfully")

    async with db_manager.session() as session:
        # Get all public_betting rows
        result = await session.execute(text("""
            SELECT id, sport_code, home_team, away_team, game_date
            FROM public_betting
            WHERE master_game_id IS NULL
            ORDER BY game_date, sport_code
        """))
        pb_rows = result.fetchall()
        logger.info(f"Public betting rows to match: {len(pb_rows)}")

        matched = 0
        unmatched = 0
        unmatched_details = []

        for row in pb_rows:
            pb_id = str(row[0])
            sport_code = row[1]
            home_nick = clean_nickname(row[2])
            away_nick = clean_nickname(row[3])
            game_date = row[4]

            if not sport_code or not game_date:
                unmatched += 1
                unmatched_details.append(f"  Missing sport/date: {row[2]} vs {row[3]}")
                continue

            # Find all master_games for this sport on this date
            games_result = await session.execute(text("""
                SELECT mg.id,
                       ht.canonical_name as home_name,
                       at_.canonical_name as away_name
                FROM master_games mg
                LEFT JOIN master_teams ht ON mg.home_master_team_id = ht.id
                LEFT JOIN master_teams at_ ON mg.away_master_team_id = at_.id
                WHERE mg.sport_code = :sport
                AND DATE(mg.scheduled_at) = :gdate
            """), {"sport": sport_code, "gdate": game_date})
            candidates = games_result.fetchall()

            # Try to find a match
            best_match = None
            for cand in candidates:
                mgid, home_name, away_name = str(cand[0]), cand[1] or "", cand[2] or ""

                home_ok = team_matches(home_nick, home_name)
                away_ok = team_matches(away_nick, away_name)

                if home_ok and away_ok:
                    best_match = mgid
                    break

                # Try swapped (home/away might be reversed)
                if team_matches(home_nick, away_name) and team_matches(away_nick, home_name):
                    best_match = mgid
                    break

            if best_match:
                await session.execute(text("""
                    UPDATE public_betting SET master_game_id = :mgid WHERE id = :pbid
                """), {"mgid": best_match, "pbid": pb_id})
                matched += 1
            else:
                unmatched += 1
                cand_count = len(candidates)
                unmatched_details.append(
                    f"  {sport_code} {game_date}: {row[2]} vs {row[3]} ({cand_count} candidates)"
                )

        await session.commit()

        logger.info(f"\n✅ RESULTS:")
        logger.info(f"  Matched: {matched} / {len(pb_rows)}")
        logger.info(f"  Unmatched: {unmatched}")

        if unmatched_details:
            logger.info(f"\n⚠️  UNMATCHED DETAILS (first 30):")
            for detail in unmatched_details[:30]:
                logger.info(detail)


if __name__ == "__main__":
    asyncio.run(fix_public_betting())