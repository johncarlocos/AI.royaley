"""
ROYALEY - Verify Master Data Architecture (All 12 Tables)
Runs diagnostic queries to confirm the health of the entire master data layer.

Run: python -m scripts.verify_master_data
"""

import asyncio
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text
from app.core.database import db_manager

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


async def verify():
    await db_manager.initialize()

    async with db_manager.async_session() as session:

        print("\n" + "=" * 70)
        print("   ROYALEY — MASTER DATA ARCHITECTURE VERIFICATION REPORT")
        print("=" * 70)

        # ── 1. TABLE COUNTS ──
        print("\n📊 TABLE COUNTS:")
        tables = [
            "master_teams", "master_players", "master_games",
            "master_odds", "ml_training_dataset",
            "team_mappings", "player_mappings", "game_mappings",
            "venue_mappings", "odds_mappings",
            "source_registry", "mapping_audit_log",
        ]
        for t in tables:
            try:
                r = await session.execute(text(f"SELECT COUNT(*) FROM {t}"))
                c = r.scalar()
                print(f"  {t:<30} {c:>10,}")
            except Exception:
                print(f"  {t:<30} {'⚠️ TABLE NOT FOUND':>10}")

        # ── 2. MAPPING COVERAGE ──
        print("\n📈 MAPPING COVERAGE:")

        # Teams mapped
        try:
            r = await session.execute(text(
                "SELECT COUNT(*) FROM teams WHERE master_team_id IS NOT NULL"
            ))
            mapped = r.scalar()
            r = await session.execute(text("SELECT COUNT(*) FROM teams"))
            total = r.scalar()
            pct = (100 * mapped / total) if total > 0 else 0
            icon = "✅" if pct > 95 else "⚠️" if pct > 80 else "❌"
            print(f"  {icon} Teams mapped:      {mapped:>10,} / {total:>10,} ({pct:.1f}%)")
        except Exception as e:
            print(f"  ⚠️  Teams: {e}")

        # Games mapped
        try:
            r = await session.execute(text(
                "SELECT COUNT(*) FROM games WHERE master_game_id IS NOT NULL"
            ))
            mapped = r.scalar()
            r = await session.execute(text("SELECT COUNT(*) FROM games"))
            total = r.scalar()
            pct = (100 * mapped / total) if total > 0 else 0
            icon = "✅" if pct > 95 else "⚠️" if pct > 80 else "❌"
            print(f"  {icon} Games mapped:      {mapped:>10,} / {total:>10,} ({pct:.1f}%)")
        except Exception as e:
            print(f"  ⚠️  Games: {e}")

        # Odds linked (raw odds with master_game_id)
        try:
            r = await session.execute(text(
                "SELECT COUNT(*) FROM odds WHERE master_game_id IS NOT NULL"
            ))
            linked = r.scalar()
            r = await session.execute(text("SELECT COUNT(*) FROM odds"))
            total = r.scalar()
            pct = (100 * linked / total) if total > 0 else 0
            icon = "✅" if pct > 90 else "⚠️" if pct > 70 else "❌"
            print(f"  {icon} Odds linked:       {linked:>10,} / {total:>10,} ({pct:.1f}%)")
        except Exception as e:
            print(f"  ⚠️  Odds: {e}")

        # Public betting linked
        try:
            r = await session.execute(text(
                "SELECT COUNT(*) FROM public_betting WHERE master_game_id IS NOT NULL"
            ))
            linked = r.scalar()
            r = await session.execute(text("SELECT COUNT(*) FROM public_betting"))
            total = r.scalar()
            pct = (100 * linked / total) if total > 0 else 0
            icon = "✅" if pct > 70 else "⚠️" if pct > 40 else "❌"
            print(f"  {icon} Public betting:    {linked:>10,} / {total:>10,} ({pct:.1f}%)")
        except Exception as e:
            print(f"  ⚠️  Public betting: {e}")

        # ── 3. DEDUPLICATION ──
        print("\n🔗 DEDUPLICATION:")
        try:
            r = await session.execute(text("SELECT COUNT(*) FROM games"))
            source_games = r.scalar()
            r = await session.execute(text("SELECT COUNT(*) FROM master_games"))
            master_games = r.scalar()
            if master_games > 0:
                ratio = source_games / master_games
                removed = source_games - master_games
                print(f"  Source games:      {source_games:>10,}")
                print(f"  Master games:      {master_games:>10,}")
                print(f"  Dedup ratio:       {ratio:>10.2f}x ({removed:,} duplicates removed)")
        except Exception:
            pass

        try:
            r = await session.execute(text("SELECT COUNT(*) FROM odds WHERE master_game_id IS NOT NULL"))
            source_odds = r.scalar()
            r = await session.execute(text("SELECT COUNT(*) FROM master_odds"))
            master_odds = r.scalar()
            if master_odds > 0:
                ratio = source_odds / master_odds
                print(f"  Source odds:       {source_odds:>10,}")
                print(f"  Master odds:       {master_odds:>10,}")
                print(f"  Odds dedup ratio:  {ratio:>10.2f}x")
        except Exception:
            pass

        # ── 4. MASTER ODDS HEALTH ──
        print("\n📉 MASTER ODDS BY SPORT:")
        try:
            r = await session.execute(text("""
                SELECT mg.sport_code,
                       COUNT(DISTINCT mo.master_game_id) as games_with_odds,
                       COUNT(*) as total_master_odds,
                       COUNT(*) FILTER (WHERE mo.is_sharp) as sharp_odds,
                       COUNT(*) FILTER (WHERE mo.closing_line IS NOT NULL OR mo.closing_odds_home IS NOT NULL) as with_closing
                FROM master_odds mo
                JOIN master_games mg ON mo.master_game_id = mg.id
                GROUP BY mg.sport_code
                ORDER BY mg.sport_code
            """))
            rows = r.fetchall()
            for row in rows:
                sport, games_w, total_mo, sharp, closing = row
                print(f"  {sport:<8} {total_mo:>8,} master odds  |  {games_w:>6,} games  |  "
                      f"⭐{sharp:>5,} sharp  |  📊{closing:>6,} with closing")
        except Exception as e:
            print(f"  ⚠️  Master odds query failed: {e}")

        # ── 5. SPORTSBOOK PRIORITIES ──
        print("\n📚 SPORTSBOOK PRIORITIES:")
        try:
            r = await session.execute(text("""
                SELECT name, key, priority, is_sharp, is_active
                FROM sportsbooks
                ORDER BY priority ASC, name
            """))
            rows = r.fetchall()
            for row in rows:
                name, key, prio, sharp, active = row
                sharp_icon = "⭐" if sharp else "  "
                active_icon = "✅" if active else "❌"
                print(f"  {sharp_icon} {active_icon} {name:<30} key={key:<20} priority={prio}")
        except Exception:
            pass

        # ── 6. ML TRAINING READINESS ──
        print("\n🤖 ML TRAINING READINESS:")
        try:
            r = await session.execute(text("SELECT COUNT(*) FROM ml_training_dataset"))
            total_rows = r.scalar()
            r = await session.execute(text(
                "SELECT COUNT(*) FROM ml_training_dataset WHERE num_books_with_odds > 0"
            ))
            with_odds = r.scalar()
            r = await session.execute(text(
                "SELECT COUNT(*) FROM ml_training_dataset WHERE home_score IS NOT NULL"
            ))
            with_scores = r.scalar()
            r = await session.execute(text(
                "SELECT COUNT(*) FROM ml_training_dataset WHERE num_books_with_odds > 0 AND home_score IS NOT NULL"
            ))
            ml_ready = r.scalar()
            r = await session.execute(text(
                "SELECT COUNT(*) FROM ml_training_dataset WHERE pinnacle_spread IS NOT NULL"
            ))
            with_pinnacle = r.scalar()

            print(f"  Total rows:         {total_rows:>10,}")
            print(f"  With odds:          {with_odds:>10,} ({_pct(with_odds, total_rows)})")
            print(f"  With final scores:  {with_scores:>10,} ({_pct(with_scores, total_rows)})")
            print(f"  With Pinnacle:      {with_pinnacle:>10,} ({_pct(with_pinnacle, total_rows)})")
            print(f"  🏆 ML-TRAINABLE:    {ml_ready:>10,} (odds + scores)")
        except Exception as e:
            print(f"  ⚠️  ML training dataset not yet built: {e}")
            print(f"      Run: python -m scripts.build_ml_training_data")

        # ── 7. ML TRAINING BY SPORT ──
        print("\n🏆 ML-TRAINABLE BY SPORT:")
        try:
            r = await session.execute(text("""
                SELECT sport_code,
                       COUNT(*) as total,
                       COUNT(*) FILTER (WHERE num_books_with_odds > 0 AND home_score IS NOT NULL) as trainable,
                       COUNT(*) FILTER (WHERE pinnacle_spread IS NOT NULL) as with_pinnacle
                FROM ml_training_dataset
                GROUP BY sport_code
                ORDER BY sport_code
            """))
            rows = r.fetchall()
            for row in rows:
                sport, total, trainable, pinnacle = row
                icon = "✅" if trainable > 1000 else "⚠️" if trainable > 100 else "❌"
                print(f"  {icon} {sport:<8} {trainable:>8,} trainable / {total:>8,} total  |  ⭐{pinnacle:>6,} Pinnacle")
        except Exception:
            pass

        # ── 8. FEATURE COMPLETENESS ──
        print("\n📋 FEATURE COMPLETENESS (ML Training Dataset):")
        try:
            r = await session.execute(text("""
                SELECT
                    COUNT(*) as total,
                    COUNT(spread_close) as has_spread,
                    COUNT(moneyline_home) as has_ml,
                    COUNT(total_close) as has_total,
                    COUNT(public_spread_home_pct) as has_betting,
                    COUNT(temperature_f) as has_weather,
                    COUNT(home_injuries_out) as has_injuries,
                    COUNT(no_vig_prob_home) as has_novig
                FROM ml_training_dataset
            """))
            row = r.fetchone()
            if row and row[0] > 0:
                total = row[0]
                features = [
                    ("Spread (close)", row[1]),
                    ("Moneyline", row[2]),
                    ("Total (close)", row[3]),
                    ("Public betting", row[4]),
                    ("Weather", row[5]),
                    ("Injuries", row[6]),
                    ("No-vig probability", row[7]),
                ]
                for name, count in features:
                    pct = 100 * count / total
                    icon = "✅" if pct > 80 else "⚠️" if pct > 40 else "❌"
                    print(f"  {icon} {name:<25} {count:>8,} ({pct:.1f}%)")
        except Exception:
            pass

        print("\n" + "=" * 70)
        print("   VERIFICATION COMPLETE")
        print("=" * 70 + "\n")

    await db_manager.close()


def _pct(part, total):
    if total == 0:
        return "0.0%"
    return f"{100 * part / total:.1f}%"


if __name__ == "__main__":
    asyncio.run(verify())