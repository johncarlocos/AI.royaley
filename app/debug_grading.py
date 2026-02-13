"""
Debug grading - shows exactly why games aren't being graded.
Usage: docker exec royaley_api python debug_grading.py
"""
import asyncio
import httpx
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from app.core.config import settings, ODDS_API_SPORT_KEYS

async def debug():
    engine = create_async_engine(settings.DATABASE_URL, echo=False)
    async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as db:
        # 1. Upcoming games needing grading
        print("=" * 90)
        print("1. UPCOMING GAMES PAST GAME TIME (status=scheduled, should be graded)")
        print("=" * 90)
        rows = await db.execute(text("""
            SELECT ug.id, s.code, ug.home_team_name, ug.away_team_name,
                   ug.scheduled_at, ug.status, ug.home_score, ug.away_score,
                   ug.external_id,
                   (SELECT COUNT(*) FROM predictions p WHERE p.upcoming_game_id = ug.id) as pred_count
            FROM upcoming_games ug
            JOIN sports s ON ug.sport_id = s.id
            WHERE ug.scheduled_at < NOW()
            ORDER BY ug.scheduled_at DESC
            LIMIT 40
        """))
        for r in rows.fetchall():
            flag = "***" if r.status == 'scheduled' and r.pred_count > 0 else "   "
            print(f"  {flag} {r.code:6s} | {r.status:10s} | {r.scheduled_at} | "
                  f"{r.home_team_name:30s} vs {r.away_team_name:30s} | "
                  f"preds={r.pred_count} | score={r.home_score}-{r.away_score}")

        # 2. Prediction results status
        print("\n" + "=" * 90)
        print("2. PREDICTION RESULTS STATUS")
        print("=" * 90)
        res = await db.execute(text("""
            SELECT
                COALESCE(pr.actual_result, 'NO_RESULT_ROW') as status,
                COUNT(*) as cnt
            FROM predictions p
            LEFT JOIN prediction_results pr ON pr.prediction_id = p.id
            GROUP BY COALESCE(pr.actual_result, 'NO_RESULT_ROW')
        """))
        for r in res.fetchall():
            print(f"  {r.status:20s}: {r.cnt}")

        # 3. Thresholds
        print("\n" + "=" * 90)
        print("3. TIMING CHECK")
        print("=" * 90)
        t = await db.execute(text("""
            SELECT NOW() as now_utc,
                   NOW() - INTERVAL '3 hours' as grading_threshold,
                   (SELECT COUNT(*) FROM upcoming_games WHERE status='scheduled' AND scheduled_at < NOW() - INTERVAL '3 hours') as eligible_3h,
                   (SELECT COUNT(*) FROM upcoming_games WHERE status='scheduled' AND scheduled_at < NOW()) as eligible_now,
                   (SELECT COUNT(*) FROM upcoming_games WHERE status='scheduled') as total_scheduled,
                   (SELECT COUNT(*) FROM upcoming_games WHERE status='completed') as total_completed
        """))
        r = t.fetchone()
        print(f"  NOW (UTC):              {r.now_utc}")
        print(f"  Grading threshold (3h): {r.grading_threshold}")
        print(f"  Past 3h + scheduled:    {r.eligible_3h}  ← grader looks at these")
        print(f"  Past now + scheduled:   {r.eligible_now}")
        print(f"  Total scheduled:        {r.total_scheduled}")
        print(f"  Total completed:        {r.total_completed}")

        # 4. Odds API scores vs DB names
        print("\n" + "=" * 90)
        print("4. ODDS API SCORES vs DB TEAM NAMES (matching check)")
        print("=" * 90)

        api_key = settings.ODDS_API_KEY

        sport_rows = await db.execute(text("""
            SELECT DISTINCT s.code, s.api_key, s.id as sport_id
            FROM upcoming_games ug
            JOIN sports s ON ug.sport_id = s.id
            WHERE ug.status = 'scheduled'
              AND ug.scheduled_at < NOW()
        """))

        async with httpx.AsyncClient(timeout=30.0) as client:
            # Get tennis tournament keys
            tennis_keys = {}
            try:
                resp = await client.get("https://api.the-odds-api.com/v4/sports", params={"apiKey": api_key})
                for s in resp.json():
                    if s.get('active'):
                        k = s.get('key', '')
                        if k.startswith('tennis_wta'):
                            tennis_keys['WTA'] = k
                        elif k.startswith('tennis_atp'):
                            tennis_keys['ATP'] = k
            except:
                pass

            for sport_code, db_api_key, sport_id in sport_rows.fetchall():
                if sport_code in tennis_keys:
                    api_sport_key = tennis_keys[sport_code]
                else:
                    api_sport_key = db_api_key or ODDS_API_SPORT_KEYS.get(sport_code)

                if not api_sport_key:
                    print(f"\n  {sport_code}: No API key, skipping")
                    continue

                print(f"\n  === {sport_code} (api_key: {api_sport_key}) ===")

                try:
                    resp = await client.get(
                        f"https://api.the-odds-api.com/v4/sports/{api_sport_key}/scores",
                        params={"apiKey": api_key, "daysFrom": 3},
                    )
                    if resp.status_code != 200:
                        print(f"    API Error: {resp.status_code}")
                        continue

                    scores = resp.json()
                    print(f"    API returned {len(scores)} games")

                    for game in scores:
                        completed = game.get("completed", False)
                        if not completed:
                            continue

                        api_home = game.get("home_team", "")
                        api_away = game.get("away_team", "")
                        commence = game.get("commence_time", "")
                        game_scores = game.get("scores", [])

                        score_str = ""
                        for sc in game_scores:
                            score_str += f"{sc['name']}:{sc.get('score','?')} "

                        print(f"\n    COMPLETED: {api_home} vs {api_away}")
                        print(f"    Commence:  {commence}")
                        print(f"    Scores:    {score_str}")

                        # Try exact match (what grader does)
                        exact = await db.execute(text("""
                            SELECT ug.id, ug.home_team_name, ug.away_team_name,
                                   ug.scheduled_at, ug.status,
                                   ABS(EXTRACT(EPOCH FROM (ug.scheduled_at - :commence::timestamptz))) as time_diff_sec
                            FROM upcoming_games ug
                            WHERE ug.sport_id = :sid
                              AND ug.home_team_name = :home
                              AND ug.status = 'scheduled'
                            ORDER BY ABS(EXTRACT(EPOCH FROM (ug.scheduled_at - :commence::timestamptz)))
                            LIMIT 3
                        """), {"sid": sport_id, "home": api_home, "commence": commence or "2000-01-01T00:00:00Z"})

                        exact_matches = exact.fetchall()
                        if exact_matches:
                            for m in exact_matches:
                                within = "YES" if m.time_diff_sec < 7200 else f"NO ({m.time_diff_sec:.0f}s > 7200s)"
                                print(f"    ✅ EXACT MATCH: {m.home_team_name} vs {m.away_team_name} | {m.scheduled_at} | within 2h: {within}")
                        else:
                            print(f"    ❌ NO EXACT MATCH for home_team_name = '{api_home}'")

                            # Try fuzzy match
                            fuzzy = await db.execute(text("""
                                SELECT ug.home_team_name, ug.away_team_name, ug.scheduled_at, ug.status,
                                       ABS(EXTRACT(EPOCH FROM (ug.scheduled_at - :commence::timestamptz))) as time_diff_sec
                                FROM upcoming_games ug
                                WHERE ug.sport_id = :sid
                                  AND ug.status = 'scheduled'
                                  AND ug.scheduled_at BETWEEN :commence::timestamptz - INTERVAL '4 hours'
                                                        AND :commence::timestamptz + INTERVAL '4 hours'
                                ORDER BY ug.scheduled_at
                                LIMIT 5
                            """), {"sid": sport_id, "commence": commence or "2000-01-01T00:00:00Z"})

                            fuzzy_matches = fuzzy.fetchall()
                            if fuzzy_matches:
                                print(f"    🔍 NEARBY GAMES (same sport, within 4h):")
                                for m in fuzzy_matches:
                                    print(f"       DB: '{m.home_team_name}' vs '{m.away_team_name}' | {m.scheduled_at} | {m.status}")
                            else:
                                print(f"    🔍 No nearby games found at all for this sport/time")

                except Exception as e:
                    print(f"    Error: {e}")

    await engine.dispose()
    print("\n" + "=" * 90)
    print("DONE - Check mismatches above to identify the grading issue")
    print("=" * 90)

asyncio.run(debug())