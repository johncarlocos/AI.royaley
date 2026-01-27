#!/usr/bin/env python3
"""
AI PRO SPORTS - Database Validation & Feature Builder
=====================================================

This script:
1. Validates all raw data tables
2. Reports data quality issues
3. Checks if data is sufficient for game_features
4. Builds the game_features table

Run with:
    docker-compose exec api python scripts/validate_and_build_features.py --validate
    docker-compose exec api python scripts/validate_and_build_features.py --build
    docker-compose exec api python scripts/validate_and_build_features.py --all
"""

import asyncio
import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from sqlalchemy import select, func, and_, or_, text, distinct
from sqlalchemy.ext.asyncio import AsyncSession

console = Console()


# ═══════════════════════════════════════════════════════════════════════════════
# PART 1: DATA VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

async def get_table_counts(session: AsyncSession) -> Dict[str, int]:
    """Get row counts for all tables."""
    tables = [
        'sports', 'teams', 'players', 'venues', 'seasons',
        'games', 'odds', 'sportsbooks', 'injuries', 'game_injuries',
        'weather_data', 'odds_movements', 'closing_lines',
        'player_stats', 'team_stats', 'game_features', 'predictions'
    ]
    
    counts = {}
    for table in tables:
        try:
            result = await session.execute(text(f"SELECT COUNT(*) FROM {table}"))
            counts[table] = result.scalar() or 0
        except Exception as e:
            counts[table] = -1  # Table doesn't exist
    
    return counts


async def validate_games_table(session: AsyncSession) -> Dict:
    """Validate the games table for quality issues."""
    issues = {
        "zero_score_finals": 0,
        "missing_scores": 0,
        "missing_teams": 0,
        "future_games": 0,
        "completed_games": 0,
        "by_sport": {}
    }
    
    # Check for 0-0 final games
    result = await session.execute(text("""
        SELECT COUNT(*) FROM games 
        WHERE status = 'final' 
        AND home_score = 0 AND away_score = 0
    """))
    issues["zero_score_finals"] = result.scalar() or 0
    
    # Check for missing scores in final games
    result = await session.execute(text("""
        SELECT COUNT(*) FROM games 
        WHERE status = 'final' 
        AND (home_score IS NULL OR away_score IS NULL)
    """))
    issues["missing_scores"] = result.scalar() or 0
    
    # Check for missing team references
    result = await session.execute(text("""
        SELECT COUNT(*) FROM games 
        WHERE home_team_id IS NULL OR away_team_id IS NULL
    """))
    issues["missing_teams"] = result.scalar() or 0
    
    # Count completed vs future games
    result = await session.execute(text("""
        SELECT COUNT(*) FROM games WHERE status = 'final'
    """))
    issues["completed_games"] = result.scalar() or 0
    
    result = await session.execute(text("""
        SELECT COUNT(*) FROM games WHERE status = 'scheduled'
    """))
    issues["future_games"] = result.scalar() or 0
    
    # Games by sport
    result = await session.execute(text("""
        SELECT s.code, COUNT(g.id) as total,
               SUM(CASE WHEN g.status = 'final' THEN 1 ELSE 0 END) as completed,
               SUM(CASE WHEN g.status = 'final' AND g.home_score > 0 THEN 1 ELSE 0 END) as with_scores
        FROM games g
        JOIN sports s ON g.sport_id = s.id
        GROUP BY s.code
        ORDER BY s.code
    """))
    for row in result.fetchall():
        issues["by_sport"][row[0]] = {
            "total": row[1],
            "completed": row[2],
            "with_scores": row[3]
        }
    
    return issues


async def validate_teams_table(session: AsyncSession) -> Dict:
    """Validate the teams table."""
    result = await session.execute(text("""
        SELECT s.code, COUNT(t.id) as team_count
        FROM teams t
        JOIN sports s ON t.sport_id = s.id
        GROUP BY s.code
        ORDER BY s.code
    """))
    
    return {"by_sport": {row[0]: row[1] for row in result.fetchall()}}


async def validate_odds_table(session: AsyncSession) -> Dict:
    """Validate the odds table."""
    issues = {
        "total": 0,
        "with_spread": 0,
        "with_total": 0,
        "with_moneyline": 0,
        "sportsbooks": [],
        "by_sport": {}
    }
    
    result = await session.execute(text("SELECT COUNT(*) FROM odds"))
    issues["total"] = result.scalar() or 0
    
    result = await session.execute(text("""
        SELECT COUNT(*) FROM odds WHERE home_spread IS NOT NULL
    """))
    issues["with_spread"] = result.scalar() or 0
    
    result = await session.execute(text("""
        SELECT COUNT(*) FROM odds WHERE total_line IS NOT NULL
    """))
    issues["with_total"] = result.scalar() or 0
    
    result = await session.execute(text("""
        SELECT COUNT(*) FROM odds WHERE home_ml IS NOT NULL
    """))
    issues["with_moneyline"] = result.scalar() or 0
    
    # Sportsbooks
    result = await session.execute(text("""
        SELECT sb.name, COUNT(o.id) as odds_count
        FROM odds o
        JOIN sportsbooks sb ON o.sportsbook_id = sb.id
        GROUP BY sb.name
        ORDER BY odds_count DESC
        LIMIT 10
    """))
    issues["sportsbooks"] = [(row[0], row[1]) for row in result.fetchall()]
    
    return issues


async def validate_team_stats(session: AsyncSession) -> Dict:
    """Validate team_stats table - critical for features.
    
    Actual schema: team_id, stat_type, value (individual rows per stat)
    """
    issues = {
        "total": 0,
        "by_sport": {},
        "sample_stat_types": [],
        "teams_with_stats": 0
    }
    
    result = await session.execute(text("SELECT COUNT(*) FROM team_stats"))
    issues["total"] = result.scalar() or 0
    
    # Check what stat_types exist
    result = await session.execute(text("""
        SELECT DISTINCT stat_type
        FROM team_stats
        ORDER BY stat_type
        LIMIT 30
    """))
    issues["sample_stat_types"] = [row[0] for row in result.fetchall()]
    
    # Count teams with stats
    result = await session.execute(text("""
        SELECT COUNT(DISTINCT team_id) FROM team_stats
    """))
    issues["teams_with_stats"] = result.scalar() or 0
    
    # By sport
    result = await session.execute(text("""
        SELECT s.code, COUNT(ts.id) as stat_count, COUNT(DISTINCT ts.team_id) as teams
        FROM team_stats ts
        JOIN teams t ON ts.team_id = t.id
        JOIN sports s ON t.sport_id = s.id
        GROUP BY s.code
        ORDER BY s.code
    """))
    issues["by_sport"] = {row[0]: {"records": row[1], "teams": row[2]} for row in result.fetchall()}
    
    return issues


async def validate_injuries(session: AsyncSession) -> Dict:
    """Validate injuries table."""
    result = await session.execute(text("SELECT COUNT(*) FROM injuries"))
    total = result.scalar() or 0
    
    result = await session.execute(text("""
        SELECT status, COUNT(*) FROM injuries GROUP BY status
    """))
    by_status = {row[0]: row[1] for row in result.fetchall()}
    
    return {"total": total, "by_status": by_status}


async def validate_weather(session: AsyncSession) -> Dict:
    """Validate weather_data table."""
    result = await session.execute(text("SELECT COUNT(*) FROM weather_data"))
    total = result.scalar() or 0
    
    return {"total": total}


async def validate_closing_lines(session: AsyncSession) -> Dict:
    """Validate closing_lines table - critical for CLV tracking."""
    result = await session.execute(text("SELECT COUNT(*) FROM closing_lines"))
    total = result.scalar() or 0
    
    result = await session.execute(text("""
        SELECT sb.name, COUNT(cl.id) as count
        FROM closing_lines cl
        JOIN sportsbooks sb ON cl.sportsbook_id = sb.id
        GROUP BY sb.name
        ORDER BY count DESC
        LIMIT 5
    """))
    by_sportsbook = [(row[0], row[1]) for row in result.fetchall()]
    
    return {"total": total, "by_sportsbook": by_sportsbook}


async def run_validation(session: AsyncSession) -> Dict:
    """Run all validations and return results."""
    console.print("\n[bold cyan]═══ AI PRO SPORTS - DATA VALIDATION ═══[/bold cyan]\n")
    
    results = {}
    
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
        task = progress.add_task("Checking table counts...", total=None)
        results["table_counts"] = await get_table_counts(session)
        
        progress.update(task, description="Validating games table...")
        results["games"] = await validate_games_table(session)
        
        progress.update(task, description="Validating teams table...")
        results["teams"] = await validate_teams_table(session)
        
        progress.update(task, description="Validating odds table...")
        results["odds"] = await validate_odds_table(session)
        
        progress.update(task, description="Validating team_stats table...")
        results["team_stats"] = await validate_team_stats(session)
        
        progress.update(task, description="Validating injuries table...")
        results["injuries"] = await validate_injuries(session)
        
        progress.update(task, description="Validating weather_data table...")
        results["weather"] = await validate_weather(session)
        
        progress.update(task, description="Validating closing_lines table...")
        results["closing_lines"] = await validate_closing_lines(session)
    
    return results


def display_validation_results(results: Dict):
    """Display validation results in a nice format."""
    
    # Table Counts
    console.print("\n[bold]1. TABLE ROW COUNTS[/bold]")
    table = Table(show_header=True)
    table.add_column("Table", style="cyan")
    table.add_column("Rows", justify="right")
    table.add_column("Status", justify="center")
    
    for tbl, count in sorted(results["table_counts"].items()):
        if count == -1:
            status = "❌ Missing"
        elif count == 0:
            status = "⚠️ Empty"
        else:
            status = "✅ OK"
        table.add_row(tbl, str(count) if count >= 0 else "N/A", status)
    
    console.print(table)
    
    # Games Validation
    console.print("\n[bold]2. GAMES TABLE VALIDATION[/bold]")
    games = results["games"]
    
    issues_table = Table(show_header=True)
    issues_table.add_column("Check", style="cyan")
    issues_table.add_column("Count", justify="right")
    issues_table.add_column("Status", justify="center")
    
    issues_table.add_row(
        "Games with 0-0 scores (final)", 
        str(games["zero_score_finals"]),
        "🔴 CRITICAL" if games["zero_score_finals"] > 0 else "✅ OK"
    )
    issues_table.add_row(
        "Games missing scores (final)", 
        str(games["missing_scores"]),
        "🔴 CRITICAL" if games["missing_scores"] > 0 else "✅ OK"
    )
    issues_table.add_row(
        "Games missing team references", 
        str(games["missing_teams"]),
        "🔴 CRITICAL" if games["missing_teams"] > 0 else "✅ OK"
    )
    issues_table.add_row(
        "Completed games", 
        str(games["completed_games"]),
        "✅ OK" if games["completed_games"] > 100 else "⚠️ LOW"
    )
    issues_table.add_row(
        "Future/scheduled games", 
        str(games["future_games"]),
        "✅ OK"
    )
    
    console.print(issues_table)
    
    # Games by sport
    console.print("\n[bold]3. GAMES BY SPORT[/bold]")
    sport_table = Table(show_header=True)
    sport_table.add_column("Sport", style="cyan")
    sport_table.add_column("Total", justify="right")
    sport_table.add_column("Completed", justify="right")
    sport_table.add_column("With Scores", justify="right")
    sport_table.add_column("ML Ready %", justify="right")
    
    for sport, data in games["by_sport"].items():
        pct = round(100 * data["with_scores"] / max(data["completed"], 1), 1)
        sport_table.add_row(
            sport,
            str(data["total"]),
            str(data["completed"]),
            str(data["with_scores"]),
            f"{pct}%"
        )
    
    console.print(sport_table)
    
    # Odds Validation
    console.print("\n[bold]4. ODDS DATA[/bold]")
    odds = results["odds"]
    console.print(f"  Total odds records: {odds['total']:,}")
    console.print(f"  With spread: {odds['with_spread']:,}")
    console.print(f"  With total: {odds['with_total']:,}")
    console.print(f"  With moneyline: {odds['with_moneyline']:,}")
    
    if odds["sportsbooks"]:
        console.print("\n  Top Sportsbooks:")
        for sb, count in odds["sportsbooks"][:5]:
            console.print(f"    - {sb}: {count:,} records")
    
    # Team Stats
    console.print("\n[bold]5. TEAM STATS[/bold]")
    ts = results["team_stats"]
    console.print(f"  Total records: {ts['total']:,}")
    console.print(f"  Teams with stats: {ts['teams_with_stats']}")
    
    if ts["sample_stat_types"]:
        console.print(f"  Available stat types ({len(ts['sample_stat_types'])}):")
        console.print(f"    {', '.join(ts['sample_stat_types'][:15])}...")
    
    console.print("\n  By Sport:")
    for sport, data in ts["by_sport"].items():
        console.print(f"    - {sport}: {data['records']:,} records for {data['teams']} teams")
    
    # Injuries
    console.print("\n[bold]6. INJURIES[/bold]")
    inj = results["injuries"]
    console.print(f"  Total: {inj['total']:,}")
    for status, count in inj.get("by_status", {}).items():
        console.print(f"    - {status}: {count:,}")
    
    # Weather
    console.print("\n[bold]7. WEATHER DATA[/bold]")
    console.print(f"  Total records: {results['weather']['total']:,}")
    
    # Closing Lines
    console.print("\n[bold]8. CLOSING LINES (CLV Tracking)[/bold]")
    cl = results["closing_lines"]
    console.print(f"  Total: {cl['total']:,}")
    for sb, count in cl.get("by_sportsbook", []):
        console.print(f"    - {sb}: {count:,}")
    
    # Summary
    console.print("\n" + "=" * 60)
    console.print("[bold]VALIDATION SUMMARY[/bold]")
    console.print("=" * 60)
    
    critical_issues = []
    warnings = []
    
    if games["zero_score_finals"] > 0:
        critical_issues.append(f"🔴 {games['zero_score_finals']} games with 0-0 scores marked as final")
    
    if games["missing_scores"] > 0:
        critical_issues.append(f"🔴 {games['missing_scores']} final games missing scores")
    
    if results["table_counts"].get("team_stats", 0) == 0:
        critical_issues.append("🔴 team_stats table is empty - cannot compute features")
    
    if results["table_counts"].get("odds", 0) == 0:
        critical_issues.append("🔴 odds table is empty - cannot compute line movement features")
    
    if games["completed_games"] < 100:
        warnings.append(f"⚠️ Only {games['completed_games']} completed games - need more for ML training")
    
    if results["table_counts"].get("game_features", 0) == 0:
        warnings.append("⚠️ game_features table is empty - features not yet built")
    
    if critical_issues:
        console.print("\n[bold red]CRITICAL ISSUES (Must Fix):[/bold red]")
        for issue in critical_issues:
            console.print(f"  {issue}")
    
    if warnings:
        console.print("\n[bold yellow]WARNINGS:[/bold yellow]")
        for warning in warnings:
            console.print(f"  {warning}")
    
    if not critical_issues and not warnings:
        console.print("\n[bold green]✅ ALL CHECKS PASSED - Ready to build features![/bold green]")
    
    return len(critical_issues) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2: CHECK FEATURE READINESS
# ═══════════════════════════════════════════════════════════════════════════════

async def check_feature_readiness(session: AsyncSession) -> Dict:
    """Check if we have all data needed to build features."""
    
    console.print("\n[bold cyan]═══ FEATURE READINESS CHECK ═══[/bold cyan]\n")
    
    readiness = {
        "elo_features": False,
        "form_features": False,
        "rest_features": False,
        "h2h_features": False,
        "line_features": False,
        "weather_features": False,
        "injury_features": False,
        "team_stat_features": False,
    }
    
    checks = Table(show_header=True)
    checks.add_column("Feature Category", style="cyan")
    checks.add_column("Required Data", style="white")
    checks.add_column("Status", justify="center")
    checks.add_column("Notes")
    
    # ELO Features - Need: games with scores, teams
    result = await session.execute(text("""
        SELECT COUNT(*) FROM games 
        WHERE status = 'final' AND home_score > 0
    """))
    games_with_scores = result.scalar() or 0
    readiness["elo_features"] = games_with_scores >= 100
    checks.add_row(
        "ELO Ratings (6)",
        "games with scores",
        "✅" if readiness["elo_features"] else "❌",
        f"{games_with_scores} games"
    )
    
    # Form Features - Need: recent games per team
    result = await session.execute(text("""
        SELECT COUNT(DISTINCT home_team_id) + COUNT(DISTINCT away_team_id) 
        FROM games WHERE status = 'final'
    """))
    teams_with_games = (result.scalar() or 0) // 2
    readiness["form_features"] = teams_with_games >= 20
    checks.add_row(
        "Recent Form (10)",
        "games per team",
        "✅" if readiness["form_features"] else "❌",
        f"{teams_with_games} teams with games"
    )
    
    # Rest Features - Need: game dates
    readiness["rest_features"] = games_with_scores >= 50
    checks.add_row(
        "Rest/Travel (7)",
        "game dates",
        "✅" if readiness["rest_features"] else "❌",
        "Computed from game dates"
    )
    
    # H2H Features - Need: historical matchups
    readiness["h2h_features"] = games_with_scores >= 100
    checks.add_row(
        "Head-to-Head (5)",
        "historical games",
        "✅" if readiness["h2h_features"] else "❌",
        "Computed from past games"
    )
    
    # Line Movement Features - Need: odds data
    result = await session.execute(text("SELECT COUNT(*) FROM odds"))
    odds_count = result.scalar() or 0
    readiness["line_features"] = odds_count >= 100
    checks.add_row(
        "Line Movement (8)",
        "odds records",
        "✅" if readiness["line_features"] else "❌",
        f"{odds_count} odds records"
    )
    
    # Weather Features - Need: weather_data
    result = await session.execute(text("SELECT COUNT(*) FROM weather_data"))
    weather_count = result.scalar() or 0
    readiness["weather_features"] = weather_count >= 10 or True  # Optional
    checks.add_row(
        "Weather (6)",
        "weather records",
        "✅" if weather_count > 0 else "⚠️",
        f"{weather_count} records (optional)"
    )
    
    # Injury Features - Need: injuries data
    result = await session.execute(text("SELECT COUNT(*) FROM injuries"))
    injury_count = result.scalar() or 0
    readiness["injury_features"] = injury_count >= 10 or True  # Optional
    checks.add_row(
        "Injuries (4)",
        "injury records",
        "✅" if injury_count > 0 else "⚠️",
        f"{injury_count} records (optional)"
    )
    
    # Team Stats Features - Need: team_stats with stats JSONB
    result = await session.execute(text("""
        SELECT COUNT(*) FROM team_stats WHERE stats IS NOT NULL AND stats != '{}'::jsonb
    """))
    stats_count = result.scalar() or 0
    readiness["team_stat_features"] = stats_count >= 20
    checks.add_row(
        "Team Stats (8-20)",
        "team_stats records",
        "✅" if readiness["team_stat_features"] else "❌",
        f"{stats_count} stat records"
    )
    
    console.print(checks)
    
    # Overall readiness
    required_features = ["elo_features", "form_features", "rest_features", "h2h_features", "line_features"]
    all_required_ready = all(readiness[f] for f in required_features)
    
    console.print("\n[bold]Feature Readiness Summary:[/bold]")
    if all_required_ready:
        console.print("[bold green]✅ ALL REQUIRED DATA AVAILABLE - Can build features![/bold green]")
    else:
        missing = [f for f in required_features if not readiness[f]]
        console.print(f"[bold red]❌ MISSING DATA for: {', '.join(missing)}[/bold red]")
    
    return readiness


# ═══════════════════════════════════════════════════════════════════════════════
# PART 3: BUILD GAME FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

async def build_game_features(session: AsyncSession, sport_code: str = None, limit: int = None):
    """Build game_features from raw data."""
    
    console.print("\n[bold cyan]═══ BUILDING GAME FEATURES ═══[/bold cyan]\n")
    
    # Get games that need features
    where_clause = "WHERE g.status = 'final' AND g.home_score > 0"
    if sport_code:
        where_clause += f" AND s.code = '{sport_code}'"
    
    limit_clause = f"LIMIT {limit}" if limit else ""
    
    # Count games needing features
    result = await session.execute(text(f"""
        SELECT COUNT(*) FROM games g
        JOIN sports s ON g.sport_id = s.id
        LEFT JOIN game_features gf ON g.id = gf.game_id
        {where_clause}
        AND gf.id IS NULL
    """))
    games_to_process = result.scalar() or 0
    
    console.print(f"Games needing features: {games_to_process}")
    
    if games_to_process == 0:
        console.print("[yellow]No games need feature building.[/yellow]")
        return
    
    # Get games
    result = await session.execute(text(f"""
        SELECT 
            g.id as game_id,
            s.code as sport_code,
            g.home_team_id,
            g.away_team_id,
            g.scheduled_at,
            g.home_score,
            g.away_score,
            g.status
        FROM games g
        JOIN sports s ON g.sport_id = s.id
        LEFT JOIN game_features gf ON g.id = gf.game_id
        {where_clause}
        AND gf.id IS NULL
        ORDER BY g.scheduled_at
        {limit_clause}
    """))
    
    games = result.fetchall()
    console.print(f"Processing {len(games)} games...")
    
    built_count = 0
    error_count = 0
    
    with Progress() as progress:
        task = progress.add_task("Building features...", total=len(games))
        
        for game in games:
            try:
                # Build features for this game
                features = await compute_features_for_game(
                    session,
                    game_id=game[0],
                    sport_code=game[1],
                    home_team_id=game[2],
                    away_team_id=game[3],
                    game_date=game[4],
                    home_score=game[5],
                    away_score=game[6]
                )
                
                # Insert into game_features
                await session.execute(text("""
                    INSERT INTO game_features (game_id, features, computed_at)
                    VALUES (:game_id, :features, NOW())
                    ON CONFLICT (game_id) DO UPDATE SET
                        features = :features,
                        computed_at = NOW()
                """), {
                    "game_id": str(game[0]),
                    "features": json.dumps(features)
                })
                
                built_count += 1
                
            except Exception as e:
                error_count += 1
                if error_count <= 5:
                    console.print(f"[red]Error on game {game[0]}: {e}[/red]")
            
            progress.update(task, advance=1)
    
    await session.commit()
    
    console.print(f"\n[bold green]✅ Built features for {built_count} games[/bold green]")
    if error_count > 0:
        console.print(f"[yellow]⚠️ {error_count} errors encountered[/yellow]")


async def compute_features_for_game(
    session: AsyncSession,
    game_id: str,
    sport_code: str,
    home_team_id: str,
    away_team_id: str,
    game_date: datetime,
    home_score: int,
    away_score: int
) -> Dict:
    """Compute all features for a single game."""
    
    features = {}
    
    # ─────────────────────────────────────────────────────────────────────
    # 1. ELO FEATURES (Simplified - using win/loss record as proxy)
    # ─────────────────────────────────────────────────────────────────────
    home_elo = await get_team_elo(session, home_team_id, game_date)
    away_elo = await get_team_elo(session, away_team_id, game_date)
    
    features["home_elo"] = round(home_elo, 1)
    features["away_elo"] = round(away_elo, 1)
    features["elo_diff"] = round(home_elo - away_elo, 1)
    features["elo_win_prob"] = round(1 / (1 + 10 ** ((away_elo - home_elo) / 400)), 4)
    
    # ─────────────────────────────────────────────────────────────────────
    # 2. RECENT FORM FEATURES
    # ─────────────────────────────────────────────────────────────────────
    home_form = await get_recent_form(session, home_team_id, game_date)
    away_form = await get_recent_form(session, away_team_id, game_date)
    
    features["home_wins_last_5"] = home_form["wins_5"]
    features["away_wins_last_5"] = away_form["wins_5"]
    features["home_wins_last_10"] = home_form["wins_10"]
    features["away_wins_last_10"] = away_form["wins_10"]
    features["home_margin_last_5"] = round(home_form["margin_5"], 2)
    features["away_margin_last_5"] = round(away_form["margin_5"], 2)
    features["home_streak"] = home_form["streak"]
    features["away_streak"] = away_form["streak"]
    
    # ─────────────────────────────────────────────────────────────────────
    # 3. REST FEATURES
    # ─────────────────────────────────────────────────────────────────────
    home_rest = await get_rest_days(session, home_team_id, game_date)
    away_rest = await get_rest_days(session, away_team_id, game_date)
    
    features["home_rest_days"] = home_rest
    features["away_rest_days"] = away_rest
    features["rest_advantage"] = home_rest - away_rest
    features["home_b2b"] = 1 if home_rest <= 1 else 0
    features["away_b2b"] = 1 if away_rest <= 1 else 0
    
    # ─────────────────────────────────────────────────────────────────────
    # 4. HEAD-TO-HEAD FEATURES
    # ─────────────────────────────────────────────────────────────────────
    h2h = await get_h2h_record(session, home_team_id, away_team_id, game_date)
    
    features["h2h_home_wins"] = h2h["home_wins"]
    features["h2h_away_wins"] = h2h["away_wins"]
    features["h2h_home_win_pct"] = round(h2h["home_win_pct"], 3)
    features["h2h_avg_margin"] = round(h2h["avg_margin"], 2)
    
    # ─────────────────────────────────────────────────────────────────────
    # 5. LINE MOVEMENT FEATURES
    # ─────────────────────────────────────────────────────────────────────
    odds = await get_odds_data(session, game_id)
    
    features["opening_spread"] = odds.get("opening_spread", 0)
    features["current_spread"] = odds.get("current_spread", 0)
    features["spread_movement"] = round(odds.get("spread_movement", 0), 1)
    features["opening_total"] = odds.get("opening_total", 0)
    features["current_total"] = odds.get("current_total", 0)
    features["total_movement"] = round(odds.get("total_movement", 0), 1)
    features["home_ml"] = odds.get("home_ml", -110)
    features["away_ml"] = odds.get("away_ml", -110)
    
    # ─────────────────────────────────────────────────────────────────────
    # 6. TEAM STATS FEATURES (if available)
    # ─────────────────────────────────────────────────────────────────────
    home_stats = await get_team_stats(session, home_team_id)
    away_stats = await get_team_stats(session, away_team_id)
    
    features["home_ppg"] = round(home_stats.get("points_per_game", 0), 2)
    features["away_ppg"] = round(away_stats.get("points_per_game", 0), 2)
    features["home_ppg_allowed"] = round(home_stats.get("points_allowed_per_game", 0), 2)
    features["away_ppg_allowed"] = round(away_stats.get("points_allowed_per_game", 0), 2)
    features["home_ypg"] = round(home_stats.get("yards_per_game", 0), 2)
    features["away_ypg"] = round(away_stats.get("yards_per_game", 0), 2)
    
    # ─────────────────────────────────────────────────────────────────────
    # 7. WEATHER FEATURES (outdoor sports)
    # ─────────────────────────────────────────────────────────────────────
    weather = await get_weather_data(session, game_id)
    
    features["temperature"] = weather.get("temperature", 70)
    features["humidity"] = weather.get("humidity", 50)
    features["wind_speed"] = weather.get("wind_speed", 0)
    features["precipitation_pct"] = weather.get("precipitation_pct", 0)
    features["is_dome"] = 1 if weather.get("is_dome", False) else 0
    
    # ─────────────────────────────────────────────────────────────────────
    # 8. INJURY FEATURES
    # ─────────────────────────────────────────────────────────────────────
    home_injuries = await get_injury_impact(session, game_id, home_team_id, "home")
    away_injuries = await get_injury_impact(session, game_id, away_team_id, "away")
    
    features["home_injury_score"] = round(home_injuries["injury_score"], 3)
    features["away_injury_score"] = round(away_injuries["injury_score"], 3)
    features["home_players_out"] = home_injuries["players_out"]
    features["away_players_out"] = away_injuries["players_out"]
    
    # ─────────────────────────────────────────────────────────────────────
    # 9. TARGET VARIABLES (actual outcomes)
    # ─────────────────────────────────────────────────────────────────────
    features["home_score"] = home_score
    features["away_score"] = away_score
    features["actual_margin"] = home_score - away_score
    features["actual_total"] = home_score + away_score
    features["home_won"] = 1 if home_score > away_score else 0
    
    # Cover calculations
    spread = features.get("current_spread", 0) or 0
    total = features.get("current_total", 0) or 0
    features["home_covered"] = 1 if (home_score - away_score) > -spread else 0
    features["over_hit"] = 1 if total > 0 and (home_score + away_score) > total else 0
    
    return features


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

async def get_team_elo(session: AsyncSession, team_id: str, before_date: datetime) -> float:
    """Get team's ELO rating before a given date."""
    # Simplified: Calculate from win percentage
    result = await session.execute(text("""
        SELECT 
            SUM(CASE 
                WHEN (home_team_id = :team_id AND home_score > away_score) OR
                     (away_team_id = :team_id AND away_score > home_score) 
                THEN 1 ELSE 0 END) as wins,
            COUNT(*) as games
        FROM games
        WHERE (home_team_id = :team_id OR away_team_id = :team_id)
          AND status = 'final'
          AND home_score > 0
          AND scheduled_at < :before_date
    """), {"team_id": str(team_id), "before_date": before_date})
    
    row = result.fetchone()
    if row and row[1] > 0:
        win_pct = row[0] / row[1]
        return 1500 + (win_pct - 0.5) * 400  # Simple ELO approximation
    return 1500  # Default


async def get_recent_form(session: AsyncSession, team_id: str, before_date: datetime) -> Dict:
    """Get team's recent form statistics."""
    result = await session.execute(text("""
        SELECT 
            home_team_id, away_team_id, home_score, away_score
        FROM games
        WHERE (home_team_id = :team_id OR away_team_id = :team_id)
          AND status = 'final'
          AND home_score > 0
          AND scheduled_at < :before_date
        ORDER BY scheduled_at DESC
        LIMIT 10
    """), {"team_id": str(team_id), "before_date": before_date})
    
    games = result.fetchall()
    
    if not games:
        return {
            "wins_5": 0, "wins_10": 0, "margin_5": 0, "margin_10": 0, "streak": 0
        }
    
    wins_5 = 0
    wins_10 = 0
    margins = []
    streak = 0
    streak_type = None
    
    for i, g in enumerate(games):
        is_home = str(g[0]) == str(team_id)
        if is_home:
            won = g[2] > g[3]
            margin = g[2] - g[3]
        else:
            won = g[3] > g[2]
            margin = g[3] - g[2]
        
        if i < 5:
            wins_5 += 1 if won else 0
        wins_10 += 1 if won else 0
        margins.append(margin)
        
        # Streak calculation
        if i == 0:
            streak_type = won
            streak = 1 if won else -1
        elif won == streak_type:
            streak += 1 if won else -1
    
    return {
        "wins_5": wins_5,
        "wins_10": wins_10,
        "margin_5": sum(margins[:5]) / max(len(margins[:5]), 1),
        "margin_10": sum(margins) / max(len(margins), 1),
        "streak": streak
    }


async def get_rest_days(session: AsyncSession, team_id: str, game_date: datetime) -> int:
    """Get days since team's last game."""
    result = await session.execute(text("""
        SELECT scheduled_at
        FROM games
        WHERE (home_team_id = :team_id OR away_team_id = :team_id)
          AND scheduled_at < :game_date
          AND status = 'final'
        ORDER BY scheduled_at DESC
        LIMIT 1
    """), {"team_id": str(team_id), "game_date": game_date})
    
    row = result.fetchone()
    if row:
        last_game = row[0]
        return (game_date - last_game).days
    return 7  # Default


async def get_h2h_record(session: AsyncSession, home_team_id: str, away_team_id: str, before_date: datetime) -> Dict:
    """Get head-to-head record between two teams."""
    result = await session.execute(text("""
        SELECT 
            home_team_id, away_team_id, home_score, away_score
        FROM games
        WHERE ((home_team_id = :home_id AND away_team_id = :away_id) OR
               (home_team_id = :away_id AND away_team_id = :home_id))
          AND status = 'final'
          AND home_score > 0
          AND scheduled_at < :before_date
        ORDER BY scheduled_at DESC
        LIMIT 10
    """), {"home_id": str(home_team_id), "away_id": str(away_team_id), "before_date": before_date})
    
    games = result.fetchall()
    
    if not games:
        return {"home_wins": 0, "away_wins": 0, "home_win_pct": 0.5, "avg_margin": 0}
    
    home_wins = 0
    margins = []
    
    for g in games:
        if str(g[0]) == str(home_team_id):
            # Current home team was home
            if g[2] > g[3]:
                home_wins += 1
            margins.append(g[2] - g[3])
        else:
            # Current home team was away
            if g[3] > g[2]:
                home_wins += 1
            margins.append(g[3] - g[2])
    
    return {
        "home_wins": home_wins,
        "away_wins": len(games) - home_wins,
        "home_win_pct": home_wins / len(games),
        "avg_margin": sum(margins) / len(margins)
    }


async def get_odds_data(session: AsyncSession, game_id: str) -> Dict:
    """Get odds data for a game.
    
    Actual odds table columns:
    - home_line, away_line (spread)
    - total (over/under line)
    - home_odds, away_odds (spread juice)
    - over_odds, under_odds (total juice)
    - is_opening (boolean for opening line)
    """
    # Get opening odds
    result_open = await session.execute(text("""
        SELECT home_line, total, home_odds, away_odds
        FROM odds
        WHERE game_id = :game_id AND bet_type = 'spread' AND is_opening = true
        ORDER BY recorded_at ASC
        LIMIT 1
    """), {"game_id": str(game_id)})
    opening = result_open.fetchone()
    
    # Get current odds (most recent)
    result_current = await session.execute(text("""
        SELECT home_line, total, home_odds, away_odds
        FROM odds
        WHERE game_id = :game_id AND bet_type = 'spread'
        ORDER BY recorded_at DESC
        LIMIT 1
    """), {"game_id": str(game_id)})
    current = result_current.fetchone()
    
    # Get totals
    result_total = await session.execute(text("""
        SELECT total FROM odds
        WHERE game_id = :game_id AND bet_type = 'total'
        ORDER BY recorded_at DESC
        LIMIT 1
    """), {"game_id": str(game_id)})
    total_row = result_total.fetchone()
    
    if not opening and not current:
        return {}
    
    opening_spread = opening[0] if opening and opening[0] else 0
    opening_total = opening[1] if opening and opening[1] else (total_row[0] if total_row else 0)
    current_spread = current[0] if current and current[0] else opening_spread
    current_total = current[1] if current and current[1] else (total_row[0] if total_row else opening_total)
    
    return {
        "opening_spread": opening_spread,
        "current_spread": current_spread,
        "spread_movement": current_spread - opening_spread if opening_spread else 0,
        "opening_total": opening_total,
        "current_total": current_total,
        "total_movement": current_total - opening_total if opening_total else 0,
        "home_ml": current[2] if current and current[2] else -110,
        "away_ml": current[3] if current and current[3] else -110
    }


async def get_team_stats(session: AsyncSession, team_id: str) -> Dict:
    """Get team statistics from team_stats table.
    
    team_stats uses individual rows with (stat_type, value) pairs,
    not JSONB. This function pivots them into a dict.
    """
    result = await session.execute(text("""
        SELECT stat_type, value
        FROM team_stats
        WHERE team_id = :team_id
    """), {"team_id": str(team_id)})
    
    stats = {}
    for row in result.fetchall():
        stats[row[0]] = row[1]
    
    return stats


async def get_weather_data(session: AsyncSession, game_id: str) -> Dict:
    """Get weather data for a game."""
    result = await session.execute(text("""
        SELECT temperature_f, humidity_pct, wind_speed_mph, 
               precipitation_pct, is_dome, conditions
        FROM weather_data
        WHERE game_id = :game_id
        LIMIT 1
    """), {"game_id": str(game_id)})
    
    row = result.fetchone()
    if row:
        return {
            "temperature": row[0] or 70,
            "humidity": row[1] or 50,
            "wind_speed": row[2] or 0,
            "precipitation_pct": row[3] or 0,
            "is_dome": row[4] or False,
            "conditions": row[5] or "Clear"
        }
    return {"temperature": 70, "humidity": 50, "wind_speed": 0, "is_dome": True}


async def get_injury_impact(session: AsyncSession, game_id: str, team_id: str, side: str) -> Dict:
    """Get injury impact for a team in a game."""
    result = await session.execute(text("""
        SELECT 
            COUNT(*) as total_injuries,
            SUM(CASE WHEN i.status IN ('Out', 'IR') THEN 1 ELSE 0 END) as players_out,
            SUM(CASE WHEN i.is_starter THEN 1 ELSE 0 END) as starters_out,
            AVG(gi.impact_on_game) as avg_impact
        FROM game_injuries gi
        JOIN injuries i ON gi.injury_id = i.id
        WHERE gi.game_id = :game_id
          AND gi.team_side = :side
    """), {"game_id": str(game_id), "side": side})
    
    row = result.fetchone()
    if row and row[0] > 0:
        # Calculate injury score (1.0 = fully healthy, lower = more injuries)
        players_out = row[1] or 0
        starters_out = row[2] or 0
        injury_score = max(0.5, 1.0 - (starters_out * 0.1) - (players_out * 0.02))
        return {
            "injury_score": injury_score,
            "players_out": players_out,
            "starters_out": starters_out
        }
    return {"injury_score": 1.0, "players_out": 0, "starters_out": 0}


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

async def main():
    parser = argparse.ArgumentParser(description="AI PRO SPORTS Data Validation & Feature Builder")
    parser.add_argument("--validate", action="store_true", help="Run data validation")
    parser.add_argument("--check-readiness", action="store_true", help="Check feature readiness")
    parser.add_argument("--build", action="store_true", help="Build game_features")
    parser.add_argument("--all", action="store_true", help="Run all steps")
    parser.add_argument("--sport", type=str, help="Filter by sport code (e.g., NFL, NBA)")
    parser.add_argument("--limit", type=int, help="Limit number of games to process")
    
    args = parser.parse_args()
    
    if not any([args.validate, args.check_readiness, args.build, args.all]):
        parser.print_help()
        return
    
    from app.core.database import db_manager
    
    await db_manager.initialize()
    
    async with db_manager.session() as session:
        if args.validate or args.all:
            results = await run_validation(session)
            is_valid = display_validation_results(results)
        
        if args.check_readiness or args.all:
            await check_feature_readiness(session)
        
        if args.build or args.all:
            await build_game_features(
                session, 
                sport_code=args.sport,
                limit=args.limit
            )


if __name__ == "__main__":
    asyncio.run(main())