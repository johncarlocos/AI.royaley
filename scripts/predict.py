#!/usr/bin/env python3
"""
ROYALEY - Prediction Script
Phase 2: CLI for generating predictions

Usage:
    # Generate predictions for today's games
    python predict.py --sport NFL
    
    # Generate predictions for specific date
    python predict.py --sport NBA --date 2025-01-15
    
    # Generate predictions for all sports
    python predict.py --all
    
    # Output to file
    python predict.py --sport NFL --output predictions.json
"""

import asyncio
import argparse
import logging
import sys
import json
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from app.core.config import settings
from app.core.database import db_manager
from app.models import (
    Sport, Game, Team, Odds, MLModel, Prediction as PredictionModel,
    GameStatus, MLFramework, SignalTier
)
from sqlalchemy import select, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

console = Console()
logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


BET_TYPES = ["spread", "moneyline", "total"]


async def get_upcoming_games(
    session: AsyncSession,
    sport_code: str,
    target_date: date = None,
) -> List[Game]:
    """Get upcoming games for a sport."""
    target_date = target_date or date.today()
    
    # Get sport
    result = await session.execute(
        select(Sport).where(Sport.code == sport_code)
    )
    sport = result.scalar_one_or_none()
    
    if not sport:
        console.print(f"[red]Sport not found: {sport_code}[/red]")
        return []
    
    # Get games for the target date
    start_dt = datetime.combine(target_date, datetime.min.time())
    end_dt = datetime.combine(target_date, datetime.max.time())
    
    query = (
        select(Game)
        .where(
            and_(
                Game.sport_id == sport.id,
                Game.scheduled_at >= start_dt,
                Game.scheduled_at <= end_dt,
                Game.status == GameStatus.SCHEDULED,
            )
        )
        .order_by(Game.scheduled_at)
    )
    
    result = await session.execute(query)
    return result.scalars().all()


async def get_production_model(
    session: AsyncSession,
    sport_code: str,
    bet_type: str,
) -> Optional[MLModel]:
    """Get production model for sport/bet_type."""
    # Get sport
    result = await session.execute(
        select(Sport).where(Sport.code == sport_code)
    )
    sport = result.scalar_one_or_none()
    
    if not sport:
        return None
    
    # Get production model
    query = (
        select(MLModel)
        .where(
            and_(
                MLModel.sport_id == sport.id,
                MLModel.bet_type == bet_type,
                MLModel.is_production == True,
            )
        )
        .order_by(MLModel.created_at.desc())
        .limit(1)
    )
    
    result = await session.execute(query)
    return result.scalar_one_or_none()


async def generate_prediction(
    session: AsyncSession,
    game: Game,
    model: MLModel,
    bet_type: str,
) -> Dict[str, Any]:
    """Generate prediction for a game."""
    try:
        # Get teams
        home_team = await session.get(Team, game.home_team_id)
        away_team = await session.get(Team, game.away_team_id)
        
        if not home_team or not away_team:
            return None
        
        # Get current odds
        odds = await get_game_odds(session, game.id, bet_type)
        
        # For now, use a simple ELO-based prediction
        # In production, this would load the actual model and make predictions
        elo_diff = home_team.elo_rating - away_team.elo_rating
        home_advantage = settings.get_sports_config(
            model.sport.code if model.sport else "NFL"
        ).get("home_advantage", 2.5)
        
        # Simple probability calculation based on ELO
        win_prob = 1 / (1 + 10 ** (-(elo_diff + home_advantage * 10) / 400))
        
        # Adjust for bet type
        if bet_type == "spread":
            # Estimate spread cover probability
            if odds and odds.home_line:
                spread = odds.home_line
                cover_prob = win_prob + (spread / 100) * 0.1  # Rough adjustment
                cover_prob = max(0.2, min(0.8, cover_prob))
            else:
                cover_prob = win_prob
            probability = cover_prob
            predicted_side = "home" if cover_prob > 0.5 else "away"
            
        elif bet_type == "moneyline":
            probability = win_prob
            predicted_side = "home" if win_prob > 0.5 else "away"
            
        else:  # total
            # Rough total prediction
            if odds and odds.total:
                total = odds.total
                # Higher scoring teams = more likely over
                avg_points = (home_team.elo_rating + away_team.elo_rating) / 2
                over_prob = 0.5 + (avg_points - 1500) / 1000
                over_prob = max(0.3, min(0.7, over_prob))
            else:
                over_prob = 0.5
            probability = over_prob
            predicted_side = "over" if over_prob > 0.5 else "under"
        
        # Calculate edge
        if odds:
            if bet_type == "spread":
                implied_prob = american_to_prob(
                    odds.home_odds if predicted_side == "home" else odds.away_odds
                )
            elif bet_type == "moneyline":
                implied_prob = american_to_prob(
                    odds.home_odds if predicted_side == "home" else odds.away_odds
                )
            else:
                implied_prob = american_to_prob(
                    odds.over_odds if predicted_side == "over" else odds.under_odds
                )
            edge = probability - implied_prob
        else:
            implied_prob = 0.5
            edge = 0
        
        # Determine signal tier
        if edge >= 0.10:
            tier = "A"
        elif edge >= 0.05:
            tier = "B"
        elif edge >= 0.02:
            tier = "C"
        else:
            tier = "D"
        
        return {
            "game_id": str(game.id),
            "scheduled_at": game.scheduled_at.isoformat(),
            "home_team": home_team.name,
            "away_team": away_team.name,
            "home_abbr": home_team.abbreviation,
            "away_abbr": away_team.abbreviation,
            "bet_type": bet_type,
            "predicted_side": predicted_side,
            "probability": round(probability, 4),
            "implied_probability": round(implied_prob, 4),
            "edge": round(edge, 4),
            "signal_tier": tier,
            "home_elo": home_team.elo_rating,
            "away_elo": away_team.elo_rating,
            "line": odds.home_line if odds and bet_type == "spread" else (
                odds.total if odds and bet_type == "total" else None
            ),
            "odds": odds.home_odds if odds and predicted_side == "home" else (
                odds.away_odds if odds else None
            ),
            "model_id": str(model.id) if model else None,
            "model_version": model.version if model else None,
        }
        
    except Exception as e:
        logger.error(f"Error generating prediction: {e}")
        return None


async def get_game_odds(
    session: AsyncSession,
    game_id,
    bet_type: str,
) -> Optional[Odds]:
    """Get current odds for a game."""
    query = (
        select(Odds)
        .where(
            and_(
                Odds.game_id == game_id,
                or_(
                    Odds.bet_type == bet_type,
                    Odds.bet_type == "spreads" if bet_type == "spread" else Odds.bet_type == bet_type,
                    Odds.bet_type == "totals" if bet_type == "total" else Odds.bet_type == bet_type,
                )
            )
        )
        .order_by(Odds.recorded_at.desc())
        .limit(1)
    )
    
    result = await session.execute(query)
    return result.scalar_one_or_none()


def american_to_prob(american_odds: int) -> float:
    """Convert American odds to implied probability."""
    if american_odds is None:
        return 0.5
    if american_odds > 0:
        return 100 / (american_odds + 100)
    else:
        return -american_odds / (-american_odds + 100)


async def generate_predictions(
    sport_code: str,
    target_date: date = None,
    bet_types: List[str] = None,
) -> List[Dict[str, Any]]:
    """Generate predictions for a sport."""
    bet_types = bet_types or BET_TYPES
    target_date = target_date or date.today()
    predictions = []
    
    await db_manager.initialize()
    
    async with db_manager.session() as session:
        # Get games
        games = await get_upcoming_games(session, sport_code, target_date)
        
        if not games:
            console.print(f"[yellow]No games found for {sport_code} on {target_date}[/yellow]")
            return []
        
        console.print(f"[cyan]Found {len(games)} games for {sport_code}[/cyan]")
        
        for bet_type in bet_types:
            # Get model
            model = await get_production_model(session, sport_code, bet_type)
            
            if not model:
                console.print(f"[yellow]No production model for {sport_code} {bet_type}[/yellow]")
                # Still generate predictions using ELO
            
            for game in games:
                pred = await generate_prediction(session, game, model, bet_type)
                if pred:
                    predictions.append(pred)
    
    return predictions


def print_predictions(predictions: List[Dict[str, Any]], sport_code: str):
    """Print predictions table."""
    if not predictions:
        console.print("[yellow]No predictions generated[/yellow]")
        return
    
    table = Table(title=f"{sport_code} Predictions")
    
    table.add_column("Time", style="cyan")
    table.add_column("Matchup", style="white")
    table.add_column("Bet", style="blue")
    table.add_column("Pick", style="green")
    table.add_column("Prob", style="yellow")
    table.add_column("Edge", style="magenta")
    table.add_column("Tier", style="red")
    
    for pred in predictions:
        # Format time
        dt = datetime.fromisoformat(pred["scheduled_at"])
        time_str = dt.strftime("%I:%M %p")
        
        # Format matchup
        matchup = f"{pred['away_abbr']} @ {pred['home_abbr']}"
        
        # Format pick
        if pred["bet_type"] == "spread":
            pick = f"{pred['predicted_side'].upper()} {pred.get('line', '')}"
        elif pred["bet_type"] == "moneyline":
            pick = pred['predicted_side'].upper()
        else:
            pick = f"{pred['predicted_side'].upper()} {pred.get('line', '')}"
        
        # Format edge with color
        edge = pred["edge"]
        edge_str = f"{edge*100:+.1f}%"
        if edge >= 0.05:
            edge_str = f"[green]{edge_str}[/green]"
        elif edge >= 0.02:
            edge_str = f"[yellow]{edge_str}[/yellow]"
        else:
            edge_str = f"[red]{edge_str}[/red]"
        
        # Format tier
        tier = pred["signal_tier"]
        if tier == "A":
            tier_str = f"[bold green]{tier}[/bold green]"
        elif tier == "B":
            tier_str = f"[green]{tier}[/green]"
        elif tier == "C":
            tier_str = f"[yellow]{tier}[/yellow]"
        else:
            tier_str = f"[red]{tier}[/red]"
        
        table.add_row(
            time_str,
            matchup,
            pred["bet_type"],
            pick,
            f"{pred['probability']*100:.1f}%",
            edge_str,
            tier_str,
        )
    
    console.print(table)
    
    # Summary
    tier_counts = {}
    for pred in predictions:
        tier = pred["signal_tier"]
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
    
    console.print(f"\n[cyan]Signal Tiers:[/cyan] " + ", ".join(
        f"{t}: {c}" for t, c in sorted(tier_counts.items())
    ))


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="ROYALEY Prediction Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict.py --sport NFL
  python predict.py --sport NBA --date 2025-01-15
  python predict.py --all
  python predict.py --sport NFL --output predictions.json
        """
    )
    
    parser.add_argument(
        "--sport", "-s",
        type=str,
        help="Sport code (NFL, NBA, NHL, etc.)"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Generate predictions for all sports"
    )
    parser.add_argument(
        "--date", "-d",
        type=str,
        help="Target date (YYYY-MM-DD), default: today"
    )
    parser.add_argument(
        "--bet-type", "-b",
        type=str,
        choices=BET_TYPES,
        help="Specific bet type only"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file (JSON)"
    )
    parser.add_argument(
        "--min-edge",
        type=float,
        default=0.0,
        help="Minimum edge filter (default: 0)"
    )
    parser.add_argument(
        "--tier",
        type=str,
        choices=["A", "B", "C", "D"],
        help="Filter by signal tier"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate arguments
    if not args.all and not args.sport:
        parser.print_help()
        console.print("\n[yellow]Specify --sport or --all[/yellow]")
        return
    
    # Parse date
    target_date = None
    if args.date:
        try:
            target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
        except ValueError:
            console.print(f"[red]Invalid date format: {args.date}[/red]")
            return
    else:
        target_date = date.today()
    
    # Get sports
    sport_codes = [args.sport] if args.sport else settings.SUPPORTED_SPORTS
    bet_types = [args.bet_type] if args.bet_type else BET_TYPES
    
    # Print header
    console.print(Panel(
        f"[bold green]ROYALEY Predictions[/bold green]\n"
        f"Date: {target_date}\n"
        f"Sports: {', '.join(sport_codes)}\n"
        f"Bet Types: {', '.join(bet_types)}",
        title="Prediction Configuration"
    ))
    
    all_predictions = []
    
    try:
        for sport_code in sport_codes:
            predictions = await generate_predictions(
                sport_code=sport_code,
                target_date=target_date,
                bet_types=bet_types,
            )
            
            # Apply filters
            if args.min_edge > 0:
                predictions = [p for p in predictions if p["edge"] >= args.min_edge]
            
            if args.tier:
                predictions = [p for p in predictions if p["signal_tier"] == args.tier]
            
            if predictions:
                print_predictions(predictions, sport_code)
                all_predictions.extend(predictions)
            
            console.print("")
        
        # Output to file
        if args.output and all_predictions:
            with open(args.output, 'w') as f:
                json.dump(all_predictions, f, indent=2)
            console.print(f"[green]Predictions saved to {args.output}[/green]")
        
        # Final summary
        console.print(Panel(
            f"[bold]Total Predictions: {len(all_predictions)}[/bold]",
            title="Summary"
        ))
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        if args.verbose:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
