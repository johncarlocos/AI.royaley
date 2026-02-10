#!/usr/bin/env python3
"""
ROYALEY - Phase 7a: Score Predictions
Scores past predictions against actual game results.

Usage:
    python scripts/score_predictions.py --date today
    python scripts/score_predictions.py --date 2025-02-10 --predictions predictions.json
    python scripts/score_predictions.py --predictions-dir paper_trading/ --all
"""

import asyncio
import argparse
import logging
import sys
import json
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from app.core.database import db_manager
from app.models import Game, Sport, Team, GameStatus
from sqlalchemy import select, and_

console = Console()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


async def get_game_result(session, game_id: str) -> Optional[Dict]:
    """Get actual game result from database."""
    from uuid import UUID
    try:
        game = await session.get(Game, UUID(game_id))
    except Exception:
        return None

    if not game or game.status not in [GameStatus.FINAL, GameStatus.COMPLETED]:
        return None

    home_team = await session.get(Team, game.home_team_id)
    away_team = await session.get(Team, game.away_team_id)

    return {
        "game_id": str(game.id),
        "home_score": game.home_score,
        "away_score": game.away_score,
        "home_team": home_team.name if home_team else "",
        "away_team": away_team.name if away_team else "",
        "margin": (game.home_score or 0) - (game.away_score or 0),
        "total_points": (game.home_score or 0) + (game.away_score or 0),
        "home_won": (game.home_score or 0) > (game.away_score or 0),
    }


def score_prediction(prediction: Dict, result: Dict) -> Dict:
    """Score a single prediction against actual result."""
    bet_type = prediction["bet_type"]
    pred_side = prediction["predicted_side"]
    pred_prob = prediction["probability"]
    implied_prob = prediction["implied_probability"]
    edge = prediction["edge"]
    tier = prediction["signal_tier"]
    stake = prediction.get("recommended_stake", 0)
    odds = prediction.get("odds", -110)

    won = False
    actual_outcome = None

    if bet_type == "moneyline":
        actual_outcome = "home" if result["home_won"] else "away"
        won = (pred_side == actual_outcome)

    elif bet_type == "spread":
        line = prediction.get("line", 0)
        if line is not None:
            margin = result["margin"]
            # Home covers if margin > -line (positive line means home is underdog)
            home_covers = margin > -line
            actual_outcome = "home" if home_covers else "away"
            won = (pred_side == actual_outcome)
            # Push
            if abs(margin + line) < 0.001:
                won = None  # Push
                actual_outcome = "push"

    elif bet_type == "total":
        line = prediction.get("line", 0)
        if line is not None:
            total = result["total_points"]
            went_over = total > line
            actual_outcome = "over" if went_over else "under"
            won = (pred_side == actual_outcome)
            # Push
            if abs(total - line) < 0.001:
                won = None
                actual_outcome = "push"

    # Calculate P&L
    if won is True:
        if odds > 0:
            profit = stake * (odds / 100.0)
        else:
            profit = stake * (100.0 / abs(odds))
    elif won is False:
        profit = -stake
    else:
        profit = 0  # Push

    # CLV: Compare predicted prob vs closing implied prob
    clv = pred_prob - implied_prob

    return {
        **prediction,
        "actual_outcome": actual_outcome,
        "won": won,
        "profit_loss": round(profit, 2),
        "clv": round(clv, 4),
        "home_score": result["home_score"],
        "away_score": result["away_score"],
        "scored_at": datetime.utcnow().isoformat(),
    }


async def score_predictions_file(
    predictions_path: str,
    output_path: str = None,
) -> List[Dict]:
    """Score all predictions in a JSON file."""

    with open(predictions_path) as f:
        predictions = json.load(f)

    console.print(f"[cyan]Scoring {len(predictions)} predictions from {predictions_path}[/cyan]")

    await db_manager.initialize()
    scored = []

    async with db_manager.session() as session:
        for pred in predictions:
            game_id = pred.get("game_id")
            if not game_id:
                continue

            result = await get_game_result(session, game_id)
            if result is None:
                pred["won"] = None
                pred["actual_outcome"] = "pending"
                pred["profit_loss"] = 0
                scored.append(pred)
                continue

            scored_pred = score_prediction(pred, result)
            scored.append(scored_pred)

    # Print results
    _print_scored_predictions(scored)

    # Save
    if output_path is None:
        output_path = predictions_path.replace(".json", "_scored.json")

    with open(output_path, 'w') as f:
        json.dump(scored, f, indent=2)
    console.print(f"\n[green]Scored predictions saved to {output_path}[/green]")

    # Append to running log
    log_path = Path("paper_trading") / "scored_log.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for s in scored:
        if s.get("won") is not None:
            rows.append({
                "date": s.get("scheduled_at", "")[:10],
                "sport": s.get("sport", ""),
                "bet_type": s.get("bet_type", ""),
                "matchup": f"{s.get('away_abbr', '')} @ {s.get('home_abbr', '')}",
                "pick": s.get("predicted_side", ""),
                "probability": s.get("probability", 0),
                "edge": s.get("edge", 0),
                "tier": s.get("signal_tier", ""),
                "won": s.get("won"),
                "profit_loss": s.get("profit_loss", 0),
                "clv": s.get("clv", 0),
            })

    if rows:
        new_df = pd.DataFrame(rows)
        if log_path.exists():
            existing = pd.read_csv(log_path)
            combined = pd.concat([existing, new_df], ignore_index=True)
        else:
            combined = new_df
        combined.to_csv(log_path, index=False)
        console.print(f"[green]Updated running log: {log_path} ({len(combined)} total bets)[/green]")

    return scored


def _print_scored_predictions(scored: List[Dict]):
    """Print scored predictions table."""
    resolved = [s for s in scored if s.get("won") is not None]
    pending = [s for s in scored if s.get("won") is None or s.get("actual_outcome") == "pending"]

    if not resolved:
        console.print("[yellow]No resolved predictions yet[/yellow]")
        return

    table = Table(title="Scored Predictions")
    table.add_column("Matchup", style="white")
    table.add_column("Bet", style="blue")
    table.add_column("Pick", style="cyan")
    table.add_column("Prob", style="yellow")
    table.add_column("Result", style="white")
    table.add_column("W/L", style="bold")
    table.add_column("P&L", style="green")
    table.add_column("CLV", style="magenta")

    for s in resolved:
        matchup = f"{s.get('away_abbr', '?')} @ {s.get('home_abbr', '?')}"
        result_str = f"{s.get('home_score', '?')}-{s.get('away_score', '?')}"

        if s["won"] is True:
            wl_str = "[green]WIN[/green]"
        elif s["won"] is False:
            wl_str = "[red]LOSS[/red]"
        else:
            wl_str = "[yellow]PUSH[/yellow]"

        pnl = s.get("profit_loss", 0)
        pnl_str = f"${pnl:+,.2f}"
        if pnl > 0:
            pnl_str = f"[green]{pnl_str}[/green]"
        elif pnl < 0:
            pnl_str = f"[red]{pnl_str}[/red]"

        clv = s.get("clv", 0)
        clv_str = f"{clv*100:+.1f}%"

        table.add_row(
            matchup, s.get("bet_type", ""), s.get("predicted_side", "").upper(),
            f"{s.get('probability', 0)*100:.1f}%", result_str,
            wl_str, pnl_str, clv_str,
        )

    console.print(table)

    # Summary
    wins = sum(1 for s in resolved if s.get("won") is True)
    losses = sum(1 for s in resolved if s.get("won") is False)
    pushes = sum(1 for s in resolved if s.get("won") is None and s.get("actual_outcome") == "push")
    total_pnl = sum(s.get("profit_loss", 0) for s in resolved)
    avg_clv = np.mean([s.get("clv", 0) for s in resolved])

    wr = wins / (wins + losses) if (wins + losses) > 0 else 0

    console.print(Panel(
        f"Resolved: {len(resolved)} | Pending: {len(pending)}\n"
        f"Record: {wins}W-{losses}L-{pushes}P ({wr*100:.1f}%)\n"
        f"P&L: ${total_pnl:+,.2f}\n"
        f"Avg CLV: {avg_clv*100:+.1f}%",
        title="Scoring Summary"
    ))


async def main():
    parser = argparse.ArgumentParser(description="ROYALEY Phase 7a: Score Predictions")
    parser.add_argument("--predictions", "-p", type=str, help="Predictions JSON file")
    parser.add_argument("--date", "-d", type=str, help="Date to score (today, yesterday, YYYY-MM-DD)")
    parser.add_argument("--predictions-dir", type=str, default="paper_trading")
    parser.add_argument("--output", "-o", type=str, default=None)
    parser.add_argument("--all", action="store_true", help="Score all unscored predictions")

    args = parser.parse_args()

    if args.predictions:
        await score_predictions_file(args.predictions, args.output)
    elif args.date:
        if args.date == "today":
            d = date.today()
        elif args.date == "yesterday":
            d = date.today() - timedelta(days=1)
        else:
            d = datetime.strptime(args.date, "%Y-%m-%d").date()

        pred_file = Path(args.predictions_dir) / f"predictions_{d}.json"
        if pred_file.exists():
            await score_predictions_file(str(pred_file), args.output)
        else:
            console.print(f"[yellow]No prediction file found: {pred_file}[/yellow]")
    elif args.all:
        pred_dir = Path(args.predictions_dir)
        if pred_dir.exists():
            for f in sorted(pred_dir.glob("predictions_*.json")):
                if "_scored" not in f.name:
                    console.print(f"\n{'='*60}")
                    await score_predictions_file(str(f))
        else:
            console.print(f"[yellow]Directory not found: {pred_dir}[/yellow]")
    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
