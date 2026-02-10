#!/usr/bin/env python3
"""
ROYALEY - Phase 5: Walk-Forward Backtesting
Full simulation using trained models with fractional Kelly sizing.

Usage:
    python scripts/run_backtest.py --models-dir /nvme0n1-disk/royaley/models
    python scripts/run_backtest.py --models-dir /nvme0n1-disk/royaley/models --sports NFL,NBA --kelly 0.25
    python scripts/run_backtest.py --models-dir /nvme0n1-disk/royaley/models --start-date 2024-01-01 --end-date 2025-01-01
"""

import asyncio
import argparse
import logging
import sys
import pickle
import json
from datetime import datetime, date
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from scripts.evaluate_models import (
    discover_models, load_and_predict, load_validation_data,
    SPORTS, BET_TYPES, TARGET_COLUMNS,
)

console = Console()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@dataclass
class Bet:
    """Single bet record."""
    date: str
    sport: str
    bet_type: str
    framework: str
    predicted_prob: float
    implied_prob: float
    edge: float
    signal_tier: str
    kelly_fraction: float
    stake: float
    odds: int  # American odds
    result: int  # 1=win, 0=loss
    profit_loss: float
    bankroll_after: float


@dataclass
class BacktestResult:
    """Full backtest results."""
    bets: List[Bet] = field(default_factory=list)
    initial_bankroll: float = 10000.0
    final_bankroll: float = 10000.0
    total_bets: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    roi: float = 0.0
    total_profit: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    tier_a_bets: int = 0
    tier_a_wins: int = 0
    tier_a_winrate: float = 0.0
    tier_b_bets: int = 0
    tier_b_wins: int = 0
    tier_b_winrate: float = 0.0
    avg_edge: float = 0.0
    avg_kelly: float = 0.0
    daily_pnl: Dict[str, float] = field(default_factory=dict)
    bankroll_curve: List[float] = field(default_factory=list)


def american_to_decimal(american: int) -> float:
    """Convert American odds to decimal odds."""
    if american > 0:
        return 1.0 + american / 100.0
    else:
        return 1.0 + 100.0 / abs(american)


def american_to_implied(american: int) -> float:
    """Convert American odds to implied probability."""
    if american > 0:
        return 100.0 / (american + 100.0)
    else:
        return abs(american) / (abs(american) + 100.0)


def kelly_criterion(pred_prob: float, decimal_odds: float, fraction: float = 0.25) -> float:
    """
    Fractional Kelly Criterion.
    Returns bet size as fraction of bankroll.
    """
    b = decimal_odds - 1.0  # Net odds
    p = pred_prob
    q = 1.0 - p

    if b <= 0:
        return 0.0

    kelly_full = (b * p - q) / b

    if kelly_full <= 0:
        return 0.0

    return kelly_full * fraction


def classify_tier(edge: float, pred_prob: float) -> str:
    """Classify prediction into signal tier."""
    if pred_prob >= 0.65 and edge >= 0.10:
        return "A"
    elif pred_prob >= 0.60 and edge >= 0.05:
        return "B"
    elif pred_prob >= 0.55 and edge >= 0.02:
        return "C"
    else:
        return "D"


def run_backtest(
    models_dir: Path,
    csv_dir: str = None,
    sports: List[str] = None,
    bet_types: List[str] = None,
    initial_bankroll: float = 10000.0,
    kelly_fraction: float = 0.25,
    min_edge: float = 0.03,
    min_probability: float = 0.55,
    max_bet_pct: float = 0.02,
    signal_tiers: List[str] = None,
    start_date: str = None,
    end_date: str = None,
    verbose: bool = False,
) -> BacktestResult:
    """Run full walk-forward backtest."""

    sports = sports or SPORTS
    bet_types = bet_types or BET_TYPES
    signal_tiers = signal_tiers or ["A", "B"]

    console.print(Panel(
        f"[bold cyan]ROYALEY Phase 5: Walk-Forward Backtesting[/bold cyan]\n"
        f"Bankroll: ${initial_bankroll:,.0f} | Kelly: {kelly_fraction}\n"
        f"Min edge: {min_edge:.0%} | Min prob: {min_probability:.0%}\n"
        f"Max bet: {max_bet_pct:.0%} of bankroll\n"
        f"Tiers: {', '.join(signal_tiers)}\n"
        f"Sports: {', '.join(sports)}\n"
        f"Date range: {start_date or 'all'} to {end_date or 'all'}",
        title="Backtest Configuration"
    ))

    result = BacktestResult(initial_bankroll=initial_bankroll)
    bankroll = initial_bankroll
    peak_bankroll = bankroll
    max_drawdown = 0.0

    all_bets = []

    for sport in sports:
        for bet_type in bet_types:
            console.print(f"\n[cyan]Backtesting {sport}/{bet_type}...[/cyan]")

            # Load data
            val_result = load_validation_data(sport, bet_type, csv_dir)
            if val_result is None:
                console.print(f"  [yellow]No data for {sport}/{bet_type}[/yellow]")
                continue

            val_df, feature_columns, target_col = val_result

            # Date filtering
            date_col = None
            for dc in ['date', 'scheduled_at', 'game_date']:
                if dc in val_df.columns:
                    date_col = dc
                    break

            if date_col and start_date:
                val_df[date_col] = pd.to_datetime(val_df[date_col], errors='coerce')
                val_df = val_df[val_df[date_col] >= start_date]
            if date_col and end_date:
                val_df = val_df[val_df[date_col] <= end_date]

            if len(val_df) < 10:
                continue

            y_true = val_df[target_col].values

            # Load ensemble config for this sport/bet_type
            config_path = models_dir / sport / bet_type / "ensemble_config.json"
            ensemble_config = None
            if config_path.exists():
                with open(config_path) as f:
                    ensemble_config = json.load(f)

            # Discover available models for this combo
            all_models = discover_models(models_dir)
            combo_models = [m for m in all_models
                           if m["sport"] == sport
                           and m["bet_type"] == bet_type
                           and m["framework"] != "meta_ensemble"]

            if not combo_models:
                console.print(f"  [yellow]No models for {sport}/{bet_type}[/yellow]")
                continue

            # Get predictions from each framework
            fw_predictions = {}
            for model_info in combo_models:
                fw = model_info["framework"]
                try:
                    preds = load_and_predict(model_info, val_df, feature_columns)
                    if preds is not None:
                        # Apply calibration
                        cal_path = models_dir / fw / sport / bet_type / "calibrator.pkl"
                        if cal_path.exists():
                            with open(cal_path, 'rb') as f:
                                cal_data = pickle.load(f)
                            calibrator = cal_data["calibrator"]
                            cal_method = cal_data.get("method", "isotonic")
                            if cal_method == "isotonic":
                                preds = np.clip(calibrator.predict(preds), 0.001, 0.999)
                            elif cal_method == "platt":
                                preds = np.clip(calibrator.predict_proba(preds.reshape(-1, 1))[:, 1], 0.001, 0.999)

                        fw_predictions[fw] = preds
                except Exception as e:
                    logger.debug(f"  {fw} failed: {e}")

            if not fw_predictions:
                continue

            # Combine using ensemble weights
            if ensemble_config and ensemble_config.get("weights"):
                weights = ensemble_config["weights"]
                # Only use frameworks we have predictions for
                active_weights = {fw: w for fw, w in weights.items() if fw in fw_predictions}
                if active_weights:
                    total_w = sum(active_weights.values())
                    if total_w > 0:
                        combined_probs = sum(
                            (w / total_w) * fw_predictions[fw]
                            for fw, w in active_weights.items()
                        )
                    else:
                        combined_probs = np.mean(list(fw_predictions.values()), axis=0)
                else:
                    combined_probs = np.mean(list(fw_predictions.values()), axis=0)
                framework_used = "ensemble"
            else:
                # Equal weight all frameworks
                combined_probs = np.mean(list(fw_predictions.values()), axis=0)
                framework_used = "equal_ensemble"

            combined_probs = np.clip(combined_probs, 0.001, 0.999)

            # Simulate betting on each game
            standard_odds = -110  # Standard vig
            implied_prob = american_to_implied(standard_odds)
            decimal_odds = american_to_decimal(standard_odds)

            n_bets_this = 0
            for i in range(len(val_df)):
                if np.isnan(y_true[i]) or np.isnan(combined_probs[i]):
                    continue

                pred_prob = combined_probs[i]
                edge = pred_prob - implied_prob
                tier = classify_tier(edge, pred_prob)

                # Apply filters
                if edge < min_edge:
                    continue
                if pred_prob < min_probability:
                    continue
                if tier not in signal_tiers:
                    continue

                # Kelly sizing
                kelly_size = kelly_criterion(pred_prob, decimal_odds, kelly_fraction)
                if kelly_size <= 0:
                    continue

                # Cap at max bet percentage
                kelly_size = min(kelly_size, max_bet_pct)
                stake = bankroll * kelly_size

                if stake < 1.0:  # Minimum $1 bet
                    continue

                # Resolve bet
                won = int(y_true[i])
                if won:
                    profit = stake * (decimal_odds - 1.0)
                else:
                    profit = -stake

                bankroll += profit

                # Track drawdown
                if bankroll > peak_bankroll:
                    peak_bankroll = bankroll
                drawdown = (peak_bankroll - bankroll) / peak_bankroll
                if drawdown > max_drawdown:
                    max_drawdown = drawdown

                # Get date
                game_date = ""
                if date_col and date_col in val_df.columns:
                    d = val_df.iloc[i][date_col]
                    game_date = str(d)[:10] if pd.notna(d) else ""

                bet = Bet(
                    date=game_date, sport=sport, bet_type=bet_type,
                    framework=framework_used, predicted_prob=round(pred_prob, 4),
                    implied_prob=round(implied_prob, 4), edge=round(edge, 4),
                    signal_tier=tier, kelly_fraction=round(kelly_size, 4),
                    stake=round(stake, 2), odds=standard_odds,
                    result=won, profit_loss=round(profit, 2),
                    bankroll_after=round(bankroll, 2),
                )
                all_bets.append(bet)
                n_bets_this += 1

            if n_bets_this > 0:
                sport_wins = sum(1 for b in all_bets[-n_bets_this:] if b.result == 1)
                sport_pnl = sum(b.profit_loss for b in all_bets[-n_bets_this:])
                console.print(
                    f"  {n_bets_this} bets | {sport_wins}W-{n_bets_this-sport_wins}L "
                    f"({sport_wins/n_bets_this*100:.1f}%) | P&L: ${sport_pnl:+,.2f}"
                )

    # Compile results
    result.bets = all_bets
    result.final_bankroll = bankroll
    result.total_bets = len(all_bets)
    result.wins = sum(1 for b in all_bets if b.result == 1)
    result.losses = result.total_bets - result.wins
    result.win_rate = result.wins / result.total_bets if result.total_bets > 0 else 0
    result.total_profit = bankroll - initial_bankroll
    result.roi = result.total_profit / initial_bankroll if initial_bankroll > 0 else 0
    result.max_drawdown_pct = max_drawdown

    # Tier breakdown
    tier_a = [b for b in all_bets if b.signal_tier == "A"]
    tier_b = [b for b in all_bets if b.signal_tier == "B"]
    result.tier_a_bets = len(tier_a)
    result.tier_a_wins = sum(1 for b in tier_a if b.result == 1)
    result.tier_a_winrate = result.tier_a_wins / result.tier_a_bets if result.tier_a_bets > 0 else 0
    result.tier_b_bets = len(tier_b)
    result.tier_b_wins = sum(1 for b in tier_b if b.result == 1)
    result.tier_b_winrate = result.tier_b_wins / result.tier_b_bets if result.tier_b_bets > 0 else 0

    result.avg_edge = np.mean([b.edge for b in all_bets]) if all_bets else 0
    result.avg_kelly = np.mean([b.kelly_fraction for b in all_bets]) if all_bets else 0

    # Bankroll curve
    curve = [initial_bankroll]
    for b in all_bets:
        curve.append(b.bankroll_after)
    result.bankroll_curve = curve

    # Daily P&L
    daily = {}
    for b in all_bets:
        d = b.date[:10]
        daily[d] = daily.get(d, 0) + b.profit_loss
    result.daily_pnl = daily

    # Sharpe ratio (daily)
    if daily:
        daily_returns = list(daily.values())
        if len(daily_returns) > 1:
            mean_r = np.mean(daily_returns)
            std_r = np.std(daily_returns)
            result.sharpe_ratio = (mean_r / std_r * np.sqrt(252)) if std_r > 0 else 0

    return result


def print_backtest_results(result: BacktestResult):
    """Print comprehensive backtest report."""

    # Main metrics
    console.print(Panel(
        f"[bold]Bankroll:[/bold] ${result.initial_bankroll:,.0f} → ${result.final_bankroll:,.2f}\n"
        f"[bold]P&L:[/bold] ${result.total_profit:+,.2f}\n"
        f"[bold]ROI:[/bold] {result.roi*100:+.2f}%\n"
        f"[bold]Total Bets:[/bold] {result.total_bets} ({result.wins}W-{result.losses}L)\n"
        f"[bold]Win Rate:[/bold] {result.win_rate*100:.1f}%\n"
        f"[bold]Max Drawdown:[/bold] {result.max_drawdown_pct*100:.1f}%\n"
        f"[bold]Sharpe Ratio:[/bold] {result.sharpe_ratio:.2f}\n"
        f"[bold]Avg Edge:[/bold] {result.avg_edge*100:.1f}%\n"
        f"[bold]Avg Kelly:[/bold] {result.avg_kelly*100:.2f}%",
        title="[bold green]Backtest Results[/bold green]" if result.roi > 0 else "[bold red]Backtest Results[/bold red]",
    ))

    # Tier breakdown
    table = Table(title="Signal Tier Performance")
    table.add_column("Tier", style="bold")
    table.add_column("Bets", style="cyan")
    table.add_column("Wins", style="green")
    table.add_column("Win Rate", style="yellow")
    table.add_column("Target", style="dim")
    table.add_column("Status", style="white")

    for tier, bets, wins, wr, target in [
        ("A", result.tier_a_bets, result.tier_a_wins, result.tier_a_winrate, 0.60),
        ("B", result.tier_b_bets, result.tier_b_wins, result.tier_b_winrate, 0.57),
    ]:
        status = "[green]✓ PASS[/green]" if wr >= target else "[red]✗ FAIL[/red]"
        table.add_row(
            tier, str(bets), str(wins),
            f"{wr*100:.1f}%", f"{target*100:.0f}%", status,
        )

    console.print(table)

    # Sport breakdown
    sport_stats = {}
    for b in result.bets:
        key = b.sport
        if key not in sport_stats:
            sport_stats[key] = {"bets": 0, "wins": 0, "pnl": 0}
        sport_stats[key]["bets"] += 1
        sport_stats[key]["wins"] += b.result
        sport_stats[key]["pnl"] += b.profit_loss

    if sport_stats:
        sport_table = Table(title="Performance by Sport")
        sport_table.add_column("Sport", style="cyan")
        sport_table.add_column("Bets", style="white")
        sport_table.add_column("Win Rate", style="yellow")
        sport_table.add_column("P&L", style="green")

        for sport, stats in sorted(sport_stats.items(), key=lambda x: -x[1]["pnl"]):
            wr = stats["wins"] / stats["bets"] if stats["bets"] > 0 else 0
            pnl_str = f"${stats['pnl']:+,.2f}"
            if stats["pnl"] > 0:
                pnl_str = f"[green]{pnl_str}[/green]"
            else:
                pnl_str = f"[red]{pnl_str}[/red]"

            sport_table.add_row(sport, str(stats["bets"]), f"{wr*100:.1f}%", pnl_str)

        console.print(sport_table)

    # Go/No-Go assessment
    checks = [
        ("ROI ≥ 2%", result.roi >= 0.02),
        ("Tier A Win Rate ≥ 58%", result.tier_a_winrate >= 0.58 or result.tier_a_bets == 0),
        ("Max Drawdown < 20%", result.max_drawdown_pct < 0.20),
        (f"Total Bets ≥ 50", result.total_bets >= 50),
    ]

    all_pass = all(c[1] for c in checks)
    verdict = "[bold green]✅ GO — Ready for paper trading[/bold green]" if all_pass else "[bold red]❌ NO-GO — Models need improvement[/bold red]"

    console.print(f"\n[bold]Go/No-Go Assessment:[/bold]")
    for label, passed in checks:
        status = "[green]✓[/green]" if passed else "[red]✗[/red]"
        console.print(f"  {status} {label}")
    console.print(f"\n  {verdict}")


def save_backtest(result: BacktestResult, output_dir: str = "."):
    """Save backtest results to files."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Bets CSV
    bets_data = [{
        "date": b.date, "sport": b.sport, "bet_type": b.bet_type,
        "framework": b.framework, "predicted_prob": b.predicted_prob,
        "implied_prob": b.implied_prob, "edge": b.edge,
        "signal_tier": b.signal_tier, "kelly_fraction": b.kelly_fraction,
        "stake": b.stake, "odds": b.odds, "result": b.result,
        "profit_loss": b.profit_loss, "bankroll_after": b.bankroll_after,
    } for b in result.bets]
    pd.DataFrame(bets_data).to_csv(out / "backtest_bets.csv", index=False)

    # Summary JSON
    summary = {
        "initial_bankroll": result.initial_bankroll,
        "final_bankroll": result.final_bankroll,
        "total_bets": result.total_bets,
        "wins": result.wins, "losses": result.losses,
        "win_rate": round(result.win_rate, 4),
        "roi": round(result.roi, 4),
        "total_profit": round(result.total_profit, 2),
        "max_drawdown_pct": round(result.max_drawdown_pct, 4),
        "sharpe_ratio": round(result.sharpe_ratio, 2),
        "tier_a_bets": result.tier_a_bets, "tier_a_winrate": round(result.tier_a_winrate, 4),
        "tier_b_bets": result.tier_b_bets, "tier_b_winrate": round(result.tier_b_winrate, 4),
        "avg_edge": round(result.avg_edge, 4),
        "run_date": datetime.utcnow().isoformat(),
    }
    with open(out / "backtest_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    # Bankroll curve CSV
    pd.DataFrame({"step": range(len(result.bankroll_curve)),
                   "bankroll": result.bankroll_curve}).to_csv(
        out / "bankroll_curve.csv", index=False)

    console.print(f"\n[green]Results saved to {out}/[/green]")


def main():
    parser = argparse.ArgumentParser(description="ROYALEY Phase 5: Walk-Forward Backtesting")
    parser.add_argument("--models-dir", "-m", type=str, default="/app/models")
    parser.add_argument("--csv-dir", type=str, default=None)
    parser.add_argument("--sports", type=str, default=None,
                        help="Comma-separated sports (e.g., NFL,NBA)")
    parser.add_argument("--bet-types", type=str, default=None)
    parser.add_argument("--bankroll", type=float, default=10000.0)
    parser.add_argument("--kelly", type=float, default=0.25)
    parser.add_argument("--min-edge", type=float, default=0.03)
    parser.add_argument("--min-prob", type=float, default=0.55)
    parser.add_argument("--max-bet-pct", type=float, default=0.02)
    parser.add_argument("--tiers", type=str, default="A,B")
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="backtest_results")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    sports = args.sports.split(",") if args.sports else SPORTS
    bet_types = args.bet_types.split(",") if args.bet_types else BET_TYPES
    tiers = args.tiers.split(",")

    result = run_backtest(
        models_dir=Path(args.models_dir),
        csv_dir=args.csv_dir,
        sports=sports,
        bet_types=bet_types,
        initial_bankroll=args.bankroll,
        kelly_fraction=args.kelly,
        min_edge=args.min_edge,
        min_probability=args.min_prob,
        max_bet_pct=args.max_bet_pct,
        signal_tiers=tiers,
        start_date=args.start_date,
        end_date=args.end_date,
        verbose=args.verbose,
    )

    print_backtest_results(result)
    save_backtest(result, args.output_dir)


if __name__ == "__main__":
    main()
