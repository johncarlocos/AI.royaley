#!/usr/bin/env python3
"""
ROYALEY - Phase 7b: Model Drift Monitor
Detects performance degradation and alerts when models need retraining.

Usage:
    python scripts/drift_monitor.py
    python scripts/drift_monitor.py --log paper_trading/scored_log.csv --window 7
    python scripts/drift_monitor.py --detailed
"""

import argparse
import logging
import sys
import json
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ============================================================================
# DRIFT DETECTION THRESHOLDS
# ============================================================================

# NO-GO triggers
MAX_CONSECUTIVE_LOSING_DAYS = 7
MAX_ACCURACY_DROP_VS_BACKTEST = 0.05  # 5% drop
MIN_LIVE_ACCURACY = 0.50  # Below coin flip = broken
MIN_CLV_ROLLING = -0.03   # Average CLV below -3% = edge gone

# Warning triggers
WARN_ACCURACY_DROP = 0.03  # 3% drop from backtest
WARN_ROI_THRESHOLD = -0.05  # -5% ROI
WARN_TIER_A_MIN_WINRATE = 0.55

# Backtest benchmarks (loaded from backtest_summary.json if available)
DEFAULT_BACKTEST_BENCHMARKS = {
    "accuracy": 0.55,
    "tier_a_winrate": 0.60,
    "roi": 0.05,
}


def load_scored_log(log_path: str) -> Optional[pd.DataFrame]:
    """Load the running scored predictions log."""
    path = Path(log_path)
    if not path.exists():
        console.print(f"[yellow]Scored log not found: {log_path}[/yellow]")
        return None

    df = pd.read_csv(path)
    if len(df) == 0:
        console.print("[yellow]Scored log is empty[/yellow]")
        return None

    # Parse dates
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

    return df


def load_backtest_benchmarks(path: str = "backtest_results/backtest_summary.json") -> Dict:
    """Load backtest benchmarks for comparison."""
    p = Path(path)
    if p.exists():
        with open(p) as f:
            data = json.load(f)
        return {
            "accuracy": data.get("win_rate", 0.55),
            "tier_a_winrate": data.get("tier_a_winrate", 0.60),
            "roi": data.get("roi", 0.05),
        }
    return DEFAULT_BACKTEST_BENCHMARKS


def analyze_drift(df: pd.DataFrame, benchmarks: Dict, window: int = 7) -> Dict:
    """Analyze model drift across multiple dimensions."""
    alerts = []
    warnings = []
    metrics = {}

    # Overall stats
    total_bets = len(df)
    if total_bets == 0:
        return {"alerts": ["No bets to analyze"], "warnings": [], "metrics": {}}

    wins = df['won'].sum() if 'won' in df.columns else 0
    overall_accuracy = wins / total_bets
    total_pnl = df['profit_loss'].sum() if 'profit_loss' in df.columns else 0
    total_staked = df['profit_loss'].abs().sum()  # Approximate
    overall_roi = total_pnl / max(total_staked, 1)

    metrics["total_bets"] = total_bets
    metrics["overall_accuracy"] = round(overall_accuracy, 4)
    metrics["overall_pnl"] = round(total_pnl, 2)
    metrics["overall_roi"] = round(overall_roi, 4)

    # ── CHECK 1: Rolling accuracy ──
    if 'date' in df.columns and df['date'].notna().any():
        df_sorted = df.sort_values('date')
        recent = df_sorted.tail(window * 5)  # Approximate last N days of bets

        if len(recent) >= 10:
            recent_accuracy = recent['won'].mean()
            metrics["recent_accuracy"] = round(recent_accuracy, 4)

            backtest_acc = benchmarks.get("accuracy", 0.55)
            acc_drop = backtest_acc - recent_accuracy

            if recent_accuracy < MIN_LIVE_ACCURACY:
                alerts.append(
                    f"🚨 CRITICAL: Recent accuracy {recent_accuracy:.1%} is BELOW coin flip! "
                    f"Models may be broken."
                )
            elif acc_drop >= MAX_ACCURACY_DROP_VS_BACKTEST:
                alerts.append(
                    f"🚨 Accuracy dropped {acc_drop:.1%} vs backtest "
                    f"(live: {recent_accuracy:.1%}, backtest: {backtest_acc:.1%}). "
                    f"Exceeds {MAX_ACCURACY_DROP_VS_BACKTEST:.0%} threshold."
                )
            elif acc_drop >= WARN_ACCURACY_DROP:
                warnings.append(
                    f"⚠️ Accuracy down {acc_drop:.1%} vs backtest "
                    f"(live: {recent_accuracy:.1%}, backtest: {backtest_acc:.1%})"
                )

    # ── CHECK 2: Consecutive losing days ──
    if 'date' in df.columns and 'profit_loss' in df.columns:
        daily_pnl = df.groupby(df['date'].dt.date)['profit_loss'].sum()
        if len(daily_pnl) > 0:
            consecutive_losses = 0
            max_consecutive = 0
            for d_pnl in daily_pnl.values:
                if d_pnl < 0:
                    consecutive_losses += 1
                    max_consecutive = max(max_consecutive, consecutive_losses)
                else:
                    consecutive_losses = 0

            metrics["max_consecutive_losing_days"] = max_consecutive
            metrics["current_streak"] = consecutive_losses

            if max_consecutive >= MAX_CONSECUTIVE_LOSING_DAYS:
                alerts.append(
                    f"🚨 NO-GO: {max_consecutive} consecutive losing days! "
                    f"Threshold: {MAX_CONSECUTIVE_LOSING_DAYS}"
                )
            elif max_consecutive >= MAX_CONSECUTIVE_LOSING_DAYS - 2:
                warnings.append(
                    f"⚠️ {max_consecutive} consecutive losing days approaching limit "
                    f"({MAX_CONSECUTIVE_LOSING_DAYS})"
                )

    # ── CHECK 3: CLV (Closing Line Value) ──
    if 'clv' in df.columns:
        avg_clv = df['clv'].mean()
        metrics["avg_clv"] = round(avg_clv, 4)

        if avg_clv < MIN_CLV_ROLLING:
            alerts.append(
                f"🚨 Average CLV is {avg_clv:.1%} (below {MIN_CLV_ROLLING:.1%}). "
                f"Edge has likely disappeared."
            )
        elif avg_clv < 0:
            warnings.append(
                f"⚠️ Negative average CLV ({avg_clv:.1%}). Monitor closely."
            )

    # ── CHECK 4: Tier A performance ──
    if 'tier' in df.columns:
        tier_a = df[df['tier'] == 'A']
        if len(tier_a) >= 5:
            tier_a_wr = tier_a['won'].mean()
            metrics["tier_a_winrate"] = round(tier_a_wr, 4)
            metrics["tier_a_bets"] = len(tier_a)

            if tier_a_wr < WARN_TIER_A_MIN_WINRATE:
                warnings.append(
                    f"⚠️ Tier A win rate {tier_a_wr:.1%} below {WARN_TIER_A_MIN_WINRATE:.0%} "
                    f"({len(tier_a)} bets)"
                )

    # ── CHECK 5: Sport-level breakdown ──
    if 'sport' in df.columns:
        sport_stats = {}
        for sport, group in df.groupby('sport'):
            if len(group) >= 5:
                wr = group['won'].mean()
                pnl = group['profit_loss'].sum()
                sport_stats[sport] = {"accuracy": round(wr, 4), "pnl": round(pnl, 2), "bets": len(group)}

                if wr < 0.45:
                    warnings.append(
                        f"⚠️ {sport} accuracy very low: {wr:.1%} ({len(group)} bets)"
                    )

        metrics["sport_breakdown"] = sport_stats

    # ── CHECK 6: Bet-type breakdown ──
    if 'bet_type' in df.columns:
        bt_stats = {}
        for bt, group in df.groupby('bet_type'):
            if len(group) >= 5:
                wr = group['won'].mean()
                pnl = group['profit_loss'].sum()
                bt_stats[bt] = {"accuracy": round(wr, 4), "pnl": round(pnl, 2), "bets": len(group)}

        metrics["bet_type_breakdown"] = bt_stats

    # ── CHECK 7: Weekly trend ──
    if 'date' in df.columns and df['date'].notna().any():
        df['week'] = df['date'].dt.isocalendar().week.astype(int)
        weekly = df.groupby('week').agg(
            accuracy=('won', 'mean'),
            pnl=('profit_loss', 'sum'),
            bets=('won', 'count'),
        )
        if len(weekly) >= 2:
            trend = weekly['accuracy'].diff().iloc[-1] if len(weekly) > 1 else 0
            metrics["weekly_accuracy_trend"] = round(trend, 4)
            if trend < -0.05:
                warnings.append(f"⚠️ Week-over-week accuracy dropped by {abs(trend):.1%}")

    return {
        "alerts": alerts,
        "warnings": warnings,
        "metrics": metrics,
    }


def print_drift_report(analysis: Dict, benchmarks: Dict):
    """Print comprehensive drift monitoring report."""

    metrics = analysis["metrics"]
    alerts = analysis["alerts"]
    warnings = analysis["warnings"]

    # Status determination
    if alerts:
        status = "[bold red]🚨 ALERTS DETECTED — ACTION REQUIRED[/bold red]"
    elif warnings:
        status = "[bold yellow]⚠️ WARNINGS — MONITOR CLOSELY[/bold yellow]"
    else:
        status = "[bold green]✅ ALL CLEAR — Models performing within bounds[/bold green]"

    console.print(Panel(status, title="Drift Monitor Status"))

    # Alerts
    if alerts:
        console.print("\n[red bold]ALERTS:[/red bold]")
        for a in alerts:
            console.print(f"  {a}")

    if warnings:
        console.print("\n[yellow bold]WARNINGS:[/yellow bold]")
        for w in warnings:
            console.print(f"  {w}")

    # Core metrics table
    table = Table(title="Performance Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Live", style="white")
    table.add_column("Backtest", style="dim")
    table.add_column("Status", style="bold")

    def status_cell(live_val, bench_val, higher_better=True, threshold=0.03):
        if live_val is None:
            return "[dim]N/A[/dim]"
        diff = live_val - bench_val if higher_better else bench_val - live_val
        if diff >= 0:
            return "[green]✓[/green]"
        elif abs(diff) < threshold:
            return "[yellow]~[/yellow]"
        else:
            return "[red]✗[/red]"

    rows = [
        ("Total Bets", str(metrics.get("total_bets", 0)), "-", ""),
        ("Overall Accuracy", f"{metrics.get('overall_accuracy', 0):.1%}",
         f"{benchmarks.get('accuracy', 0):.1%}",
         status_cell(metrics.get('overall_accuracy'), benchmarks.get('accuracy', 0.55))),
        ("Recent Accuracy", f"{metrics.get('recent_accuracy', 0):.1%}",
         f"{benchmarks.get('accuracy', 0):.1%}",
         status_cell(metrics.get('recent_accuracy'), benchmarks.get('accuracy', 0.55))),
        ("Overall P&L", f"${metrics.get('overall_pnl', 0):+,.2f}", "-", ""),
        ("Overall ROI", f"{metrics.get('overall_roi', 0):.1%}",
         f"{benchmarks.get('roi', 0):.1%}",
         status_cell(metrics.get('overall_roi'), benchmarks.get('roi', 0.05))),
        ("Avg CLV", f"{metrics.get('avg_clv', 0):.2%}", ">0%",
         status_cell(metrics.get('avg_clv', 0), 0)),
        ("Tier A Win Rate", f"{metrics.get('tier_a_winrate', 0):.1%}",
         f"{benchmarks.get('tier_a_winrate', 0.60):.0%}",
         status_cell(metrics.get('tier_a_winrate'), benchmarks.get('tier_a_winrate', 0.60))),
        ("Max Losing Streak", f"{metrics.get('max_consecutive_losing_days', 0)} days",
         f"<{MAX_CONSECUTIVE_LOSING_DAYS}", ""),
    ]

    for r in rows:
        table.add_row(*r)

    console.print(table)

    # Sport breakdown
    sport_stats = metrics.get("sport_breakdown", {})
    if sport_stats:
        sport_table = Table(title="By Sport")
        sport_table.add_column("Sport", style="cyan")
        sport_table.add_column("Bets", style="white")
        sport_table.add_column("Accuracy", style="yellow")
        sport_table.add_column("P&L", style="green")

        for sport, stats in sorted(sport_stats.items(), key=lambda x: -x[1]["pnl"]):
            pnl_str = f"${stats['pnl']:+,.2f}"
            if stats['pnl'] >= 0:
                pnl_str = f"[green]{pnl_str}[/green]"
            else:
                pnl_str = f"[red]{pnl_str}[/red]"
            sport_table.add_row(
                sport, str(stats["bets"]),
                f"{stats['accuracy']:.1%}", pnl_str,
            )
        console.print(sport_table)

    # Recommendation
    if alerts:
        console.print(Panel(
            "[red]RECOMMENDED ACTIONS:[/red]\n"
            "1. PAUSE live betting immediately\n"
            "2. Review recent predictions for systematic errors\n"
            "3. Check if market conditions have changed\n"
            "4. Consider retraining models with recent data\n"
            "5. Run Phase 2 evaluation on latest data to confirm drift",
            title="Action Items"
        ))
    elif warnings:
        console.print(Panel(
            "[yellow]RECOMMENDED ACTIONS:[/yellow]\n"
            "1. Continue paper trading but increase monitoring frequency\n"
            "2. Review underperforming sports/bet_types\n"
            "3. Compare recent feature distributions vs training data\n"
            "4. Schedule model refresh within 1-2 weeks",
            title="Action Items"
        ))


def main():
    parser = argparse.ArgumentParser(description="ROYALEY Phase 7b: Drift Monitor")
    parser.add_argument("--log", "-l", type=str, default="paper_trading/scored_log.csv",
                        help="Path to scored predictions log")
    parser.add_argument("--backtest", type=str, default="backtest_results/backtest_summary.json",
                        help="Path to backtest summary for benchmarks")
    parser.add_argument("--window", "-w", type=int, default=7,
                        help="Rolling window size in days")
    parser.add_argument("--detailed", action="store_true")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Save drift report as JSON")

    args = parser.parse_args()

    # Load data
    df = load_scored_log(args.log)
    if df is None:
        return

    benchmarks = load_backtest_benchmarks(args.backtest)

    # Analyze
    analysis = analyze_drift(df, benchmarks, window=args.window)

    # Print
    print_drift_report(analysis, benchmarks)

    # Save
    if args.output:
        # Make serializable
        report = {
            "run_at": datetime.utcnow().isoformat(),
            "alerts": analysis["alerts"],
            "warnings": analysis["warnings"],
            "metrics": {k: v for k, v in analysis["metrics"].items()
                       if not isinstance(v, (pd.Series, pd.DataFrame))},
        }
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        console.print(f"\n[green]Drift report saved to {args.output}[/green]")


if __name__ == "__main__":
    main()
