#!/usr/bin/env python3
"""
ROYALEY - Master Pipeline Runner
Runs Phases 2-7 in sequence with go/no-go gates.

Usage:
    python scripts/run_pipeline.py --all                         # Full pipeline
    python scripts/run_pipeline.py --phase 2                     # Single phase
    python scripts/run_pipeline.py --from-phase 3 --to-phase 5   # Phase range
    python scripts/run_pipeline.py --all --sport NFL              # Filter by sport
    python scripts/run_pipeline.py --daily                        # Daily prediction workflow
"""

import asyncio
import argparse
import logging
import sys
import subprocess
import json
from datetime import datetime, date
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

MODELS_DIR = "/app/models"
SCRIPTS_DIR = Path(__file__).parent


def run_command(cmd: List[str], description: str, check: bool = True) -> int:
    """Run a command and display status."""
    console.print(f"\n[cyan]Running: {description}[/cyan]")
    console.print(f"[dim]$ {' '.join(cmd)}[/dim]")

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode == 0:
        console.print(f"[green]✓ {description} completed successfully[/green]")
    else:
        console.print(f"[red]✗ {description} failed (exit code: {result.returncode})[/red]")
        if check:
            raise RuntimeError(f"Pipeline failed at: {description}")

    return result.returncode


def phase_gate(phase: int, condition: bool, message: str):
    """Go/No-Go gate between phases."""
    if condition:
        console.print(Panel(
            f"[green]✅ Phase {phase} GATE PASSED: {message}[/green]",
            title=f"Phase {phase} → Phase {phase + 1}"
        ))
    else:
        console.print(Panel(
            f"[red]❌ Phase {phase} GATE FAILED: {message}[/red]\n"
            f"Pipeline halted. Fix issues before proceeding.",
            title=f"Phase {phase} BLOCKED"
        ))
        raise RuntimeError(f"Phase {phase} gate failed: {message}")


async def run_phase_2(args) -> bool:
    """Phase 2: Model Evaluation."""
    console.print(Panel("[bold]PHASE 2: MODEL EVALUATION[/bold]", style="cyan"))

    cmd = [
        sys.executable, str(SCRIPTS_DIR / "evaluate_models.py"),
        "--models-dir", args.models_dir,
        "--output", "model_scorecard.csv",
        "--verbose",
    ]
    if args.sport:
        cmd.extend(["--sport", args.sport])
    if args.csv_dir:
        cmd.extend(["--csv-dir", args.csv_dir])

    rc = run_command(cmd, "Model Evaluation & Scorecard")

    # Verify scorecard exists and has passing models
    scorecard = Path("model_scorecard.csv")
    if scorecard.exists():
        import pandas as pd
        df = pd.read_csv(scorecard)
        loaded = df[df["load_success"] == True]
        passed = loaded[loaded["passes_threshold"] == True]
        console.print(f"  Models loaded: {len(loaded)}, Passed threshold: {len(passed)}")

        phase_gate(2, len(passed) > 0, f"{len(passed)} models passed evaluation thresholds")
        return True

    phase_gate(2, False, "Scorecard not generated")
    return False


async def run_phase_3(args) -> bool:
    """Phase 3: Probability Calibration."""
    console.print(Panel("[bold]PHASE 3: PROBABILITY CALIBRATION[/bold]", style="cyan"))

    cmd = [
        sys.executable, str(SCRIPTS_DIR / "calibrate_models.py"),
        "--models-dir", args.models_dir,
        "--scorecard", "model_scorecard.csv",
        "--method", "isotonic",
        "--verbose",
    ]
    if args.sport:
        cmd.extend(["--sport", args.sport])
    if args.csv_dir:
        cmd.extend(["--csv-dir", args.csv_dir])

    rc = run_command(cmd, "Probability Calibration")

    # Verify calibration report
    report = Path("calibration_report.csv")
    if report.exists():
        import pandas as pd
        df = pd.read_csv(report)
        improved = df[df["ece_after"] < df["ece_before"]]
        console.print(f"  Models calibrated: {len(df)}, Improved: {len(improved)}")
        phase_gate(3, len(df) > 0, f"{len(df)} models calibrated")
        return True

    # Calibration is optional - continue even if it doesn't improve things
    console.print("[yellow]Calibration report not found, continuing anyway[/yellow]")
    return True


async def run_phase_4(args) -> bool:
    """Phase 4: Ensemble Optimization."""
    console.print(Panel("[bold]PHASE 4: ENSEMBLE OPTIMIZATION[/bold]", style="cyan"))

    cmd = [
        sys.executable, str(SCRIPTS_DIR / "optimize_ensemble.py"),
        "--models-dir", args.models_dir,
        "--method", "scipy",
        "--verbose",
    ]
    if args.sport:
        cmd.extend(["--sport", args.sport])
    if args.csv_dir:
        cmd.extend(["--csv-dir", args.csv_dir])

    rc = run_command(cmd, "Ensemble Weight Optimization")

    report = Path("ensemble_optimization_report.csv")
    if report.exists():
        import pandas as pd
        df = pd.read_csv(report)
        console.print(f"  Combinations optimized: {len(df)}")
        phase_gate(4, len(df) > 0, f"{len(df)} sport/bet combos optimized")
        return True

    console.print("[yellow]Ensemble optimization report not found, continuing[/yellow]")
    return True


async def run_phase_5(args) -> bool:
    """Phase 5: Walk-Forward Backtesting."""
    console.print(Panel("[bold]PHASE 5: WALK-FORWARD BACKTESTING[/bold]", style="cyan"))

    cmd = [
        sys.executable, str(SCRIPTS_DIR / "run_backtest.py"),
        "--models-dir", args.models_dir,
        "--kelly", str(args.kelly),
        "--min-edge", str(args.min_edge),
        "--tiers", "A,B",
        "--output-dir", "backtest_results",
        "--verbose",
    ]
    if args.sport:
        cmd.extend(["--sports", args.sport])
    if args.csv_dir:
        cmd.extend(["--csv-dir", args.csv_dir])

    rc = run_command(cmd, "Walk-Forward Backtesting")

    # Check results
    summary_path = Path("backtest_results/backtest_summary.json")
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)

        roi = summary.get("roi", 0)
        win_rate = summary.get("win_rate", 0)
        max_dd = summary.get("max_drawdown_pct", 1.0)
        total_bets = summary.get("total_bets", 0)

        console.print(f"  ROI: {roi*100:+.2f}% | Win Rate: {win_rate*100:.1f}% | "
                      f"Max DD: {max_dd*100:.1f}% | Bets: {total_bets}")

        # Go/No-Go checks
        checks = [
            roi >= 0.02,
            max_dd < 0.20,
            total_bets >= 20,
        ]
        all_pass = all(checks)

        if all_pass:
            phase_gate(5, True, f"ROI={roi*100:+.1f}%, DD={max_dd*100:.1f}%, {total_bets} bets")
        else:
            issues = []
            if roi < 0.02:
                issues.append(f"ROI {roi*100:+.1f}% < 2%")
            if max_dd >= 0.20:
                issues.append(f"Drawdown {max_dd*100:.1f}% ≥ 20%")
            if total_bets < 20:
                issues.append(f"Only {total_bets} bets (need ≥20)")
            phase_gate(5, False, "; ".join(issues))

        return all_pass

    console.print("[yellow]Backtest summary not found[/yellow]")
    return True  # Don't block if no data


async def run_phase_6(args) -> bool:
    """Phase 6: Promote Models & Wire Prediction Engine."""
    console.print(Panel("[bold]PHASE 6: PROMOTE MODELS[/bold]", style="cyan"))

    cmd = [
        sys.executable, str(SCRIPTS_DIR / "promote_models.py"),
        "--scorecard", "model_scorecard.csv",
        "--min-accuracy", "0.53",
        "--manifest-only",  # Just generate manifest, skip DB for now
    ]

    rc = run_command(cmd, "Promote Production Models")

    manifest = Path("production_manifest.json")
    if manifest.exists():
        with open(manifest) as f:
            data = json.load(f)
        n_models = len(data.get("models", {}))
        console.print(f"  Production manifest: {n_models} models")
        phase_gate(6, n_models > 0, f"{n_models} models in production manifest")
        return True

    phase_gate(6, False, "Production manifest not generated")
    return False


async def run_daily_workflow(args):
    """Daily prediction workflow (Phase 7 operational)."""
    console.print(Panel(
        "[bold green]ROYALEY Daily Prediction Workflow[/bold green]\n"
        f"Date: {date.today()}\n"
        f"Models: {args.models_dir}",
        title="Daily Run"
    ))

    today = date.today().isoformat()
    paper_dir = Path("paper_trading")
    paper_dir.mkdir(exist_ok=True)

    # Step 1: Generate predictions
    console.print("\n[cyan]Step 1: Generate Predictions[/cyan]")
    pred_file = paper_dir / f"predictions_{today}.json"

    cmd = [
        sys.executable, str(SCRIPTS_DIR / "predict.py"),
        "--all",
        "--min-edge", str(args.min_edge),
        "--tier", "A",
        "--kelly", str(args.kelly),
        "--models-dir", args.models_dir,
        "--output", str(pred_file),
    ]
    run_command(cmd, "Generate predictions", check=False)

    # Step 2: Score yesterday's predictions
    console.print("\n[cyan]Step 2: Score Yesterday's Predictions[/cyan]")
    cmd = [
        sys.executable, str(SCRIPTS_DIR / "score_predictions.py"),
        "--date", "yesterday",
        "--predictions-dir", str(paper_dir),
    ]
    run_command(cmd, "Score yesterday's predictions", check=False)

    # Step 3: Drift monitoring
    console.print("\n[cyan]Step 3: Drift Check[/cyan]")
    cmd = [
        sys.executable, str(SCRIPTS_DIR / "drift_monitor.py"),
        "--log", str(paper_dir / "scored_log.csv"),
    ]
    run_command(cmd, "Drift monitoring", check=False)

    console.print(Panel(
        f"[green]Daily workflow complete![/green]\n"
        f"Predictions: {pred_file}\n"
        f"Run this daily for 2-4 weeks before live deployment.",
        title="Done"
    ))


async def run_full_pipeline(args):
    """Run complete pipeline Phases 2-6."""
    console.print(Panel(
        "[bold cyan]ROYALEY FULL PIPELINE: Phases 2 → 6[/bold cyan]\n"
        f"Models: {args.models_dir}\n"
        f"Sport: {args.sport or 'ALL'}\n"
        f"Kelly: {args.kelly} | Min Edge: {args.min_edge}",
        title="Pipeline Configuration",
        style="bold cyan"
    ))

    start_time = datetime.now()
    phases = {
        2: ("Model Evaluation", run_phase_2),
        3: ("Probability Calibration", run_phase_3),
        4: ("Ensemble Optimization", run_phase_4),
        5: ("Walk-Forward Backtesting", run_phase_5),
        6: ("Promote Models", run_phase_6),
    }

    from_phase = args.from_phase or 2
    to_phase = args.to_phase or 6

    completed = []
    for phase_num in range(from_phase, to_phase + 1):
        if phase_num not in phases:
            continue

        name, func = phases[phase_num]
        try:
            success = await func(args)
            completed.append((phase_num, name, success))
        except RuntimeError as e:
            completed.append((phase_num, name, False))
            console.print(f"\n[red]Pipeline halted at Phase {phase_num}: {e}[/red]")
            break

    # Summary
    elapsed = (datetime.now() - start_time).total_seconds()
    console.print(Panel(
        "\n".join(
            f"  {'✅' if s else '❌'} Phase {n}: {name}"
            for n, name, s in completed
        ) + f"\n\n  Elapsed: {elapsed/60:.1f} minutes",
        title="Pipeline Summary"
    ))


async def main():
    parser = argparse.ArgumentParser(
        description="ROYALEY Master Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Pipeline mode
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--all", action="store_true", help="Run full pipeline (Phase 2-6)")
    mode.add_argument("--phase", type=int, choices=[2, 3, 4, 5, 6],
                      help="Run a single phase")
    mode.add_argument("--daily", action="store_true",
                      help="Run daily prediction workflow (Phase 7)")

    # Phase range
    parser.add_argument("--from-phase", type=int, default=None)
    parser.add_argument("--to-phase", type=int, default=None)

    # Common options
    parser.add_argument("--models-dir", "-m", type=str, default=MODELS_DIR)
    parser.add_argument("--csv-dir", type=str, default=None)
    parser.add_argument("--sport", "-s", type=str, default=None)
    parser.add_argument("--kelly", type=float, default=0.25)
    parser.add_argument("--min-edge", type=float, default=0.03)
    parser.add_argument("--bankroll", type=float, default=10000.0)

    args = parser.parse_args()

    if args.daily:
        await run_daily_workflow(args)
    elif args.phase:
        args.from_phase = args.phase
        args.to_phase = args.phase
        await run_full_pipeline(args)
    elif args.all or args.from_phase or args.to_phase:
        await run_full_pipeline(args)
    else:
        parser.print_help()
        console.print("\n[yellow]Use --all for full pipeline, --daily for daily workflow[/yellow]")


if __name__ == "__main__":
    asyncio.run(main())
