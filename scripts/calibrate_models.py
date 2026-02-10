#!/usr/bin/env python3
"""
ROYALEY - Phase 3: Probability Calibration
Applies calibration to surviving models from Phase 2 scorecard.

Usage:
    python scripts/calibrate_models.py --models-dir /nvme0n1-disk/royaley/models --scorecard model_scorecard.csv
    python scripts/calibrate_models.py --models-dir /nvme0n1-disk/royaley/models --method isotonic
    python scripts/calibrate_models.py --models-dir /nvme0n1-disk/royaley/models --sport NFL
"""

import asyncio
import argparse
import logging
import sys
import pickle
import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

# Reuse evaluation utilities
from scripts.evaluate_models import (
    discover_models, load_and_predict, load_validation_data,
    _compute_ece, SPORTS, BET_TYPES, TARGET_COLUMNS,
)

console = Console()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def calibrate_isotonic(y_true: np.ndarray, y_proba: np.ndarray) -> object:
    """Fit isotonic regression calibrator."""
    from sklearn.isotonic import IsotonicRegression
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(y_proba, y_true)
    return calibrator


def calibrate_platt(y_true: np.ndarray, y_proba: np.ndarray) -> object:
    """Fit Platt scaling (logistic regression on raw probs)."""
    from sklearn.linear_model import LogisticRegression
    calibrator = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
    calibrator.fit(y_proba.reshape(-1, 1), y_true)
    return calibrator


def calibrate_temperature(y_true: np.ndarray, y_proba: np.ndarray) -> Dict:
    """Fit temperature scaling — single parameter optimization."""
    from scipy.optimize import minimize_scalar

    def neg_log_likelihood(temperature):
        scaled = np.clip(y_proba ** (1.0 / temperature), 0.001, 0.999)
        nll = -(y_true * np.log(scaled) + (1 - y_true) * np.log(1 - scaled)).mean()
        return nll

    result = minimize_scalar(neg_log_likelihood, bounds=(0.1, 10.0), method='bounded')
    return {"temperature": result.x, "method": "temperature_scaling"}


def apply_calibration(calibrator: object, y_proba: np.ndarray, method: str) -> np.ndarray:
    """Apply a fitted calibrator to raw probabilities."""
    if method == "isotonic":
        calibrated = calibrator.predict(y_proba)
    elif method == "platt":
        calibrated = calibrator.predict_proba(y_proba.reshape(-1, 1))[:, 1]
    elif method == "temperature":
        temp = calibrator["temperature"]
        calibrated = y_proba ** (1.0 / temp)
    else:
        calibrated = y_proba

    return np.clip(calibrated, 0.001, 0.999)


def generate_reliability_data(y_true: np.ndarray, y_proba: np.ndarray,
                               n_bins: int = 10) -> List[Dict]:
    """Generate reliability diagram data points."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bins = []
    for i in range(n_bins):
        mask = (y_proba >= bin_boundaries[i]) & (y_proba < bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        bins.append({
            "bin_center": (bin_boundaries[i] + bin_boundaries[i + 1]) / 2,
            "avg_predicted": float(y_proba[mask].mean()),
            "avg_actual": float(y_true[mask].mean()),
            "count": int(mask.sum()),
        })
    return bins


async def calibrate_all_models(
    models_dir: Path,
    scorecard_path: str = None,
    csv_dir: str = None,
    method: str = "isotonic",
    sport_filter: str = None,
    min_composite: float = 0.0,
    verbose: bool = False,
) -> Dict[str, Dict]:
    """Run calibration pipeline."""

    console.print(Panel(
        "[bold cyan]ROYALEY Phase 3: Probability Calibration[/bold cyan]\n"
        f"Method: {method}\n"
        f"Models: {models_dir}\n"
        f"Scorecard: {scorecard_path or 'None (calibrating all)'}",
        title="Configuration"
    ))

    # Load scorecard to filter models
    passing_models = None
    if scorecard_path and Path(scorecard_path).exists():
        sc_df = pd.read_csv(scorecard_path)
        sc_df = sc_df[sc_df["load_success"] == True]
        if min_composite > 0:
            sc_df = sc_df[sc_df["composite_score"] >= min_composite]
        if sport_filter:
            sc_df = sc_df[sc_df["sport"] == sport_filter.upper()]
        passing_models = set()
        for _, row in sc_df.iterrows():
            passing_models.add((row["sport"], row["bet_type"], row["framework"]))
        console.print(f"  Scorecard loaded: {len(passing_models)} models to calibrate")

    # Discover models
    all_models = discover_models(models_dir)
    if sport_filter:
        all_models = [m for m in all_models if m["sport"] == sport_filter.upper()]

    if passing_models:
        all_models = [m for m in all_models
                      if (m["sport"], m["bet_type"], m["framework"]) in passing_models]

    console.print(f"  Models to calibrate: {len(all_models)}")

    results = {}
    calibration_report = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console
    ) as progress:
        task = progress.add_task("Calibrating", total=len(all_models))

        for model_info in all_models:
            sport = model_info["sport"]
            bet_type = model_info["bet_type"]
            framework = model_info["framework"]
            key = f"{framework}/{sport}/{bet_type}"

            progress.update(task, description=key)

            try:
                # Load validation data
                val_result = load_validation_data(sport, bet_type, csv_dir)
                if val_result is None:
                    logger.warning(f"No validation data for {key}")
                    progress.advance(task)
                    continue

                val_df, feature_columns, target_col = val_result
                y_true = val_df[target_col].values

                # Split: 60% for calibration fitting, 40% for evaluation
                n = len(val_df)
                cal_split = int(n * 0.6)
                cal_df = val_df.iloc[:cal_split].copy()
                eval_df = val_df.iloc[cal_split:].copy()

                y_cal_true = y_true[:cal_split]
                y_eval_true = y_true[cal_split:]

                # Get raw predictions on calibration set
                y_cal_proba = load_and_predict(model_info, cal_df, feature_columns)
                if y_cal_proba is None:
                    progress.advance(task)
                    continue

                # Get raw predictions on eval set
                y_eval_proba = load_and_predict(model_info, eval_df, feature_columns)
                if y_eval_proba is None:
                    progress.advance(task)
                    continue

                # Filter NaN
                cal_valid = ~(np.isnan(y_cal_true) | np.isnan(y_cal_proba))
                eval_valid = ~(np.isnan(y_eval_true) | np.isnan(y_eval_proba))

                y_cal_true = y_cal_true[cal_valid]
                y_cal_proba = y_cal_proba[cal_valid]
                y_eval_true = y_eval_true[eval_valid]
                y_eval_proba = y_eval_proba[eval_valid]

                if len(y_cal_true) < 20 or len(y_eval_true) < 10:
                    progress.advance(task)
                    continue

                # ECE before calibration
                ece_before = _compute_ece(y_eval_true, y_eval_proba)
                reliability_before = generate_reliability_data(y_eval_true, y_eval_proba)

                # Fit calibrator
                if method == "isotonic":
                    calibrator = calibrate_isotonic(y_cal_true, y_cal_proba)
                elif method == "platt":
                    calibrator = calibrate_platt(y_cal_true, y_cal_proba)
                elif method == "temperature":
                    calibrator = calibrate_temperature(y_cal_true, y_cal_proba)
                else:
                    # Try all methods, pick best
                    iso_cal = calibrate_isotonic(y_cal_true, y_cal_proba)
                    platt_cal = calibrate_platt(y_cal_true, y_cal_proba)
                    temp_cal = calibrate_temperature(y_cal_true, y_cal_proba)

                    iso_ece = _compute_ece(y_eval_true, apply_calibration(iso_cal, y_eval_proba, "isotonic"))
                    platt_ece = _compute_ece(y_eval_true, apply_calibration(platt_cal, y_eval_proba, "platt"))
                    temp_ece = _compute_ece(y_eval_true, apply_calibration(temp_cal, y_eval_proba, "temperature"))

                    best = min([("isotonic", iso_ece, iso_cal),
                                ("platt", platt_ece, platt_cal),
                                ("temperature", temp_ece, temp_cal)],
                               key=lambda x: x[1])
                    method = best[0]
                    calibrator = best[2]

                # Apply calibration to eval set
                y_eval_calibrated = apply_calibration(calibrator, y_eval_proba, method)

                # ECE after calibration
                ece_after = _compute_ece(y_eval_true, y_eval_calibrated)
                reliability_after = generate_reliability_data(y_eval_true, y_eval_calibrated)

                improvement = ((ece_before - ece_after) / ece_before * 100) if ece_before > 0 else 0

                # Save calibrator to model directory
                # Save to framework-specific dir AND sport-level dir
                save_paths = []

                # Framework dir: {models_dir}/{framework}/{sport}/{bet_type}/calibrator.pkl
                fw_cal_path = models_dir / framework / sport / bet_type / "calibrator.pkl"
                fw_cal_path.parent.mkdir(parents=True, exist_ok=True)
                with open(fw_cal_path, 'wb') as f:
                    pickle.dump({"calibrator": calibrator, "method": method,
                                 "ece_before": ece_before, "ece_after": ece_after,
                                 "trained_at": datetime.utcnow().isoformat(),
                                 "n_calibration_samples": len(y_cal_true)}, f)
                save_paths.append(str(fw_cal_path))

                # Sport-level dir: {models_dir}/{sport}/{bet_type}/calibrator.pkl
                sport_cal_path = models_dir / sport / bet_type / "calibrator.pkl"
                sport_cal_path.parent.mkdir(parents=True, exist_ok=True)
                with open(sport_cal_path, 'wb') as f:
                    pickle.dump({"calibrator": calibrator, "method": method,
                                 "ece_before": ece_before, "ece_after": ece_after,
                                 "framework": framework,
                                 "trained_at": datetime.utcnow().isoformat(),
                                 "n_calibration_samples": len(y_cal_true)}, f)
                save_paths.append(str(sport_cal_path))

                report_entry = {
                    "sport": sport, "bet_type": bet_type, "framework": framework,
                    "method": method,
                    "ece_before": round(ece_before, 4),
                    "ece_after": round(ece_after, 4),
                    "improvement_pct": round(improvement, 1),
                    "n_cal_samples": len(y_cal_true),
                    "n_eval_samples": len(y_eval_true),
                    "calibrator_path": save_paths[0],
                }
                calibration_report.append(report_entry)

                results[key] = report_entry

                if verbose:
                    emoji = "✅" if ece_after < ece_before else "⚠️"
                    console.print(
                        f"  {emoji} {key}: ECE {ece_before:.4f} → {ece_after:.4f} "
                        f"({improvement:+.1f}%) [{method}]"
                    )

            except Exception as e:
                logger.error(f"Calibration failed for {key}: {e}")
                if verbose:
                    traceback.print_exc()

            progress.advance(task)

    # Print results table
    _print_calibration_report(calibration_report)

    # Save report
    report_path = "calibration_report.csv"
    pd.DataFrame(calibration_report).to_csv(report_path, index=False)
    console.print(f"\n[green]Calibration report saved to {report_path}[/green]")

    return results


def _print_calibration_report(report: List[Dict]):
    """Print calibration results table."""
    if not report:
        console.print("[yellow]No models were calibrated.[/yellow]")
        return

    table = Table(title="Calibration Results")
    table.add_column("Sport", style="cyan")
    table.add_column("Bet", style="blue")
    table.add_column("Framework", style="magenta")
    table.add_column("Method", style="white")
    table.add_column("ECE Before", style="red")
    table.add_column("ECE After", style="green")
    table.add_column("Improvement", style="yellow")
    table.add_column("Samples", style="dim")

    for r in sorted(report, key=lambda x: x["improvement_pct"], reverse=True):
        imp_str = f"{r['improvement_pct']:+.1f}%"
        if r["improvement_pct"] > 0:
            imp_str = f"[green]{imp_str}[/green]"
        else:
            imp_str = f"[red]{imp_str}[/red]"

        table.add_row(
            r["sport"], r["bet_type"], r["framework"], r["method"],
            f"{r['ece_before']:.4f}", f"{r['ece_after']:.4f}",
            imp_str, str(r["n_cal_samples"]),
        )

    console.print(table)

    # Summary
    avg_before = np.mean([r["ece_before"] for r in report])
    avg_after = np.mean([r["ece_after"] for r in report])
    improved = sum(1 for r in report if r["ece_after"] < r["ece_before"])

    console.print(Panel(
        f"Models calibrated: {len(report)}\n"
        f"Models improved: {improved}/{len(report)}\n"
        f"Average ECE: {avg_before:.4f} → {avg_after:.4f}",
        title="Calibration Summary"
    ))


async def main():
    parser = argparse.ArgumentParser(description="ROYALEY Phase 3: Probability Calibration")
    parser.add_argument("--models-dir", "-m", type=str, default="/nvme0n1-disk/royaley/models")
    parser.add_argument("--scorecard", type=str, default=None,
                        help="Phase 2 scorecard CSV to filter models")
    parser.add_argument("--csv-dir", type=str, default=None)
    parser.add_argument("--method", type=str, default="isotonic",
                        choices=["isotonic", "platt", "temperature", "auto"],
                        help="Calibration method (default: isotonic)")
    parser.add_argument("--sport", "-s", type=str, default=None)
    parser.add_argument("--min-composite", type=float, default=0.0,
                        help="Minimum composite score from scorecard")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    await calibrate_all_models(
        models_dir=Path(args.models_dir),
        scorecard_path=args.scorecard,
        csv_dir=args.csv_dir,
        method=args.method,
        sport_filter=args.sport,
        min_composite=args.min_composite,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    asyncio.run(main())
