#!/usr/bin/env python3
"""
ROYALEY - Phase 4: Meta-Ensemble Weight Optimization
Optimizes framework weights per sport/bet_type combination.

Usage:
    python scripts/optimize_ensemble.py --models-dir /nvme0n1-disk/royaley/models
    python scripts/optimize_ensemble.py --models-dir /nvme0n1-disk/royaley/models --sport NFL --bet-type spread
"""

import asyncio
import argparse
import logging
import sys
import pickle
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from scripts.evaluate_models import (
    discover_models, load_and_predict, load_validation_data,
    _compute_ece, SPORTS, BET_TYPES, TARGET_COLUMNS,
)

console = Console()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

FRAMEWORKS_ORDER = ["h2o", "sklearn", "autogluon", "tensorflow", "quantum"]


def optimize_weights_scipy(
    predictions: Dict[str, np.ndarray],
    y_true: np.ndarray,
    metric: str = "log_loss",
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Optimize ensemble weights using scipy minimize.
    Returns (optimal_weights, metrics).
    """
    from sklearn.metrics import log_loss as sk_log_loss, accuracy_score, roc_auc_score

    fw_names = sorted(predictions.keys())
    n_fw = len(fw_names)
    fw_preds = [predictions[fw] for fw in fw_names]

    if n_fw == 1:
        return {fw_names[0]: 1.0}, _eval_combined(fw_preds[0], y_true)

    def objective(raw_weights):
        # Softmax to ensure positive weights summing to 1
        exp_w = np.exp(raw_weights - np.max(raw_weights))
        weights = exp_w / exp_w.sum()

        combined = sum(w * p for w, p in zip(weights, fw_preds))
        combined = np.clip(combined, 0.001, 0.999)

        if metric == "log_loss":
            return sk_log_loss(y_true, combined)
        elif metric == "neg_accuracy":
            return -accuracy_score(y_true, combined > 0.5)
        elif metric == "brier":
            return np.mean((y_true - combined) ** 2)
        else:
            return sk_log_loss(y_true, combined)

    # Multi-start optimization
    best_result = None
    best_value = float('inf')

    for _ in range(10):
        x0 = np.random.randn(n_fw) * 0.5
        result = minimize(objective, x0, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-6})
        if result.fun < best_value:
            best_value = result.fun
            best_result = result

    # Also try equal weights
    equal_x = np.zeros(n_fw)
    equal_val = objective(equal_x)
    if equal_val < best_value:
        best_result = type('Result', (), {'x': equal_x, 'fun': equal_val})()

    # Convert to weights
    exp_w = np.exp(best_result.x - np.max(best_result.x))
    optimal_weights = exp_w / exp_w.sum()

    weight_dict = {fw: float(w) for fw, w in zip(fw_names, optimal_weights)}

    # Compute combined predictions with optimal weights
    combined = sum(weight_dict[fw] * predictions[fw] for fw in fw_names)
    combined = np.clip(combined, 0.001, 0.999)

    metrics = _eval_combined(combined, y_true)

    return weight_dict, metrics


def optimize_weights_grid(
    predictions: Dict[str, np.ndarray],
    y_true: np.ndarray,
    metric: str = "log_loss",
    step: float = 0.05,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Grid search over weight space (for 2-3 frameworks)."""
    from sklearn.metrics import log_loss as sk_log_loss, accuracy_score

    fw_names = sorted(predictions.keys())
    n_fw = len(fw_names)
    fw_preds = [predictions[fw] for fw in fw_names]

    if n_fw > 4:
        return optimize_weights_scipy(predictions, y_true, metric)

    best_weights = None
    best_val = float('inf')

    # Generate weight grid
    steps = np.arange(0, 1.0 + step, step)

    if n_fw == 2:
        for w0 in steps:
            w1 = 1.0 - w0
            if w1 < 0:
                continue
            weights = [w0, w1]
            combined = sum(w * p for w, p in zip(weights, fw_preds))
            combined = np.clip(combined, 0.001, 0.999)
            val = sk_log_loss(y_true, combined) if metric == "log_loss" else -accuracy_score(y_true, combined > 0.5)
            if val < best_val:
                best_val = val
                best_weights = weights

    elif n_fw == 3:
        for w0 in steps:
            for w1 in steps:
                w2 = 1.0 - w0 - w1
                if w2 < -0.001 or w2 > 1.001:
                    continue
                w2 = max(0, w2)
                weights = [w0, w1, w2]
                combined = sum(w * p for w, p in zip(weights, fw_preds))
                combined = np.clip(combined, 0.001, 0.999)
                val = sk_log_loss(y_true, combined) if metric == "log_loss" else -accuracy_score(y_true, combined > 0.5)
                if val < best_val:
                    best_val = val
                    best_weights = weights

    elif n_fw == 4:
        # Coarser grid for 4 frameworks
        coarse_steps = np.arange(0, 1.05, 0.1)
        for w0 in coarse_steps:
            for w1 in coarse_steps:
                for w2 in coarse_steps:
                    w3 = 1.0 - w0 - w1 - w2
                    if w3 < -0.01 or w3 > 1.01:
                        continue
                    w3 = max(0, w3)
                    weights = [w0, w1, w2, w3]
                    combined = sum(w * p for w, p in zip(weights, fw_preds))
                    combined = np.clip(combined, 0.001, 0.999)
                    val = sk_log_loss(y_true, combined) if metric == "log_loss" else -accuracy_score(y_true, combined > 0.5)
                    if val < best_val:
                        best_val = val
                        best_weights = weights

    weight_dict = {fw: float(w) for fw, w in zip(fw_names, best_weights)}
    combined = sum(weight_dict[fw] * predictions[fw] for fw in fw_names)
    combined = np.clip(combined, 0.001, 0.999)
    metrics = _eval_combined(combined, y_true)

    return weight_dict, metrics


def _eval_combined(combined: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    """Evaluate combined predictions."""
    from sklearn.metrics import accuracy_score, roc_auc_score, log_loss as sk_log_loss, brier_score_loss

    y_pred_class = (combined > 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred_class),
        "auc_roc": roc_auc_score(y_true, combined) if len(np.unique(y_true)) > 1 else 0.5,
        "log_loss": sk_log_loss(y_true, combined),
        "brier_score": brier_score_loss(y_true, combined),
        "ece": _compute_ece(y_true, combined),
    }


async def optimize_all_ensembles(
    models_dir: Path,
    csv_dir: str = None,
    sport_filter: str = None,
    bet_type_filter: str = None,
    method: str = "scipy",
    verbose: bool = False,
) -> Dict:
    """Optimize ensemble weights for all sport/bet_type combos."""

    console.print(Panel(
        "[bold cyan]ROYALEY Phase 4: Ensemble Weight Optimization[/bold cyan]\n"
        f"Models: {models_dir}\n"
        f"Optimization: {method}\n"
        f"Filter: sport={sport_filter or 'ALL'}, bet_type={bet_type_filter or 'ALL'}",
        title="Configuration"
    ))

    all_models = discover_models(models_dir)

    # Group models by sport/bet_type
    groups: Dict[str, List[Dict]] = {}
    for m in all_models:
        if m["framework"] == "meta_ensemble":
            continue  # Skip existing ensembles
        key = f"{m['sport']}/{m['bet_type']}"
        if sport_filter and m["sport"] != sport_filter.upper():
            continue
        if bet_type_filter and m["bet_type"] != bet_type_filter:
            continue
        if key not in groups:
            groups[key] = []
        groups[key].append(m)

    console.print(f"  Found {len(groups)} sport/bet_type combinations")

    ensemble_configs = {}
    report = []

    for key, models in sorted(groups.items()):
        sport, bet_type = key.split("/")
        console.print(f"\n[cyan]Optimizing {key} ({len(models)} frameworks)...[/cyan]")

        # Load validation data
        val_result = load_validation_data(sport, bet_type, csv_dir)
        if val_result is None:
            console.print(f"  [yellow]No validation data for {key}[/yellow]")
            continue

        val_df, feature_columns, target_col = val_result
        y_true = val_df[target_col].values

        # Get predictions from each framework
        fw_predictions = {}
        for model_info in models:
            framework = model_info["framework"]
            try:
                preds = load_and_predict(model_info, val_df, feature_columns)
                if preds is not None:
                    valid = ~(np.isnan(preds) | np.isnan(y_true))
                    if valid.sum() > 20:
                        fw_predictions[framework] = preds
                        if verbose:
                            acc = np.mean((preds[valid] > 0.5) == y_true[valid])
                            console.print(f"  {framework}: {valid.sum()} predictions, acc={acc:.3f}")
            except Exception as e:
                logger.warning(f"  {framework} failed: {e}")

        if len(fw_predictions) < 1:
            console.print(f"  [yellow]No valid predictions for {key}[/yellow]")
            continue

        # Filter to common valid samples
        common_valid = np.ones(len(y_true), dtype=bool)
        for fw, preds in fw_predictions.items():
            common_valid &= ~np.isnan(preds)
        common_valid &= ~np.isnan(y_true)

        if common_valid.sum() < 20:
            console.print(f"  [yellow]Too few common samples for {key}: {common_valid.sum()}[/yellow]")
            continue

        y_valid = y_true[common_valid]
        fw_preds_valid = {fw: preds[common_valid] for fw, preds in fw_predictions.items()}

        # Apply calibration if calibrators exist
        for fw in fw_preds_valid:
            cal_path = models_dir / fw / sport / bet_type / "calibrator.pkl"
            if cal_path.exists():
                try:
                    with open(cal_path, 'rb') as f:
                        cal_data = pickle.load(f)
                    cal_method = cal_data.get("method", "isotonic")
                    calibrator = cal_data["calibrator"]
                    if cal_method == "isotonic":
                        fw_preds_valid[fw] = np.clip(calibrator.predict(fw_preds_valid[fw]), 0.001, 0.999)
                    elif cal_method == "platt":
                        fw_preds_valid[fw] = np.clip(calibrator.predict_proba(fw_preds_valid[fw].reshape(-1, 1))[:, 1], 0.001, 0.999)
                    elif cal_method == "temperature":
                        temp = calibrator["temperature"]
                        fw_preds_valid[fw] = np.clip(fw_preds_valid[fw] ** (1.0 / temp), 0.001, 0.999)
                except Exception:
                    pass

        # Evaluate individual frameworks
        individual_metrics = {}
        best_single_acc = 0
        best_single_fw = None
        for fw, preds in fw_preds_valid.items():
            m = _eval_combined(preds, y_valid)
            individual_metrics[fw] = m
            if m["accuracy"] > best_single_acc:
                best_single_acc = m["accuracy"]
                best_single_fw = fw
            console.print(
                f"  {fw}: acc={m['accuracy']:.3f} auc={m['auc_roc']:.3f} "
                f"ll={m['log_loss']:.3f}"
            )

        # Optimize weights
        if method == "grid" and len(fw_preds_valid) <= 4:
            optimal_weights, ensemble_metrics = optimize_weights_grid(fw_preds_valid, y_valid)
        else:
            optimal_weights, ensemble_metrics = optimize_weights_scipy(fw_preds_valid, y_valid)

        # Compare ensemble vs best single
        ensemble_acc = ensemble_metrics["accuracy"]
        improvement = ensemble_acc - best_single_acc

        console.print(f"\n  [bold]Optimal Weights:[/bold]")
        for fw, w in sorted(optimal_weights.items(), key=lambda x: -x[1]):
            bar = "█" * int(w * 30) + "░" * (30 - int(w * 30))
            console.print(f"    {fw:12s} {bar} {w:.3f}")

        console.print(
            f"\n  Ensemble: acc={ensemble_acc:.3f} | "
            f"Best single ({best_single_fw}): acc={best_single_acc:.3f} | "
            f"Δ = {improvement:+.3f}"
        )

        # Decision: use ensemble only if it beats best single by >= 0.005
        use_ensemble = improvement >= 0.005
        if use_ensemble:
            console.print(f"  [green]✓ Using ENSEMBLE (beats best single by {improvement*100:+.1f}%)[/green]")
            final_weights = optimal_weights
        else:
            console.print(f"  [yellow]→ Using BEST SINGLE ({best_single_fw}) — ensemble gain too small[/yellow]")
            final_weights = {fw: (1.0 if fw == best_single_fw else 0.0) for fw in fw_preds_valid}

        # Save ensemble config
        config = {
            "sport": sport,
            "bet_type": bet_type,
            "weights": final_weights,
            "use_ensemble": use_ensemble,
            "best_single_framework": best_single_fw,
            "best_single_accuracy": best_single_acc,
            "ensemble_accuracy": ensemble_acc,
            "improvement": improvement,
            "individual_metrics": {fw: m for fw, m in individual_metrics.items()},
            "ensemble_metrics": ensemble_metrics,
            "n_samples": int(common_valid.sum()),
            "optimized_at": datetime.utcnow().isoformat(),
        }

        # Save to disk
        config_dir = models_dir / sport / bet_type
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / "ensemble_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)

        ensemble_configs[key] = config
        report.append({
            "sport": sport, "bet_type": bet_type,
            "n_frameworks": len(fw_preds_valid),
            "use_ensemble": use_ensemble,
            "best_single": best_single_fw,
            "best_single_acc": round(best_single_acc, 4),
            "ensemble_acc": round(ensemble_acc, 4),
            "improvement": round(improvement, 4),
            "weights": json.dumps({k: round(v, 3) for k, v in final_weights.items()}),
        })

    # Print final report
    _print_optimization_report(report)

    # Save summary
    pd.DataFrame(report).to_csv("ensemble_optimization_report.csv", index=False)
    console.print(f"\n[green]Report saved to ensemble_optimization_report.csv[/green]")

    return ensemble_configs


def _print_optimization_report(report: List[Dict]):
    """Print optimization results."""
    if not report:
        console.print("[yellow]No ensembles optimized.[/yellow]")
        return

    table = Table(title="Ensemble Optimization Results")
    table.add_column("Sport", style="cyan")
    table.add_column("Bet", style="blue")
    table.add_column("# FW", style="dim")
    table.add_column("Strategy", style="magenta")
    table.add_column("Best Single", style="yellow")
    table.add_column("Ensemble Acc", style="green")
    table.add_column("Δ", style="white")

    for r in report:
        strategy = "ENSEMBLE" if r["use_ensemble"] else f"SINGLE ({r['best_single']})"
        delta = f"{r['improvement']*100:+.1f}%"
        if r["improvement"] > 0:
            delta = f"[green]{delta}[/green]"
        else:
            delta = f"[red]{delta}[/red]"

        table.add_row(
            r["sport"], r["bet_type"], str(r["n_frameworks"]),
            strategy, f"{r['best_single_acc']:.3f}",
            f"{r['ensemble_acc']:.3f}", delta,
        )

    console.print(table)

    n_ensemble = sum(1 for r in report if r["use_ensemble"])
    console.print(Panel(
        f"Combinations optimized: {len(report)}\n"
        f"Using ensemble: {n_ensemble}\n"
        f"Using best single: {len(report) - n_ensemble}",
        title="Optimization Summary"
    ))


async def main():
    parser = argparse.ArgumentParser(description="ROYALEY Phase 4: Ensemble Optimization")
    parser.add_argument("--models-dir", "-m", type=str, default="/nvme0n1-disk/royaley/models")
    parser.add_argument("--csv-dir", type=str, default=None)
    parser.add_argument("--sport", "-s", type=str, default=None)
    parser.add_argument("--bet-type", "-b", type=str, default=None)
    parser.add_argument("--method", type=str, default="scipy",
                        choices=["scipy", "grid"])
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    await optimize_all_ensembles(
        models_dir=Path(args.models_dir),
        csv_dir=args.csv_dir,
        sport_filter=args.sport,
        bet_type_filter=args.bet_type,
        method=args.method,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    asyncio.run(main())
