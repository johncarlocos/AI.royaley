#!/usr/bin/env python3
"""
ROYALEY - Evaluation Patch v4: Scoring & Display Fixes
=======================================================
Fixes applied ON TOP of v3 patches (which must be applied first):

1. Suppress PerformanceWarnings during feature engineering
2. Add tiered pass system: PRODUCTION / SIGNAL / LEAKAGE_SUSPECT  
3. Fix composite score (ROI normalization, leakage penalty)
4. Add data leakage detection (tennis home=winner bias)
5. Better summary showing models with real signal

Usage:
    # Apply AFTER v3 patch:
    docker cp patch_evaluation_v4.py royaley_api:/app/scripts/
    docker exec royaley_api python /app/scripts/patch_evaluation_v4.py
    docker exec -it royaley_api python scripts/evaluate_models.py --verbose
"""

import sys
from pathlib import Path

EVAL_SCRIPT = Path("/app/scripts/evaluate_models.py")


def apply_patches():
    if not EVAL_SCRIPT.exists():
        print(f"ERROR: {EVAL_SCRIPT} not found")
        sys.exit(1)

    content = EVAL_SCRIPT.read_text()
    changes = []

    # ══════════════════════════════════════════════════════════════════
    # FIX 1: Suppress PerformanceWarnings
    # ══════════════════════════════════════════════════════════════════
    if 'import warnings' not in content:
        old_imports = "import numpy as np\nimport pandas as pd"
        new_imports = """import warnings
import numpy as np
import pandas as pd

# Suppress DataFrame fragmentation warnings during feature engineering
# (we call df.copy() at the end to defragment)
warnings.filterwarnings('ignore', message='.*DataFrame is highly fragmented.*')"""
        if old_imports in content:
            content = content.replace(old_imports, new_imports)
            changes.append("Added warnings suppression for PerformanceWarning")
        else:
            print("  WARNING: Could not find import block for warnings insertion")
    else:
        print("  SKIP: warnings already imported")

    # ══════════════════════════════════════════════════════════════════
    # FIX 2: Add tiered threshold constants
    # ══════════════════════════════════════════════════════════════════
    old_thresholds = """# Minimum acceptable thresholds
MIN_ACCURACY = 0.52
MIN_AUC = 0.52
MAX_CALIBRATION_ERROR = 0.15"""

    new_thresholds = """# Minimum acceptable thresholds
MIN_ACCURACY = 0.52
MIN_AUC = 0.52
MAX_CALIBRATION_ERROR = 0.15

# Tiered pass criteria (more nuanced than binary pass/fail)
# PRODUCTION: ready to bet with real money
# SIGNAL: model has real discriminative power, needs calibration work
# LEAKAGE_SUSPECT: high accuracy from class imbalance, not model skill
TIER_SIGNAL_AUC = 0.55        # AUC alone proves ranking ability
TIER_LEAKAGE_CLASS_PCT = 0.88  # If >88% predictions are one class → suspect"""

    if old_thresholds in content:
        content = content.replace(old_thresholds, new_thresholds)
        changes.append("Added tiered threshold constants (SIGNAL/LEAKAGE)")
    else:
        print("  SKIP: thresholds already modified")

    # ══════════════════════════════════════════════════════════════════
    # FIX 3: Add tier/leakage fields to ModelScore dataclass
    # ══════════════════════════════════════════════════════════════════
    old_dataclass_end = """    # Ranking
    composite_score: float = 0.0
    rank: int = 0
    passes_threshold: bool = False"""

    new_dataclass_end = """    # Ranking
    composite_score: float = 0.0
    rank: int = 0
    passes_threshold: bool = False
    tier: str = ""          # "PRODUCTION", "SIGNAL", "WEAK", "LEAKAGE_SUSPECT"
    leakage_suspect: bool = False
    dominant_class_pct: float = 0.0  # % of predictions in majority class"""

    if old_dataclass_end in content and 'tier: str' not in content:
        content = content.replace(old_dataclass_end, new_dataclass_end)
        changes.append("Added tier/leakage fields to ModelScore")
    else:
        print("  SKIP: ModelScore already has tier fields")

    # ══════════════════════════════════════════════════════════════════
    # FIX 4: Fix composite_score - realistic ROI range + leakage penalty
    # ══════════════════════════════════════════════════════════════════
    old_composite = '''def compute_composite_score(metrics: Dict[str, float]) -> float:
    """
    Composite score for ranking models.
    Weighted combination: AUC(30%) + Accuracy(25%) + ROI(25%) + (1-ECE)(10%) + (1-Brier)(10%)
    """
    auc = metrics.get("auc_roc", 0.5)
    acc = metrics.get("accuracy", 0.5)
    roi = max(metrics.get("simulated_roi", -1.0), -1.0)  # Clamp
    ece = min(metrics.get("ece", 1.0), 1.0)
    brier = min(metrics.get("brier_score", 1.0), 1.0)

    # Normalize ROI to 0-1 range (assume -50% to +50%)
    roi_norm = (roi + 0.5) / 1.0
    roi_norm = max(0, min(1, roi_norm))

    score = (
        0.30 * auc +
        0.25 * acc +
        0.25 * roi_norm +
        0.10 * (1.0 - ece) +
        0.10 * (1.0 - brier)
    )

    return round(score, 6)'''

    new_composite = '''def compute_composite_score(metrics: Dict[str, float],
                            leakage_suspect: bool = False) -> float:
    """
    Composite score for ranking models.
    AUC(35%) + Accuracy(20%) + ROI(20%) + (1-ECE)(15%) + (1-Brier)(10%)
    
    Changes from v3:
    - AUC weighted higher (best single metric for model quality)
    - ECE weighted higher (calibration matters for betting)
    - ROI capped at realistic +30% (prevents tennis leakage inflation)
    - Leakage penalty halves score for suspect models
    """
    auc = metrics.get("auc_roc", 0.5)
    acc = metrics.get("accuracy", 0.5)
    roi = max(metrics.get("simulated_roi", -1.0), -1.0)
    ece = min(metrics.get("ece", 1.0), 1.0)
    brier = min(metrics.get("brier_score", 1.0), 1.0)

    # Normalize ROI to 0-1 range with REALISTIC bounds (-20% to +30%)
    # Any ROI > 30% is almost certainly leakage/overfitting
    roi_capped = max(-0.20, min(0.30, roi))
    roi_norm = (roi_capped + 0.20) / 0.50
    roi_norm = max(0, min(1, roi_norm))

    score = (
        0.35 * auc +
        0.20 * acc +
        0.20 * roi_norm +
        0.15 * (1.0 - ece) +
        0.10 * (1.0 - brier)
    )

    # Leakage penalty: models that predict one class >88% of the time
    # get a heavy penalty (likely exploiting class imbalance, not skill)
    if leakage_suspect:
        score *= 0.50

    return round(score, 6)'''

    if old_composite in content:
        content = content.replace(old_composite, new_composite)
        changes.append("Fixed composite_score: realistic ROI cap + AUC-weighted + leakage penalty")
    else:
        print("  SKIP: composite_score already modified")

    # ══════════════════════════════════════════════════════════════════
    # FIX 5: Replace threshold check with tiered system + leakage detection
    # ══════════════════════════════════════════════════════════════════
    old_threshold_block = """                # Composite score
                score.composite_score = compute_composite_score(metrics)

                # Threshold check
                score.passes_threshold = (
                    score.accuracy >= MIN_ACCURACY and
                    score.auc_roc >= MIN_AUC and
                    score.ece <= MAX_CALIBRATION_ERROR
                )

                if verbose:
                    status = "✅" if score.passes_threshold else "❌"
                    console.print(
                        f"  {status} {framework}/{sport}/{bet_type}: "
                        f"acc={score.accuracy:.3f} auc={score.auc_roc:.3f} "
                        f"roi={score.simulated_roi:+.3f} ece={score.ece:.3f}"
                    )"""

    new_threshold_block = """                # Leakage detection: if model predicts one class >88% of the time,
                # "accuracy" comes from class imbalance not model skill
                pred_classes = (y_pred_proba > 0.5).astype(int)
                majority_pct = max(pred_classes.mean(), 1 - pred_classes.mean())
                score.dominant_class_pct = majority_pct
                score.leakage_suspect = majority_pct > TIER_LEAKAGE_CLASS_PCT

                # Composite score (with leakage penalty)
                score.composite_score = compute_composite_score(
                    metrics, leakage_suspect=score.leakage_suspect
                )

                # Tiered threshold check
                score.passes_threshold = (
                    score.accuracy >= MIN_ACCURACY and
                    score.auc_roc >= MIN_AUC and
                    score.ece <= MAX_CALIBRATION_ERROR and
                    not score.leakage_suspect
                )

                # Assign tier
                if score.leakage_suspect:
                    score.tier = "LEAK?"
                elif score.passes_threshold:
                    score.tier = "PROD"
                elif score.auc_roc >= TIER_SIGNAL_AUC:
                    score.tier = "SIGNAL"
                else:
                    score.tier = "WEAK"

                if verbose:
                    tier_colors = {"PROD": "green", "SIGNAL": "yellow", "LEAK?": "red", "WEAK": "dim"}
                    color = tier_colors.get(score.tier, "white")
                    icon = {"PROD": "✅", "SIGNAL": "📊", "LEAK?": "⚠️", "WEAK": "❌"}.get(score.tier, "❌")
                    console.print(
                        f"  {icon} [{color}]{framework}/{sport}/{bet_type}[/{color}]: "
                        f"acc={score.accuracy:.3f} auc={score.auc_roc:.3f} "
                        f"roi={score.simulated_roi:+.3f} ece={score.ece:.3f} "
                        f"[{color}][{score.tier}][/{color}]"
                    )"""

    if old_threshold_block in content:
        content = content.replace(old_threshold_block, new_threshold_block)
        changes.append("Added tiered pass system (PROD/SIGNAL/LEAK?/WEAK) + leakage detection")
    else:
        print("  SKIP: threshold block already modified")

    # ══════════════════════════════════════════════════════════════════
    # FIX 6: Update scorecard display to show tiers
    # ══════════════════════════════════════════════════════════════════
    old_pass_str = '        pass_str = "[green]✓[/green]" if s.passes_threshold else "[red]✗[/red]"'
    new_pass_str = '''        tier_colors = {"PROD": "green", "SIGNAL": "yellow", "LEAK?": "red", "WEAK": "dim"}
        color = tier_colors.get(s.tier, "white")
        pass_str = f"[{color}]{s.tier}[/{color}]"'''

    if old_pass_str in content:
        content = content.replace(old_pass_str, new_pass_str)
        changes.append("Updated scorecard display with tier labels")
    else:
        print("  SKIP: pass_str display already modified")

    # Also update the column header
    old_col = '    table.add_column("Pass?", style="white")'
    new_col = '    table.add_column("Tier", style="white")'
    if old_col in content:
        content = content.replace(old_col, new_col)

    # ══════════════════════════════════════════════════════════════════
    # FIX 7: Update summary to show tiered breakdown
    # ══════════════════════════════════════════════════════════════════
    old_summary = '''def _print_summary(scores: List[ModelScore]):
    """Print summary statistics."""
    loaded = [s for s in scores if s.load_success]
    passed = [s for s in loaded if s.passes_threshold]
    failed = [s for s in loaded if not s.passes_threshold]
    errors = [s for s in scores if not s.load_success]

    console.print(Panel(
        f"[bold]Total Models:[/bold] {len(scores)}\\n"
        f"[green]Loaded Successfully:[/green] {len(loaded)}\\n"
        f"[green]Passed Thresholds:[/green] {len(passed)} (acc≥{MIN_ACCURACY}, auc≥{MIN_AUC}, ece≤{MAX_CALIBRATION_ERROR})\\n"
        f"[red]Below Threshold:[/red] {len(failed)}\\n"
        f"[red]Load Errors:[/red] {len(errors)}",
        title="Summary"
    ))

    # Best per sport/bet_type
    if passed:
        console.print("\\n[cyan]Best Model per Sport/Bet-Type:[/cyan]")
        best_table = Table()
        best_table.add_column("Sport", style="cyan")
        best_table.add_column("Bet Type", style="blue")
        best_table.add_column("Framework", style="magenta")
        best_table.add_column("Accuracy", style="yellow")
        best_table.add_column("AUC", style="yellow")
        best_table.add_column("ROI", style="green")
        best_table.add_column("Composite", style="bold white")

        seen = set()
        for s in passed:
            key = (s.sport, s.bet_type)
            if key in seen:
                continue
            seen.add(key)
            best_table.add_row(
                s.sport, s.bet_type, s.framework,
                f"{s.accuracy:.3f}", f"{s.auc_roc:.3f}",
                f"{s.simulated_roi*100:+.1f}%", f"{s.composite_score:.4f}",
            )'''

    new_summary = '''def _print_summary(scores: List[ModelScore]):
    """Print summary statistics with tiered breakdown."""
    loaded = [s for s in scores if s.load_success]
    errors = [s for s in scores if not s.load_success]

    # Tiered counts
    prod = [s for s in loaded if s.tier == "PROD"]
    signal = [s for s in loaded if s.tier == "SIGNAL"]
    leakage = [s for s in loaded if s.tier == "LEAK?"]
    weak = [s for s in loaded if s.tier == "WEAK"]

    console.print(Panel(
        f"[bold]Total Models:[/bold] {len(scores)}\\n"
        f"[green]Loaded Successfully:[/green] {len(loaded)}\\n"
        f"[red]Load Errors:[/red] {len(errors)}\\n"
        f"\\n"
        f"[bold]Model Tiers:[/bold]\\n"
        f"  [green]🟢 PRODUCTION:[/green]  {len(prod):>3}  (acc≥{MIN_ACCURACY}, auc≥{MIN_AUC}, ece≤{MAX_CALIBRATION_ERROR})\\n"
        f"  [yellow]🟡 SIGNAL:[/yellow]      {len(signal):>3}  (auc≥{TIER_SIGNAL_AUC}, needs calibration/threshold work)\\n"
        f"  [red]🔴 LEAK?:[/red]       {len(leakage):>3}  (>88% predictions in one class)\\n"
        f"  [dim]⚪ WEAK:[/dim]        {len(weak):>3}  (no discriminative power)",
        title="Summary"
    ))

    # Show SIGNAL models (these have edge but need work)
    if signal:
        console.print("\\n[yellow]📊 SIGNAL Models (have edge, need calibration):[/yellow]")
        sig_table = Table()
        sig_table.add_column("Sport", style="cyan")
        sig_table.add_column("Bet Type", style="blue")
        sig_table.add_column("Framework", style="magenta")
        sig_table.add_column("AUC", style="yellow")
        sig_table.add_column("Acc", style="yellow")
        sig_table.add_column("ECE", style="red")
        sig_table.add_column("ROI", style="green")
        sig_table.add_column("Issue", style="dim")

        for s in sorted(signal, key=lambda x: x.auc_roc, reverse=True):
            issues = []
            if s.accuracy < MIN_ACCURACY:
                issues.append(f"acc<{MIN_ACCURACY}")
            if s.ece > MAX_CALIBRATION_ERROR:
                issues.append(f"ece>{MAX_CALIBRATION_ERROR}")
            sig_table.add_row(
                s.sport, s.bet_type, s.framework,
                f"{s.auc_roc:.3f}", f"{s.accuracy:.3f}", f"{s.ece:.3f}",
                f"{s.simulated_roi*100:+.1f}%",
                ", ".join(issues) if issues else "threshold border",
            )
        console.print(sig_table)

    # Best PRODUCTION models per sport/bet_type
    if prod:
        console.print("\\n[green]✅ PRODUCTION Models (ready for deployment):[/green]")
        best_table = Table()
        best_table.add_column("Sport", style="cyan")
        best_table.add_column("Bet Type", style="blue")
        best_table.add_column("Framework", style="magenta")
        best_table.add_column("Accuracy", style="yellow")
        best_table.add_column("AUC", style="yellow")
        best_table.add_column("ROI", style="green")
        best_table.add_column("Composite", style="bold white")

        seen = set()
        for s in prod:
            key = (s.sport, s.bet_type)
            if key in seen:
                continue
            seen.add(key)
            best_table.add_row(
                s.sport, s.bet_type, s.framework,
                f"{s.accuracy:.3f}", f"{s.auc_roc:.3f}",
                f"{s.simulated_roi*100:+.1f}%", f"{s.composite_score:.4f}",
            )'''

    if old_summary in content:
        content = content.replace(old_summary, new_summary)
        changes.append("Updated summary with tiered breakdown + SIGNAL model details")
    else:
        print("  SKIP: summary already modified")

    # ══════════════════════════════════════════════════════════════════
    # FIX 8: Update CSV export to include new fields
    # ══════════════════════════════════════════════════════════════════
    old_csv_fields = '''            "composite_score": round(s.composite_score, 6),
            "passes_threshold": s.passes_threshold,'''
    new_csv_fields = '''            "composite_score": round(s.composite_score, 6),
            "passes_threshold": s.passes_threshold,
            "tier": s.tier,
            "leakage_suspect": s.leakage_suspect,
            "dominant_class_pct": round(s.dominant_class_pct, 3),'''
    if old_csv_fields in content:
        content = content.replace(old_csv_fields, new_csv_fields)
        changes.append("Added tier/leakage fields to CSV export")
    else:
        print("  SKIP: CSV fields already modified")

    # ── WRITE ──
    if changes:
        EVAL_SCRIPT.write_text(content)
        print(f"\n{'='*60}")
        print(f"  PATCH v4 APPLIED ({len(changes)} changes)")
        print(f"{'='*60}")
        for i, c in enumerate(changes, 1):
            print(f"  {i}. {c}")
        print(f"\n  Re-run evaluation:")
        print(f"    python scripts/evaluate_models.py --verbose")
    else:
        print("\n  No changes needed - all v4 patches already applied")


if __name__ == "__main__":
    print("=" * 60)
    print("  ROYALEY Evaluation Patch v4")
    print("  Fixes: warnings + tiers + leakage + scoring")
    print("=" * 60)
    apply_patches()