#!/usr/bin/env python3
"""
ROYALEY - Phase 6a: Promote Models to Production
Promotes the best model per sport/bet_type based on scorecard rankings.

Usage:
    python scripts/promote_models.py --scorecard model_scorecard.csv
    python scripts/promote_models.py --scorecard model_scorecard.csv --min-accuracy 0.55 --dry-run
"""

import asyncio
import argparse
import logging
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from app.core.database import db_manager
from app.models import MLModel, Sport
from sqlalchemy import select, and_, update

console = Console()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def select_production_models(
    scorecard_path: str,
    min_accuracy: float = 0.55,
    min_auc: float = 0.53,
    max_ece: float = 0.12,
    strategy: str = "best_composite",
) -> Dict[str, Dict]:
    """
    Select the best model per sport/bet_type from scorecard.
    Returns dict keyed by 'SPORT/bet_type'.
    """
    df = pd.read_csv(scorecard_path)
    df = df[df["load_success"] == True]

    # Apply thresholds
    df = df[df["accuracy"] >= min_accuracy]
    df = df[df["auc_roc"] >= min_auc]
    df = df[df["ece"] <= max_ece]

    if len(df) == 0:
        console.print("[red]No models pass the minimum thresholds![/red]")
        return {}

    # Group by sport/bet_type, pick best
    selections = {}

    for (sport, bet_type), group in df.groupby(["sport", "bet_type"]):
        if strategy == "best_composite":
            best = group.loc[group["composite_score"].idxmax()]
        elif strategy == "best_accuracy":
            best = group.loc[group["accuracy"].idxmax()]
        elif strategy == "best_roi":
            best = group.loc[group["simulated_roi"].idxmax()]
        else:
            best = group.loc[group["composite_score"].idxmax()]

        key = f"{sport}/{bet_type}"
        selections[key] = {
            "sport": sport,
            "bet_type": bet_type,
            "framework": best["framework"],
            "accuracy": best["accuracy"],
            "auc_roc": best["auc_roc"],
            "simulated_roi": best["simulated_roi"],
            "composite_score": best["composite_score"],
            "model_path": best["model_path"],
            "ece": best["ece"],
        }

    return selections


async def promote_to_database(
    selections: Dict[str, Dict],
    dry_run: bool = False,
) -> int:
    """Update is_production flag in database."""

    if dry_run:
        console.print("[yellow]DRY RUN — no database changes[/yellow]")
        return 0

    await db_manager.initialize()
    promoted = 0

    async with db_manager.session() as session:
        for key, sel in selections.items():
            sport_code = sel["sport"]
            bet_type = sel["bet_type"]
            model_path = sel["model_path"]

            # Get sport ID
            result = await session.execute(
                select(Sport).where(Sport.code == sport_code)
            )
            sport = result.scalar_one_or_none()
            if not sport:
                console.print(f"  [yellow]Sport not found in DB: {sport_code}[/yellow]")
                continue

            # Demote current production models for this sport/bet_type
            await session.execute(
                update(MLModel)
                .where(
                    and_(
                        MLModel.sport_id == sport.id,
                        MLModel.bet_type == bet_type,
                        MLModel.is_production == True,
                    )
                )
                .values(is_production=False)
            )

            # Find and promote the new model
            result = await session.execute(
                select(MLModel)
                .where(
                    and_(
                        MLModel.sport_id == sport.id,
                        MLModel.bet_type == bet_type,
                        MLModel.file_path == model_path,
                    )
                )
                .order_by(MLModel.created_at.desc())
                .limit(1)
            )
            model = result.scalar_one_or_none()

            if model:
                model.is_production = True
                promoted += 1
                console.print(f"  [green]✓ Promoted {sport_code}/{bet_type} → {sel['framework']}[/green]")
            else:
                # Model not in DB yet — create record
                new_model = MLModel(
                    sport_id=sport.id,
                    bet_type=bet_type,
                    framework=sel["framework"],
                    file_path=model_path,
                    is_production=True,
                    version="1.0",
                    accuracy=sel["accuracy"],
                    auc=sel["auc_roc"],
                )
                session.add(new_model)
                promoted += 1
                console.print(f"  [green]✓ Created & promoted {sport_code}/{bet_type} → {sel['framework']}[/green]")

        await session.commit()

    return promoted


def save_production_manifest(selections: Dict[str, Dict], output_path: str = "production_manifest.json"):
    """Save production model manifest for predict.py to load."""
    manifest = {
        "promoted_at": datetime.utcnow().isoformat(),
        "models": {}
    }

    for key, sel in selections.items():
        manifest["models"][key] = {
            "sport": sel["sport"],
            "bet_type": sel["bet_type"],
            "framework": sel["framework"],
            "model_path": sel["model_path"],
            "accuracy": sel["accuracy"],
            "auc_roc": sel["auc_roc"],
            "simulated_roi": sel["simulated_roi"],
        }

    with open(output_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    console.print(f"\n[green]Production manifest saved to {output_path}[/green]")


async def main():
    parser = argparse.ArgumentParser(description="ROYALEY Phase 6a: Promote Models")
    parser.add_argument("--scorecard", type=str, required=True,
                        help="Phase 2 scorecard CSV")
    parser.add_argument("--min-accuracy", type=float, default=0.55)
    parser.add_argument("--min-auc", type=float, default=0.53)
    parser.add_argument("--max-ece", type=float, default=0.12)
    parser.add_argument("--strategy", type=str, default="best_composite",
                        choices=["best_composite", "best_accuracy", "best_roi"])
    parser.add_argument("--dry-run", action="store_true",
                        help="Print selections without updating DB")
    parser.add_argument("--manifest-only", action="store_true",
                        help="Save manifest file only, skip DB update")

    args = parser.parse_args()

    # Select best models
    selections = select_production_models(
        args.scorecard,
        min_accuracy=args.min_accuracy,
        min_auc=args.min_auc,
        max_ece=args.max_ece,
        strategy=args.strategy,
    )

    if not selections:
        return

    # Print selections
    table = Table(title="Production Model Selections")
    table.add_column("Sport", style="cyan")
    table.add_column("Bet Type", style="blue")
    table.add_column("Framework", style="magenta")
    table.add_column("Accuracy", style="yellow")
    table.add_column("AUC", style="yellow")
    table.add_column("ROI", style="green")
    table.add_column("Composite", style="bold white")

    for key, sel in sorted(selections.items()):
        table.add_row(
            sel["sport"], sel["bet_type"], sel["framework"],
            f"{sel['accuracy']:.3f}", f"{sel['auc_roc']:.3f}",
            f"{sel['simulated_roi']*100:+.1f}%",
            f"{sel['composite_score']:.4f}",
        )

    console.print(table)
    console.print(f"\n  Total selections: {len(selections)}")

    # Save manifest (always)
    save_production_manifest(selections)

    # Update database
    if not args.manifest_only:
        promoted = await promote_to_database(selections, dry_run=args.dry_run)
        console.print(f"\n  Models promoted in DB: {promoted}")


if __name__ == "__main__":
    asyncio.run(main())
