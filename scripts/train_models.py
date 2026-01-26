#!/usr/bin/env python3
"""
ROYALEY - Model Training Script
Phase 2: CLI for ML model training

Usage:
    # Train single model
    python train_models.py --sport NFL --bet-type spread --framework h2o
    
    # Train all models for a sport
    python train_models.py --sport NFL --all-bet-types
    
    # Train all models for all sports
    python train_models.py --all
    
    # Use mock trainers (for testing)
    python train_models.py --sport NFL --bet-type spread --mock
"""

import asyncio
import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from app.core.config import settings
from app.services.ml.training_service import (
    TrainingService,
    TrainingResult,
    get_training_service,
)

console = Console()
logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


BET_TYPES = ["spread", "moneyline", "total"]
FRAMEWORKS = ["h2o", "sklearn", "autogluon"]


async def train_single(
    sport_code: str,
    bet_type: str,
    framework: str,
    use_mock: bool = False,
    **kwargs,
) -> TrainingResult:
    """Train a single model."""
    service = get_training_service(use_mock=use_mock)
    
    console.print(f"[cyan]Training {sport_code} {bet_type} using {framework}...[/cyan]")
    
    result = await service.train_model(
        sport_code=sport_code,
        bet_type=bet_type,
        framework=framework,
        **kwargs,
    )
    
    return result


async def train_sport(
    sport_code: str,
    framework: str,
    use_mock: bool = False,
    **kwargs,
) -> List[TrainingResult]:
    """Train all bet types for a sport."""
    results = []
    
    for bet_type in BET_TYPES:
        result = await train_single(
            sport_code=sport_code,
            bet_type=bet_type,
            framework=framework,
            use_mock=use_mock,
            **kwargs,
        )
        results.append(result)
    
    return results


async def train_all(
    framework: str,
    sport_codes: List[str] = None,
    use_mock: bool = False,
    **kwargs,
) -> List[TrainingResult]:
    """Train all models for all sports."""
    sport_codes = sport_codes or settings.SUPPORTED_SPORTS
    results = []
    
    total = len(sport_codes) * len(BET_TYPES)
    current = 0
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Training models (0/{total})", total=total)
        
        for sport_code in sport_codes:
            for bet_type in BET_TYPES:
                progress.update(
                    task,
                    description=f"Training {sport_code} {bet_type} ({current+1}/{total})"
                )
                
                result = await train_single(
                    sport_code=sport_code,
                    bet_type=bet_type,
                    framework=framework,
                    use_mock=use_mock,
                    **kwargs,
                )
                results.append(result)
                current += 1
                progress.advance(task)
    
    return results


def print_results(results: List[TrainingResult]):
    """Print training results table."""
    table = Table(title="Training Results")
    
    table.add_column("Sport", style="cyan")
    table.add_column("Bet Type", style="blue")
    table.add_column("Framework", style="magenta")
    table.add_column("Status", style="green")
    table.add_column("AUC", style="yellow")
    table.add_column("Accuracy", style="yellow")
    table.add_column("Samples", style="white")
    table.add_column("Time (s)", style="white")
    
    successful = 0
    failed = 0
    
    for r in results:
        status = "✅ Success" if r.success else f"❌ {r.error_message[:30]}..."
        
        if r.success:
            successful += 1
        else:
            failed += 1
        
        table.add_row(
            r.sport_code,
            r.bet_type,
            r.framework,
            status if r.success else f"[red]{status}[/red]",
            f"{r.auc:.4f}" if r.success else "-",
            f"{r.accuracy:.4f}" if r.success else "-",
            str(r.training_samples) if r.success else "-",
            f"{r.training_duration_seconds:.1f}" if r.success else "-",
        )
    
    console.print(table)
    console.print(f"\n[green]Successful: {successful}[/green] | [red]Failed: {failed}[/red]")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="ROYALEY Model Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_models.py --sport NFL --bet-type spread
  python train_models.py --sport NBA --all-bet-types
  python train_models.py --all --framework sklearn
  python train_models.py --sport NFL --bet-type spread --mock
        """
    )
    
    parser.add_argument(
        "--sport", "-s",
        type=str,
        help="Sport code (NFL, NBA, NHL, etc.)"
    )
    parser.add_argument(
        "--bet-type", "-b",
        type=str,
        choices=BET_TYPES,
        help="Bet type (spread, moneyline, total)"
    )
    parser.add_argument(
        "--framework", "-f",
        type=str,
        choices=FRAMEWORKS,
        default="h2o",
        help="ML framework (default: h2o)"
    )
    parser.add_argument(
        "--all-bet-types",
        action="store_true",
        help="Train all bet types for the sport"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Train all models for all sports"
    )
    parser.add_argument(
        "--sports",
        type=str,
        nargs="+",
        help="Specific sports to train (with --all)"
    )
    parser.add_argument(
        "--max-runtime",
        type=int,
        default=3600,
        help="Maximum training time in seconds (default: 3600)"
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=500,
        help="Minimum training samples required (default: 500)"
    )
    parser.add_argument(
        "--no-walk-forward",
        action="store_true",
        help="Disable walk-forward validation"
    )
    parser.add_argument(
        "--no-calibrate",
        action="store_true",
        help="Disable probability calibration"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save to database (dry run)"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock trainers (for testing)"
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
    
    if args.sport and not args.bet_type and not args.all_bet_types:
        parser.print_help()
        console.print("\n[yellow]Specify --bet-type or --all-bet-types[/yellow]")
        return
    
    # Print header
    console.print(Panel(
        f"[bold green]ROYALEY Model Training[/bold green]\n"
        f"Framework: {args.framework}\n"
        f"Max Runtime: {args.max_runtime}s\n"
        f"Mock Mode: {'Yes' if args.mock else 'No'}",
        title="Training Configuration"
    ))
    
    # Training options
    train_kwargs = {
        "max_runtime_secs": args.max_runtime,
        "min_samples": args.min_samples,
        "use_walk_forward": not args.no_walk_forward,
        "calibrate": not args.no_calibrate,
        "save_to_db": not args.no_save,
    }
    
    start_time = datetime.now()
    results = []
    
    try:
        if args.all:
            # Train all
            results = await train_all(
                framework=args.framework,
                sport_codes=args.sports,
                use_mock=args.mock,
                **train_kwargs,
            )
        elif args.all_bet_types:
            # Train all bet types for sport
            results = await train_sport(
                sport_code=args.sport,
                framework=args.framework,
                use_mock=args.mock,
                **train_kwargs,
            )
        else:
            # Train single model
            result = await train_single(
                sport_code=args.sport,
                bet_type=args.bet_type,
                framework=args.framework,
                use_mock=args.mock,
                **train_kwargs,
            )
            results = [result]
        
        # Print results
        print_results(results)
        
        # Print summary
        duration = (datetime.now() - start_time).total_seconds()
        console.print(f"\n[cyan]Total time: {duration:.1f}s[/cyan]")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Training interrupted[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        if args.verbose:
            import traceback
            traceback.print_exc()
    finally:
        # Cleanup
        service = get_training_service()
        service.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
