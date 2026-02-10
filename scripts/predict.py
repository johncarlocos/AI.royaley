#!/usr/bin/env python3
"""
ROYALEY - Phase 6b: Production Prediction Engine
Generates predictions using ACTUAL TRAINED MODELS (not ELO fallback).

Loads models from /nvme0n1-disk/royaley/models, applies calibration,
combines via optimized ensemble weights, classifies signal tiers,
and computes Kelly sizing.

Usage:
    python scripts/predict.py --sport NFL
    python scripts/predict.py --sport NBA --date 2025-02-15
    python scripts/predict.py --all --min-edge 0.03 --tier A
    python scripts/predict.py --all --output predictions.json
"""

import asyncio
import argparse
import logging
import sys
import pickle
import json
import traceback
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from app.core.config import settings
from app.core.database import db_manager
from app.models import (
    Sport, Game, Team, Odds, MLModel, Prediction as PredictionModel,
    GameStatus, GameFeature,
)
from sqlalchemy import select, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

console = Console()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ============================================================================
# CONFIGURATION
# ============================================================================

MODELS_DIR = Path("/app/models")
BET_TYPES = ["spread", "moneyline", "total"]
FRAMEWORKS = ["h2o", "sklearn", "autogluon", "tensorflow", "quantum"]

# Signal tier thresholds
TIER_THRESHOLDS = {
    "A": {"min_prob": 0.65, "min_edge": 0.10, "kelly_mult": 1.0, "max_bet_pct": 0.020},
    "B": {"min_prob": 0.60, "min_edge": 0.05, "kelly_mult": 0.75, "max_bet_pct": 0.015},
    "C": {"min_prob": 0.55, "min_edge": 0.02, "kelly_mult": 0.50, "max_bet_pct": 0.010},
    "D": {"min_prob": 0.00, "min_edge": 0.00, "kelly_mult": 0.00, "max_bet_pct": 0.000},
}


# ============================================================================
# MODEL LOADER — Loads actual trained models from disk
# ============================================================================

class ProductionModelLoader:
    """
    Loads and caches production models for prediction.
    Replaces the old ELO-based fallback with real trained models.
    """

    def __init__(self, models_dir: Path = MODELS_DIR):
        self.models_dir = models_dir
        self._model_cache: Dict[str, Any] = {}
        self._scaler_cache: Dict[str, Any] = {}
        self._calibrator_cache: Dict[str, Any] = {}
        self._ensemble_config_cache: Dict[str, Dict] = {}
        self._manifest: Dict = {}

        # Load production manifest if available
        manifest_path = Path("production_manifest.json")
        if manifest_path.exists():
            with open(manifest_path) as f:
                self._manifest = json.load(f).get("models", {})
            logger.info(f"Loaded production manifest: {len(self._manifest)} models")

    def get_ensemble_config(self, sport: str, bet_type: str) -> Optional[Dict]:
        """Load ensemble configuration for sport/bet_type."""
        key = f"{sport}/{bet_type}"
        if key in self._ensemble_config_cache:
            return self._ensemble_config_cache[key]

        config_path = self.models_dir / sport / bet_type / "ensemble_config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            self._ensemble_config_cache[key] = config
            return config

        return None

    def load_model(self, framework: str, sport: str, bet_type: str) -> Optional[Any]:
        """Load a trained model from disk."""
        cache_key = f"{framework}/{sport}/{bet_type}"
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]

        model = None
        model_dir = self.models_dir / framework / sport / bet_type

        try:
            if framework == "sklearn":
                model_path = model_dir / "model.pkl"
                if model_path.exists():
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)

            elif framework == "tensorflow":
                import tensorflow as tf
                dense_path = model_dir / "dense_model"
                if dense_path.exists():
                    model = tf.keras.models.load_model(str(dense_path))
                else:
                    for ext in [".h5", ".keras"]:
                        files = list(model_dir.glob(f"*{ext}"))
                        if files:
                            model = tf.keras.models.load_model(str(files[0]))
                            break

            elif framework == "h2o":
                import h2o
                if not h2o.cluster():
                    h2o.init(nthreads=-1, max_mem_size="4G", verbose=False)

                mojo_dir = model_dir / "mojo"
                if mojo_dir.exists():
                    mojo_files = list(mojo_dir.glob("*.zip"))
                    if mojo_files:
                        model = h2o.import_mojo(str(mojo_files[0]))
                if model is None:
                    # Load best model (prefer StackedEnsemble)
                    for d in sorted(model_dir.iterdir()):
                        if d.is_dir() and "StackedEnsemble" in d.name:
                            model = h2o.load_model(str(d))
                            break
                    if model is None:
                        for d in sorted(model_dir.iterdir()):
                            if d.is_dir() and d.name not in ["mojo"]:
                                model = h2o.load_model(str(d))
                                break

            elif framework == "autogluon":
                from autogluon.tabular import TabularPredictor
                if model_dir.exists():
                    model = TabularPredictor.load(str(model_dir))

            elif framework == "quantum":
                model_path = model_dir / "model.pkl"
                if model_path.exists():
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)

        except Exception as e:
            logger.warning(f"Failed to load {cache_key}: {e}")
            return None

        if model is not None:
            self._model_cache[cache_key] = model
            logger.info(f"Loaded model: {cache_key}")

        return model

    def load_scaler(self, framework: str, sport: str, bet_type: str) -> Optional[Any]:
        """Load scaler for a model."""
        cache_key = f"{framework}/{sport}/{bet_type}"
        if cache_key in self._scaler_cache:
            return self._scaler_cache[cache_key]

        scaler_path = self.models_dir / framework / sport / bet_type / "scaler.pkl"
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            self._scaler_cache[cache_key] = scaler
            return scaler
        return None

    def load_calibrator(self, framework: str, sport: str, bet_type: str) -> Optional[Dict]:
        """Load probability calibrator."""
        cache_key = f"{framework}/{sport}/{bet_type}"
        if cache_key in self._calibrator_cache:
            return self._calibrator_cache[cache_key]

        # Try framework-specific calibrator first
        cal_path = self.models_dir / framework / sport / bet_type / "calibrator.pkl"
        if not cal_path.exists():
            # Fallback to sport-level calibrator
            cal_path = self.models_dir / sport / bet_type / "calibrator.pkl"

        if cal_path.exists():
            with open(cal_path, 'rb') as f:
                cal_data = pickle.load(f)
            self._calibrator_cache[cache_key] = cal_data
            return cal_data
        return None

    def predict_single_framework(
        self, framework: str, sport: str, bet_type: str,
        features: np.ndarray, feature_names: List[str] = None,
    ) -> Optional[np.ndarray]:
        """Generate predictions from a single framework model."""

        model = self.load_model(framework, sport, bet_type)
        if model is None:
            return None

        X = features.copy()

        # Apply scaler if exists
        scaler = self.load_scaler(framework, sport, bet_type)
        if scaler is not None and framework in ["sklearn", "tensorflow", "quantum"]:
            try:
                X = scaler.transform(X)
            except Exception:
                pass

        # Replace NaN/inf
        X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)

        try:
            if framework == "sklearn":
                if hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(X)[:, 1]
                else:
                    probs = model.predict(X)

            elif framework == "tensorflow":
                probs = model.predict(X.astype(np.float32), verbose=0)
                if probs.ndim == 2 and probs.shape[1] == 1:
                    probs = probs.ravel()
                elif probs.ndim == 2 and probs.shape[1] == 2:
                    probs = probs[:, 1]

            elif framework == "h2o":
                import h2o
                h2o_df = h2o.H2OFrame(pd.DataFrame(features, columns=feature_names) if feature_names else features)
                preds = model.predict(h2o_df).as_data_frame()
                if 'p1' in preds.columns:
                    probs = preds['p1'].values
                elif preds.shape[1] >= 3:
                    probs = preds.iloc[:, 2].values
                else:
                    probs = preds.iloc[:, 0].values

            elif framework == "autogluon":
                df_input = pd.DataFrame(features, columns=feature_names) if feature_names else pd.DataFrame(features)
                pred_proba = model.predict_proba(df_input)
                if isinstance(pred_proba, pd.DataFrame) and pred_proba.shape[1] == 2:
                    probs = pred_proba.iloc[:, 1].values
                else:
                    probs = pred_proba.values.ravel()

            elif framework == "quantum":
                if hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(X)
                    if probs.ndim == 2:
                        probs = probs[:, 1]
                else:
                    probs = model.predict(X)

            else:
                return None

            probs = np.clip(np.asarray(probs, dtype=float), 0.001, 0.999)

            # Apply calibration
            cal_data = self.load_calibrator(framework, sport, bet_type)
            if cal_data:
                method = cal_data.get("method", "isotonic")
                calibrator = cal_data["calibrator"]
                try:
                    if method == "isotonic":
                        probs = np.clip(calibrator.predict(probs), 0.001, 0.999)
                    elif method == "platt":
                        probs = np.clip(calibrator.predict_proba(probs.reshape(-1, 1))[:, 1], 0.001, 0.999)
                    elif method == "temperature":
                        temp = calibrator["temperature"]
                        probs = np.clip(probs ** (1.0 / temp), 0.001, 0.999)
                except Exception as e:
                    logger.warning(f"Calibration failed for {framework}/{sport}/{bet_type}: {e}")

            return probs

        except Exception as e:
            logger.warning(f"Prediction failed for {framework}/{sport}/{bet_type}: {e}")
            return None

    def predict_ensemble(
        self, sport: str, bet_type: str,
        features: np.ndarray, feature_names: List[str] = None,
    ) -> Optional[Tuple[np.ndarray, Dict[str, float], Dict[str, np.ndarray]]]:
        """
        Generate ensemble predictions combining all available frameworks.
        Returns (combined_probs, weights_used, framework_predictions).
        """
        config = self.get_ensemble_config(sport, bet_type)

        # Collect predictions from each framework
        fw_predictions = {}
        for fw in FRAMEWORKS:
            preds = self.predict_single_framework(fw, sport, bet_type, features, feature_names)
            if preds is not None:
                fw_predictions[fw] = preds

        if not fw_predictions:
            return None

        # Determine weights
        if config and config.get("weights"):
            weights = config["weights"]
        else:
            # Equal weights for available frameworks
            weights = {fw: 1.0 / len(fw_predictions) for fw in fw_predictions}

        # Normalize weights to available frameworks
        active_weights = {fw: w for fw, w in weights.items() if fw in fw_predictions}
        total_w = sum(active_weights.values())
        if total_w <= 0:
            active_weights = {fw: 1.0 / len(fw_predictions) for fw in fw_predictions}
            total_w = 1.0

        # Weighted combination
        combined = sum(
            (w / total_w) * fw_predictions[fw]
            for fw, w in active_weights.items()
        )
        combined = np.clip(combined, 0.001, 0.999)

        return combined, active_weights, fw_predictions


# ============================================================================
# PREDICTION GENERATION
# ============================================================================

def american_to_prob(odds: int) -> float:
    """Convert American odds to implied probability."""
    if odds is None:
        return 0.524  # Standard -110
    if odds > 0:
        return 100.0 / (odds + 100.0)
    else:
        return abs(odds) / (abs(odds) + 100.0)


def kelly_criterion(pred_prob: float, implied_prob: float, fraction: float = 0.25) -> float:
    """Fractional Kelly Criterion."""
    if implied_prob <= 0 or implied_prob >= 1:
        return 0.0
    decimal_odds = 1.0 / implied_prob
    b = decimal_odds - 1.0
    p = pred_prob
    q = 1.0 - p
    if b <= 0:
        return 0.0
    kelly_full = (b * p - q) / b
    return max(0.0, kelly_full * fraction)


def classify_signal_tier(pred_prob: float, edge: float) -> str:
    """Classify prediction into signal tier."""
    for tier in ["A", "B", "C", "D"]:
        thresh = TIER_THRESHOLDS[tier]
        if pred_prob >= thresh["min_prob"] and edge >= thresh["min_edge"]:
            return tier
    return "D"


async def get_upcoming_games(
    session: AsyncSession, sport_code: str, target_date: date,
) -> List[Game]:
    """Get upcoming games for a sport on target date."""
    result = await session.execute(
        select(Sport).where(Sport.code == sport_code)
    )
    sport = result.scalar_one_or_none()
    if not sport:
        return []

    start_dt = datetime.combine(target_date, datetime.min.time())
    end_dt = datetime.combine(target_date, datetime.max.time())

    query = (
        select(Game)
        .where(and_(
            Game.sport_id == sport.id,
            Game.scheduled_at >= start_dt,
            Game.scheduled_at <= end_dt,
            Game.status == GameStatus.SCHEDULED,
        ))
        .order_by(Game.scheduled_at)
    )
    result = await session.execute(query)
    return result.scalars().all()


async def get_game_features(
    session: AsyncSession, game_id, feature_names: List[str] = None,
) -> Optional[Dict[str, float]]:
    """Get feature vector for a game from database."""
    result = await session.execute(
        select(GameFeature).where(GameFeature.game_id == game_id)
    )
    features = result.scalars().all()

    if not features:
        return None

    feature_dict = {}
    for f in features:
        feature_dict[f.feature_name] = f.feature_value

    return feature_dict


async def get_game_odds(
    session: AsyncSession, game_id, bet_type: str,
) -> Optional[Odds]:
    """Get latest odds for a game."""
    bt_variants = [bet_type]
    if bet_type == "spread":
        bt_variants.append("spreads")
    elif bet_type == "total":
        bt_variants.append("totals")

    query = (
        select(Odds)
        .where(and_(
            Odds.game_id == game_id,
            Odds.bet_type.in_(bt_variants),
        ))
        .order_by(Odds.recorded_at.desc())
        .limit(1)
    )
    result = await session.execute(query)
    return result.scalar_one_or_none()


async def generate_predictions(
    sport_code: str,
    target_date: date = None,
    bet_types: List[str] = None,
    models_dir: Path = MODELS_DIR,
    kelly_fraction: float = 0.25,
    bankroll: float = 10000.0,
) -> List[Dict[str, Any]]:
    """Generate predictions for all games of a sport on target date."""

    target_date = target_date or date.today()
    bet_types = bet_types or BET_TYPES
    predictions = []

    loader = ProductionModelLoader(models_dir)
    await db_manager.initialize()

    async with db_manager.session() as session:
        games = await get_upcoming_games(session, sport_code, target_date)
        if not games:
            console.print(f"[yellow]No games found for {sport_code} on {target_date}[/yellow]")
            return []

        console.print(f"[cyan]Found {len(games)} games for {sport_code}[/cyan]")

        for game in games:
            # Get teams
            home_team = await session.get(Team, game.home_team_id)
            away_team = await session.get(Team, game.away_team_id)
            if not home_team or not away_team:
                continue

            # Get features for this game
            game_features = await get_game_features(session, game.id)

            for bet_type in bet_types:
                # Get odds
                odds = await get_game_odds(session, game.id, bet_type)

                # Determine implied probability from odds
                if odds:
                    if bet_type in ["spread", "moneyline"]:
                        home_implied = american_to_prob(odds.home_odds)
                        away_implied = american_to_prob(odds.away_odds)
                    else:
                        home_implied = american_to_prob(getattr(odds, 'over_odds', -110))
                        away_implied = american_to_prob(getattr(odds, 'under_odds', -110))
                else:
                    home_implied = 0.524  # Standard -110
                    away_implied = 0.524

                # Build feature vector
                if game_features:
                    feature_names = sorted(game_features.keys())
                    feature_values = np.array([[game_features[k] for k in feature_names]])
                else:
                    # Fallback: build minimal features from team data
                    feature_names = ["home_elo", "away_elo", "elo_diff", "home_advantage"]
                    feature_values = np.array([[
                        home_team.elo_rating or 1500,
                        away_team.elo_rating or 1500,
                        (home_team.elo_rating or 1500) - (away_team.elo_rating or 1500),
                        1.0,
                    ]])

                # Get ensemble prediction from REAL MODELS
                ensemble_result = loader.predict_ensemble(
                    sport_code, bet_type, feature_values, feature_names
                )

                if ensemble_result:
                    combined_prob, weights_used, fw_preds = ensemble_result
                    pred_prob = float(combined_prob[0])
                    framework_used = "ensemble"
                else:
                    # No models available — skip (DO NOT fall back to ELO)
                    logger.warning(
                        f"No models available for {sport_code}/{bet_type}. Skipping."
                    )
                    continue

                # Determine predicted side
                if bet_type == "moneyline":
                    predicted_side = "home" if pred_prob > 0.5 else "away"
                    display_prob = pred_prob if pred_prob > 0.5 else (1.0 - pred_prob)
                elif bet_type == "spread":
                    predicted_side = "home" if pred_prob > 0.5 else "away"
                    display_prob = pred_prob if pred_prob > 0.5 else (1.0 - pred_prob)
                else:  # total
                    predicted_side = "over" if pred_prob > 0.5 else "under"
                    display_prob = pred_prob if pred_prob > 0.5 else (1.0 - pred_prob)

                # Calculate edge
                if predicted_side in ["home", "over"]:
                    implied = home_implied
                else:
                    implied = away_implied
                edge = display_prob - implied

                # Signal tier
                tier = classify_signal_tier(display_prob, edge)

                # Kelly sizing
                tier_config = TIER_THRESHOLDS[tier]
                kelly_size = kelly_criterion(display_prob, implied, kelly_fraction * tier_config["kelly_mult"])
                kelly_size = min(kelly_size, tier_config["max_bet_pct"])
                recommended_stake = bankroll * kelly_size if kelly_size > 0 else 0

                predictions.append({
                    "game_id": str(game.id),
                    "scheduled_at": game.scheduled_at.isoformat(),
                    "home_team": home_team.name,
                    "away_team": away_team.name,
                    "home_abbr": home_team.abbreviation or home_team.name[:3].upper(),
                    "away_abbr": away_team.abbreviation or away_team.name[:3].upper(),
                    "bet_type": bet_type,
                    "predicted_side": predicted_side,
                    "probability": round(display_prob, 4),
                    "implied_probability": round(implied, 4),
                    "edge": round(edge, 4),
                    "signal_tier": tier,
                    "kelly_fraction": round(kelly_size, 4),
                    "recommended_stake": round(recommended_stake, 2),
                    "framework": framework_used,
                    "weights_used": {k: round(v, 3) for k, v in weights_used.items()} if ensemble_result else {},
                    "line": getattr(odds, 'home_line', None) if odds and bet_type == "spread" else (
                        getattr(odds, 'total', None) if odds and bet_type == "total" else None
                    ),
                    "odds": odds.home_odds if odds and predicted_side in ["home", "over"] else (
                        odds.away_odds if odds else -110
                    ),
                })

    return predictions


def print_predictions(predictions: List[Dict], sport_code: str):
    """Print predictions table with full details."""
    if not predictions:
        console.print("[yellow]No predictions generated[/yellow]")
        return

    table = Table(title=f"{sport_code} Predictions — Using Trained Models")
    table.add_column("Time", style="cyan")
    table.add_column("Matchup", style="white")
    table.add_column("Bet", style="blue")
    table.add_column("Pick", style="green")
    table.add_column("Prob", style="yellow")
    table.add_column("Edge", style="magenta")
    table.add_column("Tier", style="red")
    table.add_column("Kelly", style="dim")
    table.add_column("Stake", style="bold")

    for pred in predictions:
        dt = datetime.fromisoformat(pred["scheduled_at"])
        time_str = dt.strftime("%I:%M %p")
        matchup = f"{pred['away_abbr']} @ {pred['home_abbr']}"

        if pred["bet_type"] == "spread" and pred.get("line"):
            pick = f"{pred['predicted_side'].upper()} {pred['line']}"
        elif pred["bet_type"] == "total" and pred.get("line"):
            pick = f"{pred['predicted_side'].upper()} {pred['line']}"
        else:
            pick = pred['predicted_side'].upper()

        edge_str = f"{pred['edge']*100:+.1f}%"
        if pred["edge"] >= 0.05:
            edge_str = f"[green]{edge_str}[/green]"
        elif pred["edge"] >= 0.02:
            edge_str = f"[yellow]{edge_str}[/yellow]"
        else:
            edge_str = f"[red]{edge_str}[/red]"

        tier = pred["signal_tier"]
        tier_colors = {"A": "bold green", "B": "green", "C": "yellow", "D": "red"}
        tier_str = f"[{tier_colors.get(tier, 'white')}]{tier}[/{tier_colors.get(tier, 'white')}]"

        stake_str = f"${pred['recommended_stake']:,.0f}" if pred['recommended_stake'] > 0 else "-"

        table.add_row(
            time_str, matchup, pred["bet_type"], pick,
            f"{pred['probability']*100:.1f}%", edge_str, tier_str,
            f"{pred['kelly_fraction']*100:.2f}%", stake_str,
        )

    console.print(table)

    # Summary
    tier_counts = {}
    total_stake = 0
    for p in predictions:
        t = p["signal_tier"]
        tier_counts[t] = tier_counts.get(t, 0) + 1
        total_stake += p["recommended_stake"]

    console.print(
        f"\n[cyan]Signal Tiers:[/cyan] " +
        ", ".join(f"{t}: {c}" for t, c in sorted(tier_counts.items())) +
        f" | Total Recommended Stake: ${total_stake:,.2f}"
    )


async def main():
    parser = argparse.ArgumentParser(
        description="ROYALEY Production Prediction Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--sport", "-s", type=str, help="Sport code")
    parser.add_argument("--all", "-a", action="store_true", help="All sports")
    parser.add_argument("--date", "-d", type=str, help="Target date YYYY-MM-DD")
    parser.add_argument("--bet-type", "-b", type=str, choices=BET_TYPES)
    parser.add_argument("--output", "-o", type=str, help="Output JSON file")
    parser.add_argument("--min-edge", type=float, default=0.0)
    parser.add_argument("--tier", type=str, choices=["A", "B", "C", "D"])
    parser.add_argument("--models-dir", type=str, default=str(MODELS_DIR))
    parser.add_argument("--kelly", type=float, default=0.25)
    parser.add_argument("--bankroll", type=float, default=10000.0)
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.all and not args.sport:
        parser.print_help()
        console.print("\n[yellow]Specify --sport or --all[/yellow]")
        return

    target_date = None
    if args.date:
        target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
    else:
        target_date = date.today()

    sport_codes = [args.sport.upper()] if args.sport else settings.SUPPORTED_SPORTS
    bet_types = [args.bet_type] if args.bet_type else BET_TYPES

    console.print(Panel(
        f"[bold green]ROYALEY Predictions — TRAINED MODELS[/bold green]\n"
        f"Date: {target_date}\n"
        f"Sports: {', '.join(sport_codes)}\n"
        f"Bet Types: {', '.join(bet_types)}\n"
        f"Models: {args.models_dir}\n"
        f"Kelly: {args.kelly} | Bankroll: ${args.bankroll:,.0f}",
        title="Prediction Configuration"
    ))

    all_predictions = []

    try:
        for sport_code in sport_codes:
            predictions = await generate_predictions(
                sport_code=sport_code,
                target_date=target_date,
                bet_types=bet_types,
                models_dir=Path(args.models_dir),
                kelly_fraction=args.kelly,
                bankroll=args.bankroll,
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

        # Save output
        if args.output and all_predictions:
            with open(args.output, 'w') as f:
                json.dump(all_predictions, f, indent=2)
            console.print(f"[green]Predictions saved to {args.output}[/green]")

        console.print(Panel(
            f"[bold]Total Predictions: {len(all_predictions)}[/bold]",
            title="Summary"
        ))

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        if args.verbose:
            traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())