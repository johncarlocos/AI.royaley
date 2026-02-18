"""
ROYALEY - Probability Calibrator
Trains and applies post-hoc probability calibration using actual prediction results.

Problem:
  Raw model outputs are systematically overconfident at the top:
    Tier A (claimed 58%+) → actual 53.8%
    Tier B (claimed 55-58%) → actual 50.9%
    Tier C (claimed 52-55%) → actual 55.6%

Solution:
  1. Pull all graded predictions with their raw probability + actual result
  2. Fit Platt scaling (logistic regression): P(win | raw_prob)
  3. Apply as post-processing step in model_loader.py

Usage:
    # Train calibrator on historical results
    docker exec royaley_api python -m app.pipeline.calibrator --train

    # Show reliability diagram (diagnostic)
    docker exec royaley_api python -m app.pipeline.calibrator --diagnose

    # Test: map a raw probability through calibrator
    docker exec royaley_api python -m app.pipeline.calibrator --test 0.58
"""

import asyncio
import argparse
import logging
import pickle
import sys
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from app.core.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("royaley.calibrator")

CALIBRATOR_PATH = Path("/app/models/global_calibrator.pkl")
MIN_SAMPLES_FOR_TRAINING = 50  # Need at least this many graded predictions


# =============================================================================
# PLATT SCALING CALIBRATOR
# =============================================================================

class PlattCalibrator:
    """
    Platt scaling: fits a logistic regression P(win | raw_prob) = sigmoid(a * raw_prob + b).
    
    More stable than isotonic regression for small samples (< 500).
    Automatically handles overconfidence by learning the mapping from
    predicted probabilities to actual win rates.
    """
    
    def __init__(self):
        self.a: float = 1.0  # slope (< 1 = shrink toward 50%)
        self.b: float = 0.0  # intercept
        self.n_samples: int = 0
        self.trained: bool = False
        self.reliability_bins: Dict = {}  # diagnostic info
        self.trained_at: Optional[str] = None
    
    def fit(self, raw_probs: np.ndarray, outcomes: np.ndarray) -> "PlattCalibrator":
        """
        Fit Platt scaling parameters using maximum likelihood.
        
        Args:
            raw_probs: array of predicted probabilities (0-1)
            outcomes: array of binary outcomes (1 = win, 0 = loss)
        """
        from datetime import datetime, timezone
        
        n = len(raw_probs)
        if n < MIN_SAMPLES_FOR_TRAINING:
            logger.warning(f"Only {n} samples (need {MIN_SAMPLES_FOR_TRAINING}). Using fallback shrinkage.")
            self.a = 0.667  # matches existing SHRINKAGE_FACTOR
            self.b = 0.5 * (1 - 0.667)  # centers at 0.5
            self.n_samples = n
            return self
        
        # Platt scaling: fit logistic regression on log-odds space
        # P(y=1 | x) = sigmoid(a * logit(x) + b)
        # where logit(x) = log(x / (1-x))
        
        # Clip to avoid log(0)
        eps = 1e-4
        clipped = np.clip(raw_probs, eps, 1 - eps)
        logits = np.log(clipped / (1 - clipped))
        
        # Use scipy if available, else gradient descent
        try:
            from scipy.optimize import minimize
            
            def neg_log_likelihood(params):
                a, b = params
                z = a * logits + b
                # Numerically stable sigmoid
                z_clipped = np.clip(z, -30, 30)
                p = 1 / (1 + np.exp(-z_clipped))
                p = np.clip(p, eps, 1 - eps)
                nll = -np.mean(outcomes * np.log(p) + (1 - outcomes) * np.log(1 - p))
                return nll
            
            result = minimize(neg_log_likelihood, x0=[1.0, 0.0], method='Nelder-Mead')
            self.a, self.b = result.x
            
        except ImportError:
            # Fallback: simple gradient descent
            a, b = 1.0, 0.0
            lr = 0.01
            for _ in range(2000):
                z = a * logits + b
                z_clipped = np.clip(z, -30, 30)
                p = 1 / (1 + np.exp(-z_clipped))
                p = np.clip(p, eps, 1 - eps)
                
                # Gradients
                err = p - outcomes
                grad_a = np.mean(err * logits)
                grad_b = np.mean(err)
                
                a -= lr * grad_a
                b -= lr * grad_b
            
            self.a, self.b = a, b
        
        self.n_samples = n
        self.trained = True
        self.trained_at = datetime.now(timezone.utc).isoformat()
        
        # Compute reliability bins for diagnostics
        self._compute_reliability(raw_probs, outcomes)
        
        logger.info(f"Platt calibrator trained: a={self.a:.4f}, b={self.b:.4f} on {n} samples")
        return self
    
    def calibrate(self, raw_prob: float) -> float:
        """Apply calibration to a single probability."""
        eps = 1e-4
        clipped = np.clip(raw_prob, eps, 1 - eps)
        logit = np.log(clipped / (1 - clipped))
        z = self.a * logit + self.b
        z_clipped = np.clip(z, -30, 30)
        calibrated = 1 / (1 + np.exp(-z_clipped))
        
        # Hard caps: no sports prediction should exceed these bounds
        return float(np.clip(calibrated, 0.38, 0.62))
    
    def calibrate_batch(self, raw_probs: np.ndarray) -> np.ndarray:
        """Apply calibration to an array of probabilities."""
        return np.array([self.calibrate(p) for p in raw_probs])
    
    def _compute_reliability(self, raw_probs: np.ndarray, outcomes: np.ndarray):
        """Compute reliability diagram bins."""
        bins = [(0.40, 0.45), (0.45, 0.50), (0.50, 0.525), (0.525, 0.55),
                (0.55, 0.58), (0.58, 0.62), (0.62, 1.0)]
        
        self.reliability_bins = {}
        for lo, hi in bins:
            mask = (raw_probs >= lo) & (raw_probs < hi)
            n = mask.sum()
            if n > 0:
                actual = outcomes[mask].mean()
                predicted = raw_probs[mask].mean()
                calibrated = self.calibrate_batch(raw_probs[mask]).mean()
                self.reliability_bins[f"{lo:.3f}-{hi:.3f}"] = {
                    "n": int(n),
                    "predicted": round(float(predicted), 4),
                    "actual": round(float(actual), 4),
                    "calibrated": round(float(calibrated), 4),
                    "gap": round(float(predicted - actual), 4),
                }
    
    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Platt Calibrator: a={self.a:.4f}, b={self.b:.4f}",
            f"  Trained on {self.n_samples} samples at {self.trained_at or 'N/A'}",
            f"  Effect: {'shrinks toward 50%' if abs(self.a) < 1 else 'expands from 50%'}",
            "",
            "  Reliability Diagram:",
            f"  {'Bin':<14} {'N':>4} {'Predicted':>10} {'Actual':>8} {'Calibrated':>11} {'Gap':>6}",
            f"  {'-'*55}",
        ]
        for bin_name, info in self.reliability_bins.items():
            gap_str = f"{info['gap']:+.1%}"
            lines.append(
                f"  {bin_name:<14} {info['n']:>4} {info['predicted']:>10.1%} "
                f"{info['actual']:>8.1%} {info['calibrated']:>11.1%} {gap_str:>6}"
            )
        
        # Show mapping examples
        lines.extend(["", "  Example mappings (raw → calibrated):"])
        for p in [0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62]:
            cal = self.calibrate(p)
            lines.append(f"    {p:.0%} → {cal:.1%}")
        
        return "\n".join(lines)


# =============================================================================
# LOAD / SAVE
# =============================================================================

def load_calibrator(path: Optional[Path] = None) -> Optional[PlattCalibrator]:
    """Load calibrator from disk."""
    p = path or CALIBRATOR_PATH
    if p.exists():
        try:
            with open(p, "rb") as f:
                cal = pickle.load(f)
            if isinstance(cal, PlattCalibrator) and cal.trained:
                logger.info(f"Loaded calibrator: a={cal.a:.4f}, b={cal.b:.4f} ({cal.n_samples} samples)")
                return cal
        except Exception as e:
            logger.warning(f"Failed to load calibrator from {p}: {e}")
    return None


def save_calibrator(cal: PlattCalibrator, path: Optional[Path] = None):
    """Save calibrator to disk."""
    p = path or CALIBRATOR_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        pickle.dump(cal, f)
    logger.info(f"Saved calibrator to {p}")


# =============================================================================
# TRAINING: Pull data from DB + fit calibrator
# =============================================================================

async def train_calibrator() -> PlattCalibrator:
    """
    Pull all graded predictions from DB and train a global calibrator.
    
    Uses the 'probability' column (model's predicted probability) as raw input,
    and 'actual_result' (win/loss) as the target.
    """
    engine = create_async_engine(settings.DATABASE_URL, echo=False)
    async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    async with async_session() as db:
        result = await db.execute(text("""
            SELECT 
                p.probability,
                p.odds_at_prediction,
                p.bet_type,
                p.signal_tier,
                pr.actual_result,
                s.code as sport_code
            FROM predictions p
            JOIN prediction_results pr ON pr.prediction_id = p.id
            JOIN upcoming_games ug ON ug.id = p.upcoming_game_id
            JOIN sports s ON s.id = ug.sport_id
            WHERE pr.actual_result IN ('win', 'loss')
              AND p.probability IS NOT NULL
              AND p.probability > 0 AND p.probability < 1
            ORDER BY pr.graded_at DESC
        """))
        rows = result.fetchall()
    
    await engine.dispose()
    
    if len(rows) < MIN_SAMPLES_FOR_TRAINING:
        logger.warning(f"Only {len(rows)} graded predictions (need {MIN_SAMPLES_FOR_TRAINING}). "
                       f"Calibrator will use fallback parameters.")
    
    raw_probs = np.array([float(r.probability) for r in rows])
    outcomes = np.array([1.0 if r.actual_result == 'win' else 0.0 for r in rows])
    
    logger.info(f"Training calibrator on {len(rows)} graded predictions")
    logger.info(f"  Overall: {outcomes.mean():.1%} win rate, avg predicted: {raw_probs.mean():.1%}")
    
    # Log per-tier breakdown
    tiers = {}
    for r in rows:
        t = r.signal_tier or 'D'
        if t not in tiers:
            tiers[t] = {'probs': [], 'outcomes': []}
        tiers[t]['probs'].append(float(r.probability))
        tiers[t]['outcomes'].append(1.0 if r.actual_result == 'win' else 0.0)
    
    for t in sorted(tiers.keys()):
        n = len(tiers[t]['outcomes'])
        wr = np.mean(tiers[t]['outcomes'])
        ap = np.mean(tiers[t]['probs'])
        logger.info(f"  Tier {t}: n={n}, predicted={ap:.1%}, actual={wr:.1%}, gap={ap-wr:+.1%}")
    
    # Fit calibrator
    cal = PlattCalibrator()
    cal.fit(raw_probs, outcomes)
    
    # Save
    save_calibrator(cal)
    
    # Print summary
    print("\n" + "=" * 60)
    print("CALIBRATOR TRAINING COMPLETE")
    print("=" * 60)
    print(cal.summary())
    print("=" * 60)
    
    return cal


async def diagnose():
    """Print diagnostic info about current calibrator and data."""
    cal = load_calibrator()
    if cal:
        print("\n" + "=" * 60)
        print("CURRENT CALIBRATOR")
        print("=" * 60)
        print(cal.summary())
    else:
        print("No calibrator found. Run --train first.")
    
    # Also show raw data stats
    engine = create_async_engine(settings.DATABASE_URL, echo=False)
    async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    async with async_session() as db:
        result = await db.execute(text("""
            SELECT 
                p.signal_tier,
                COUNT(*) as n,
                ROUND(AVG(p.probability)::numeric, 4) as avg_predicted,
                ROUND(AVG(CASE WHEN pr.actual_result = 'win' THEN 1.0 ELSE 0.0 END)::numeric, 4) as actual_winrate,
                ROUND(AVG(p.probability)::numeric - AVG(CASE WHEN pr.actual_result = 'win' THEN 1.0 ELSE 0.0 END)::numeric, 4) as overconfidence
            FROM predictions p
            JOIN prediction_results pr ON pr.prediction_id = p.id
            WHERE pr.actual_result IN ('win', 'loss')
            GROUP BY p.signal_tier
            ORDER BY p.signal_tier
        """))
        rows = result.fetchall()
    
    await engine.dispose()
    
    if rows:
        print("\n" + "=" * 60)
        print("RAW PREDICTION ACCURACY BY TIER")
        print("=" * 60)
        print(f"  {'Tier':<8} {'N':>5} {'Predicted':>10} {'Actual':>8} {'Gap':>8}")
        print(f"  {'-'*42}")
        for r in rows:
            print(f"  {r.signal_tier:<8} {r.n:>5} {float(r.avg_predicted):>10.1%} "
                  f"{float(r.actual_winrate):>8.1%} {float(r.overconfidence):>+8.1%}")


def test_calibrator(raw_prob: float):
    """Test what a raw probability maps to after calibration."""
    cal = load_calibrator()
    if not cal:
        print("No calibrator found. Run --train first.")
        return
    
    calibrated = cal.calibrate(raw_prob)
    print(f"Raw: {raw_prob:.1%} → Calibrated: {calibrated:.1%}")
    print(f"  (Platt a={cal.a:.4f}, b={cal.b:.4f})")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="ROYALEY Probability Calibrator")
    parser.add_argument("--train", action="store_true", help="Train calibrator on historical results")
    parser.add_argument("--diagnose", action="store_true", help="Show reliability diagram and diagnostics")
    parser.add_argument("--test", type=float, metavar="PROB", help="Test calibration of a probability (0-1)")
    args = parser.parse_args()
    
    if args.train:
        asyncio.run(train_calibrator())
    elif args.diagnose:
        asyncio.run(diagnose())
    elif args.test is not None:
        test_calibrator(args.test)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()