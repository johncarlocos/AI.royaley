"""
ROYALEY - Live Feature Builder
Computes ML features for upcoming games from DB + odds data.

Builds the same 87 features used during training:
- Team rolling stats (last 10/5 games)
- Odds features (spread, total, moneyline)
- Game context (day, month, night game)
- Derived composite features
- H2H history
"""

import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

# Exact feature order expected by sklearn models (from scaler.feature_names_in_)
FEATURE_NAMES_87 = [
    "away_3_in_4_nights", "away_avg_margin_last10", "away_avg_pts_allowed_last10",
    "away_avg_pts_last10", "away_away_win_pct", "away_is_back_to_back",
    "away_is_revenge", "away_letdown_spot", "away_power_rating", "away_rest_days",
    "away_season_game_num", "away_streak", "away_win_pct_last10", "away_wins_last10",
    "away_wins_last5", "consensus_spread", "consensus_total", "day_of_week",
    "h2h_home_avg_margin", "h2h_home_wins_last5", "h2h_total_avg",
    "home_3_in_4_nights", "home_avg_margin_last10", "home_avg_pts_allowed_last10",
    "home_avg_pts_last10", "home_home_win_pct", "home_is_back_to_back",
    "home_is_revenge", "home_letdown_spot", "home_power_rating", "home_rest_days",
    "home_season_game_num", "home_streak", "home_win_pct_last10", "home_wins_last10",
    "home_wins_last5", "implied_home_prob", "is_night_game", "moneyline_away_close",
    "moneyline_home_close", "moneyline_home_open", "month", "no_vig_home_prob",
    "num_books", "power_rating_diff", "rest_advantage", "spread_close",
    "spread_movement", "spread_open", "total_close", "total_movement", "total_open",
    # Derived features (52-86)
    "momentum_diff", "recent_form_diff", "recent_form5_diff", "scoring_diff",
    "defense_diff", "margin_diff", "venue_strength_diff", "spread_value",
    "margin_value", "line_move_direction", "total_move_direction",
    "spread_move_magnitude", "revenge_edge", "rest_power_combo", "spot_danger",
    "combined_strength", "combined_value", "home_momentum_trend",
    "away_momentum_trend", "momentum_trend_diff", "win_pct_diff",
    "expected_margin_vs_spread", "scoring_sum", "defense_sum", "pace_proxy",
    "total_value", "offensive_mismatch", "total_line_move", "h2h_total_vs_line",
    "margin_sum", "rest_total", "b2b_fatigue_count", "has_odds", "has_spread_odds",
    "has_total_odds",
]


async def build_features_for_game(
    db: AsyncSession,
    sport_id: UUID,
    home_team_id: UUID,
    away_team_id: UUID,
    game_time: datetime,
    odds_data: Dict,
) -> Optional[Dict[str, float]]:
    """
    Build the full 87-feature vector for an upcoming game.
    
    Args:
        db: Database session
        sport_id: Sport UUID
        home_team_id: Home team UUID
        away_team_id: Away team UUID
        game_time: Scheduled game time
        odds_data: Dict with keys like consensus_spread, consensus_total, etc.
    
    Returns:
        Dict mapping feature name → value, or None on failure
    """
    try:
        # 1. Get team rolling stats
        home_stats = await _get_team_rolling_stats(db, sport_id, home_team_id, game_time, is_home=True)
        away_stats = await _get_team_rolling_stats(db, sport_id, away_team_id, game_time, is_home=False)
        
        # 2. Get H2H stats
        h2h = await _get_h2h_stats(db, sport_id, home_team_id, away_team_id, game_time)
        
        # 3. Build base features dict
        features = {}
        
        # === AWAY TEAM STATS ===
        features["away_3_in_4_nights"] = away_stats.get("three_in_four", 0)
        features["away_avg_margin_last10"] = away_stats.get("avg_margin_last10", 0)
        features["away_avg_pts_allowed_last10"] = away_stats.get("avg_pts_allowed_last10", 0)
        features["away_avg_pts_last10"] = away_stats.get("avg_pts_last10", 0)
        features["away_away_win_pct"] = away_stats.get("venue_win_pct", 0.5)
        features["away_is_back_to_back"] = away_stats.get("is_b2b", 0)
        features["away_is_revenge"] = 0  # Would need schedule analysis
        features["away_letdown_spot"] = 0  # Would need schedule analysis
        features["away_power_rating"] = away_stats.get("power_rating", 0)
        features["away_rest_days"] = away_stats.get("rest_days", 3)
        features["away_season_game_num"] = away_stats.get("season_game_num", 40)
        features["away_streak"] = away_stats.get("streak", 0)
        features["away_win_pct_last10"] = away_stats.get("win_pct_last10", 0.5)
        features["away_wins_last10"] = away_stats.get("wins_last10", 5)
        features["away_wins_last5"] = away_stats.get("wins_last5", 2.5)
        
        # === ODDS FEATURES ===
        consensus_spread = odds_data.get("consensus_spread", 0)
        consensus_total = odds_data.get("consensus_total", 0)
        spread_open = odds_data.get("spread_open", consensus_spread)
        spread_close = odds_data.get("spread_close", consensus_spread)
        total_open = odds_data.get("total_open", consensus_total)
        total_close = odds_data.get("total_close", consensus_total)
        ml_home = odds_data.get("moneyline_home_close", -110)
        ml_away = odds_data.get("moneyline_away_close", -110)
        ml_home_open = odds_data.get("moneyline_home_open", ml_home)
        num_books = odds_data.get("num_books", 5)
        
        features["consensus_spread"] = consensus_spread
        features["consensus_total"] = consensus_total
        
        # === GAME CONTEXT ===
        features["day_of_week"] = game_time.weekday()
        
        # === H2H ===
        features["h2h_home_avg_margin"] = h2h.get("home_avg_margin", 0)
        features["h2h_home_wins_last5"] = h2h.get("home_wins_last5", 2.5)
        features["h2h_total_avg"] = h2h.get("total_avg", consensus_total)
        
        # === HOME TEAM STATS ===
        features["home_3_in_4_nights"] = home_stats.get("three_in_four", 0)
        features["home_avg_margin_last10"] = home_stats.get("avg_margin_last10", 0)
        features["home_avg_pts_allowed_last10"] = home_stats.get("avg_pts_allowed_last10", 0)
        features["home_avg_pts_last10"] = home_stats.get("avg_pts_last10", 0)
        features["home_home_win_pct"] = home_stats.get("venue_win_pct", 0.5)
        features["home_is_back_to_back"] = home_stats.get("is_b2b", 0)
        features["home_is_revenge"] = 0
        features["home_letdown_spot"] = 0
        features["home_power_rating"] = home_stats.get("power_rating", 0)
        features["home_rest_days"] = home_stats.get("rest_days", 3)
        features["home_season_game_num"] = home_stats.get("season_game_num", 40)
        features["home_streak"] = home_stats.get("streak", 0)
        features["home_win_pct_last10"] = home_stats.get("win_pct_last10", 0.5)
        features["home_wins_last10"] = home_stats.get("wins_last10", 5)
        features["home_wins_last5"] = home_stats.get("wins_last5", 2.5)
        
        # === ODDS-DERIVED ===
        implied_home = _ml_to_implied(ml_home)
        implied_away = _ml_to_implied(ml_away)
        no_vig_home = implied_home / (implied_home + implied_away) if (implied_home + implied_away) > 0 else 0.5
        
        features["implied_home_prob"] = implied_home
        features["is_night_game"] = 1 if game_time.hour >= 18 else 0
        features["moneyline_away_close"] = ml_away
        features["moneyline_home_close"] = ml_home
        features["moneyline_home_open"] = ml_home_open
        features["month"] = game_time.month
        features["no_vig_home_prob"] = no_vig_home
        features["num_books"] = num_books
        features["power_rating_diff"] = features["home_power_rating"] - features["away_power_rating"]
        features["rest_advantage"] = features["home_rest_days"] - features["away_rest_days"]
        features["spread_close"] = spread_close
        features["spread_movement"] = spread_close - spread_open if spread_open else 0
        features["spread_open"] = spread_open
        features["total_close"] = total_close
        features["total_movement"] = total_close - total_open if total_open else 0
        features["total_open"] = total_open
        
        # === DERIVED FEATURES (52-86) ===
        features["momentum_diff"] = features["home_streak"] - features["away_streak"]
        features["recent_form_diff"] = features["home_wins_last10"] - features["away_wins_last10"]
        features["recent_form5_diff"] = features["home_wins_last5"] - features["away_wins_last5"]
        features["scoring_diff"] = features["home_avg_pts_last10"] - features["away_avg_pts_last10"]
        features["defense_diff"] = features["away_avg_pts_allowed_last10"] - features["home_avg_pts_allowed_last10"]
        features["margin_diff"] = features["home_avg_margin_last10"] - features["away_avg_margin_last10"]
        features["venue_strength_diff"] = features["home_home_win_pct"] - features["away_away_win_pct"]
        features["spread_value"] = features["power_rating_diff"] + features["spread_close"]
        features["margin_value"] = features["margin_diff"] + features["spread_close"]
        features["line_move_direction"] = np.sign(features["spread_movement"]) if features["spread_movement"] != 0 else 0
        features["total_move_direction"] = np.sign(features["total_movement"]) if features["total_movement"] != 0 else 0
        features["spread_move_magnitude"] = abs(features["spread_movement"])
        features["revenge_edge"] = int(features["home_is_revenge"]) - int(features["away_is_revenge"])
        features["rest_power_combo"] = features["rest_advantage"] * features["power_rating_diff"] / 10 if features["power_rating_diff"] != 0 else 0
        features["spot_danger"] = 0  # Would need schedule analysis
        
        # Combined strength (normalized)
        components = []
        pr_diff = features["power_rating_diff"]
        if pr_diff != 0:
            components.append(pr_diff / max(abs(pr_diff), 1))
        components.append(features["momentum_diff"] / 10)
        components.append(features["recent_form_diff"] / 10)
        features["combined_strength"] = sum(components) / max(len(components), 1)
        
        # Combined value
        value_comps = []
        sv = features["spread_value"]
        if sv != 0:
            value_comps.append(sv / max(abs(sv), 1))
        mv = features["margin_value"]
        if mv != 0:
            value_comps.append(mv / max(abs(mv), 1))
        features["combined_value"] = sum(value_comps) / max(len(value_comps), 1)
        
        # Momentum trends
        features["home_momentum_trend"] = features["home_wins_last5"] - (features["home_wins_last10"] / 2)
        features["away_momentum_trend"] = features["away_wins_last5"] - (features["away_wins_last10"] / 2)
        features["momentum_trend_diff"] = features["home_momentum_trend"] - features["away_momentum_trend"]
        features["win_pct_diff"] = features["home_win_pct_last10"] - features["away_win_pct_last10"]
        features["expected_margin_vs_spread"] = features["margin_diff"] + features["spread_close"]
        
        # Total-specific features
        features["scoring_sum"] = features["home_avg_pts_last10"] + features["away_avg_pts_last10"]
        features["defense_sum"] = features["home_avg_pts_allowed_last10"] + features["away_avg_pts_allowed_last10"]
        features["pace_proxy"] = (features["scoring_sum"] + features["defense_sum"]) / 2
        features["total_value"] = features["pace_proxy"] - features["consensus_total"] if features["consensus_total"] else 0
        features["offensive_mismatch"] = features["scoring_diff"] + features["defense_diff"]
        features["total_line_move"] = features["total_movement"]
        features["h2h_total_vs_line"] = features["h2h_total_avg"] - features["consensus_total"] if features["consensus_total"] else 0
        features["margin_sum"] = features["home_avg_margin_last10"] + features["away_avg_margin_last10"]
        features["rest_total"] = features["home_rest_days"] + features["away_rest_days"]
        features["b2b_fatigue_count"] = int(features["home_is_back_to_back"]) + int(features["away_is_back_to_back"])
        features["has_odds"] = 1 if num_books > 0 else 0
        features["has_spread_odds"] = 1 if consensus_spread != 0 else 0
        features["has_total_odds"] = 1 if consensus_total != 0 else 0
        
        # Validate we have all 87 features
        missing = [f for f in FEATURE_NAMES_87 if f not in features]
        if missing:
            logger.warning(f"Missing features (defaulting to 0): {missing}")
            for f in missing:
                features[f] = 0.0
        
        return features
        
    except Exception as e:
        logger.error(f"Feature building failed: {e}", exc_info=True)
        return None


async def _get_team_rolling_stats(
    db: AsyncSession,
    sport_id: UUID,
    team_id: UUID,
    before_date: datetime,
    is_home: bool,
) -> Dict:
    """
    Compute rolling stats for a team from their last completed games.
    """
    try:
        # Get last 10 completed games for this team
        result = await db.execute(
            text("""
                SELECT
                    g.home_team_id, g.away_team_id,
                    g.home_score, g.away_score,
                    g.scheduled_at
                FROM games g
                WHERE g.sport_id = :sport_id
                  AND (g.home_team_id = :team_id OR g.away_team_id = :team_id)
                  AND g.status = 'completed'
                  AND g.scheduled_at < :before
                  AND g.home_score IS NOT NULL
                  AND g.away_score IS NOT NULL
                ORDER BY g.scheduled_at DESC
                LIMIT 15
            """),
            {"sport_id": sport_id, "team_id": team_id, "before": before_date},
        )
        games = result.fetchall()
        
        if not games:
            return _default_team_stats()
        
        # Parse games into structured data
        parsed = []
        for g in games:
            is_home_team = (str(g.home_team_id) == str(team_id))
            team_score = g.home_score if is_home_team else g.away_score
            opp_score = g.away_score if is_home_team else g.home_score
            won = team_score > opp_score
            margin = team_score - opp_score
            parsed.append({
                "is_home_team": is_home_team,
                "team_score": team_score,
                "opp_score": opp_score,
                "won": won,
                "margin": margin,
                "date": g.scheduled_at,
            })
        
        last10 = parsed[:10]
        last5 = parsed[:5]
        
        # Wins
        wins_last10 = sum(1 for g in last10 if g["won"])
        wins_last5 = sum(1 for g in last5 if g["won"])
        win_pct_last10 = wins_last10 / max(len(last10), 1)
        
        # Scoring
        avg_pts_last10 = np.mean([g["team_score"] for g in last10]) if last10 else 0
        avg_pts_allowed_last10 = np.mean([g["opp_score"] for g in last10]) if last10 else 0
        avg_margin_last10 = np.mean([g["margin"] for g in last10]) if last10 else 0
        
        # Venue-specific win pct (home win % or away win %)
        if is_home:
            venue_games = [g for g in parsed if g["is_home_team"]]
        else:
            venue_games = [g for g in parsed if not g["is_home_team"]]
        venue_win_pct = sum(1 for g in venue_games if g["won"]) / max(len(venue_games), 1) if venue_games else 0.5
        
        # Streak (positive = winning, negative = losing)
        streak = 0
        if parsed:
            streak_dir = 1 if parsed[0]["won"] else -1
            for g in parsed:
                if (g["won"] and streak_dir > 0) or (not g["won"] and streak_dir < 0):
                    streak += streak_dir
                else:
                    break
        
        # Rest days
        rest_days = 3  # default
        if parsed:
            delta = before_date - parsed[0]["date"]
            rest_days = max(0, delta.days)
        
        # Back to back
        is_b2b = 1 if rest_days <= 1 else 0
        
        # 3 in 4 nights
        three_in_four = 0
        if len(parsed) >= 2:
            recent_3 = [g for g in parsed[:3] if (before_date - g["date"]).days <= 4]
            three_in_four = 1 if len(recent_3) >= 2 else 0
        
        # Season game number (count games this season - rough approximation)
        season_game_num = len(parsed)  # From the 15 we fetched, actual would be more
        
        # Power rating: weighted average margin (recent games weighted more)
        weights = [1.0 / (i + 1) for i in range(len(last10))]
        total_weight = sum(weights)
        power_rating = sum(g["margin"] * w for g, w in zip(last10, weights)) / total_weight if total_weight > 0 else 0
        
        return {
            "wins_last10": wins_last10,
            "wins_last5": wins_last5,
            "win_pct_last10": win_pct_last10,
            "avg_pts_last10": avg_pts_last10,
            "avg_pts_allowed_last10": avg_pts_allowed_last10,
            "avg_margin_last10": avg_margin_last10,
            "venue_win_pct": venue_win_pct,
            "streak": streak,
            "rest_days": rest_days,
            "is_b2b": is_b2b,
            "three_in_four": three_in_four,
            "season_game_num": season_game_num,
            "power_rating": power_rating,
        }
        
    except Exception as e:
        logger.error(f"Team stats failed for {team_id}: {e}")
        return _default_team_stats()


async def _get_h2h_stats(
    db: AsyncSession,
    sport_id: UUID,
    home_team_id: UUID,
    away_team_id: UUID,
    before_date: datetime,
) -> Dict:
    """Get head-to-head stats between two teams."""
    try:
        result = await db.execute(
            text("""
                SELECT home_team_id, home_score, away_score
                FROM games
                WHERE sport_id = :sport_id
                  AND ((home_team_id = :home AND away_team_id = :away)
                    OR (home_team_id = :away AND away_team_id = :home))
                  AND status = 'completed'
                  AND scheduled_at < :before
                  AND home_score IS NOT NULL
                ORDER BY scheduled_at DESC
                LIMIT 5
            """),
            {
                "sport_id": sport_id,
                "home": home_team_id,
                "away": away_team_id,
                "before": before_date,
            },
        )
        games = result.fetchall()
        
        if not games:
            return {"home_wins_last5": 2.5, "home_avg_margin": 0, "total_avg": 0}
        
        home_wins = 0
        margins = []
        totals = []
        for g in games:
            is_our_home = str(g.home_team_id) == str(home_team_id)
            if is_our_home:
                margin = g.home_score - g.away_score
                home_wins += 1 if margin > 0 else 0
            else:
                margin = g.away_score - g.home_score
                home_wins += 1 if margin > 0 else 0
            margins.append(margin)
            totals.append(g.home_score + g.away_score)
        
        return {
            "home_wins_last5": home_wins,
            "home_avg_margin": np.mean(margins) if margins else 0,
            "total_avg": np.mean(totals) if totals else 0,
        }
        
    except Exception as e:
        logger.error(f"H2H stats failed: {e}")
        return {"home_wins_last5": 2.5, "home_avg_margin": 0, "total_avg": 0}


def _default_team_stats() -> Dict:
    """Return neutral/default stats when no history available."""
    return {
        "wins_last10": 5,
        "wins_last5": 2.5,
        "win_pct_last10": 0.5,
        "avg_pts_last10": 0,
        "avg_pts_allowed_last10": 0,
        "avg_margin_last10": 0,
        "venue_win_pct": 0.5,
        "streak": 0,
        "rest_days": 3,
        "is_b2b": 0,
        "three_in_four": 0,
        "season_game_num": 40,
        "power_rating": 0,
    }


def _ml_to_implied(american_odds: float) -> float:
    """Convert American odds to implied probability."""
    if american_odds is None or american_odds == 0:
        return 0.5
    if american_odds > 0:
        return 100 / (american_odds + 100)
    else:
        return abs(american_odds) / (abs(american_odds) + 100)


def features_to_array(features: Dict[str, float]) -> np.ndarray:
    """Convert feature dict to numpy array in correct column order."""
    return np.array([[features.get(f, 0.0) for f in FEATURE_NAMES_87]])