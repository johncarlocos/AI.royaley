"""
ROYALEY - Public Predictions API
No authentication required - read-only access for frontend dashboard.
Returns opening snapshot (from predictions table) + current consensus (from upcoming_odds).
"""
from datetime import datetime, date, timedelta
from typing import Optional, List
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db

router = APIRouter(tags=["public"])

class PublicPrediction(BaseModel):
    id: str
    game_id: str
    sport: Optional[str] = None
    home_team: Optional[str] = None
    away_team: Optional[str] = None
    game_time: Optional[str] = None
    bet_type: str
    predicted_side: str
    probability: float
    edge: Optional[float] = None
    signal_tier: Optional[str] = None
    line_at_prediction: Optional[float] = None
    odds_at_prediction: Optional[int] = None
    kelly_fraction: Optional[float] = None
    prediction_hash: Optional[str] = None
    created_at: Optional[str] = None
    result: Optional[str] = "pending"
    clv: Optional[float] = None
    profit_loss: Optional[float] = None
    # Opening snapshot (both sides, captured at prediction time)
    home_line_open: Optional[float] = None
    away_line_open: Optional[float] = None
    home_odds_open: Optional[int] = None
    away_odds_open: Optional[int] = None
    total_open: Optional[float] = None
    over_odds_open: Optional[int] = None
    under_odds_open: Optional[int] = None
    home_ml_open: Optional[int] = None
    away_ml_open: Optional[int] = None
    # Current consensus (latest from upcoming_odds, Pinnacle preferred)
    current_home_line: Optional[float] = None
    current_away_line: Optional[float] = None
    current_home_odds: Optional[int] = None
    current_away_odds: Optional[int] = None
    current_total: Optional[float] = None
    current_over_odds: Optional[int] = None
    current_under_odds: Optional[int] = None
    current_home_ml: Optional[int] = None
    current_away_ml: Optional[int] = None
    # Team records (season W-L)
    home_record: Optional[str] = None
    away_record: Optional[str] = None

class PublicPredictionsResponse(BaseModel):
    predictions: List[PublicPrediction]
    total: int
    page: int
    per_page: int

class DashboardStats(BaseModel):
    total_predictions: int = 0
    tier_a_count: int = 0
    pending_count: int = 0
    graded_today: int = 0
    win_rate: float = 0.0
    roi: float = 0.0
    top_picks: list = []
    recent_activity: list = []
    best_performers: list = []
    areas_to_monitor: list = []


# ---- Unified game join: upcoming_games first, fallback to legacy games ----
GAME_JOIN = """
    LEFT JOIN upcoming_games ug ON p.upcoming_game_id = ug.id
    LEFT JOIN games g ON p.game_id = g.id
    JOIN sports s ON s.id = COALESCE(ug.sport_id, g.sport_id)
"""

TEAM_NAMES = """
    COALESCE(ug.home_team_name, ht_legacy.name) as home_team,
    COALESCE(ug.away_team_name, at_legacy.name) as away_team,
"""

TEAM_JOIN_LEGACY = """
    LEFT JOIN teams ht_legacy ON g.home_team_id = ht_legacy.id
    LEFT JOIN teams at_legacy ON g.away_team_id = at_legacy.id
"""

GAME_TIME = "COALESCE(ug.scheduled_at, g.scheduled_at) as game_time"

# CTE for current consensus odds (Pinnacle preferred, fallback to average)
CURRENT_ODDS_CTE = """
WITH current_odds AS (
    SELECT 
        upcoming_game_id,
        bet_type,
        COALESCE(
            MAX(CASE WHEN sportsbook_key = 'pinnacle' THEN home_line END),
            AVG(home_line)
        ) as curr_home_line,
        COALESCE(
            MAX(CASE WHEN sportsbook_key = 'pinnacle' THEN away_line END),
            AVG(away_line)
        ) as curr_away_line,
        COALESCE(
            MAX(CASE WHEN sportsbook_key = 'pinnacle' THEN home_odds END),
            AVG(home_odds)
        ) as curr_home_odds,
        COALESCE(
            MAX(CASE WHEN sportsbook_key = 'pinnacle' THEN away_odds END),
            AVG(away_odds)
        ) as curr_away_odds,
        COALESCE(
            MAX(CASE WHEN sportsbook_key = 'pinnacle' THEN total END),
            AVG(total)
        ) as curr_total,
        COALESCE(
            MAX(CASE WHEN sportsbook_key = 'pinnacle' THEN over_odds END),
            AVG(over_odds)
        ) as curr_over_odds,
        COALESCE(
            MAX(CASE WHEN sportsbook_key = 'pinnacle' THEN under_odds END),
            AVG(under_odds)
        ) as curr_under_odds,
        COALESCE(
            MAX(CASE WHEN sportsbook_key = 'pinnacle' THEN home_ml END),
            AVG(home_ml)
        ) as curr_home_ml,
        COALESCE(
            MAX(CASE WHEN sportsbook_key = 'pinnacle' THEN away_ml END),
            AVG(away_ml)
        ) as curr_away_ml
    FROM upcoming_odds
    GROUP BY upcoming_game_id, bet_type
),
team_records AS (
    SELECT team_id, sport_id,
        SUM(CASE WHEN won THEN 1 ELSE 0 END) as wins,
        SUM(CASE WHEN NOT won THEN 1 ELSE 0 END) as losses
    FROM (
        SELECT home_team_id as team_id, g.sport_id, home_score > away_score as won
        FROM games g
        JOIN sports s ON s.id = g.sport_id
        WHERE home_score IS NOT NULL AND away_score IS NOT NULL
            AND scheduled_at >= CASE s.code
                WHEN 'NFL' THEN DATE '2025-09-01'
                WHEN 'NCAAF' THEN DATE '2025-08-15'
                WHEN 'NBA' THEN DATE '2025-10-15'
                WHEN 'NCAAB' THEN DATE '2025-11-01'
                WHEN 'NHL' THEN DATE '2025-10-01'
                WHEN 'MLB' THEN DATE '2025-03-20'
                WHEN 'WNBA' THEN DATE '2025-05-01'
                WHEN 'CFL' THEN DATE '2025-06-01'
                WHEN 'ATP' THEN DATE '2025-01-01'
                WHEN 'WTA' THEN DATE '2025-01-01'
                ELSE NOW() - INTERVAL '180 days'
            END
        UNION ALL
        SELECT away_team_id as team_id, g.sport_id, away_score > home_score as won
        FROM games g
        JOIN sports s ON s.id = g.sport_id
        WHERE home_score IS NOT NULL AND away_score IS NOT NULL
            AND scheduled_at >= CASE s.code
                WHEN 'NFL' THEN DATE '2025-09-01'
                WHEN 'NCAAF' THEN DATE '2025-08-15'
                WHEN 'NBA' THEN DATE '2025-10-15'
                WHEN 'NCAAB' THEN DATE '2025-11-01'
                WHEN 'NHL' THEN DATE '2025-10-01'
                WHEN 'MLB' THEN DATE '2025-03-20'
                WHEN 'WNBA' THEN DATE '2025-05-01'
                WHEN 'CFL' THEN DATE '2025-06-01'
                WHEN 'ATP' THEN DATE '2025-01-01'
                WHEN 'WTA' THEN DATE '2025-01-01'
                ELSE NOW() - INTERVAL '180 days'
            END
    ) t
    GROUP BY team_id, sport_id
)
"""


def _safe_int(val):
    """Safely convert to int, handling None and float."""
    if val is None:
        return None
    try:
        return int(round(float(val)))
    except (ValueError, TypeError):
        return None

def _safe_float(val):
    """Safely convert to float, handling None."""
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None

def _snap_line(val):
    """Round line to nearest 0.5 (standard sportsbook precision)."""
    if val is None:
        return None
    try:
        return round(float(val) * 2) / 2
    except (ValueError, TypeError):
        return None


@router.get("/predictions", response_model=PublicPredictionsResponse)
async def get_public_predictions(
    sport: Optional[str] = Query(None),
    bet_type: Optional[str] = Query(None),
    signal_tier: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    per_page: int = Query(100, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
):
    where_clauses = []
    params = {}
    if sport:
        where_clauses.append("s.code = :sport")
        params["sport"] = sport.upper()
    if bet_type:
        where_clauses.append("p.bet_type = :bet_type")
        params["bet_type"] = bet_type
    if signal_tier:
        where_clauses.append("CAST(p.signal_tier AS TEXT) = :signal_tier")
        params["signal_tier"] = signal_tier.upper()
    where_sql = ("AND " + " AND ".join(where_clauses)) if where_clauses else ""

    count_result = await db.execute(text(f"""
        SELECT COUNT(*)
        FROM predictions p
        {GAME_JOIN}
        {TEAM_JOIN_LEGACY}
        WHERE (p.upcoming_game_id IS NOT NULL OR p.game_id IS NOT NULL)
        {where_sql}
    """), params)
    total = count_result.scalar() or 0

    offset = (page - 1) * per_page
    data_params = {**params, "lim": per_page, "off": offset}
    result = await db.execute(text(f"""
        {CURRENT_ODDS_CTE}
        SELECT
            p.id,
            COALESCE(p.upcoming_game_id, p.game_id) as game_id,
            s.code as sport_code,
            {TEAM_NAMES}
            {GAME_TIME},
            p.bet_type, p.predicted_side, p.probability, p.edge,
            CAST(p.signal_tier AS TEXT) as signal_tier,
            p.line_at_prediction, p.odds_at_prediction,
            p.kelly_fraction, p.prediction_hash, p.created_at,
            -- Opening snapshot (from predictions table)
            p.home_line_open, p.away_line_open,
            p.home_odds_open, p.away_odds_open,
            p.total_open, p.over_odds_open, p.under_odds_open,
            p.home_ml_open, p.away_ml_open,
            -- Current consensus (from upcoming_odds CTE)
            co.curr_home_line, co.curr_away_line,
            co.curr_home_odds, co.curr_away_odds,
            co.curr_total, co.curr_over_odds, co.curr_under_odds,
            co.curr_home_ml, co.curr_away_ml,
            -- Grading results (from prediction_results)
            CAST(pr.actual_result AS TEXT) as actual_result,
            pr.clv as result_clv,
            pr.profit_loss,
            pr.closing_line as result_closing_line,
            pr.closing_odds as result_closing_odds,
            -- Team records
            htr.wins as home_wins, htr.losses as home_losses,
            atr.wins as away_wins, atr.losses as away_losses
        FROM predictions p
        {GAME_JOIN}
        {TEAM_JOIN_LEGACY}
        LEFT JOIN current_odds co 
            ON co.upcoming_game_id = p.upcoming_game_id 
            AND co.bet_type = p.bet_type
        LEFT JOIN prediction_results pr ON pr.prediction_id = p.id
        LEFT JOIN team_records htr ON htr.team_id = COALESCE(ug.home_team_id, g.home_team_id) AND htr.sport_id = COALESCE(ug.sport_id, g.sport_id)
        LEFT JOIN team_records atr ON atr.team_id = COALESCE(ug.away_team_id, g.away_team_id) AND atr.sport_id = COALESCE(ug.sport_id, g.sport_id)
        WHERE (p.upcoming_game_id IS NOT NULL OR p.game_id IS NOT NULL)
        {where_sql}
        ORDER BY game_time ASC, p.created_at DESC
        LIMIT :lim OFFSET :off
    """), data_params)

    predictions = []
    for row in result.fetchall():
        predictions.append(PublicPrediction(
            id=str(row.id),
            game_id=str(row.game_id),
            sport=row.sport_code,
            home_team=row.home_team,
            away_team=row.away_team,
            game_time=(row.game_time.isoformat() + 'Z') if row.game_time else None,
            bet_type=row.bet_type,
            predicted_side=row.predicted_side,
            probability=float(row.probability),
            edge=_safe_float(row.edge),
            signal_tier=row.signal_tier,
            line_at_prediction=_snap_line(row.line_at_prediction),
            odds_at_prediction=_safe_int(row.odds_at_prediction),
            kelly_fraction=_safe_float(row.kelly_fraction),
            prediction_hash=row.prediction_hash,
            created_at=(row.created_at.isoformat() + 'Z') if row.created_at else None,
            result=row.actual_result if row.actual_result and row.actual_result != 'pending' else "pending",
            clv=_safe_float(row.result_clv),
            profit_loss=_safe_float(row.profit_loss),
            # Opening snapshot
            home_line_open=_snap_line(row.home_line_open),
            away_line_open=_snap_line(row.away_line_open),
            home_odds_open=_safe_int(row.home_odds_open),
            away_odds_open=_safe_int(row.away_odds_open),
            total_open=_snap_line(row.total_open),
            over_odds_open=_safe_int(row.over_odds_open),
            under_odds_open=_safe_int(row.under_odds_open),
            home_ml_open=_safe_int(row.home_ml_open),
            away_ml_open=_safe_int(row.away_ml_open),
            # Current consensus
            current_home_line=_snap_line(row.curr_home_line),
            current_away_line=_snap_line(row.curr_away_line),
            current_home_odds=_safe_int(row.curr_home_odds),
            current_away_odds=_safe_int(row.curr_away_odds),
            current_total=_snap_line(row.curr_total),
            current_over_odds=_safe_int(row.curr_over_odds),
            current_under_odds=_safe_int(row.curr_under_odds),
            current_home_ml=_safe_int(row.curr_home_ml),
            current_away_ml=_safe_int(row.curr_away_ml),
            home_record=f"{row.home_wins}-{row.home_losses}" if row.home_wins is not None else None,
            away_record=f"{row.away_wins}-{row.away_losses}" if row.away_wins is not None else None,
        ))

    return PublicPredictionsResponse(
        predictions=predictions, total=total, page=page, per_page=per_page
    )


@router.get("/dashboard/stats", response_model=DashboardStats)
async def get_dashboard_stats(db: AsyncSession = Depends(get_db)):
    today = date.today()

    total_predictions = (await db.execute(
        text("SELECT COUNT(*) FROM predictions")
    )).scalar() or 0

    tier_a_count = (await db.execute(
        text("SELECT COUNT(*) FROM predictions WHERE CAST(signal_tier AS TEXT) = 'A'")
    )).scalar() or 0

    # All predictions are pending until grading is implemented
    graded_count = (await db.execute(
        text("SELECT COUNT(*) FROM prediction_results WHERE actual_result != 'pending'")
    )).scalar() or 0
    pending_count = total_predictions - graded_count

    wins = (await db.execute(
        text("SELECT COUNT(*) FROM prediction_results WHERE actual_result = 'win'")
    )).scalar() or 0
    losses = (await db.execute(
        text("SELECT COUNT(*) FROM prediction_results WHERE actual_result = 'loss'")
    )).scalar() or 0

    win_rate = round(wins / (wins + losses) * 100, 1) if (wins + losses) > 0 else 0.0

    graded_today = (await db.execute(
        text("SELECT COUNT(*) FROM prediction_results WHERE actual_result != 'pending' AND graded_at::date = CURRENT_DATE")
    )).scalar() or 0

    avg_clv = (await db.execute(
        text("SELECT AVG(clv) FROM prediction_results WHERE clv IS NOT NULL")
    )).scalar()

    total_pnl = (await db.execute(
        text("SELECT SUM(profit_loss) FROM prediction_results WHERE profit_loss IS NOT NULL")
    )).scalar()
    roi = round((total_pnl or 0) / max(graded_count * 100, 1) * 100, 1) if graded_count > 0 else 0.0

    # Top picks (upcoming, highest probability)
    top_rows = (await db.execute(text(f"""
        SELECT
            p.id, s.code as sport,
            {TEAM_NAMES}
            p.bet_type, p.predicted_side, p.probability, p.edge,
            CAST(p.signal_tier AS TEXT) as signal_tier,
            p.line_at_prediction,
            {GAME_TIME}
        FROM predictions p
        {GAME_JOIN}
        {TEAM_JOIN_LEGACY}
        WHERE COALESCE(ug.scheduled_at, g.scheduled_at) >= NOW() - INTERVAL '2 hours'
        ORDER BY p.probability DESC
        LIMIT 6
    """))).fetchall()

    top_picks = []
    for row in top_rows:
        side = row.predicted_side or ""
        line = row.line_at_prediction
        if row.bet_type == "spread":
            team = row.home_team if side == "home" else row.away_team
            pick_str = f"{team} {'+' if line and line > 0 else ''}{line}" if line else team
        elif row.bet_type == "total":
            pick_str = f"{'Over' if side == 'over' else 'Under'} {line}" if line else side.capitalize()
        else:
            pick_str = f"{row.home_team if side == 'home' else row.away_team} ML"
        top_picks.append({
            "sport": row.sport,
            "game": f"{row.away_team} vs {row.home_team}",
            "pick": pick_str,
            "tier": row.signal_tier or "D",
            "probability": float(row.probability),
            "time": row.game_time.strftime("%-I:%M %p") if row.game_time else "",
        })

    return DashboardStats(
        total_predictions=total_predictions,
        tier_a_count=tier_a_count,
        pending_count=pending_count,
        graded_today=graded_today,
        win_rate=win_rate,
        roi=roi,
        top_picks=top_picks,
        recent_activity=[],
        best_performers=[],
        areas_to_monitor=[],
    )


@router.get("/upcoming")
async def get_upcoming_games(
    sport: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
):
    """List upcoming games with odds and prediction counts."""
    where_clauses = []
    params = {}
    if sport:
        where_clauses.append("s.code = :sport")
        params["sport"] = sport.upper()
    where_sql = ("AND " + " AND ".join(where_clauses)) if where_clauses else ""

    result = await db.execute(text(f"""
        SELECT
            ug.id, s.code as sport, ug.home_team_name, ug.away_team_name,
            ug.scheduled_at,
            (SELECT COUNT(*) FROM upcoming_odds uo WHERE uo.upcoming_game_id = ug.id) as odds_count,
            (SELECT COUNT(*) FROM predictions p WHERE p.upcoming_game_id = ug.id) as prediction_count
        FROM upcoming_games ug
        JOIN sports s ON ug.sport_id = s.id
        WHERE ug.status = 'scheduled'
          AND ug.scheduled_at >= NOW() - INTERVAL '2 hours'
        {where_sql}
        ORDER BY ug.scheduled_at ASC
    """), params)

    games = []
    for row in result.fetchall():
        games.append({
            "id": str(row.id),
            "sport": row.sport,
            "home_team": row.home_team_name,
            "away_team": row.away_team_name,
            "scheduled_at": (row.scheduled_at.isoformat() + 'Z') if row.scheduled_at else None,
            "odds_count": row.odds_count,
            "prediction_count": row.prediction_count,
        })

    return {"games": games, "total": len(games)}


# ============================================================================
# BETTING SUMMARY - Public endpoint for Betting page
# ============================================================================

class BettingSummaryBet(BaseModel):
    id: str
    game_id: str
    sport: Optional[str] = None
    home_team: Optional[str] = None
    away_team: Optional[str] = None
    game_time: Optional[str] = None
    bet_type: str
    predicted_side: str
    pick_team: Optional[str] = None  # 'home' or 'away'
    line: Optional[float] = None
    odds: Optional[int] = None
    probability: float
    edge: Optional[float] = None
    signal_tier: Optional[str] = None
    stake: float = 100.0
    result: Optional[str] = "pending"
    profit_loss: Optional[float] = None
    clv: Optional[float] = None


class BettingSummaryStats(BaseModel):
    initial_bankroll: float = 10000.0
    current_bankroll: float = 10000.0
    total_bets: int = 0
    graded_bets: int = 0
    pending_bets: int = 0
    wins: int = 0
    losses: int = 0
    pushes: int = 0
    win_rate: float = 0.0
    roi: float = 0.0
    avg_edge: float = 0.0
    avg_clv: float = 0.0
    total_pnl: float = 0.0


class BettingSummaryResponse(BaseModel):
    stats: BettingSummaryStats
    bets: List[BettingSummaryBet]
    equity_curve: List[dict]


@router.get("/betting-summary", response_model=BettingSummaryResponse)
async def get_betting_summary(
    sport: Optional[str] = Query(None),
    tiers: Optional[str] = Query(None, description="Comma-separated tiers, e.g. A,B"),
    stake: float = Query(100.0, ge=1),
    initial_bankroll: float = Query(10000.0, ge=100),
    db: AsyncSession = Depends(get_db),
):
    """
    Public betting summary. Treats each prediction as a flat-stake bet.
    Returns stats, bet list, and equity curve for the bankroll chart.
    """
    where_clauses = []
    params: dict = {}
    if sport:
        where_clauses.append("s.code = :sport")
        params["sport"] = sport.upper()
    if tiers:
        tier_list = [t.strip().upper() for t in tiers.split(",") if t.strip()]
        if tier_list:
            placeholders = ", ".join([f":tier_{i}" for i in range(len(tier_list))])
            where_clauses.append(f"CAST(p.signal_tier AS TEXT) IN ({placeholders})")
            for i, t in enumerate(tier_list):
                params[f"tier_{i}"] = t
    where_sql = ("AND " + " AND ".join(where_clauses)) if where_clauses else ""

    result = await db.execute(text(f"""
        SELECT
            p.id,
            COALESCE(p.upcoming_game_id, p.game_id) as game_id,
            s.code as sport_code,
            {TEAM_NAMES}
            {GAME_TIME},
            p.bet_type, p.predicted_side, p.probability, p.edge,
            CAST(p.signal_tier AS TEXT) as signal_tier,
            p.line_at_prediction, p.odds_at_prediction,
            -- Determine pick_team from predicted_side
            CASE
                WHEN p.predicted_side ILIKE '%%' || COALESCE(ug.home_team_name, ht_legacy.name) || '%%' THEN 'home'
                WHEN p.predicted_side ILIKE '%%' || COALESCE(ug.away_team_name, at_legacy.name) || '%%' THEN 'away'
                WHEN p.predicted_side ILIKE 'over%%' THEN 'over'
                WHEN p.predicted_side ILIKE 'under%%' THEN 'under'
                ELSE NULL
            END as pick_team,
            -- Grading results
            CAST(pr.actual_result AS TEXT) as actual_result,
            pr.clv as result_clv,
            pr.profit_loss,
            pr.graded_at
        FROM predictions p
        {GAME_JOIN}
        {TEAM_JOIN_LEGACY}
        LEFT JOIN prediction_results pr ON pr.prediction_id = p.id
        WHERE (p.upcoming_game_id IS NOT NULL OR p.game_id IS NOT NULL)
        {where_sql}
        ORDER BY game_time ASC, p.created_at ASC
    """), params)

    bets = []
    graded_bets_for_curve = []
    total_wins = 0
    total_losses = 0
    total_pushes = 0
    total_pending = 0
    total_pnl = 0.0
    sum_edge = 0.0
    sum_clv = 0.0
    clv_count = 0

    for row in result.fetchall():
        actual = row.actual_result if row.actual_result and row.actual_result != 'pending' else "pending"
        edge_val = _safe_float(row.edge) or 0.0
        pnl = _safe_float(row.profit_loss) or 0.0
        clv_val = _safe_float(row.result_clv)

        if actual == 'win':
            total_wins += 1
        elif actual == 'loss':
            total_losses += 1
        elif actual == 'push':
            total_pushes += 1
        else:
            total_pending += 1

        if actual != 'pending':
            total_pnl += pnl
            if clv_val is not None:
                sum_clv += clv_val
                clv_count += 1
            graded_bets_for_curve.append({
                "graded_at": row.graded_at,
                "pnl": pnl,
                "game_time": row.game_time,
            })

        sum_edge += edge_val

        # Determine pick_team for frontend highlighting
        pick_team = row.pick_team
        if pick_team in ('over', 'under'):
            pick_team = None  # totals don't highlight a team

        bets.append(BettingSummaryBet(
            id=str(row.id),
            game_id=str(row.game_id),
            sport=row.sport_code,
            home_team=row.home_team,
            away_team=row.away_team,
            game_time=(row.game_time.isoformat() + 'Z') if row.game_time else None,
            bet_type=row.bet_type,
            predicted_side=row.predicted_side,
            pick_team=pick_team,
            line=_snap_line(row.line_at_prediction),
            odds=_safe_int(row.odds_at_prediction),
            probability=float(row.probability),
            edge=edge_val,
            signal_tier=row.signal_tier,
            stake=stake,
            result=actual,
            profit_loss=pnl if actual != 'pending' else None,
            clv=clv_val,
        ))

    total_bets = len(bets)
    graded_count = total_wins + total_losses + total_pushes
    decided = total_wins + total_losses
    win_rate = round(total_wins / decided * 100, 1) if decided > 0 else 0.0
    roi = round(total_pnl / max(graded_count * stake, 1) * 100, 1) if graded_count > 0 else 0.0
    avg_edge = round(sum_edge / max(total_bets, 1) * 100, 1)
    avg_clv = round(sum_clv / max(clv_count, 1) * 100, 2) if clv_count > 0 else 0.0
    current_bankroll = initial_bankroll + total_pnl

    # Build equity curve: chronological running balance
    equity_curve = []
    running = initial_bankroll
    # Sort graded bets by game_time (or graded_at)
    graded_bets_for_curve.sort(key=lambda x: x["game_time"] or x["graded_at"] or datetime.min)
    for gb in graded_bets_for_curve:
        running += gb["pnl"]
        dt = gb["game_time"] or gb["graded_at"]
        equity_curve.append({
            "date": dt.strftime("%b %d") if dt else "?",
            "value": round(running, 2),
        })
    # If no graded bets yet, show flat line
    if not equity_curve:
        today = date.today()
        for i in range(30, -1, -1):
            d = today - timedelta(days=i)
            equity_curve.append({"date": d.strftime("%b %d"), "value": initial_bankroll})

    stats = BettingSummaryStats(
        initial_bankroll=initial_bankroll,
        current_bankroll=round(current_bankroll, 2),
        total_bets=total_bets,
        graded_bets=graded_count,
        pending_bets=total_pending,
        wins=total_wins,
        losses=total_losses,
        pushes=total_pushes,
        win_rate=win_rate,
        roi=roi,
        avg_edge=avg_edge,
        avg_clv=avg_clv,
        total_pnl=round(total_pnl, 2),
    )

    return BettingSummaryResponse(stats=stats, bets=bets, equity_curve=equity_curve)