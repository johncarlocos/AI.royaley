"""
ROYALEY - Public Predictions API
No authentication required - read-only access for frontend dashboard.
"""
from datetime import datetime, date
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

    count_result = await db.execute(text(f"SELECT COUNT(*) FROM predictions p JOIN games g ON p.game_id = g.id JOIN sports s ON g.sport_id = s.id WHERE 1=1 {where_sql}"), params)
    total = count_result.scalar() or 0
    offset = (page - 1) * per_page
    data_params = {**params, "lim": per_page, "off": offset}
    result = await db.execute(text(f"""
        SELECT p.id, p.game_id, s.code as sport_code, ht.name as home_team, at2.name as away_team,
            g.scheduled_at as game_time, p.bet_type, p.predicted_side, p.probability, p.edge,
            CAST(p.signal_tier AS TEXT) as signal_tier, p.line_at_prediction, p.odds_at_prediction,
            p.kelly_fraction, p.prediction_hash, p.created_at,
            pr.actual_result as result, pr.clv, pr.profit_loss
        FROM predictions p JOIN games g ON p.game_id = g.id JOIN sports s ON g.sport_id = s.id
        JOIN teams ht ON g.home_team_id = ht.id JOIN teams at2 ON g.away_team_id = at2.id
        LEFT JOIN prediction_results pr ON pr.prediction_id = p.id
        WHERE 1=1 {where_sql}
        ORDER BY g.scheduled_at DESC, p.created_at DESC LIMIT :lim OFFSET :off
    """), data_params)
    predictions = []
    for row in result.fetchall():
        predictions.append(PublicPrediction(
            id=str(row.id), game_id=str(row.game_id), sport=row.sport_code,
            home_team=row.home_team, away_team=row.away_team,
            game_time=row.game_time.isoformat() if row.game_time else None,
            bet_type=row.bet_type, predicted_side=row.predicted_side,
            probability=float(row.probability),
            edge=float(row.edge) if row.edge else None,
            signal_tier=row.signal_tier,
            line_at_prediction=float(row.line_at_prediction) if row.line_at_prediction else None,
            odds_at_prediction=int(row.odds_at_prediction) if row.odds_at_prediction else None,
            kelly_fraction=float(row.kelly_fraction) if row.kelly_fraction else None,
            prediction_hash=row.prediction_hash,
            created_at=row.created_at.isoformat() if row.created_at else None,
            result=str(row.result) if row.result else "pending",
            clv=float(row.clv) if row.clv else None,
            profit_loss=float(row.profit_loss) if row.profit_loss else None,
        ))
    return PublicPredictionsResponse(predictions=predictions, total=total, page=page, per_page=per_page)

@router.get("/dashboard/stats", response_model=DashboardStats)
async def get_dashboard_stats(db: AsyncSession = Depends(get_db)):
    today = date.today()
    total_predictions = (await db.execute(text("SELECT COUNT(*) FROM predictions"))).scalar() or 0
    tier_a_count = (await db.execute(text("SELECT COUNT(*) FROM predictions WHERE CAST(signal_tier AS TEXT) = 'A'"))).scalar() or 0
    pending_count = (await db.execute(text("SELECT COUNT(*) FROM predictions p LEFT JOIN prediction_results pr ON pr.prediction_id = p.id WHERE pr.id IS NULL"))).scalar() or 0
    graded_today = (await db.execute(text("SELECT COUNT(*) FROM prediction_results WHERE DATE(graded_at) = :today"), {"today": today})).scalar() or 0

    win_rate = 0.0
    wr_row = (await db.execute(text("SELECT COUNT(*) FILTER (WHERE actual_result = 'win') as wins, COUNT(*) as total FROM prediction_results"))).fetchone()
    if wr_row and wr_row.total > 0:
        win_rate = round((wr_row.wins / wr_row.total) * 100, 1)

    roi = 0.0
    roi_row = (await db.execute(text("SELECT COALESCE(SUM(profit_loss), 0) as total_pl, COUNT(*) as total_bets FROM prediction_results"))).fetchone()
    if roi_row and roi_row.total_bets > 0:
        roi = round((roi_row.total_pl / roi_row.total_bets) * 100, 1)

    # Top picks (upcoming, highest edge)
    top_rows = (await db.execute(text("""
        SELECT p.id, s.code as sport, ht.name as home_team, at2.name as away_team,
            p.bet_type, p.predicted_side, p.probability, p.edge,
            CAST(p.signal_tier AS TEXT) as signal_tier, p.line_at_prediction, g.scheduled_at as game_time
        FROM predictions p JOIN games g ON p.game_id = g.id JOIN sports s ON g.sport_id = s.id
        JOIN teams ht ON g.home_team_id = ht.id JOIN teams at2 ON g.away_team_id = at2.id
        LEFT JOIN prediction_results pr ON pr.prediction_id = p.id
        WHERE pr.id IS NULL AND g.scheduled_at >= NOW() - INTERVAL '2 hours'
        ORDER BY p.edge DESC NULLS LAST, p.probability DESC LIMIT 6
    """))).fetchall()
    top_picks = []
    for row in top_rows:
        side, line = row.predicted_side or "", row.line_at_prediction
        if row.bet_type == "spread":
            team = row.home_team if side == "home" else row.away_team
            pick_str = f"{team} {'+' if line and line > 0 else ''}{line}" if line else team
        elif row.bet_type == "total":
            pick_str = f"{'Over' if side == 'over' else 'Under'} {line}" if line else side.capitalize()
        else:
            pick_str = f"{row.home_team if side == 'home' else row.away_team} ML"
        top_picks.append({"sport": row.sport, "game": f"{row.away_team} vs {row.home_team}", "pick": pick_str,
            "tier": row.signal_tier or "D", "time": row.game_time.strftime("%-I:%M %p") if row.game_time else ""})

    # Recent graded activity
    recent_rows = (await db.execute(text("""
        SELECT s.code as sport, ht.name as home_team, at2.name as away_team,
            p.bet_type, p.predicted_side, p.line_at_prediction, pr.actual_result, pr.profit_loss, pr.graded_at
        FROM prediction_results pr JOIN predictions p ON pr.prediction_id = p.id
        JOIN games g ON p.game_id = g.id JOIN sports s ON g.sport_id = s.id
        JOIN teams ht ON g.home_team_id = ht.id JOIN teams at2 ON g.away_team_id = at2.id
        ORDER BY pr.graded_at DESC LIMIT 5
    """))).fetchall()
    recent_activity = []
    for row in recent_rows:
        side, line = row.predicted_side or "", row.line_at_prediction
        team = row.home_team if side == "home" else row.away_team
        r = str(row.actual_result).upper() if row.actual_result else "PENDING"
        pl = row.profit_loss or 0
        desc = f"{team} {'+' if line and line > 0 else ''}{line}" if row.bet_type == "spread" and line else (f"{'Over' if side == 'over' else 'Under'} {line}" if row.bet_type == "total" and line else f"{team} ML")
        diff = datetime.utcnow() - row.graded_at if row.graded_at else None
        t = f"{int(diff.total_seconds()/60)}m ago" if diff and diff.total_seconds() < 3600 else (f"{int(diff.total_seconds()/3600)}h ago" if diff and diff.total_seconds() < 86400 else f"{int(diff.total_seconds()/86400)}d ago") if diff else ""
        icon = "win" if r == "WIN" else ("loss" if r == "LOSS" else "pending")
        recent_activity.append({"icon": icon, "text": f"{desc} {r} ({'+' if pl >= 0 else ''}{pl:.1f} units)", "time": t})

    # Best performers by tier (tier A win rate, etc.)
    bp_rows = (await db.execute(text("""
        SELECT CAST(p.signal_tier AS TEXT) as tier,
            COUNT(*) FILTER (WHERE pr.actual_result = 'win') as wins,
            COUNT(*) as total,
            ROUND(AVG(pr.profit_loss)::numeric, 2) as avg_pl
        FROM predictions p JOIN prediction_results pr ON pr.prediction_id = p.id
        GROUP BY p.signal_tier ORDER BY tier
    """))).fetchall()
    best_performers = []
    areas_to_monitor = []
    for row in bp_rows:
        wr = round(row.wins / row.total * 100, 1) if row.total > 0 else 0
        item = {"label": f"Tier {row.tier} Predictions", "value": f"{wr}% Win Rate", "color": "success" if wr >= 55 else ("warning" if wr >= 50 else "error")}
        if wr >= 55:
            best_performers.append(item)
        else:
            areas_to_monitor.append(item)

    return DashboardStats(
        total_predictions=total_predictions, tier_a_count=tier_a_count, pending_count=pending_count,
        graded_today=graded_today, win_rate=win_rate, roi=roi,
        top_picks=top_picks, recent_activity=recent_activity,
        best_performers=best_performers, areas_to_monitor=areas_to_monitor,
    )
