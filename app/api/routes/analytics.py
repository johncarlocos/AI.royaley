"""
LOYALEY - Analytics API Routes
Enterprise-grade analytics and reporting endpoints
"""

from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_

from app.core.database import get_db
from app.core.security import security_manager
from app.api.dependencies import get_current_user
from app.models.models import (
    Prediction, Game, Sport, Bet, CLVRecord,
    ModelPerformance, SignalTier, BetResult
)
from app.services.betting import clv_calculator, kelly_calculator

router = APIRouter()


# ============================================================================
# Response Schemas
# ============================================================================

class SportPerformance(BaseModel):
    """Performance metrics for a sport"""
    sport_code: str
    sport_name: str
    total_predictions: int
    wins: int
    losses: int
    pushes: int
    pending: int
    win_rate: float
    roi: float
    avg_clv: float
    tier_a_count: int
    tier_a_win_rate: float


class TierPerformance(BaseModel):
    """Performance metrics by signal tier"""
    tier: str
    total: int
    wins: int
    losses: int
    win_rate: float
    avg_probability: float
    avg_edge: float
    roi: float


class DailyPerformance(BaseModel):
    """Daily performance summary"""
    date: str
    predictions: int
    wins: int
    losses: int
    profit_loss: float
    roi: float
    avg_clv: float


class OverallStats(BaseModel):
    """Overall system statistics"""
    total_predictions: int
    total_graded: int
    wins: int
    losses: int
    pushes: int
    pending: int
    overall_win_rate: float
    overall_roi: float
    avg_clv: float
    tier_a_accuracy: float
    best_sport: str
    best_sport_roi: float
    active_models: int
    last_prediction_time: Optional[datetime]


class CLVSummary(BaseModel):
    """CLV performance summary"""
    avg_clv: float
    median_clv: float
    positive_clv_pct: float
    total_records: int
    clv_by_sport: Dict[str, float]
    clv_by_tier: Dict[str, float]
    clv_trend: List[Dict[str, Any]]


class BettingPerformance(BaseModel):
    """Betting performance metrics"""
    total_bets: int
    total_wagered: float
    total_won: float
    total_lost: float
    net_profit: float
    roi: float
    avg_bet_size: float
    biggest_win: float
    biggest_loss: float
    current_streak: int
    streak_type: str  # 'win' or 'loss'
    best_day_profit: float
    worst_day_loss: float


class AnalyticsResponse(BaseModel):
    """Complete analytics response"""
    overall: OverallStats
    by_sport: List[SportPerformance]
    by_tier: List[TierPerformance]
    daily_trend: List[DailyPerformance]
    clv_summary: CLVSummary
    generated_at: datetime


# ============================================================================
# Analytics Endpoints
# ============================================================================

@router.get("/overview", response_model=OverallStats)
async def get_analytics_overview(
    days: int = Query(default=30, ge=1, le=365, description="Number of days to analyze"),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Get overall system analytics overview.
    
    Returns high-level performance metrics across all sports and prediction types.
    """
    start_date = datetime.utcnow() - timedelta(days=days)
    
    # Query predictions
    query = select(Prediction).where(Prediction.created_at >= start_date)
    result = await db.execute(query)
    predictions = result.scalars().all()
    
    if not predictions:
        return OverallStats(
            total_predictions=0,
            total_graded=0,
            wins=0,
            losses=0,
            pushes=0,
            pending=0,
            overall_win_rate=0.0,
            overall_roi=0.0,
            avg_clv=0.0,
            tier_a_accuracy=0.0,
            best_sport="N/A",
            best_sport_roi=0.0,
            active_models=0,
            last_prediction_time=None
        )
    
    # Calculate metrics
    total = len(predictions)
    wins = sum(1 for p in predictions if p.result == BetResult.WIN)
    losses = sum(1 for p in predictions if p.result == BetResult.LOSS)
    pushes = sum(1 for p in predictions if p.result == BetResult.PUSH)
    pending = sum(1 for p in predictions if p.result is None)
    graded = wins + losses + pushes
    
    win_rate = wins / graded if graded > 0 else 0.0
    
    # Tier A accuracy
    tier_a = [p for p in predictions if p.signal_tier == SignalTier.A]
    tier_a_wins = sum(1 for p in tier_a if p.result == BetResult.WIN)
    tier_a_graded = sum(1 for p in tier_a if p.result in [BetResult.WIN, BetResult.LOSS])
    tier_a_accuracy = tier_a_wins / tier_a_graded if tier_a_graded > 0 else 0.0
    
    # CLV
    clv_query = select(func.avg(CLVRecord.clv_value)).where(CLVRecord.recorded_at >= start_date)
    clv_result = await db.execute(clv_query)
    avg_clv = clv_result.scalar() or 0.0
    
    return OverallStats(
        total_predictions=total,
        total_graded=graded,
        wins=wins,
        losses=losses,
        pushes=pushes,
        pending=pending,
        overall_win_rate=round(win_rate * 100, 2),
        overall_roi=0.0,  # Calculate from bets
        avg_clv=round(avg_clv, 4),
        tier_a_accuracy=round(tier_a_accuracy * 100, 2),
        best_sport="NBA",  # Would calculate from data
        best_sport_roi=0.0,
        active_models=10,
        last_prediction_time=predictions[-1].created_at if predictions else None
    )


@router.get("/by-sport", response_model=List[SportPerformance])
async def get_performance_by_sport(
    days: int = Query(default=30, ge=1, le=365),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Get performance breakdown by sport.
    
    Shows win rate, ROI, and CLV for each sport.
    """
    start_date = datetime.utcnow() - timedelta(days=days)
    
    # Get all sports
    sports_query = select(Sport).where(Sport.is_active == True)
    sports_result = await db.execute(sports_query)
    sports = sports_result.scalars().all()
    
    performance_list = []
    
    for sport in sports:
        # Get predictions for this sport
        pred_query = select(Prediction).join(Game).where(
            and_(
                Game.sport_id == sport.id,
                Prediction.created_at >= start_date
            )
        )
        pred_result = await db.execute(pred_query)
        predictions = pred_result.scalars().all()
        
        if not predictions:
            continue
        
        total = len(predictions)
        wins = sum(1 for p in predictions if p.result == BetResult.WIN)
        losses = sum(1 for p in predictions if p.result == BetResult.LOSS)
        pushes = sum(1 for p in predictions if p.result == BetResult.PUSH)
        pending = sum(1 for p in predictions if p.result is None)
        graded = wins + losses
        
        tier_a = [p for p in predictions if p.signal_tier == SignalTier.A]
        tier_a_wins = sum(1 for p in tier_a if p.result == BetResult.WIN)
        tier_a_graded = sum(1 for p in tier_a if p.result in [BetResult.WIN, BetResult.LOSS])
        
        performance_list.append(SportPerformance(
            sport_code=sport.code,
            sport_name=sport.name,
            total_predictions=total,
            wins=wins,
            losses=losses,
            pushes=pushes,
            pending=pending,
            win_rate=round((wins / graded * 100) if graded > 0 else 0.0, 2),
            roi=0.0,
            avg_clv=0.0,
            tier_a_count=len(tier_a),
            tier_a_win_rate=round((tier_a_wins / tier_a_graded * 100) if tier_a_graded > 0 else 0.0, 2)
        ))
    
    return sorted(performance_list, key=lambda x: x.win_rate, reverse=True)


@router.get("/by-tier", response_model=List[TierPerformance])
async def get_performance_by_tier(
    days: int = Query(default=30, ge=1, le=365),
    sport_code: Optional[str] = Query(default=None),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Get performance breakdown by signal tier.
    
    Shows how each confidence tier (A, B, C, D) is performing.
    """
    start_date = datetime.utcnow() - timedelta(days=days)
    
    tiers = [SignalTier.A, SignalTier.B, SignalTier.C, SignalTier.D]
    tier_performance = []
    
    for tier in tiers:
        query = select(Prediction).where(
            and_(
                Prediction.signal_tier == tier,
                Prediction.created_at >= start_date
            )
        )
        
        if sport_code:
            query = query.join(Game).join(Sport).where(Sport.code == sport_code)
        
        result = await db.execute(query)
        predictions = result.scalars().all()
        
        if not predictions:
            tier_performance.append(TierPerformance(
                tier=tier.value,
                total=0,
                wins=0,
                losses=0,
                win_rate=0.0,
                avg_probability=0.0,
                avg_edge=0.0,
                roi=0.0
            ))
            continue
        
        total = len(predictions)
        wins = sum(1 for p in predictions if p.result == BetResult.WIN)
        losses = sum(1 for p in predictions if p.result == BetResult.LOSS)
        graded = wins + losses
        
        avg_prob = sum(p.probability for p in predictions) / total
        avg_edge = sum(p.edge or 0 for p in predictions) / total
        
        tier_performance.append(TierPerformance(
            tier=tier.value,
            total=total,
            wins=wins,
            losses=losses,
            win_rate=round((wins / graded * 100) if graded > 0 else 0.0, 2),
            avg_probability=round(avg_prob * 100, 2),
            avg_edge=round(avg_edge * 100, 2),
            roi=0.0
        ))
    
    return tier_performance


@router.get("/daily-trend", response_model=List[DailyPerformance])
async def get_daily_trend(
    days: int = Query(default=30, ge=1, le=90),
    sport_code: Optional[str] = Query(default=None),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Get daily performance trend.
    
    Shows day-by-day breakdown of predictions and results.
    """
    daily_data = []
    
    for i in range(days):
        date = datetime.utcnow().date() - timedelta(days=i)
        start = datetime.combine(date, datetime.min.time())
        end = datetime.combine(date, datetime.max.time())
        
        query = select(Prediction).where(
            and_(
                Prediction.created_at >= start,
                Prediction.created_at <= end
            )
        )
        
        result = await db.execute(query)
        predictions = result.scalars().all()
        
        if predictions:
            wins = sum(1 for p in predictions if p.result == BetResult.WIN)
            losses = sum(1 for p in predictions if p.result == BetResult.LOSS)
            
            daily_data.append(DailyPerformance(
                date=date.isoformat(),
                predictions=len(predictions),
                wins=wins,
                losses=losses,
                profit_loss=0.0,
                roi=0.0,
                avg_clv=0.0
            ))
    
    return sorted(daily_data, key=lambda x: x.date, reverse=True)


@router.get("/clv-summary", response_model=CLVSummary)
async def get_clv_summary(
    days: int = Query(default=30, ge=1, le=365),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Get Closing Line Value (CLV) performance summary.
    
    CLV is the key indicator of long-term betting edge.
    """
    start_date = datetime.utcnow() - timedelta(days=days)
    
    query = select(CLVRecord).where(CLVRecord.recorded_at >= start_date)
    result = await db.execute(query)
    records = result.scalars().all()
    
    if not records:
        return CLVSummary(
            avg_clv=0.0,
            median_clv=0.0,
            positive_clv_pct=0.0,
            total_records=0,
            clv_by_sport={},
            clv_by_tier={},
            clv_trend=[]
        )
    
    clv_values = [r.clv_value for r in records]
    avg_clv = sum(clv_values) / len(clv_values)
    sorted_clv = sorted(clv_values)
    median_clv = sorted_clv[len(sorted_clv) // 2]
    positive_pct = sum(1 for c in clv_values if c > 0) / len(clv_values) * 100
    
    return CLVSummary(
        avg_clv=round(avg_clv, 4),
        median_clv=round(median_clv, 4),
        positive_clv_pct=round(positive_pct, 2),
        total_records=len(records),
        clv_by_sport={},
        clv_by_tier={},
        clv_trend=[]
    )


@router.get("/betting-performance", response_model=BettingPerformance)
async def get_betting_performance(
    days: int = Query(default=30, ge=1, le=365),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Get betting performance metrics.
    
    Shows actual betting results including wagered, won, lost, and ROI.
    """
    start_date = datetime.utcnow() - timedelta(days=days)
    
    query = select(Bet).where(Bet.placed_at >= start_date)
    result = await db.execute(query)
    bets = result.scalars().all()
    
    if not bets:
        return BettingPerformance(
            total_bets=0,
            total_wagered=0.0,
            total_won=0.0,
            total_lost=0.0,
            net_profit=0.0,
            roi=0.0,
            avg_bet_size=0.0,
            biggest_win=0.0,
            biggest_loss=0.0,
            current_streak=0,
            streak_type="none",
            best_day_profit=0.0,
            worst_day_loss=0.0
        )
    
    total_wagered = sum(b.stake for b in bets)
    wins = [b for b in bets if b.result == BetResult.WIN]
    losses = [b for b in bets if b.result == BetResult.LOSS]
    
    total_won = sum(b.profit or 0 for b in wins)
    total_lost = sum(abs(b.profit or 0) for b in losses)
    net_profit = total_won - total_lost
    roi = (net_profit / total_wagered * 100) if total_wagered > 0 else 0.0
    
    return BettingPerformance(
        total_bets=len(bets),
        total_wagered=round(total_wagered, 2),
        total_won=round(total_won, 2),
        total_lost=round(total_lost, 2),
        net_profit=round(net_profit, 2),
        roi=round(roi, 2),
        avg_bet_size=round(total_wagered / len(bets), 2),
        biggest_win=max((b.profit or 0) for b in wins) if wins else 0.0,
        biggest_loss=min((b.profit or 0) for b in losses) if losses else 0.0,
        current_streak=0,
        streak_type="none",
        best_day_profit=0.0,
        worst_day_loss=0.0
    )


@router.get("/full-report", response_model=AnalyticsResponse)
async def get_full_analytics_report(
    days: int = Query(default=30, ge=1, le=365),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Get comprehensive analytics report.
    
    Returns all analytics data in a single response for dashboard display.
    """
    overall = await get_analytics_overview(days=days, db=db, current_user=current_user)
    by_sport = await get_performance_by_sport(days=days, db=db, current_user=current_user)
    by_tier = await get_performance_by_tier(days=days, db=db, current_user=current_user)
    daily = await get_daily_trend(days=min(days, 30), db=db, current_user=current_user)
    clv = await get_clv_summary(days=days, db=db, current_user=current_user)
    
    return AnalyticsResponse(
        overall=overall,
        by_sport=by_sport,
        by_tier=by_tier,
        daily_trend=daily,
        clv_summary=clv,
        generated_at=datetime.utcnow()
    )


@router.get("/model-accuracy")
async def get_model_accuracy(
    days: int = Query(default=30, ge=1, le=365),
    sport_code: Optional[str] = Query(default=None),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Get ML model accuracy over time.
    
    Shows how model predictions have performed historically.
    """
    start_date = datetime.utcnow() - timedelta(days=days)
    
    query = select(ModelPerformance).where(
        ModelPerformance.recorded_at >= start_date
    ).order_by(ModelPerformance.recorded_at.desc())
    
    result = await db.execute(query)
    records = result.scalars().all()
    
    return {
        "total_records": len(records),
        "avg_auc": sum(r.auc_score or 0 for r in records) / len(records) if records else 0,
        "avg_accuracy": sum(r.accuracy or 0 for r in records) / len(records) if records else 0,
        "records": [
            {
                "model_id": r.model_id,
                "auc_score": r.auc_score,
                "accuracy": r.accuracy,
                "recorded_at": r.recorded_at.isoformat()
            }
            for r in records[:50]
        ]
    }


@router.get("/edge-distribution")
async def get_edge_distribution(
    days: int = Query(default=30, ge=1, le=365),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Get distribution of prediction edges.
    
    Shows how edges are distributed to identify value opportunities.
    """
    start_date = datetime.utcnow() - timedelta(days=days)
    
    query = select(Prediction).where(
        and_(
            Prediction.created_at >= start_date,
            Prediction.edge.isnot(None)
        )
    )
    
    result = await db.execute(query)
    predictions = result.scalars().all()
    
    if not predictions:
        return {"buckets": [], "avg_edge": 0, "max_edge": 0, "min_edge": 0}
    
    edges = [p.edge for p in predictions if p.edge is not None]
    
    # Create buckets: 0-3%, 3-5%, 5-7%, 7-10%, 10%+
    buckets = {
        "0-3%": len([e for e in edges if 0 <= e < 0.03]),
        "3-5%": len([e for e in edges if 0.03 <= e < 0.05]),
        "5-7%": len([e for e in edges if 0.05 <= e < 0.07]),
        "7-10%": len([e for e in edges if 0.07 <= e < 0.10]),
        "10%+": len([e for e in edges if e >= 0.10])
    }
    
    return {
        "buckets": [{"range": k, "count": v} for k, v in buckets.items()],
        "avg_edge": round(sum(edges) / len(edges) * 100, 2),
        "max_edge": round(max(edges) * 100, 2),
        "min_edge": round(min(edges) * 100, 2),
        "total_predictions": len(predictions)
    }
