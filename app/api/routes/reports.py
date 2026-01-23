"""
LOYALEY - Reports API Routes
Report endpoints for dashboard and analytics
"""

from datetime import datetime, timedelta
from typing import Optional
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from app.core.database import get_db
from app.api.dependencies import get_current_user
from app.models.models import Prediction, Bet, BetResult
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================================
# Response Schemas
# ============================================================================

class DailyReport(BaseModel):
    """Daily report data for dashboard"""
    win_rate: float = 0.0
    daily_pl: float = 0.0
    weekly_roi: float = 0.0


# ============================================================================
# Endpoints
# ============================================================================

@router.get("/daily", response_model=DailyReport)
async def get_daily_report(
    date: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Get daily report for dashboard.
    Returns win_rate, daily_pl, and weekly_roi.
    """
    # Check if demo user - return empty report for demo mode
    if hasattr(current_user, 'id') and str(current_user.id) == "00000000-0000-0000-0000-000000000000":
        return DailyReport(win_rate=0.0, daily_pl=0.0, weekly_roi=0.0)
    
    try:
        # Calculate daily P/L from today's bets
        today = datetime.utcnow().date()
        start = datetime.combine(today, datetime.min.time())
        end = datetime.combine(today, datetime.max.time())
        
        # Get today's bets
        bets_query = select(Bet).where(
            and_(
                Bet.placed_at >= start,
                Bet.placed_at <= end
            )
        )
        bets_result = await db.execute(bets_query)
        today_bets = bets_result.scalars().all()
        
        daily_pl = sum(bet.profit_loss or 0.0 for bet in today_bets)
        
        # Calculate win rate from recent predictions
        week_ago = datetime.utcnow() - timedelta(days=7)
        predictions_query = select(Prediction).where(
            Prediction.locked_at >= week_ago
        )
        preds_result = await db.execute(predictions_query)
        recent_predictions = preds_result.scalars().all()
        
        graded = [p for p in recent_predictions if p.is_graded]
        wins = sum(1 for p in graded if p.result == BetResult.WIN)
        win_rate = (wins / len(graded) * 100) if graded else 0.0
        
        # Calculate weekly ROI
        week_bets_query = select(Bet).where(Bet.placed_at >= week_ago)
        week_bets_result = await db.execute(week_bets_query)
        week_bets = week_bets_result.scalars().all()
        
        total_wagered = sum(bet.stake or 0.0 for bet in week_bets)
        total_pl = sum(bet.profit_loss or 0.0 for bet in week_bets)
        weekly_roi = (total_pl / total_wagered * 100) if total_wagered > 0 else 0.0
        
        return DailyReport(
            win_rate=win_rate,
            daily_pl=daily_pl,
            weekly_roi=weekly_roi
        )
    except Exception as e:
        # Return empty report on error (or for demo mode)
        if hasattr(current_user, 'id') and str(current_user.id) == "00000000-0000-0000-0000-000000000000":
            return DailyReport(win_rate=0.0, daily_pl=0.0, weekly_roi=0.0)
        # For real users with DB errors, still return empty but log the error
        logger.error(f"Error getting daily report: {e}")
        return DailyReport(win_rate=0.0, daily_pl=0.0, weekly_roi=0.0)

